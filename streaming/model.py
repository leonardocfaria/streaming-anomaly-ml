"""Online anomaly model (per-feature EWMA mean/variance, z-score) + Page-Hinkley drift detection + windowed retraining."""
import math
from collections import deque


class EWMA:
    """Exponentially weighted mean and variance, O(1) update per event."""

    def __init__(self, alpha: float, mean: float = 0.0, var: float = 1.0):
        self.alpha, self.mean, self.var = alpha, mean, var

    def z(self, x: float) -> float:
        return (x - self.mean) / math.sqrt(self.var + 1e-12)

    def update(self, x: float) -> None:
        d = x - self.mean
        self.mean += self.alpha * d
        self.var = (1 - self.alpha) * (self.var + self.alpha * d * d)


class PageHinkley:
    """Detects a sustained shift in the mean of a signal (both directions): cumulative deviation from the
    running mean beyond `delta`, alarm when it exceeds `threshold`."""

    def __init__(self, delta: float = 0.5, threshold: float = 50.0):
        self.delta, self.threshold = delta, threshold
        self.reset()

    def reset(self):
        self.n, self.mean, self.up, self.down, self.min_up, self.max_down = 0, 0.0, 0.0, 0.0, 0.0, 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.up += x - self.mean - self.delta
        self.down += x - self.mean + self.delta
        self.min_up, self.max_down = min(self.min_up, self.up), max(self.max_down, self.down)
        return self.up - self.min_up > self.threshold or self.max_down - self.down > self.threshold


class Detector:
    """Scores each event, then updates. score = max |z| over features; anomaly if score > threshold.
    Flagged anomalies do not update the model (an attack or outage must not become the new normal)."""

    def __init__(self, n_features: int, alpha: float = 0.01, threshold: float = 5.0, window: int = 300,
                 drift_handling: bool = True, warmup: int = 200, ph_delta: float = 1.0, ph_threshold: float = 50.0):
        # ph_delta=1.0: seasonality + EWMA lag leave a ~0.6 sigma residual; delta 0.5 let it accumulate into a
        # spurious "drift" every ~1000 events (chosen on seeds 0-2; tests use seed 3)
        self.threshold, self.warmup, self.drift_handling = threshold, warmup, drift_handling
        self.alpha, self.ph_args = alpha, (ph_delta, ph_threshold)
        self.models = [EWMA(alpha) for _ in range(n_features)]
        self.ph = [PageHinkley(*self.ph_args) for _ in range(n_features)]
        self.window = deque(maxlen=window)  # recent events (all of them): the retraining set after a drift
        self.n = 0
        self.drifts: list[int] = []
        self.updates = 0

    def _fit(self, rows):
        for j, m in enumerate(self.models):
            col = [r[j] for r in rows]
            m.mean = sum(col) / len(col)
            m.var = max(sum((v - m.mean) ** 2 for v in col) / len(col), 1e-6)

    def process(self, x: list[float]) -> tuple[float, bool]:
        """Returns (score, is_anomaly) and updates the model."""
        self.n += 1
        if self.n <= self.warmup:  # cold start: learn only
            self.window.append(x)
            if self.n == self.warmup:
                self._fit(self.window)
            return 0.0, False
        zs = [m.z(v) for m, v in zip(self.models, x)]
        score = max(abs(z) for z in zs)
        anomaly = score > self.threshold
        if self.drift_handling and any(ph.update(max(min(z, 10.0), -10.0)) for ph, z in zip(self.ph, zs)):
            # Drift: the recent stream no longer matches the model. Retrain on the most recent events
            # (including those just flagged: after a level shift they are the new normal) and reset detectors.
            self.drifts.append(self.n)
            self.window.append(x)
            recent = list(self.window)[-max(30, len(self.window) // 4):]
            self._fit(recent)
            for ph in self.ph:
                ph.reset()
            return score, False
        self.window.append(x)
        if not anomaly:
            for m, v in zip(self.models, x):
                m.update(v)
            self.updates += 1
        return score, anomaly
