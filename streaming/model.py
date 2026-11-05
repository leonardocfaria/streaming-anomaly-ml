"""Online anomaly model (per-feature EWMA mean/variance, z-score) + Page-Hinkley drift detection + windowed retraining."""
import math
from statistics import median


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
    Flagged anomalies do not update the model (an attack or outage must not become the new normal).
    A sustained shift is caught by Page-Hinkley instead: the detector then collects the next `relearn` events
    (a bounded blind period: they are not judged) and refits on them only, robustly (median / MAD)."""

    def __init__(self, n_features: int, alpha: float = 0.01, threshold: float = 5.0, relearn: int = 50,
                 drift_handling: bool = True, warmup: int = 200, ph_delta: float = 1.0, ph_threshold: float = 50.0):
        # ph_delta=1.0: seasonality + EWMA lag leave a ~0.6 sigma residual; delta 0.5 let it accumulate into a
        # spurious "drift" every ~1000 events (chosen on seeds 0-2; tests use seed 3)
        self.threshold, self.relearn, self.drift_handling = threshold, relearn, drift_handling
        self.models = [EWMA(alpha) for _ in range(n_features)]
        self.ph = [PageHinkley(ph_delta, ph_threshold) for _ in range(n_features)]
        self.buffer, self.buffer_size = [], warmup  # warm-up is the first "relearn" period
        self.n, self.updates, self.blind = 0, 0, 0
        self.drifts: list[int] = []

    def _fit(self, rows):
        """Median / MAD per feature: a few spikes in the buffer cannot inflate the variance."""
        for j, m in enumerate(self.models):
            col = [r[j] for r in rows]
            med = median(col)
            m.mean, m.var = med, max((1.4826 * median(abs(v - med) for v in col)) ** 2, 1e-6)

    def process(self, x: list[float]) -> tuple[float, bool]:
        """Returns (score, is_anomaly) and updates the model."""
        self.n += 1
        if self.buffer is not None:  # warm-up or relearning after a drift: collect, don't judge
            self.buffer.append(x)
            self.blind += 1
            if len(self.buffer) >= self.buffer_size:
                self._fit(self.buffer)
                self.buffer = None
                for ph in self.ph:
                    ph.reset()
            return 0.0, False
        zs = [m.z(v) for m, v in zip(self.models, x)]
        score = max(abs(z) for z in zs)
        if self.drift_handling and any(ph.update(max(min(z, 10.0), -10.0)) for ph, z in zip(self.ph, zs)):
            self.drifts.append(self.n)
            self.buffer, self.buffer_size = [x], self.relearn  # refit on post-change data only
            self.blind += 1
            return score, False
        anomaly = score > self.threshold
        if not anomaly:
            for m, v in zip(self.models, x):
                m.update(v)
            self.updates += 1
        return score, anomaly
