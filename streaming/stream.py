"""Seeded synthetic telemetry: latency_ms and payload_kb with daily seasonality, injected anomalies and an
optional abrupt level shift (concept drift, e.g. a deploy that makes every request slower)."""
import math
import random

FEATURES = ["latency_ms", "payload_kb"]


def events(n: int, seed: int = 0, anomaly_rate: float = 0.005, shift_at: int | None = None, shift: float = 0.6,
           period: int = 2000):
    """Yields (t, [latency, payload], is_anomaly). After shift_at, latency is multiplied by (1 + shift)."""
    rng = random.Random(seed)
    for t in range(n):
        season = 1 + 0.1 * math.sin(2 * math.pi * t / period)
        latency = 100 * season * (1 + shift if shift_at is not None and t >= shift_at else 1) + rng.gauss(0, 5)
        payload = 20 + rng.gauss(0, 2)
        anomaly = rng.random() < anomaly_rate
        if anomaly:  # a spike in one feature, big enough to matter operationally
            if rng.random() < 0.5:
                latency *= rng.uniform(1.5, 3.0)
            else:
                payload *= rng.uniform(2.0, 4.0)
        yield t, [latency, payload], anomaly
