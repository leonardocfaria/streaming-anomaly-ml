# streaming-anomaly-ml Implementation Plan

## Brainstorm (answered from SPEC + repo evidence)
- **What is it?** A streaming anomaly detector that keeps working when the stream changes: events flow producer → bounded queue → consumer; an online model scores and updates on every event; a drift detector triggers windowed retraining; metrics are exposed for Prometheus.
- **Transport:** the SPEC allows "Redpanda/Kafka in docker, or a stdlib queue fallback". The stdlib `queue.Queue(maxsize)` gives the same semantics needed here (ordered, backpressure, a sentinel to end the stream) with zero dependencies and no broker to babysit. The producer/consumer interface is two functions, so a Kafka client could replace the queue in about 10 lines (not done: time box).
- **Stream:** seeded synthetic service telemetry: `latency_ms` and `payload_kb`, with daily seasonality and noise. About 0.5 % of events are injected point anomalies (labelled). An abrupt **level shift** at a given event (e.g. a deploy makes latency +60 %) is the concept drift.
- **Online model:** per-feature EWMA mean and variance (α = 0.01), updated per event *after* scoring; score = max |z| across features; anomaly if score > 5. Anomalous events do not update the model (no poisoning).
- **Drift handling:** Page-Hinkley test on the standardised residual of each feature. On detection, retrain: re-initialise the mean/variance from the last W = 300 events (window) and reset the detector. Baseline for comparison: the same model with drift handling off.
- **Metrics:** events, anomalies flagged, drift detections, queue depth, per-event processing latency (exported as Prometheus text by a tiny `http.server`), and offline evaluation (precision/recall against injected labels, false alarms after the drift).
- **Non-goals:** a Kafka/Redpanda deployment, multi-partition consumers, exactly-once semantics.

## Files
- `streaming/model.py` (EWMA model, Page-Hinkley, `Detector`), `streaming/stream.py` (generator), `streaming/pipeline.py` (producer/consumer threads, metrics, HTTP exporter), `run.py`, tests.

## Tasks (TDD)
- [x] 1. EWMA mean/var converge; model updates on every normal event; anomalies don't update it.
- [x] 2. Page-Hinkley fires on a level shift, not on stationary noise.
- [x] 3. Drift test: after the shift, with drift handling false alarms stop within a few hundred events; without it they persist; injected anomalies are still caught.
- [x] 4. Threaded pipeline processes every event in order with a bounded queue; metrics endpoint; run.py report; README; CI.
