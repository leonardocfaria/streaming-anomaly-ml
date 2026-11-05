# streaming-anomaly-ml

## Goal
Build "Streaming ML: online anomaly detection". Build a streaming pipeline (Redpanda/Kafka in docker, or a stdlib queue fallback) that scores events with an online model (river library or a hand-rolled streaming estimator), handles concept drift with windowed retraining, and exposes metrics. Keep it CPU-friendly (the cloud box has no GPU): small models and datasets, a fixed random seed, and a documented runtime.

## Definition of done
- [x] producer/consumer pipeline works locally
- [x] online model updates per event
- [x] drift handling demonstrated in a test
- [x] Tests pass with one command; CI workflow runs them
- [x] README with setup, usage, architecture and results
- [x] Self code review done
- [x] Draft PR open

