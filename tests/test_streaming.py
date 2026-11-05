import random
import urllib.request

from streaming.evaluate import summarise
from streaming.model import EWMA, Detector, PageHinkley
from streaming.pipeline import Metrics, run, serve_metrics
from streaming.stream import events


def test_ewma_converges():
    rng, m = random.Random(0), EWMA(0.01, mean=0, var=1)
    for _ in range(5000):
        m.update(rng.gauss(10, 2))
    assert abs(m.mean - 10) < 0.3 and abs(m.var ** 0.5 - 2) < 0.3


def test_page_hinkley_fires_on_shift_not_on_noise():
    rng = random.Random(1)
    ph = PageHinkley()
    assert not any(ph.update(rng.gauss(0, 1)) for _ in range(5000))
    ph.reset()
    hits = [ph.update(rng.gauss(0, 1) + (3 if t > 500 else 0)) for t in range(1000)]
    assert not any(hits[:500]) and any(hits[500:530])


def test_model_updates_per_event_but_not_on_anomalies():
    d = Detector(1, warmup=50)
    for x in [100.0 + (i % 5) for i in range(100)]:
        d.process([x])
    before, n_updates = d.models[0].mean, d.updates
    score, flagged = d.process([1000.0])
    assert flagged and d.models[0].mean == before and d.updates == n_updates  # anomaly: no update
    d.process([101.0])
    assert d.updates == n_updates + 1 and d.models[0].mean != before  # normal event: one update


def test_drift_handling_stops_false_alarms_and_keeps_catching_anomalies():
    kw = dict(n=12000, seed=3, shift_at=6000)
    with_dh = summarise(run(events(**kw), Detector(2))[0], 6000)
    without = summarise(run(events(**kw), Detector(2, drift_handling=False))[0], 6000)
    # without drift handling the model never adapts: nearly every post-shift event is a false alarm
    assert without["later_after_drift"]["false_alarms"] > 5000
    # with it, the detector re-learns within the first window and is quiet afterwards
    assert with_dh["later_after_drift"]["false_alarms"] < 20
    assert with_dh["later_after_drift"]["recall"] > 0.8 and with_dh["before_drift"]["recall"] > 0.8


def test_pipeline_processes_every_event_in_order_and_serves_metrics():
    metrics = Metrics()
    server = serve_metrics(metrics, port=0)
    results, metrics, _ = run(events(3000, seed=5), Detector(2), maxsize=10, metrics=metrics)
    assert [t for t, *_ in results] == list(range(3000))
    url = f"http://127.0.0.1:{server.server_address[1]}/metrics"
    text = urllib.request.urlopen(url).read().decode()
    server.shutdown()
    assert "stream_events_total 3000" in text and "stream_drift_detections_total" in text


def test_no_spurious_drift_on_stationary_seasonal_stream():
    d = Detector(2)
    run(events(20000, seed=4), d)  # seasonality + anomalies, no shift
    assert d.drifts == []
