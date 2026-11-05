"""Producer -> bounded queue -> consumer (stdlib threads), with Prometheus-format metrics over HTTP.

shortcut: stdlib queue instead of Kafka/Redpanda (the SPEC's fallback); swap produce()/consume() for a
Kafka producer/consumer when events must cross processes or survive restarts.
"""
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .model import Detector

STOP = object()


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.events = self.anomalies = self.drifts = 0
        self.latency_sum = 0.0
        self.queue_depth = 0

    def prometheus(self) -> str:
        with self.lock:
            return ("# TYPE stream_events_total counter\n" f"stream_events_total {self.events}\n"
                    "# TYPE stream_anomalies_total counter\n" f"stream_anomalies_total {self.anomalies}\n"
                    "# TYPE stream_drift_detections_total counter\n" f"stream_drift_detections_total {self.drifts}\n"
                    "# TYPE stream_queue_depth gauge\n" f"stream_queue_depth {self.queue_depth}\n"
                    "# TYPE stream_processing_seconds summary\n"
                    f"stream_processing_seconds_sum {self.latency_sum:.6f}\nstream_processing_seconds_count {self.events}\n")


def produce(q: queue.Queue, stream, errors: list):
    try:
        for item in stream:
            if errors:
                break  # the consumer failed: stop producing
            q.put(item)  # blocks when the queue is full: backpressure instead of unbounded memory
    except Exception as e:
        errors.append(e)
    finally:
        q.put(STOP)  # always: the consumer must never wait forever


def consume(q: queue.Queue, detector: Detector, metrics: Metrics, results: list, errors: list):
    while (item := q.get()) is not STOP:
        if errors:
            continue  # after a failure keep draining so the producer never blocks on a full queue
        try:
            _handle(item, detector, metrics, results, q)
        except Exception as e:
            errors.append(e)


def _handle(item, detector, metrics, results, q):
    t, x, label = item
    t0 = time.perf_counter()
    score, flagged = detector.process(x)
    dt = time.perf_counter() - t0
    with metrics.lock:
        metrics.events += 1
        metrics.anomalies += flagged
        metrics.drifts = len(detector.drifts)
        metrics.latency_sum += dt
        metrics.queue_depth = q.qsize()
    results.append((t, score, flagged, label))


def run(stream, detector: Detector, maxsize: int = 1000, metrics: Metrics | None = None) -> tuple[list, Metrics, float]:
    """Run producer and consumer threads to completion. Returns (results in order, metrics, seconds)."""
    q, metrics, results, errors = queue.Queue(maxsize), metrics or Metrics(), [], []
    t0 = time.perf_counter()
    threads = [threading.Thread(target=produce, args=(q, stream, errors)),
               threading.Thread(target=consume, args=(q, detector, metrics, results, errors))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    return results, metrics, time.perf_counter() - t0


def serve_metrics(metrics: Metrics, port: int = 9108) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = metrics.prometheus().encode()
            self.send_response(200 if self.path == "/metrics" else 404)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(body if self.path == "/metrics" else b"")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
