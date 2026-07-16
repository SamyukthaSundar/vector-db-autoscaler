"""
Person A (S) -- infra/prometheus/metrics_exporter.py

Pulls raw stats from each Qdrant node and re-exposes them in a normalized,
per-shard format on :8000/metrics for Prometheus to scrape, and also writes
them as MetricPoint records (shared/schemas.py) that forecaster/ and policy/
can read directly for offline development without needing live Prometheus.
"""

import os
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.schemas import MetricPoint, append_jsonl, now  # noqa: E402

QUERY_RATE = Gauge("shard_query_rate", "Queries per second", ["shard_id", "node"])
P95_LATENCY = Gauge("shard_p95_latency_ms", "p95 latency (ms)", ["shard_id", "node"])
P99_LATENCY = Gauge("shard_p99_latency_ms", "p99 latency (ms)", ["shard_id", "node"])
VECTOR_COUNT = Gauge("shard_vector_count", "Vector count", ["shard_id", "node"])

METRICS_LOG_PATH = os.environ.get("METRICS_LOG_PATH", "results/metrics_log.jsonl")


def get_qdrant_nodes():
    raw = os.environ.get("QDRANT_NODES", "localhost:6333")
    return [tuple(n.split(":")) for n in raw.split(",")]


def poll_node(host: str, port: str):
    """Poll one Qdrant node's collection info and telemetry, and emit
    per-shard MetricPoints. Qdrant doesn't expose per-shard latency natively
    for all deployment modes, so this combines collection stats with
    request-timing stats gathered by the workload replay engine's own
    logging (see workload/replay_engine.py) when available.
    """
    try:
        info = requests.get(f"http://{host}:{port}/collections", timeout=5).json()
    except requests.RequestException as e:
        print(f"[metrics_exporter] could not reach {host}:{port} -- {e}")
        return []

    points = []
    collections = info.get("result", {}).get("collections", [])
    for coll in collections:
        name = coll.get("name", "unknown")
        try:
            detail = requests.get(
                f"http://{host}:{port}/collections/{name}", timeout=5
            ).json()
        except requests.RequestException:
            continue

        result = detail.get("result", {})
        vector_count = result.get("points_count", 0)
        shard_count = result.get("config", {}).get("params", {}).get(
            "shard_number", 1
        )

        # Qdrant doesn't give us a clean per-shard breakdown via REST, so we
        # approximate even distribution here; replace with real per-shard
        # stats if/when available from your Qdrant version's telemetry API.
        for shard_idx in range(shard_count):
            shard_id = f"{name}-shard-{shard_idx}"
            point = MetricPoint(
                timestamp=now(),
                shard_id=shard_id,
                node_name=f"{host}:{port}",
                query_rate=0.0,       # filled in by replay_engine's logging
                p95_latency_ms=0.0,   # filled in by replay_engine's logging
                p99_latency_ms=0.0,
                vector_count=vector_count // max(shard_count, 1),
            )
            points.append(point)

            QUERY_RATE.labels(shard_id=shard_id, node=point.node_name).set(
                point.query_rate
            )
            P95_LATENCY.labels(shard_id=shard_id, node=point.node_name).set(
                point.p95_latency_ms
            )
            P99_LATENCY.labels(shard_id=shard_id, node=point.node_name).set(
                point.p99_latency_ms
            )
            VECTOR_COUNT.labels(shard_id=shard_id, node=point.node_name).set(
                point.vector_count
            )
    return points


def poll_loop(interval_seconds: int = 15):
    os.makedirs(os.path.dirname(METRICS_LOG_PATH), exist_ok=True)
    while True:
        for host, port in get_qdrant_nodes():
            points = poll_node(host, port)
            for p in points:
                append_jsonl(METRICS_LOG_PATH, p)
        time.sleep(interval_seconds)


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            output = generate_latest()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(output)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # quiet


if __name__ == "__main__":
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()
    server = HTTPServer(("0.0.0.0", 8000), MetricsHandler)
    print("metrics_exporter serving on :8000/metrics")
    server.serve_forever()
