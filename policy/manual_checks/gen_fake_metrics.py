"""
Throwaway script -- NOT part of the deliverable. Generates a synthetic
results/metrics_log.jsonl so policy/ can be exercised end-to-end without a
live Qdrant cluster. Mimics what infra/metrics_exporter.py + Prometheus would
have produced from a real replay.
"""
import random
import time

from shared.schemas import MetricPoint, append_jsonl

random.seed(7)

NODES = ["qdrant-node-1", "qdrant-node-2", "qdrant-node-3"]
OUT_PATH = "results/metrics_log.jsonl"

# per-node query-rate profile (qps) so nodes clearly differ:
# node-1 stays cold, node-2 hovers mid, node-3 trends hot -> should trigger
# scale_up on node-3 and (once far enough apart) a migration from node-3 to node-1
PROFILES = {
    "qdrant-node-1": lambda t: 15 + random.uniform(-3, 3),
    "qdrant-node-2": lambda t: 45 + random.uniform(-5, 5),
    "qdrant-node-3": lambda t: 60 + t * 0.6 + random.uniform(-4, 4),  # ramps up over time
}

now = time.time()
N_TICKS = 40
INTERVAL = 15  # seconds between synthetic samples

for i in range(N_TICKS):
    ts = now - (N_TICKS - i) * INTERVAL
    for node in NODES:
        rate = max(0.0, PROFILES[node](i))
        point = MetricPoint(
            timestamp=ts,
            shard_id=f"{node}-shard-agg",
            node_name=node,
            query_rate=rate,
            p95_latency_ms=80 + rate * 1.2,
            p99_latency_ms=120 + rate * 1.8,
            vector_count=100_000,
            cpu_util_percent=min(95.0, rate * 1.1),
        )
        append_jsonl(OUT_PATH, point)

print(f"Wrote {N_TICKS * len(NODES)} synthetic MetricPoint rows to {OUT_PATH}")