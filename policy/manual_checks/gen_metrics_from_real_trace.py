"""
Generates results/metrics_log.jsonl from the REAL Azure production trace
(workload/trace_data/azure_parsed.csv) instead of made-up numbers.

Still simulates the per-node split and p95/p99 latency, because that
breakdown genuinely only exists once workload/replay_engine.py has fired
real queries at a live Qdrant cluster and infra/metrics_exporter.py has
timed the responses -- the trace file itself is just one aggregate
query-rate signal, not per-shard telemetry.

What's real here: the query_rate SHAPE (same numbers N validated the
forecaster against).
What's simulated: splitting that single signal across 3 nodes with
different multipliers (to create a hot/cold imbalance worth testing
against), and the latency formula.
"""
import pandas as pd
import time

from shared.schemas import MetricPoint, append_jsonl

TRACE_PATH = "workload/trace_data/azure_parsed.csv"
OUT_PATH = "results/metrics_log.jsonl"

# Uneven multipliers per node -- simulates shard hotspotting (the exact
# problem predictive_rebalancing.py exists to catch), using real traffic
# shape rather than synthetic sine waves.
NODE_MULTIPLIERS = {
    "qdrant-node-1": 0.4,   # cooler shard
    "qdrant-node-2": 1.0,   # baseline -- matches the real trace as-is
    "qdrant-node-3": 1.6,   # hotter shard
}

df = pd.read_csv(TRACE_PATH)

# Change this to pick which 40-row slice of the real trace to replay:
#   None      -> the last 40 rows (whatever traffic level the trace happens to end on)
#   741       -> the busiest 40-row window in this file (~91 qps avg) -- good for
#                seeing scale_up / migrate_shard actually fire
WINDOW_END = 741

if WINDOW_END is not None:
    recent = df.iloc[WINDOW_END - 39: WINDOW_END + 1].reset_index(drop=True)
else:
    recent = df.tail(40).reset_index(drop=True)

now = time.time()
INTERVAL = 60  # azure_parsed.csv is sampled at 60s intervals

for i, row in recent.iterrows():
    ts = now - (len(recent) - i) * INTERVAL
    for node, mult in NODE_MULTIPLIERS.items():
        rate = max(0.0, row["query_rate"] * mult)
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

print(f"Wrote {len(recent) * len(NODE_MULTIPLIERS)} rows to {OUT_PATH}, "
      f"driven by real Azure trace query_rate (last {len(recent)} samples).")