"""
Person B (J) -- workload/replay_engine.py

Replays a parsed/generated trace against the live Qdrant cluster, logging
actual observed latency per request so policy/eval/eval_harness.py (Person D)
can compute real SLA-violation numbers rather than simulated ones.
"""

import argparse
import time
import uuid
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from shared.schemas import MetricPoint, append_jsonl, now

METRICS_LOG_PATH = "results/metrics_log.jsonl"


def load_scenario(name: str) -> pd.DataFrame:
    path = f"workload/trace_data/scenario_{name}.csv"
    return pd.read_csv(path)


def replay(df: pd.DataFrame, client: QdrantClient, collection_name: str,
           vector_size: int = 128, speed_multiplier: float = 60.0,
           node_name: str = None):
    """Replay the trace. `speed_multiplier` compresses trace time into wall
    time (default 60x -- an hour of trace plays out in a minute), since
    running traces at real-time speed isn't practical for iteration.
    """
    latencies = []
    prev_t = df["timestamp"].iloc[0]

    for _, row in df.iterrows():
        t, rate = row["timestamp"], row["query_rate"]
        sleep_time = max(0, (t - prev_t) / speed_multiplier)
        time.sleep(min(sleep_time, 2.0))  # cap sleep for fast local iteration
        prev_t = t

        n_queries = max(1, int(rate))
        for _ in range(n_queries):
            vec = np.random.rand(vector_size).tolist()
            start = time.perf_counter()
            try:
                client.search(
                    collection_name=collection_name,
                    query_vector=vec,
                    limit=10,
                )
            except Exception as e:
                print(f"[replay_engine] query failed: {e}")
                continue
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        if latencies:
            p95 = float(np.percentile(latencies[-500:], 95))
            p99 = float(np.percentile(latencies[-500:], 99))
            point = MetricPoint(
                timestamp=now(),
                shard_id=f"{node_name}-shard-agg" if node_name else f"{collection_name}-aggregate",
                node_name=node_name or "replay_engine",
                query_rate=rate,
                p95_latency_ms=p95,
                p99_latency_ms=p99,
                vector_count=0,
            )
            append_jsonl(METRICS_LOG_PATH, point)

    print(f"Replay complete. {len(latencies)} total queries issued.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="steady_state",
                     choices=["steady_state", "diurnal", "bursty"])
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--collection", default="vectors_main")
    ap.add_argument("--speed", type=float, default=60.0)
    ap.add_argument("--node-name", default=None,
                     help="e.g. qdrant-node-1 -- tags metrics as {node-name}-shard-agg "
                          "so policy/ picks them up correctly")
    args = ap.parse_args()

    df = load_scenario(args.scenario)
    client = QdrantClient(host=args.host, port=args.port)
    replay(df, client, args.collection, speed_multiplier=args.speed,
           node_name=args.node_name)