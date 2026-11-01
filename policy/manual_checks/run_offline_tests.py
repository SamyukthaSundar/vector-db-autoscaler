"""
Runs Test 1 (synthetic data) and Test 2 (real Azure trace, quiet + busy
windows) back to back, printing clear section headers so the output is
easy to read. Nothing here touches Qdrant/Docker -- fully offline.
"""
import os
import pandas as pd
import time

from shared.schemas import MetricPoint, append_jsonl
from policy.baselines.static_provisioning import StaticProvisioningPolicy
from policy.baselines.reactive_threshold import ReactiveThresholdPolicy
from policy.predictive.predictive_scaling import PredictiveScalingPolicy
from policy.predictive.predictive_rebalancing import PredictiveRebalancingPolicy

METRICS_LOG = "results/metrics_log.jsonl"


def reset_logs():
    for f in os.listdir("results"):
        if f.endswith(".jsonl"):
            os.remove(os.path.join("results", f))


def run_all_policies():
    StaticProvisioningPolicy().run(duration_seconds=1, poll_interval=1)
    ReactiveThresholdPolicy().run(duration_seconds=1, poll_interval=1)
    PredictiveScalingPolicy().run(duration_seconds=1, poll_interval=1)
    PredictiveRebalancingPolicy().run(duration_seconds=1, poll_interval=1)


def gen_synthetic():
    import random
    random.seed(7)
    nodes_profiles = {
        "qdrant-node-1": lambda t: 15 + random.uniform(-3, 3),
        "qdrant-node-2": lambda t: 45 + random.uniform(-5, 5),
        "qdrant-node-3": lambda t: 60 + t * 0.6 + random.uniform(-4, 4),
    }
    now = time.time()
    for i in range(40):
        ts = now - (40 - i) * 15
        for node, fn in nodes_profiles.items():
            rate = max(0.0, fn(i))
            point = MetricPoint(
                timestamp=ts, shard_id=f"{node}-shard-agg", node_name=node,
                query_rate=rate, p95_latency_ms=80 + rate * 1.2,
                p99_latency_ms=120 + rate * 1.8, vector_count=100_000,
                cpu_util_percent=min(95.0, rate * 1.1),
            )
            append_jsonl(METRICS_LOG, point)


def gen_from_real_trace(window_end):
    df = pd.read_csv("workload/trace_data/azure_parsed.csv")
    if window_end is not None:
        recent = df.iloc[window_end - 39: window_end + 1].reset_index(drop=True)
    else:
        recent = df.tail(40).reset_index(drop=True)
    multipliers = {"qdrant-node-1": 0.4, "qdrant-node-2": 1.0, "qdrant-node-3": 1.6}
    now = time.time()
    for i, row in recent.iterrows():
        ts = now - (len(recent) - i) * 60
        for node, mult in multipliers.items():
            rate = max(0.0, row["query_rate"] * mult)
            point = MetricPoint(
                timestamp=ts, shard_id=f"{node}-shard-agg", node_name=node,
                query_rate=rate, p95_latency_ms=80 + rate * 1.2,
                p99_latency_ms=120 + rate * 1.8, vector_count=100_000,
                cpu_util_percent=min(95.0, rate * 1.1),
            )
            append_jsonl(METRICS_LOG, point)


print("\n" + "=" * 60)
print("TEST 1: SYNTHETIC DATA (cold / medium / ramping-hot nodes)")
print("=" * 60)
reset_logs()
gen_synthetic()
run_all_policies()

print("\n" + "=" * 60)
print("TEST 2a: REAL AZURE TRACE -- quiet window (trace's last 40 rows)")
print("=" * 60)
reset_logs()
gen_from_real_trace(window_end=None)
run_all_policies()

print("\n" + "=" * 60)
print("TEST 2b: REAL AZURE TRACE -- busy window (index 741, ~91 qps avg)")
print("=" * 60)
reset_logs()
gen_from_real_trace(window_end=741)
run_all_policies()

print("\nALL TESTS COMPLETE")