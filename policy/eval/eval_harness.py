"""
Person D -- policy/eval/eval_harness.py

Shared evaluation harness. Every policy (static/reactive/predictive/
predictive_rebalance) writes its per-tick observations here in a common
format (EvalLogLine from shared/schemas.py), so generate_comparison_charts.py
can read all of them back and compare fairly.
"""

import os
import yaml

from shared.schemas import EvalLogLine, append_jsonl, now, read_jsonl, MetricPoint

RESULTS_DIR = "results"
METRICS_LOG_PATH = os.path.join(RESULTS_DIR, "metrics_log.jsonl")


class EvalHarness:
    def __init__(self, policy_name: str, config_path: str = "configs/cluster_config.yaml"):
        self.policy_name = policy_name
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.sla = self.config["sla"]
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.log_path = os.path.join(RESULTS_DIR, f"eval_{policy_name}.jsonl")

    def _latest_latency(self, shard_id: str):
        records = read_jsonl(METRICS_LOG_PATH) if os.path.exists(METRICS_LOG_PATH) else []
        matching = [r for r in records if r.get("shard_id") == shard_id]
        if not matching:
            return 0.0, 0.0
        matching.sort(key=lambda r: r["timestamp"])
        latest = matching[-1]
        return latest.get("p95_latency_ms", 0.0), latest.get("p99_latency_ms", 0.0)

    def log_tick(self, shard_id: str, active_replicas: int, action_taken: str = "noop"):
        p95, p99 = self._latest_latency(shard_id)
        sla_violation = p95 > self.sla["max_p95_latency_ms"] or p99 > self.sla["max_p99_latency_ms"]

        line = EvalLogLine(
            timestamp=now(),
            policy_name=self.policy_name,
            shard_id=shard_id,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            sla_violation=sla_violation,
            active_replicas=active_replicas,
            action_taken=action_taken,
        )
        append_jsonl(self.log_path, line)
        return line
