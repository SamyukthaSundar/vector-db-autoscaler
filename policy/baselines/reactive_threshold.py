"""
Person D -- policy/baselines/reactive_threshold.py

Reactive threshold-based autoscaler (the Manu/Milvus-style baseline): scales
up when CURRENT observed load crosses a threshold, scales down when it
drops below another threshold. This is the standard approach we're trying
to beat with predictive scaling -- it only reacts after load has already
arrived, which is exactly the gap the predictive policy targets.
"""

import time
import yaml

from infra.actuation.actuation_client import get_actuator
from shared.schemas import MetricPoint, read_jsonl
from policy.eval.eval_harness import EvalHarness

METRICS_LOG_PATH = "results/metrics_log.jsonl"


class ReactiveThresholdPolicy:
    def __init__(self, config_path: str = "configs/cluster_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.actuator = get_actuator(config_path)
        self.harness = EvalHarness(policy_name="reactive")
        self.thresholds = self.config["reactive_policy"]
        self._last_action_time = {}

    def _latest_load(self, shard_id: str) -> float:
        import os
        records = read_jsonl(METRICS_LOG_PATH) if os.path.exists(METRICS_LOG_PATH) else []
        matching = [r for r in records if r.get("shard_id") == shard_id]
        if not matching:
            return 0.0
        matching.sort(key=lambda r: r["timestamp"])
        # crude normalization: treat query_rate directly as a load proxy,
        # scaled against an assumed per-shard capacity of 100 qps
        return matching[-1]["query_rate"] / 100.0

    def _cooldown_ok(self, node_name: str) -> bool:
        last = self._last_action_time.get(node_name, 0)
        return (time.time() - last) > self.thresholds["cooldown_seconds"]

    def tick(self):
        for node_name in self.actuator.clients:
            shard_id = f"{node_name}-shard-agg"
            load = self._latest_load(shard_id)
            action_taken = "noop"

            if load >= self.thresholds["scale_up_load_threshold"] and self._cooldown_ok(node_name):
                self.actuator.scale_replica(node_name, "up")
                self._last_action_time[node_name] = time.time()
                action_taken = "scale_up"
            elif load <= self.thresholds["scale_down_load_threshold"] and self._cooldown_ok(node_name):
                self.actuator.scale_replica(node_name, "down")
                self._last_action_time[node_name] = time.time()
                action_taken = "scale_down"

            self.harness.log_tick(
                shard_id=shard_id,
                active_replicas=self.actuator.get_active_replica_count(node_name),
                action_taken=action_taken,
            )

    def run(self, duration_seconds: int = 600, poll_interval: int = 15):
        elapsed = 0
        while elapsed < duration_seconds:
            self.tick()
            time.sleep(poll_interval)
            elapsed += poll_interval
        print("Reactive threshold run complete.")


if __name__ == "__main__":
    policy = ReactiveThresholdPolicy()
    policy.run(duration_seconds=60, poll_interval=15)
