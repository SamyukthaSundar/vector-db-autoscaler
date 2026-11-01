"""
Person D -- policy/predictive/predictive_scaling.py

Predictive, uncertainty-aware scaling policy (the MagicScaler-lite piece):
consumes the forecaster's predicted distribution for near-future load and
scales PRE-EMPTIVELY when the upper confidence bound crosses the capacity
threshold -- rather than waiting for load to actually arrive like the
reactive baseline does. This predictive-vs-reactive comparison is the
project's headline result.
"""

import time
import yaml

from infra.actuation.actuation_client import get_actuator
from forecaster.forecast_service import get_forecast_service
from policy.eval.eval_harness import EvalHarness


class PredictiveScalingPolicy:
    def __init__(self, config_path: str = "configs/cluster_config.yaml", forecast_mode: str = "ewma"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.actuator = get_actuator(config_path)
        self.forecaster = get_forecast_service(mode=forecast_mode)
        self.harness = EvalHarness(policy_name="predictive")
        self.thresholds = self.config["predictive_policy"]
        self._last_action_time = {}

    def _cooldown_ok(self, node_name: str, cooldown_seconds: int = 60) -> bool:
        last = self._last_action_time.get(node_name, 0)
        return (time.time() - last) > cooldown_seconds

    def tick(self):
        for node_name in self.actuator.clients:
            shard_id = f"{node_name}-shard-agg"
            action_taken = "noop"

            try:
                result = self.forecaster.forecast(
                    shard_id, horizon_seconds=self.thresholds["forecast_horizon_seconds"]
                )
            except ValueError:
                # no history yet for this shard -- nothing to predict on
                self.harness.log_tick(
                    shard_id=shard_id,
                    active_replicas=self.actuator.get_active_replica_count(node_name),
                    action_taken=action_taken,
                )
                continue

            # normalize predicted upper bound against an assumed 100 qps/shard capacity
            predicted_upper_load = result.upper_bound / 100.0

            if predicted_upper_load >= self.thresholds["scale_up_upper_bound_threshold"] and self._cooldown_ok(node_name):
                self.actuator.scale_replica(node_name, "up")
                self._last_action_time[node_name] = time.time()
                action_taken = "scale_up"
                print(
                    f"[predictive_scaling] pre-emptive scale-up on {node_name}: "
                    f"forecast upper bound {result.upper_bound:.1f} qps "
                    f"(horizon {result.horizon_seconds}s)"
                )

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
        print("Predictive scaling run complete.")


if __name__ == "__main__":
    policy = PredictiveScalingPolicy()
    policy.run(duration_seconds=60, poll_interval=15)
