"""
Person D -- policy/predictive/predictive_rebalancing.py

THE CENTERPIECE ALGORITHM -- predictive shard rebalancing. None of the
source papers (MagicScaler, Milvus/Manu, Autopilot) implement this: using
forecasted (not just current) per-shard load to proactively migrate shards
from predicted-hot nodes to predicted-cold nodes, before the imbalance
actually causes SLA violations.

Heuristic, not a proven-optimal algorithm (be upfront about this in any
write-up) -- but it's the project's actual novel contribution.
"""

import time
import yaml
from dataclasses import dataclass

from infra.actuation.actuation_client import get_actuator
from forecaster.forecast_service import get_forecast_service
from policy.eval.eval_harness import EvalHarness


@dataclass
class ShardForecastSummary:
    shard_id: str
    node_name: str
    predicted_load: float  # upper-bound forecast, normalized 0-1 against capacity


class PredictiveRebalancingPolicy:
    def __init__(self, config_path: str = "configs/cluster_config.yaml", forecast_mode: str = "ewma"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.actuator = get_actuator(config_path)
        self.forecaster = get_forecast_service(mode=forecast_mode)
        self.harness = EvalHarness(policy_name="predictive_rebalance")
        self.thresholds = self.config["predictive_policy"]
        self._last_migration_time = {}

    def _gather_shard_forecasts(self) -> list:
        summaries = []
        for node_name in self.actuator.clients:
            shard_id = f"{node_name}-shard-agg"
            try:
                result = self.forecaster.forecast(
                    shard_id, horizon_seconds=self.thresholds["forecast_horizon_seconds"]
                )
            except ValueError:
                continue
            predicted_load = result.upper_bound / 100.0  # normalize vs assumed capacity
            summaries.append(ShardForecastSummary(shard_id, node_name, predicted_load))
        return summaries

    def _cooldown_ok(self, shard_id: str, cooldown_seconds: int = 90) -> bool:
        last = self._last_migration_time.get(shard_id, 0)
        return (time.time() - last) > cooldown_seconds

    def tick(self):
        summaries = self._gather_shard_forecasts()
        if len(summaries) < 2:
            return  # need at least 2 shards to consider rebalancing

        summaries.sort(key=lambda s: s.predicted_load)
        coldest, hottest = summaries[0], summaries[-1]
        load_delta = hottest.predicted_load - coldest.predicted_load

        for node_name in self.actuator.clients:
            shard_id = f"{node_name}-shard-agg"
            action_taken = "noop"

            if (
                shard_id == hottest.shard_id
                and load_delta >= self.thresholds["rebalance_load_delta_threshold"]
                and self._cooldown_ok(hottest.shard_id)
            ):
                self.actuator.migrate_shard(
                    shard_id=hottest.shard_id,
                    source_node=hottest.node_name,
                    target_node=coldest.node_name,
                )
                self._last_migration_time[hottest.shard_id] = time.time()
                action_taken = "migrate_shard"
                print(
                    f"[predictive_rebalancing] migrating {hottest.shard_id} from "
                    f"{hottest.node_name} (predicted load {hottest.predicted_load:.2f}) "
                    f"to {coldest.node_name} (predicted load {coldest.predicted_load:.2f})"
                )

            self.harness.log_tick(
                shard_id=shard_id,
                active_replicas=self.actuator.get_active_replica_count(node_name),
                action_taken=action_taken,
            )

    def run(self, duration_seconds: int = 600, poll_interval: int = 30):
        elapsed = 0
        while elapsed < duration_seconds:
            self.tick()
            time.sleep(poll_interval)
            elapsed += poll_interval
        print("Predictive rebalancing run complete.")


if __name__ == "__main__":
    policy = PredictiveRebalancingPolicy()
    policy.run(duration_seconds=60, poll_interval=30)
