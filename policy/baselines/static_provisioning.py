"""
Person D -- policy/baselines/static_provisioning.py

Simplest possible baseline: fixed replica count for the whole run, no
scaling or rebalancing decisions at all. This is the "before" picture --
everything else in policy/ should beat this on SLA violations, at the cost
of using more resource-hours.
"""

import time
import yaml

from infra.actuation.actuation_client import get_actuator
from policy.eval.eval_harness import EvalHarness


class StaticProvisioningPolicy:
    def __init__(self, config_path: str = "configs/cluster_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.actuator = get_actuator(config_path)
        self.harness = EvalHarness(policy_name="static")

    def run(self, duration_seconds: int = 600, poll_interval: int = 15):
        """No decisions made -- just logs the fixed replica count and
        observed SLA compliance over the run, for comparison purposes.
        """
        elapsed = 0
        while elapsed < duration_seconds:
            for node_name in self.actuator.clients:
                self.harness.log_tick(
                    shard_id=f"{node_name}-shard-agg",
                    active_replicas=self.actuator.get_active_replica_count(node_name),
                    action_taken="noop",
                )
            time.sleep(poll_interval)
            elapsed += poll_interval
        print("Static provisioning run complete.")


if __name__ == "__main__":
    policy = StaticProvisioningPolicy()
    policy.run(duration_seconds=60, poll_interval=15)
