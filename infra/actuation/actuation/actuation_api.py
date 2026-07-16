"""
Person A (S) -- infra/actuation/actuation_api.py

This is THE contract file. policy/ (Person D) calls into these functions
and should never talk to Qdrant directly. Keep the function signatures
stable once other people start building against them -- if you must change
one, ping the team first.
"""

import time
import yaml
from qdrant_client import QdrantClient

from shared.schemas import ScalingAction, now


class ActuationError(Exception):
    pass


class ClusterActuator:
    def __init__(self, config_path: str = "configs/cluster_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.collection_name = self.config["collection_name"]
        self.clients = {
            node["name"]: QdrantClient(host=node["host"], port=node["port"])
            for node in self.config["nodes"]
        }
        # tracks which nodes are "active" replicas per shard (simulated,
        # since single-node OSS Qdrant doesn't natively expose live replica
        # add/remove the way a managed cluster would)
        self.active_replicas = {node: 1 for node in self.clients}

    def scale_replica(self, node_name: str, direction: str = "up") -> ScalingAction:
        """Scale a replica up or down on the given node.

        In a full production system this would call Qdrant's cluster API to
        add/remove a peer. For this project's scope we simulate replica
        count as a resource-accounting concept that the eval harness uses
        to compute resource-hours -- the actual query serving still goes
        through the single node, but the *decision logic* being evaluated
        (when/whether to scale) is the real deliverable.
        """
        if node_name not in self.clients:
            raise ActuationError(f"Unknown node: {node_name}")

        if direction == "up":
            self.active_replicas[node_name] += 1
        elif direction == "down":
            self.active_replicas[node_name] = max(
                1, self.active_replicas[node_name] - 1
            )
        else:
            raise ActuationError(f"Invalid direction: {direction}")

        action = ScalingAction(
            timestamp=now(),
            action_type="scale_up" if direction == "up" else "scale_down",
            source_node=node_name,
            reason=f"replica {direction}-scale requested",
        )
        print(f"[actuation] {action}")
        return action

    def migrate_shard(self, shard_id: str, source_node: str, target_node: str) -> ScalingAction:
        """Migrate a shard's data from source_node to target_node.

        Qdrant supports shard transfer via its cluster API
        (PUT /collections/{name}/cluster). This wraps that call; for local
        single-node dev clusters where cluster mode isn't fully configured,
        this logs the intended action so the policy logic can still be
        tested end-to-end against the eval harness.
        """
        if source_node not in self.clients or target_node not in self.clients:
            raise ActuationError("Unknown source or target node")

        try:
            # Real call would look roughly like:
            # self.clients[source_node].http.cluster_api.update_collection_cluster_setup(...)
            # Left as a clearly-marked stub since real shard transfer requires
            # a properly peered Qdrant cluster (beyond single-node docker-compose).
            pass
        except Exception as e:
            raise ActuationError(f"Shard migration failed: {e}")

        action = ScalingAction(
            timestamp=now(),
            action_type="migrate_shard",
            shard_id=shard_id,
            source_node=source_node,
            target_node=target_node,
            reason="load-based rebalancing",
        )
        print(f"[actuation] {action}")
        return action

    def get_active_replica_count(self, node_name: str) -> int:
        return self.active_replicas.get(node_name, 1)

    def total_active_replicas(self) -> int:
        return sum(self.active_replicas.values())


if __name__ == "__main__":
    actuator = ClusterActuator()
    actuator.scale_replica("qdrant-node-1", "up")
    actuator.migrate_shard("vectors_main-shard-0", "qdrant-node-1", "qdrant-node-2")
    print("Active replicas:", actuator.active_replicas)
