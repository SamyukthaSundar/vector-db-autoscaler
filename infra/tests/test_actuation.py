"""
Person A (S) -- infra/tests/test_actuation.py

Sanity tests for the actuation API contract. Run with: pytest infra/tests/
"""

import pytest
from infra.actuation.actuation_api import ClusterActuator, ActuationError


@pytest.fixture
def actuator():
    return ClusterActuator()


def test_scale_replica_up(actuator):
    node = list(actuator.clients.keys())[0]
    before = actuator.get_active_replica_count(node)
    actuator.scale_replica(node, "up")
    after = actuator.get_active_replica_count(node)
    assert after == before + 1


def test_scale_replica_down_floor(actuator):
    node = list(actuator.clients.keys())[0]
    actuator.scale_replica(node, "down")
    actuator.scale_replica(node, "down")
    assert actuator.get_active_replica_count(node) >= 1


def test_unknown_node_raises(actuator):
    with pytest.raises(ActuationError):
        actuator.scale_replica("not-a-real-node", "up")


def test_migrate_shard_logs_action(actuator):
    nodes = list(actuator.clients.keys())
    action = actuator.migrate_shard("vectors_main-shard-0", nodes[0], nodes[1])
    assert action.action_type == "migrate_shard"
    assert action.source_node == nodes[0]
    assert action.target_node == nodes[1]
