"""
Person A (S) -- infra/actuation/actuation_client.py

Convenience singleton so policy/ modules don't each re-instantiate
ClusterActuator (and re-parse the config file) separately.
"""

from infra.actuation.actuation_api import ClusterActuator

_actuator_instance = None


def get_actuator(config_path: str = "configs/cluster_config.yaml") -> ClusterActuator:
    global _actuator_instance
    if _actuator_instance is None:
        _actuator_instance = ClusterActuator(config_path)
    return _actuator_instance
