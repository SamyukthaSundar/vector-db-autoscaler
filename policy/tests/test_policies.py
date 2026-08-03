"""
Person D -- policy/tests/test_policies.py
"""

import pandas as pd
from policy.eval.metrics import sla_violation_rate, resource_hours, thrashing_count


def test_sla_violation_rate_empty():
    assert sla_violation_rate(pd.DataFrame()) == 0.0


def test_sla_violation_rate_basic():
    df = pd.DataFrame({"sla_violation": [True, False, False, True]})
    assert sla_violation_rate(df) == 0.5


def test_resource_hours_basic():
    df = pd.DataFrame({"active_replicas": [1, 2, 2, 1]})
    hours = resource_hours(df, tick_interval_seconds=3600)  # 1 hour ticks for easy math
    assert hours == 6.0


def test_thrashing_count_basic():
    df = pd.DataFrame({"action_taken": ["noop", "scale_up", "noop", "migrate_shard"]})
    assert thrashing_count(df) == 2
