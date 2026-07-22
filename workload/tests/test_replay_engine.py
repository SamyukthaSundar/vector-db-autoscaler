"""
Person B (J) -- workload/tests/test_replay_engine.py
"""

import pandas as pd
from workload.scenarios import steady_state, diurnal, bursty


def test_steady_state_shape():
    df = steady_state.generate(duration_seconds=600)
    assert {"timestamp", "query_rate", "query_type"}.issubset(df.columns)
    assert (df["query_rate"] > 0).all()


def test_diurnal_has_daily_cycle():
    df = diurnal.generate(duration_seconds=24 * 3600)
    assert df["query_rate"].max() > df["query_rate"].min()


def test_bursty_has_spikes():
    df = bursty.generate(duration_seconds=1800, burst_prob=0.5)
    assert df["query_rate"].max() > df["query_rate"].median() * 2
