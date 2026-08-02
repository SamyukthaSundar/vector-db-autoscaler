"""
Person C (N) -- forecaster/tests/test_forecast_service.py
"""

import pytest
from forecaster.baselines.ewma_forecaster import EWMAForecaster


def test_ewma_forecast_bounds_sane():
    fc = EWMAForecaster()
    for v in [40, 42, 45, 50, 55, 60, 58, 57, 59, 61]:
        fc.update("shard-x", v)
    result = fc.forecast("shard-x")
    assert result.lower_bound <= result.point_estimate <= result.upper_bound
    assert result.shard_id == "shard-x"


def test_ewma_raises_without_history():
    fc = EWMAForecaster()
    with pytest.raises(ValueError):
        fc.forecast("never-seen-shard")
