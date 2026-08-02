"""
Person C (N) -- forecaster/forecast_service.py

This is THE contract file for the forecaster module. policy/predictive/*.py
(Person D) should only call `forecast(shard_id)` from here, never touch the
model classes in baselines/ or models/ directly -- that way N can swap the
underlying model without D needing to change anything downstream.
"""

import pandas as pd

from shared.schemas import ForecastResult, MetricPoint, read_jsonl
from forecaster.baselines.ewma_forecaster import EWMAForecaster
from forecaster.models.quantile_regression import QuantileForecaster

METRICS_LOG_PATH = "results/metrics_log.jsonl"


class ForecastService:
    """Facade over whichever forecasting model is currently active.
    Defaults to EWMA; switch `self.mode` to "quantile" once enough history
    has accumulated to fit the heavier model.
    """

    def __init__(self, mode: str = "ewma", metrics_log_path: str = METRICS_LOG_PATH):
        self.mode = mode
        self.metrics_log_path = metrics_log_path
        self.ewma = EWMAForecaster()
        self.quantile_models = {}  # shard_id -> fitted QuantileForecaster

    def _load_shard_series(self, shard_id: str) -> pd.Series:
        records = read_jsonl(self.metrics_log_path)
        points = [MetricPoint.from_dict(r) for r in records if r.get("shard_id") == shard_id]
        points.sort(key=lambda p: p.timestamp)
        return pd.Series([p.query_rate for p in points])

    def forecast(self, shard_id: str, horizon_seconds: int = 300) -> ForecastResult:
        series = self._load_shard_series(shard_id)

        if self.mode == "ewma" or len(series) < 20:
            for v in series.tolist():
                self.ewma.update(shard_id, v)
            return self.ewma.forecast(shard_id, horizon_seconds)

        if self.mode == "quantile":
            if shard_id not in self.quantile_models or len(series) % 50 == 0:
                model = QuantileForecaster()
                model.fit(series)
                self.quantile_models[shard_id] = model
            return self.quantile_models[shard_id].forecast(shard_id, series, horizon_seconds)

        raise ValueError(f"Unknown forecast mode: {self.mode}")


_service_instance = None


def get_forecast_service(mode: str = "ewma") -> ForecastService:
    global _service_instance
    if _service_instance is None:
        _service_instance = ForecastService(mode=mode)
    return _service_instance


if __name__ == "__main__":
    service = get_forecast_service(mode="ewma")
    try:
        print(service.forecast("vectors_main-shard-0"))
    except ValueError as e:
        print(f"No data yet: {e}")
