"""
Person C (N) -- forecaster/models/prophet_forecaster.py

Alternative forecaster using Facebook Prophet, which natively models
seasonality (daily/weekly cycles) and produces uncertainty intervals out of
the box -- a good fit for the diurnal scenario in particular.

Note: requires `pip install prophet`. Prophet can be slow to fit repeatedly
in an online loop -- prefer the quantile_regression.py model if you need
frequent re-fitting during live evaluation, and use this one for scenarios
with strong daily seasonality where Prophet's seasonal decomposition adds
real value.
"""

import pandas as pd
from prophet import Prophet

from shared.schemas import ForecastResult, now


class ProphetForecaster:
    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.model = Prophet(interval_width=confidence_level, daily_seasonality=True)
        self.fitted = False

    def fit(self, timestamps: pd.Series, values: pd.Series):
        df = pd.DataFrame({
            "ds": pd.to_datetime(timestamps, unit="s"),
            "y": values.values,
        })
        self.model.fit(df)
        self.fitted = True

    def forecast(self, shard_id: str, horizon_seconds: int = 300) -> ForecastResult:
        if not self.fitted:
            raise RuntimeError("Call fit() before forecast()")

        future = self.model.make_future_dataframe(
            periods=max(1, horizon_seconds // 60), freq="min"
        )
        forecast_df = self.model.predict(future)
        row = forecast_df.iloc[-1]

        return ForecastResult(
            shard_id=shard_id,
            timestamp=now(),
            horizon_seconds=horizon_seconds,
            point_estimate=max(0.0, float(row["yhat"])),
            lower_bound=max(0.0, float(row["yhat_lower"])),
            upper_bound=max(0.0, float(row["yhat_upper"])),
            confidence_level=self.confidence_level,
        )


if __name__ == "__main__":
    import numpy as np
    t = pd.Series(np.arange(0, 24 * 3600, 60))
    values = pd.Series(50 + 30 * np.sin(2 * np.pi * t / (24 * 3600)) + np.random.normal(0, 4, len(t)))
    model = ProphetForecaster()
    model.fit(t, values)
    print(model.forecast("shard-0"))
