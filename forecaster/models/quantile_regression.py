"""
Person C (N) -- forecaster/models/quantile_regression.py

Multi-scale-feature quantile regression forecaster -- the "MagicScaler-lite"
upgrade from the EWMA baseline. Uses lag features at multiple resolutions
(recent, hourly, daily aggregates) plus gradient-boosted quantile regressors
to produce a genuine predictive distribution (5th/50th/95th percentile),
rather than a Gaussian assumption around a point estimate.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from shared.schemas import ForecastResult, now


def build_multiscale_features(series: pd.Series, lookback_short: int = 5,
                                lookback_medium: int = 30, lookback_long: int = 120) -> pd.DataFrame:
    """Build multi-resolution lag features: short-term lags, medium-term
    rolling mean, long-term rolling mean -- a lightweight stand-in for
    MagicScaler's multi-scale encoder.
    """
    df = pd.DataFrame({"y": series})
    for lag in range(1, lookback_short + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["roll_mean_medium"] = df["y"].shift(1).rolling(lookback_medium).mean()
    df["roll_std_medium"] = df["y"].shift(1).rolling(lookback_medium).std()
    df["roll_mean_long"] = df["y"].shift(1).rolling(lookback_long).mean()
    return df.dropna()


class QuantileForecaster:
    QUANTILES = [0.05, 0.5, 0.95]

    def __init__(self):
        self.models = {
            q: GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=100, max_depth=3)
            for q in self.QUANTILES
        }
        self.feature_cols = None
        self.fitted = False

    def fit(self, series: pd.Series):
        feats = build_multiscale_features(series)
        if len(feats) < 20:
            raise ValueError("Not enough data to fit quantile forecaster (need 20+ points)")
        self.feature_cols = [c for c in feats.columns if c != "y"]
        X, y = feats[self.feature_cols], feats["y"]
        for q, model in self.models.items():
            model.fit(X, y)
        self.fitted = True

    def forecast(self, shard_id: str, series: pd.Series, horizon_seconds: int = 300,
                 confidence_level: float = 0.90) -> ForecastResult:
        if not self.fitted:
            raise RuntimeError("Call fit() before forecast()")

        feats = build_multiscale_features(series)
        if feats.empty:
            raise ValueError("Not enough recent data to build features for forecast")

        last_row = feats[self.feature_cols].iloc[[-1]]
        preds = {q: float(model.predict(last_row)[0]) for q, model in self.models.items()}

        return ForecastResult(
            shard_id=shard_id,
            timestamp=now(),
            horizon_seconds=horizon_seconds,
            point_estimate=preds[0.5],
            lower_bound=max(0.0, preds[0.05]),
            upper_bound=preds[0.95],
            confidence_level=confidence_level,
        )


if __name__ == "__main__":
    t = np.arange(0, 5000)
    series = pd.Series(50 + 20 * np.sin(2 * np.pi * t / 500) + np.random.normal(0, 3, len(t)))
    model = QuantileForecaster()
    model.fit(series)
    result = model.forecast("shard-0", series)
    print(result)
