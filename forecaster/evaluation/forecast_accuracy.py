"""
Person C (N) -- forecaster/evaluation/forecast_accuracy.py

Offline evaluation of forecaster accuracy against held-out trace windows:
MAE for the point estimate, and pinball (quantile) loss for the uncertainty
bounds. Run this against workload/trace_data/*.csv scenario files -- doesn't
need the live cluster.
"""

import numpy as np
import pandas as pd

from forecaster.baselines.ewma_forecaster import EWMAForecaster
from forecaster.models.quantile_regression import QuantileForecaster


def pinball_loss(y_true: float, y_pred: float, quantile: float) -> float:
    diff = y_true - y_pred
    return max(quantile * diff, (quantile - 1) * diff)


def evaluate_ewma(series: pd.Series, train_frac: float = 0.7) -> dict:
    split = int(len(series) * train_frac)
    train, test = series.iloc[:split], series.iloc[split:]

    fc = EWMAForecaster()
    for v in train:
        fc.update("eval-shard", v)

    errors, pinballs = [], []
    for actual in test:
        result = fc.forecast("eval-shard")
        errors.append(abs(actual - result.point_estimate))
        pinballs.append(pinball_loss(actual, result.upper_bound, 0.90))
        fc.update("eval-shard", actual)

    return {"mae": float(np.mean(errors)), "pinball_loss_90": float(np.mean(pinballs))}


def evaluate_quantile(series: pd.Series, train_frac: float = 0.7) -> dict:
    split = int(len(series) * train_frac)
    train, test_series = series.iloc[:split], series

    model = QuantileForecaster()
    model.fit(train)

    errors, pinballs = [], []
    for i in range(split, len(series) - 1):
        window = test_series.iloc[:i]
        result = model.forecast("eval-shard", window)
        actual = series.iloc[i]
        errors.append(abs(actual - result.point_estimate))
        pinballs.append(pinball_loss(actual, result.upper_bound, 0.90))

    return {"mae": float(np.mean(errors)), "pinball_loss_90": float(np.mean(pinballs))}


if __name__ == "__main__":
    import os

    # Try to load J's real trace-derived scenario data first.
    # Falls back to synthetic data only if it's not available yet,
    # so this script still works standalone even before J's data lands.
    trace_path = "workload/trace_data/scenario_diurnal.csv"

    if os.path.exists(trace_path):
        print(f"Using real trace data from {trace_path}")
        df = pd.read_csv(trace_path)
        series = df["query_rate"]
    else:
        print(f"{trace_path} not found -- using synthetic fallback data")
        t = np.arange(0, 3000)
        series = pd.Series(50 + 20 * np.sin(2 * np.pi * t / 300) + np.random.normal(0, 3, len(t)))

    print("EWMA baseline:", evaluate_ewma(series))
    print("Quantile model:", evaluate_quantile(series))
