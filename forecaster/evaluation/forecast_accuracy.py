"""
Person C (N) -- forecaster/evaluation/forecast_accuracy.py

Offline evaluation of forecaster accuracy against held-out trace windows:
MAE for the point estimate, and pinball (quantile) loss for the uncertainty
bounds. Run this against workload/trace_data/*.csv scenario files -- doesn't
need the live cluster.
"""

import os
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


# Every trace file worth evaluating against: synthetic scenario files (controlled,
# isolated patterns) plus real-world production traces (Azure, Alibaba).
TRACE_FILES = {
    "steady_state (synthetic)": "workload/trace_data/scenario_steady_state.csv",
    "diurnal (synthetic)": "workload/trace_data/scenario_diurnal.csv",
    "bursty (synthetic)": "workload/trace_data/scenario_bursty.csv",
    "azure (real-world)": "workload/trace_data/azure_parsed.csv",
    "alibaba (real-world)": "workload/trace_data/alibaba_parsed.csv",
}


def load_series(path: str) -> pd.Series:
    df = pd.read_csv(path)
    return df["query_rate"]


def run_all():
    results = []

    for label, path in TRACE_FILES.items():
        if not os.path.exists(path):
            print(f"[skip] {label}: {path} not found")
            continue

        series = load_series(path)
        if len(series) < 150:  # raw rows -- needs to comfortably exceed lookback_long (120)
            print(f"[skip] {label}: only {len(series)} rows, too few for quantile model's feature window")
            continue

        ewma_result = evaluate_ewma(series)

        try:
            quantile_result = evaluate_quantile(series)
        except ValueError as e:
            print(f"[skip quantile] {label}: {e}")
            continue

        results.append({
            "trace": label,
            "n_rows": len(series),
            "ewma_mae": ewma_result["mae"],
            "ewma_pinball_90": ewma_result["pinball_loss_90"],
            "quantile_mae": quantile_result["mae"],
            "quantile_pinball_90": quantile_result["pinball_loss_90"],
        })

    if not results:
        print("No trace files found -- check workload/trace_data/ has been populated.")
        return

    df_results = pd.DataFrame(results)
    print("\n=== Forecast Accuracy Comparison ===\n")
    print(df_results.to_string(index=False))

    out_path = "results/forecast_accuracy_comparison.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved comparison table to {out_path}")

    




if __name__ == "__main__":
    run_all()