"""
Person B (J) -- workload/scenarios/diurnal.py

Generates a diurnal (day/night) evaluation scenario: smooth sinusoidal
pattern mimicking typical daily traffic cycles, the classic case predictive
forecasting is supposed to shine at, since the pattern is regular enough to
forecast confidently ahead of time.
"""

import numpy as np
import pandas as pd
import os


def generate(duration_seconds: int = 24 * 3600, base_rate: float = 50.0,
             amplitude: float = 35.0, noise_std: float = 4.0) -> pd.DataFrame:
    t = np.arange(0, duration_seconds, 60)
    diurnal = base_rate + amplitude * np.sin(2 * np.pi * t / (24 * 3600) - np.pi / 2)
    query_rate = np.clip(diurnal + np.random.normal(0, noise_std, size=len(t)), 1, None)
    return pd.DataFrame({"timestamp": t, "query_rate": query_rate, "query_type": "read"})


if __name__ == "__main__":
    df = generate()
    out_path = "workload/trace_data/scenario_diurnal.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
