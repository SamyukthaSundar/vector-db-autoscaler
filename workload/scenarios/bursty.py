"""
Person B (J) -- workload/scenarios/bursty.py

Generates a bursty evaluation scenario: baseline load with sudden spikes at
random intervals. This is the scenario that's supposed to expose the
weaknesses of a purely reactive autoscaler (it only reacts after the spike
has already caused SLA violations), making the predictive-vs-reactive gap
visible.
"""

import numpy as np
import pandas as pd
import os


def generate(duration_seconds: int = 3600, base_rate: float = 40.0,
             burst_rate: float = 200.0, burst_prob: float = 0.02,
             burst_duration_steps: int = 6, noise_std: float = 3.0,
             seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_seconds, 10)
    query_rate = np.full(len(t), base_rate, dtype=float)

    i = 0
    while i < len(t):
        if rng.random() < burst_prob:
            end = min(i + burst_duration_steps, len(t))
            query_rate[i:end] = burst_rate
            i = end
        else:
            i += 1

    query_rate += rng.normal(0, noise_std, size=len(t))
    query_rate = np.clip(query_rate, 1, None)
    return pd.DataFrame({"timestamp": t, "query_rate": query_rate, "query_type": "read"})


if __name__ == "__main__":
    df = generate()
    out_path = "workload/trace_data/scenario_bursty.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
