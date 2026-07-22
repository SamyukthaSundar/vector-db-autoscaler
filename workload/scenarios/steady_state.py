"""
Person B (J) -- workload/scenarios/steady_state.py

Generates a steady-state evaluation scenario: roughly constant query rate
with small noise. Useful as a sanity-check scenario before testing bursty/
diurnal cases -- if a policy behaves badly here, something's wrong with the
policy itself, not the workload shape.
"""

import numpy as np
import pandas as pd
import os


def generate(duration_seconds: int = 3600, base_rate: float = 50.0, noise_std: float = 3.0) -> pd.DataFrame:
    t = np.arange(0, duration_seconds, 10)
    query_rate = np.clip(base_rate + np.random.normal(0, noise_std, size=len(t)), 1, None)
    return pd.DataFrame({"timestamp": t, "query_rate": query_rate, "query_type": "read"})


if __name__ == "__main__":
    df = generate()
    out_path = "workload/trace_data/scenario_steady_state.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
