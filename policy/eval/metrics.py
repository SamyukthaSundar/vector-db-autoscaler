"""
Person D -- policy/eval/metrics.py

Computes the headline comparison metrics from a policy's eval log:
- SLA violation rate
- resource-hours used (proxy for cost)
- thrashing (number of scaling/migration actions taken)
"""

import pandas as pd
from shared.schemas import read_jsonl


def load_eval_log(policy_name: str, results_dir: str = "results") -> pd.DataFrame:
    path = f"{results_dir}/eval_{policy_name}.jsonl"
    records = read_jsonl(path)
    return pd.DataFrame(records)


def sla_violation_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["sla_violation"].mean())


def resource_hours(df: pd.DataFrame, tick_interval_seconds: float = 15.0) -> float:
    """Approximate resource-hours as sum(active_replicas) * tick_interval,
    converted to hours. A crude but consistent proxy for cost across policies.
    """
    if df.empty:
        return 0.0
    return float(df["active_replicas"].sum() * tick_interval_seconds / 3600.0)


def thrashing_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int((df["action_taken"] != "noop").sum())


def summarize(policy_name: str, results_dir: str = "results") -> dict:
    df = load_eval_log(policy_name, results_dir)
    return {
        "policy": policy_name,
        "sla_violation_rate": sla_violation_rate(df),
        "resource_hours": resource_hours(df),
        "thrashing_count": thrashing_count(df),
        "n_ticks": len(df),
    }


if __name__ == "__main__":
    for policy in ["static", "reactive", "predictive", "predictive_rebalance"]:
        try:
            print(summarize(policy))
        except FileNotFoundError:
            print(f"No eval log yet for policy '{policy}' -- run it first.")
