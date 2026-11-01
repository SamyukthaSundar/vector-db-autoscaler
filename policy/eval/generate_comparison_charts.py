"""
Person D -- policy/eval/generate_comparison_charts.py

Final step: reads all four policies' eval logs and produces the comparison
charts (SLA violations, resource-hours, thrashing) that go into the
README/results write-up. Run this after all four policies have been
executed against the same scenario.
"""

import os
import matplotlib.pyplot as plt

from policy.eval.metrics import summarize

POLICIES = ["static", "reactive", "predictive", "predictive_rebalance"]
OUT_DIR = "results/charts"


def collect_summaries():
    summaries = []
    for policy in POLICIES:
        try:
            summaries.append(summarize(policy))
        except FileNotFoundError:
            print(f"Skipping '{policy}' -- no eval log found. Run it first.")
    return summaries


def plot_metric(summaries, key: str, title: str, ylabel: str, filename: str):
    labels = [s["policy"] for s in summaries]
    values = [s[key] for s in summaries]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["#888", "#e67e22", "#2980b9", "#27ae60"][:len(labels)])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    summaries = collect_summaries()
    if not summaries:
        print("No policy eval logs found -- run the policies first.")
        return

    plot_metric(summaries, "sla_violation_rate", "SLA Violation Rate by Policy",
                "Violation rate", "sla_violation_rate.png")
    plot_metric(summaries, "resource_hours", "Resource-Hours Used by Policy",
                "Resource-hours", "resource_hours.png")
    plot_metric(summaries, "thrashing_count", "Scaling/Migration Actions (Thrashing) by Policy",
                "Number of actions", "thrashing_count.png")

    print("\nSummary table:")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
