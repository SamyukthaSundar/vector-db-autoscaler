"""
Person B (J) -- workload/parsers/alibaba_trace_parser.py

Parses the Alibaba cluster trace container_usage.csv into the
normalized (timestamp, query_rate) format.
"""

import os
import pandas as pd

from workload.parsers.base_parser import BaseTraceParser


class AlibabaTraceParser(BaseTraceParser):
    def parse(self, raw_path: str) -> pd.DataFrame:
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"{raw_path} not found. Place container_usage.csv in {raw_path}"
            )

        # Read only columns 0 (timestamp) and 2 (cpu_util_percent) by integer index
        raw = pd.read_csv(
            raw_path,
            header=None,
            usecols=[0, 2],
            names=["time_stamp", "cpu_util_percent"],
            low_memory=False
        )

        # Ensure numeric types
        raw["time_stamp"] = pd.to_numeric(raw["time_stamp"], errors="coerce")
        raw["cpu_util_percent"] = pd.to_numeric(raw["cpu_util_percent"], errors="coerce")

        # Drop NaNs and invalid negative setup timestamps
        raw = raw.dropna(subset=["time_stamp", "cpu_util_percent"])
        raw = raw[raw["time_stamp"] >= 0]

        # Normalize timestamps to start from 0 seconds upwards
        min_ts = raw["time_stamp"].min()
        raw["normalized_sec"] = raw["time_stamp"] - min_ts

        # Aggregate CPU utilization across containers into 1-minute (60s) windows
        raw["time_bucket"] = (raw["normalized_sec"] // 60).astype(int) * 60

        grouped = raw.groupby("time_bucket")["cpu_util_percent"].mean().reset_index()

        df = pd.DataFrame({
            "timestamp": grouped["time_bucket"],
            "query_rate": grouped["cpu_util_percent"],
        })

        return self.validate(df)


if __name__ == "__main__":
    parser = AlibabaTraceParser()
    df = parser.parse("workload/trace_data/raw/container_usage.csv")
    out_path = "workload/trace_data/alibaba_parsed.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Successfully wrote {len(df)} rows to {out_path}")