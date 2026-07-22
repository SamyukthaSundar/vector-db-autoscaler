"""
Person B (J) -- workload/parsers/alibaba_trace_parser.py

Parses the Alibaba cluster-trace-v2017/v2018 container_usage.csv into the
normalized (timestamp, query_rate) format.

Source: https://github.com/alibaba/clusterdata
Recommended: use cluster-trace-v2017 (12-hour snapshot) rather than v2018
(48GB / 280GB extracted) given typical project timelines.
"""

import os
import pandas as pd

from workload.parsers.base_parser import BaseTraceParser

# container_usage.csv columns per Alibaba's documented schema (v2017/v2018):
# container_id, machine_id, time_stamp, cpu_util_percent, mem_util_percent,
# cpi, mem_gps, mpki, net_in, net_out, disk_io_percent
ALIBABA_COLUMNS = [
    "container_id", "machine_id", "time_stamp", "cpu_util_percent",
    "mem_util_percent", "cpi", "mem_gps", "mpki", "net_in", "net_out",
    "disk_io_percent",
]


class AlibabaTraceParser(BaseTraceParser):
    def parse(self, raw_path: str) -> pd.DataFrame:
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"{raw_path} not found. Download container_usage.csv from "
                "https://github.com/alibaba/clusterdata (cluster-trace-v2017 "
                "recommended for project timelines) and place it here."
            )

        raw = pd.read_csv(raw_path, header=None, names=ALIBABA_COLUMNS)

        # Aggregate CPU utilization across containers into 1-minute windows
        # as a proxy for query rate/load (matches the approach used by prior
        # work applying this trace to autoscaling research).
        raw["time_bucket"] = (raw["time_stamp"] // 60) * 60
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
    print(f"Wrote {len(df)} rows to {out_path}")
