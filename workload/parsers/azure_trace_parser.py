"""
Person B (J) -- workload/parsers/azure_trace_parser.py

Parses the Azure Public Dataset VM request traces into the normalized
(timestamp, query_rate) format used by the rest of the pipeline.

Download the trace from: https://github.com/Azure/AzurePublicDataset
Place the raw CSV under workload/trace_data/raw/ before running this.
"""

import os
import pandas as pd

from workload.parsers.base_parser import BaseTraceParser


class AzureTraceParser(BaseTraceParser):
    def parse(self, raw_path: str) -> pd.DataFrame:
        if not os.path.exists(raw_path):
            print(
                f"[azure_trace_parser] {raw_path} not found -- "
                "generating a small synthetic stand-in trace instead so the "
                "rest of the pipeline can still be developed/tested."
            )
            return self._synthetic_fallback()

        raw = pd.read_csv(raw_path)
        # Azure VM trace columns include a timestamp and CPU utilization
        # reading per VM; we aggregate readings into a request-rate proxy.
        # Adjust column names below to match the specific Azure trace file used.
        df = pd.DataFrame({
            "timestamp": raw["timestamp"],
            "query_rate": raw["avg cpu"] if "avg cpu" in raw.columns else raw.iloc[:, 1],
        })
        return self.validate(df)

    def _synthetic_fallback(self) -> pd.DataFrame:
        """Small diurnal + noisy synthetic trace, used only when the real
        Azure trace file hasn't been downloaded yet -- lets teammates keep
        developing forecaster/ and policy/ without blocking on the download.
        """
        import numpy as np
        t = np.arange(0, 24 * 3600, 60)  # 24 hours, 1-minute resolution
        diurnal = 50 + 40 * np.sin(2 * np.pi * t / (24 * 3600) - np.pi / 2)
        noise = np.random.normal(0, 5, size=len(t))
        query_rate = np.clip(diurnal + noise, 1, None)
        df = pd.DataFrame({"timestamp": t, "query_rate": query_rate})
        return self.validate(df)


if __name__ == "__main__":
    parser = AzureTraceParser()
    df = parser.parse("workload/trace_data/raw/azure_vm_trace.csv")
    out_path = "workload/trace_data/azure_parsed.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
