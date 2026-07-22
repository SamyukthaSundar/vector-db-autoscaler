"""
Person B (J) -- workload/parsers/base_parser.py

Common interface all trace parsers implement, so replay_engine.py and
forecaster/ can consume any trace source the same way.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseTraceParser(ABC):
    """Every parser must produce a DataFrame with columns:
    - timestamp (float, seconds relative to trace start)
    - query_rate (float, queries per second at that time)
    - query_type (str, optional -- e.g. 'read'/'write', default 'read')
    """

    @abstractmethod
    def parse(self, raw_path: str) -> pd.DataFrame:
        ...

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "query_rate"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Parsed trace missing required columns: {missing}")
        if "query_type" not in df.columns:
            df["query_type"] = "read"
        return df.sort_values("timestamp").reset_index(drop=True)
