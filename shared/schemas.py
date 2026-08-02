"""
Shared data contracts used across infra/, workload/, forecaster/, and policy/.

Everyone imports from here rather than inventing their own formats.
Owned jointly by the team -- changes here should be discussed in standup
before merging, since every module depends on these shapes.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import time
import json


@dataclass
class MetricPoint:
    """A single metrics observation, as scraped/exported by infra/ (Person A)
    and consumed by forecaster/ (Person C) and policy/ (Person D)."""
    timestamp: float                # unix epoch seconds
    shard_id: str
    node_name: str
    query_rate: float                # queries per second
    p95_latency_ms: float
    p99_latency_ms: float
    vector_count: int
    cpu_util_percent: float = 0.0

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "MetricPoint":
        return MetricPoint(**d)


@dataclass
class ForecastResult:
    """Output of forecaster/forecast_service.py (Person C), consumed by
    policy/predictive/*.py (Person D)."""
    shard_id: str
    timestamp: float                 # time the forecast was made
    horizon_seconds: int             # how far ahead this forecast looks
    point_estimate: float            # predicted query_rate (or load)
    lower_bound: float                # e.g. 5th percentile
    upper_bound: float                # e.g. 95th percentile
    confidence_level: float = 0.90

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ForecastResult":
        return ForecastResult(**d)


@dataclass
class ScalingAction:
    """An action taken by any policy (static/reactive/predictive), logged by
    policy/eval/eval_harness.py (Person D) for later comparison."""
    timestamp: float
    action_type: str                 # "scale_up" | "scale_down" | "migrate_shard" | "noop"
    shard_id: Optional[str] = None
    source_node: Optional[str] = None
    target_node: Optional[str] = None
    policy_name: str = ""            # "static" | "reactive" | "predictive"
    reason: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class EvalLogLine:
    """One row of the evaluation log. This is the schema policy/eval/eval_harness.py
    writes to disk, and generate_comparison_charts.py reads back."""
    timestamp: float
    policy_name: str
    shard_id: str
    p95_latency_ms: float
    p99_latency_ms: float
    sla_violation: bool
    active_replicas: int
    action_taken: str = "noop"

    def to_dict(self):
        return asdict(self)


def append_jsonl(path: str, record) -> None:
    """Utility: append a dataclass record as one JSON line to `path`."""
    with open(path, "a") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


def read_jsonl(path: str):
    """Utility: read a .jsonl file back into a list of dicts."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def now() -> float:
    return time.time()
