"""
Person C (N) -- forecaster/baselines/ewma_forecaster.py

Trivial baseline forecaster: exponentially-weighted moving average with a
Gaussian prediction interval. Not sophisticated, but it IS a legitimate
uncertainty-aware forecast (point estimate + bounds), and gives a fast
lower bound to compare the fancier model against.
"""

import numpy as np
import pandas as pd

from shared.schemas import ForecastResult, now


class EWMAForecaster:
    def __init__(self, alpha: float = 0.3, confidence_level: float = 0.90):
        self.alpha = alpha
        self.confidence_level = confidence_level
        self._history = {}  # shard_id -> list of recent query_rate values

    def update(self, shard_id: str, query_rate: float):
        self._history.setdefault(shard_id, []).append(query_rate)
        # keep a bounded window so this doesn't grow unbounded in a long run
        if len(self._history[shard_id]) > 500:
            self._history[shard_id] = self._history[shard_id][-500:]

    def forecast(self, shard_id: str, horizon_seconds: int = 300) -> ForecastResult:
        history = self._history.get(shard_id, [])
        if not history:
            raise ValueError(f"No history for shard {shard_id} -- call update() first")

        values = np.array(history)
        ewma = pd.Series(values).ewm(alpha=self.alpha).mean().iloc[-1]
        residuals = values - pd.Series(values).ewm(alpha=self.alpha).mean().values
        std = float(np.std(residuals)) if len(residuals) > 1 else values[-1] * 0.1

        z = 1.645 if self.confidence_level == 0.90 else 1.96  # 90% or 95%
        lower = max(0.0, ewma - z * std)
        upper = ewma + z * std

        return ForecastResult(
            shard_id=shard_id,
            timestamp=now(),
            horizon_seconds=horizon_seconds,
            point_estimate=float(ewma),
            lower_bound=float(lower),
            upper_bound=float(upper),
            confidence_level=self.confidence_level,
        )


if __name__ == "__main__":
    fc = EWMAForecaster()
    for v in [40, 42, 45, 50, 55, 60, 58]:
        fc.update("shard-0", v)
    result = fc.forecast("shard-0")
    print(result)
