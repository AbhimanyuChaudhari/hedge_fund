"""
execution/USDINR/live/features.py
Thin wrapper around the production USDINRFeatureBuilder.
"""

import sys
import pathlib
import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]   # C:/projects/hedge_fund
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.processing.usdinr.features import USDINRFeatureBuilder
from data.processing.usdinr.constants import CHALLENGER50


class LiveFeatureBuffer:
    """
    Maintains a rolling window of raw 1s OHLCV+orderbook rows
    and recomputes CHALLENGER50 features on each new tick.

    Usage
    -----
    buf = LiveFeatureBuffer(symbol="USDINR26NOVFUT", window=600)
    buf.push(tick_dict)
    vec = buf.latest_vector()   # np.ndarray shape (50,) or None
    """

    MIN_ROWS = 300

    def __init__(self, symbol: str, window: int = 600):
        self.symbol  = symbol
        self.window  = window
        self._buffer = []

    def push(self, tick: dict) -> None:
        self._buffer.append(tick)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    def latest_vector(self):
        """Returns np.ndarray shape (50,) or None if not enough data."""
        if len(self._buffer) < self.MIN_ROWS:
            return None

        df = pd.DataFrame(self._buffer)
        df["symbol"] = self.symbol

        try:
            builder = USDINRFeatureBuilder(data=df, auto_build=True)
            df_feat = builder.dataset
        except Exception as e:
            raise RuntimeError(f"FeatureBuilder failed: {e}") from e

        last    = df_feat.iloc[-1]
        missing = [c for c in CHALLENGER50 if c not in df_feat.columns]
        if missing:
            raise ValueError(f"Missing CHALLENGER50 columns: {missing}")

        return last[CHALLENGER50].to_numpy(dtype=np.float32)

    def latest_row(self):
        """Full feature row for the latest bar (all columns)."""
        if len(self._buffer) < self.MIN_ROWS:
            return None
        df = pd.DataFrame(self._buffer)
        df["symbol"] = self.symbol
        builder = USDINRFeatureBuilder(data=df, auto_build=True)
        return builder.dataset.iloc[-1]

    @property
    def n_rows(self) -> int:
        return len(self._buffer)

    @property
    def feature_names(self):
        return CHALLENGER50