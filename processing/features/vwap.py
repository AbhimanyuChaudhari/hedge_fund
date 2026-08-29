import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np

def vwap(prices: list[float], volumes: list[float]) -> float:
    total_volume = sum(volumes)
    if total_volume == 0:
        return prices[-1] if prices else 0.0
    return sum(p * v for p, v in zip(prices, volumes)) / total_volume

def rolling_vwap(df: pd.DataFrame, price_col: str = 'close', volume_col: str = 'volume') -> pd.Series:
    pv = df[price_col] * df[volume_col]
    return pv.cumsum() / df[volume_col].cumsum()

def vwap_deviation(price: float, vwap_price: float) -> float:
    if vwap_price == 0:
        return 0.0
    return (price - vwap_price) / vwap_price