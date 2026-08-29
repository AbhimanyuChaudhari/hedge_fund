import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd

def realized_variance(prices: list[float], window: int = 20) -> float:
    if len(prices) < 2:
        return 0.0
    returns = np.diff(np.log(prices[-window:]))
    return float(np.var(returns))

def realized_volatility(prices: list[float], window: int = 20) -> float:
    return float(np.sqrt(realized_variance(prices, window)))

def annualized_volatility(prices: list[float], window: int = 20, periods_per_year: int = 375 * 60) -> float:
    return realized_volatility(prices, window) * np.sqrt(periods_per_year)

def rolling_variance(df: pd.DataFrame, price_col: str = 'close', window: int = 20) -> pd.Series:
    log_returns = np.log(df[price_col]).diff()
    return log_returns.rolling(window).var()