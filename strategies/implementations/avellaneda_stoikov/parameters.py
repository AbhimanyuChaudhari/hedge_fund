from dataclasses import dataclass

@dataclass
class ASParameters:
    gamma: float = 0.1          # risk aversion — higher = tighter quotes, less inventory risk
    k: float = 1.5              # order book depth parameter — higher = orders fill less often
    max_inventory: int = 5      # max lots to hold (positive or negative)
    requote_interval: int = 5   # seconds between requotes
    session_length: float = 375 # trading minutes per day (6.25 hours)
    tick_size: float = 0.05     # NIFTY tick size
    lot_size: int = 25          # NIFTY lot size
    slippage_ticks: int = 3     # expected slippage in ticks
    brokerage_per_lot: float = 100.0  # round trip brokerage in rupees