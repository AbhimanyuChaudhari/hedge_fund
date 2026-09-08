"""
USDINR26SEPFUT — Instrument-specific configuration.
All parameters calibrated from live data (Sep 3-8, 2026, 4 days).
Update after each calibration run.
"""
from dataclasses import dataclass, field
from datetime import date


@dataclass
class USDINRConfig:
    # ── Instrument ─────────────────────────────────────────────────
    symbol:           str   = 'USDINR26SEPFUT'
    exchange:         str   = 'CDS'
    instrument_type:  str   = 'currency_futures'
    lot_size:         int   = 1000
    tick_size:        float = 0.0025
    min_lots:         int   = 20      # minimum viable lots (breakeven constraint)
    margin_rate:      float = 0.02    # 2% margin requirement

    # ── Calibrated market microstructure (4 days, Sep 3-8 2026) ───
    kappa:            float = 1.3282  # order arrival decay
    sigma:            float = 0.00065 # price volatility per second (price units)
    A:                float = 0.1250  # base arrival rate
    mu_avg:           float = 0.001108# avg drift per second
    cancel_rate:      float = 0.0184  # cancellation rate per second
    market_spread:    float = 0.0050  # median bid-ask spread (paise)
    calibration_date: str   = '2026-09-08'
    calibration_days: int   = 4

    # ── Signal parameters (validated from IC analysis) ─────────────
    # Signals ranked by IC at 120s horizon:
    # 1. cancel_imbalance  IC = -0.162 (inverted) — dominant
    # 2. raw_imbalance     IC = +0.051             — second
    # 3. weighted_mid_div  IC = +0.058             — third
    # NOTE: weighted OFI (CKS 2014) has NEGATIVE IC on USDINR
    #       due to tiny tick size — DO NOT USE for this instrument
    w_cancel:         float = 0.50   # cancel imbalance weight (inverted)
    w_imbalance:      float = 0.30   # raw imbalance weight
    w_wmid:           float = 0.20   # weighted mid divergence weight
    signal_window:    int   = 30     # bars for rolling signal
    norm_window:      int   = 300    # bars for rolling normalization

    # ── Hourly IC profile (forward 300s, 4 days) ───────────────────
    # Hour  IC      Notes
    # 9am   0.1131  good
    # 10am  0.1473  strong
    # 11am  0.1371  good
    # 12pm  0.0223  weak
    # 1pm   0.0543  moderate
    # 2pm   0.2869  strongest — FII activity before US open
    # 3pm  -0.1977  AVOID — end of session mean reversion
    avoid_hours:      list  = field(default_factory=lambda: [15])

    # ── Entry/exit parameters (validated from 4-day backtest) ──────
    score_threshold:  float = 0.55   # min combined score to trade
    stop_loss_ticks:  int   = 30     # stop loss in ticks
    take_profit_ticks:int   = 90     # take profit in ticks (3:1 R:R)
    max_hold_seconds: int   = 1800   # 30 minutes — IC peaks at 120s, hold 30min
    max_inventory:    int   = 5      # max lots

    # ── Transaction costs (at min_lots=20) ─────────────────────────
    # breakeven_spread: 0.00435 paise at 20 lots
    # breakeven_spread: 0.00294 paise at 50 lots
    # market_spread:    0.00500 paise
    # viable at:        20+ lots

    # ── Market hours (CDS, different from NSE equity) ──────────────
    market_open_ist:  int   = 9 * 3600       # 9:00am IST
    market_close_ist: int   = 17 * 3600      # 5:00pm IST
    session_secs:     int   = 8 * 3600       # 8 hours

    # ── Research findings ──────────────────────────────────────────
    # 4-day backtest best config:
    #   hold=1800s, sl=30, tp=90, thr=0.55
    #   trades=24, win_rate=45.8%, total_pnl=-₹895 (near breakeven)
    #   avg_pnl=+₹12 per trade
    # Need: 10+ days to confirm, target win_rate > 50%
    # Next calibration: after 10 trading days