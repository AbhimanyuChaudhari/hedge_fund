"""
NIFTY26SEPFUT — Instrument-specific configuration.
Calibrated from live data (Sep 3-8, 2026, 4 days).

KEY DIFFERENCES FROM USDINR:
- Raw imbalance is INVERTED (contrarian on NIFTY)
- Cancel imbalance is NORMAL (not inverted)
- Avoid 9-10am (opening imbalance is contrarian, IC=-0.14)
- Market making NOT viable (STT breakeven ~14pts vs 3.6pt spread)
- Pure directional strategy only
"""
from dataclasses import dataclass, field


@dataclass
class NIFTYConfig:
    # ── Instrument ─────────────────────────────────────────────────
    symbol:           str   = 'NIFTY26SEPFUT'
    exchange:         str   = 'NFO'
    instrument_type:  str   = 'equity_futures'
    lot_size:         int   = 25
    tick_size:        float = 0.05
    margin_rate:      float = 0.10   # ~10% margin for NIFTY futures

    # ── Calibrated microstructure (4 days, Sep 3-8 2026) ──────────
    kappa:            float = 1.2408
    sigma:            float = 1.4500  # pts per second
    market_spread:    float = 3.6000  # median pts
    calibration_date: str   = '2026-09-08'
    calibration_days: int   = 4

    # ── Signal parameters ──────────────────────────────────────────
    # NIFTY signals (opposite to USDINR):
    # 1. cancel_imb   IC = +0.053 at 120s  NORMAL   weight 0.60
    # 2. raw_imbalance IC = -0.018 at 120s  INVERTED weight 0.40
    # NOTE: wmid_div dead on NIFTY (IC < 0.001)
    # NOTE: raw_imbalance must be INVERTED for NIFTY
    w_cancel:         float = 0.60   # cancel imbalance (normal)
    w_imbalance:      float = 0.40   # raw imbalance (INVERTED)
    invert_imbalance: bool  = True   # NIFTY imbalance is contrarian
    signal_window:    int   = 30
    norm_window:      int   = 300

    # ── Hourly IC profile ──────────────────────────────────────────
    # 9am:  IC = -0.1256  AVOID — contrarian opening
    # 10am: IC = -0.1453  AVOID — strongest negative
    # 11am: IC = +0.0842  TRADE
    # 12pm: IC = -0.0015  AVOID — dead signal
    # 1pm:  IC = +0.0780  TRADE
    # 2pm:  IC = +0.0384  TRADE
    # 3pm:  IC = +0.0668  TRADE
    avoid_hours:      list  = field(default_factory=lambda: [9, 10, 12])

    # ── Entry/exit parameters ──────────────────────────────────────
    score_threshold:  float = 0.55
    stop_loss_ticks:  int   = 30     # 30 × 0.05 = 1.5 pts
    take_profit_ticks:int   = 90     # 90 × 0.05 = 4.5 pts (3:1 R:R)
    max_hold_seconds: int   = 1800   # 30 minutes
    max_inventory:    int   = 2      # max lots (capital constraint)

    # ── Market hours (NSE equity) ──────────────────────────────────
    market_open_ist:  int   = 9 * 3600 + 15 * 60   # 9:15am IST
    market_close_ist: int   = 15 * 3600 + 30 * 60  # 3:30pm IST
    session_secs:     int   = 375 * 60

    # ── Market making viability ────────────────────────────────────
    # breakeven_spread: ~14 pts (STT dominates)
    # market_spread:     3.6 pts
    # VERDICT: NOT VIABLE for market making
    # Strategy: directional only (11am-3:30pm)