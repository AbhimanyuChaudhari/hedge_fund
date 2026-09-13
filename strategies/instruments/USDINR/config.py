from dataclasses import dataclass, field


@dataclass
class USDINRConfig:
    # ── Instrument ─────────────────────────────────────────────────
    symbol:           str   = 'USDINR26SEPFUT'
    exchange:         str   = 'CDS'
    instrument_type:  str   = 'currency_futures'
    lot_size:         int   = 1000
    tick_size:        float = 0.0025
    min_lots:         int   = 20
    margin_rate:      float = 0.02

    # ── Calibrated microstructure (4 days, Sep 3-9 2026) ──────────
    kappa:            float = 1.3282
    sigma:            float = 0.00065
    A:                float = 0.1250
    cancel_rate:      float = 0.0184
    market_spread:    float = 0.0050
    calibration_date: str   = '2026-09-09'
    calibration_days: int   = 4

    # ── Kyle Lambda (OLS regression, 60s horizon) ─────────────────
    # Δprice = λ × imbalance + ε
    # t-stat = 19.07, R² = 0.005
    kyle_lambda:      float = 0.00148   # was hardcoded 0.01

    # ── Drift weights (Ridge regression, 60s horizon) ──────────────
    # µ = β_imb×imbalance + β_mom×momentum + β_cancel×cancel_imb
    # CRITICAL: momentum is NEGATIVE — USDINR mean-reverts at 60s!
    # IC of combined signal = 0.33 at 60s
    drift_w_imbalance: float =  0.1067  # was 0.6 hardcoded
    drift_w_momentum:  float = -0.7875  # was +0.4 hardcoded (WRONG SIGN!)
    drift_w_cancel:    float = -0.1058  # was 0.0 (not included)

    # ── Cancel vol weight (vol regression) ────────────────────────
    # σ²_eff = σ² + δ² × cancel_rate × weight
    # Nearly zero — cancels don't add volatility on USDINR
    cancel_vol_weight: float = 0.000018  # was 1.0 hardcoded

    # ── mu_avg (kyle_lambda × avg_imbalance) ──────────────────────
    mu_avg:            float = 0.000592  # 0.00148 × 0.40

    # ── Signal parameters (IC analysis validated) ──────────────────
    w_cancel:         float = 0.50
    w_imbalance:      float = 0.30
    w_wmid:           float = 0.20
    signal_window:    int   = 30
    norm_window:      int   = 300

    # ── Hourly IC profile (forward 300s, 4 days) ───────────────────
    # 9am=0.11, 10am=0.15, 11am=0.14, 12pm=0.02, 1pm=0.05
    # 2pm=0.29 (strongest), 3pm=-0.20 (AVOID)
    avoid_hours:      list  = field(default_factory=lambda: [15])

    # ── Entry/exit (4-day backtest validated) ──────────────────────
    score_threshold:  float = 0.55
    stop_loss_ticks:  int   = 30
    take_profit_ticks:int   = 90
    max_hold_seconds: int   = 1800
    max_inventory:    int   = 5

    # ── Market hours (CDS) ─────────────────────────────────────────
    market_open_ist:  int   = 9 * 3600
    market_close_ist: int   = 17 * 3600
    session_secs:     int   = 8 * 3600