from dataclasses import dataclass, field

@dataclass
class CSTParameters:
    # ── Core market making ─────────────────────────────────────────
    gamma:         float = 0.1    # risk aversion (TODO: calibrate notebook 10)
    sigma:         float = 2.0    # baseline volatility (updated live from data)
    kappa:         float = 1.5    # order arrival decay (TODO: calibrate notebook 10)
    A:             float = 1.0    # base arrival rate (TODO: calibrate notebook 10)
    T:             float = 1.0    # session length normalized
    max_inventory: int   = 5      # max lots per side
    tick_size:     float = 0.05   # NSE futures tick size

    # ── Drift estimation weights ───────────────────────────────────
    kyle_lambda:      float = 0.01  # price impact per OFI unit (TODO: notebook 07)
    ofi_weight:       float = 0.6   # weight on OFI signal (TODO: notebook 09)
    momentum_weight:  float = 0.4   # weight on price momentum (TODO: notebook 09)

    # ── Cancel impact ──────────────────────────────────────────────
    cancel_vol_weight: float = 1.0  # how much cancels add to σ² (TODO: notebook 08)
    cancel_drift_on:   bool  = True # include cancel drift in reservation price

    # ── Order flow estimation window ──────────────────────────────
    flow_window:   int = 60   # seconds of history for rate estimation
    recalib_secs:  int = 10   # recalibrate every N seconds

    # ── Signal thresholds ─────────────────────────────────────────
    min_edge_ticks: float = 2.0  # minimum edge in ticks before acting