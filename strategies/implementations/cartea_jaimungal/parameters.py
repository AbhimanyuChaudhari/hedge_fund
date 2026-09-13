from dataclasses import dataclass

@dataclass
class CJParameters:
    """
    Cartea-Jaimungal (2013) market making parameters.

    Key difference from A-S/GLFT/CST:
    - Alpha signal α_t embedded directly in HJB equation
    - Separate running (φ) and terminal (α_terminal) inventory penalties
    - Optimal spread has explicit signal term: 2α_t/κ
    - Signal skews quotes asymmetrically (not just shifts reservation price)

    Papers:
    - Cartea & Jaimungal (2013) "Modelling Asset Prices for Algorithmic
      and High Frequency Trading" Applied Mathematical Finance
    - Cartea & Jaimungal (2015) "Optimal Execution with Limit and
      Market Orders" Quantitative Finance
    """

    # ── Core parameters ────────────────────────────────────────────
    from dataclasses import dataclass

@dataclass
class CJParameters:
    """
    Cartea-Jaimungal (2013) market making parameters.

    Key difference from A-S/GLFT/CST:
    - Alpha signal α_t embedded directly in HJB equation
    - Separate running (φ) and terminal (α_terminal) inventory penalties
    - Optimal spread has explicit signal term: 2α_t/κ
    - Signal skews quotes asymmetrically (not just shifts reservation price)

    Papers:
    - Cartea & Jaimungal (2013) "Modelling Asset Prices for Algorithmic
      and High Frequency Trading" Applied Mathematical Finance
    - Cartea & Jaimungal (2015) "Optimal Execution with Limit and
      Market Orders" Quantitative Finance
    """

    # ── Core parameters ────────────────────────────────────────────
    gamma:          float = 0.1     # risk aversion (same as A-S)
    sigma:          float = 2.0     # price volatility
    kappa:          float = 1.5     # order arrival decay rate
    A:              float = 1.0     # base order arrival rate
    T:              float = 1.0     # session length (normalized)
    tick_size:      float = 0.05    # instrument tick size
    max_inventory:  int   = 5       # max lots

    # ── CJ specific — inventory penalties ─────────────────────────
    # CJ separates penalty into two components:
    # φ = running penalty per second of holding inventory
    # α_terminal = penalty for non-zero inventory at session end
    phi:            float = 0.001   # running inventory penalty
                                    # TODO: calibrate from data
    alpha_terminal: float = 0.01    # terminal inventory penalty
                                    # TODO: calibrate from data

    # ── Alpha signal parameters ────────────────────────────────────
    # α_t = predictive signal (OFI, imbalance, momentum)
    # Embedded directly in HJB → affects spread asymmetrically
    # When α_t > 0 (bullish): ask narrows, bid widens
    # When α_t < 0 (bearish): bid narrows, ask widens
    alpha_decay:    float = 0.5     # signal decay rate (mean reversion)
                                    # dα = -ζ×α dt + dN (OU process)
                                    # TODO: calibrate from IC decay analysis
    alpha_vol:      float = 0.1     # signal volatility
                                    # TODO: calibrate from signal std

    # ── Order arrival parameters ───────────────────────────────────
    # Same exponential model as A-S/GLFT/CST
    # λ(δ) = A × e^(-κδ)
    # Calibrated USDINR: kappa=1.3282, A=0.1250
    recalib_secs:   int   = 10      # recalibrate every N seconds
    flow_window:    int   = 60      # seconds for flow estimation