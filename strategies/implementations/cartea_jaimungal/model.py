import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
from strategies.implementations.cartea_jaimungal.parameters import CJParameters


class CJModel:
    """
    Cartea-Jaimungal (2013) market making model.

    Key improvements over A-S/GLFT/CST:
    1. Alpha signal α_t embedded in HJB — not just reservation price shift
    2. Separate running + terminal inventory penalties
    3. Optimal spread has explicit signal term: 2α_t/κ
    4. Quotes skew asymmetrically based on signal strength

    Optimal quotes (closed form solution):
        bid = S - δ*_b
        ask = S + δ*_a

    where:
        δ*_b = (1/κ)×ln(1 + κ/γ) + (2q+1)/2 × γσ²(T-t) - α_t/κ
        δ*_a = (1/κ)×ln(1 + κ/γ) - (2q-1)/2 × γσ²(T-t) + α_t/κ

    The α_t/κ term is what makes CJ different from GLFT:
    - Positive signal → bid moves up (more aggressive buy), ask moves up (less aggressive sell)
    - Negative signal → ask moves down, bid moves down
    """

    def __init__(self, params: CJParameters = None):
        self.params = params or CJParameters()

    def reservation_price(self, mid: float, q: int,
                          time_remaining: float,
                          alpha: float = 0.0) -> float:
        """
        CJ reservation price includes both inventory AND signal:
        r = S - q×γ×σ²×(T-t) + α_t×(T-t)
              ↑ inventory term   ↑ signal term (new vs A-S)
        """
        p = self.params
        return (mid
                - q * p.gamma * p.sigma**2 * time_remaining
                + alpha * time_remaining)

    def optimal_depths(self, q: int, time_remaining: float,
                       alpha: float = 0.0) -> tuple[float, float]:
        """
        CJ optimal bid/ask depths with signal term.

        δ*_b = spread_base + inventory_skew - signal_term
        δ*_a = spread_base - inventory_skew + signal_term

        where:
            spread_base  = (1/κ)×ln(1 + κ/γ)  ← same as GLFT asymptotic
            inventory_skew = (2|q|+1)/2 × γσ²(T-t)
            signal_term  = α_t / κ             ← NEW in CJ

        When α > 0 (price going up):
            bid depth decreases (quote more aggressively to buy)
            ask depth increases (quote less aggressively to sell)
        """
        p = self.params
        gamma = p.gamma
        kappa = p.kappa
        sigma = p.sigma
        T_t   = time_remaining

        # Base spread (asymptotic, same as GLFT)
        spread_base = (1/gamma) * np.log(1 + gamma/kappa)

        # Inventory skew
        inv_skew = ((2*q + 1) / 2) * gamma * sigma**2 * T_t

        # Signal term — this is CJ's key contribution
        signal_term = alpha / kappa

        # Optimal depths
        delta_bid = spread_base + inv_skew - signal_term
        delta_ask = spread_base - inv_skew + signal_term

        return max(0.0, delta_bid), max(0.0, delta_ask)

    def optimal_spread(self, time_remaining: float) -> float:
        """
        CJ symmetric spread (when q=0, α=0):
        spread = (2/κ)×ln(1 + κ/γ) + γ×σ²×(T-t)
        Same as GLFT asymptotic spread.
        """
        p = self.params
        return ((2/p.kappa) * np.log(1 + p.kappa/p.gamma) +
                p.gamma * p.sigma**2 * time_remaining)

    def optimal_quotes(self, mid: float, q: int,
                       time_remaining: float,
                       alpha: float = 0.0,
                       sigma: float = None) -> tuple[float, float]:
        """
        Full CJ optimal bid and ask prices.

        Args:
            mid:            current mid price
            q:              current inventory (lots)
            time_remaining: fraction of session remaining [0,1]
            alpha:          current alpha signal value
            sigma:          current volatility (overrides params.sigma)

        Returns:
            (bid_price, ask_price)
        """
        if sigma is not None:
            self.params.sigma = sigma

        r              = self.reservation_price(mid, q, time_remaining, alpha)
        delta_b, delta_a = self.optimal_depths(q, time_remaining, alpha)

        bid = round(r - delta_b, 4)
        ask = round(r + delta_a, 4)
        return bid, ask

    def running_penalty(self, q: int, dt: float = 1.0) -> float:
        """
        CJ running inventory penalty — charged every second.
        φ × q² × dt
        Encourages faster inventory reduction vs A-S terminal penalty.
        """
        return self.params.phi * q**2 * dt

    def terminal_penalty(self, q: int) -> float:
        """
        CJ terminal inventory penalty — charged at session end.
        α_terminal × q²
        """
        return self.params.alpha_terminal * q**2

    def alpha_update(self, alpha: float, new_signal: float,
                     dt: float = 1.0) -> float:
        """
        Update alpha signal using Ornstein-Uhlenbeck mean reversion:
        dα = -ζ×α×dt + new_signal

        Alpha decays toward zero (mean reversion) but is refreshed
        by new signal observations every second.

        Args:
            alpha:      current alpha value
            new_signal: new signal observation (from OFI, imbalance etc)
            dt:         time step in seconds

        Returns:
            updated alpha
        """
        zeta       = self.params.alpha_decay
        new_alpha  = alpha * np.exp(-zeta * dt) + new_signal
        return float(np.clip(new_alpha, -1, 1))