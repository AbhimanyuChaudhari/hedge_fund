from dataclasses import dataclass
from strategies.implementations.cartea_jaimungal.parameters import CJParameters
from strategies.implementations.cartea_jaimungal_ricci.hawkes import HawkesParameters


@dataclass
class CJRicciParameters:
    """
    Cartea-Jaimungal-Ricci parameters.
    Extends CJ with Hawkes process order arrivals.

    Key difference from CJ:
    - Order arrival rate λ(t) is dynamic (Hawkes) not fixed (Poisson)
    - κ_eff varies in real time based on recent order clustering
    - When orders cluster → κ_eff high → tighter spreads
    - When orders quiet → κ_eff low → wider spreads
    """

    # ── CJ base parameters ─────────────────────────────────────────
    gamma:          float = 0.1
    sigma:          float = 2.0
    kappa:          float = 1.5    # base κ — scaled by Hawkes
    A:              float = 1.0
    T:              float = 1.0
    tick_size:      float = 0.05
    max_inventory:  int   = 5
    phi:            float = 0.001
    alpha_terminal: float = 0.01
    alpha_decay:    float = 0.5
    alpha_vol:      float = 0.1

    # ── Hawkes parameters ──────────────────────────────────────────
    hawkes_mu_0:    float = 0.5    # baseline intensity
    hawkes_alpha:   float = 0.3    # excitation factor
    hawkes_beta:    float = 1.0    # decay rate
    hawkes_window:  int   = 300    # calibration window (seconds)
    recalib_secs:   int   = 10

    # ── Flow estimation ────────────────────────────────────────────
    flow_window:    int   = 60

    def to_cj_params(self) -> CJParameters:
        """Convert to CJ parameters (for base model)."""
        return CJParameters(
            gamma         = self.gamma,
            sigma         = self.sigma,
            kappa         = self.kappa,
            A             = self.A,
            T             = self.T,
            tick_size     = self.tick_size,
            max_inventory = self.max_inventory,
            phi           = self.phi,
            alpha_terminal= self.alpha_terminal,
            alpha_decay   = self.alpha_decay,
            alpha_vol     = self.alpha_vol,
        )

    def to_hawkes_params(self) -> HawkesParameters:
        """Convert to Hawkes parameters."""
        return HawkesParameters(
            mu_0   = self.hawkes_mu_0,
            alpha  = self.hawkes_alpha,
            beta   = self.hawkes_beta,
            window = self.hawkes_window,
        )