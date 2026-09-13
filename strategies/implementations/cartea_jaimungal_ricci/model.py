import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import time
from strategies.implementations.cartea_jaimungal.model import CJModel
from strategies.implementations.cartea_jaimungal_ricci.parameters import CJRicciParameters
from strategies.implementations.cartea_jaimungal_ricci.hawkes import HawkesProcess


class CJRicciModel:
    """
    Cartea-Jaimungal-Ricci market making model.

    Extends CJ with:
    1. Hawkes process order arrivals (self-exciting, not Poisson)
    2. Dynamic κ_eff based on real-time order clustering
    3. Spread adjusts to current market activity level

    When market is active (orders clustering):
        λ(t) > E[λ] → κ_eff increases → tighter spread OK
    When market is quiet:
        λ(t) < E[λ] → κ_eff decreases → need wider spread

    This is the Ricci extension: regime-aware market making.
    """

    def __init__(self, params: CJRicciParameters = None):
        self.params  = params or CJRicciParameters()
        self.cj      = CJModel(self.params.to_cj_params())
        self.hawkes  = HawkesProcess(self.params.to_hawkes_params())
        self._last_calib   = 0.0
        self._kappa_eff    = self.params.kappa
        self._hawkes_stats = {}

    def _update_hawkes(self, bars: list[dict]):
        """Recalibrate Hawkes every recalib_secs seconds."""
        now = time.time()
        if now - self._last_calib < self.params.recalib_secs:
            return
        self._hawkes_stats = self.hawkes.calibrate_from_bars(
            bars, self.params.tick_size
        )
        self._last_calib = now

    def _get_kappa_eff(self) -> float:
        """Get current effective κ from Hawkes intensity."""
        t         = time.time()
        intensity = self.hawkes.intensity(t)
        return self.hawkes.effective_kappa(intensity, self.params.kappa)

    def optimal_quotes(self, mid: float, q: int,
                       time_remaining: float,
                       alpha: float = 0.0,
                       sigma: float = None,
                       bars: list[dict] = None) -> tuple[float, float]:
        """
        CJ-Ricci optimal quotes with dynamic κ from Hawkes.

        Same as CJ but κ scales with current order flow intensity.
        """
        if bars:
            self._update_hawkes(bars)

        # Dynamic κ from Hawkes
        self._kappa_eff       = self._get_kappa_eff()
        self.cj.params.kappa  = self._kappa_eff

        if sigma is not None:
            self.cj.params.sigma = sigma

        return self.cj.optimal_quotes(
            mid=mid, q=q,
            time_remaining=time_remaining,
            alpha=alpha
        )

    def get_diagnostics(self) -> dict:
        """Return current Hawkes diagnostics for monitoring."""
        return {
            'kappa_eff':       round(self._kappa_eff, 4),
            'hawkes_intensity': round(self.hawkes.intensity(time.time()), 4),
            'stationary_rate': round(self.hawkes.stationary_intensity(), 4),
            'branching_ratio': round(self.hawkes.branching_ratio(), 4),
            **self._hawkes_stats,
        }