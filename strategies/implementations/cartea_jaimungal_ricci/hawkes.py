import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
from dataclasses import dataclass


@dataclass
class HawkesParameters:
    """
    Hawkes process parameters for order arrival modeling.

    Standard exponential Hawkes process:
    λ(t) = µ_0 + Σ_{t_i < t} α_h × e^(-β × (t - t_i))

    where:
        µ_0  = baseline intensity (orders/second without excitation)
        α_h  = excitation factor (how much each order excites future orders)
        β    = decay rate (how fast excitation fades)
        t_i  = past event times

    Stationarity condition: α_h < β (branching ratio < 1)
    Branching ratio: n = α_h/β (average events triggered per event)

    Calibrated from USDINR: needs 15+ days of data
    """
    mu_0:  float = 0.5    # baseline intensity (orders/second)
                           # TODO: calibrate from tick frequency
    alpha: float = 0.3    # excitation factor
                           # TODO: calibrate from MLE on order timestamps
    beta:  float = 1.0    # decay rate
                           # TODO: calibrate from MLE
    window: int  = 300    # seconds of history for intensity estimation


class HawkesProcess:
    """
    Exponential Hawkes process for modeling order arrival clustering.

    Key insight: orders come in bursts.
    After a fill, the probability of another fill is temporarily higher.
    This clustering is captured by the self-exciting Hawkes process.

    Used in CJ-Ricci to replace fixed Poisson arrival rate κ
    with dynamic, self-exciting rate λ(t).
    """

    def __init__(self, params: HawkesParameters = None):
        self.params     = params or HawkesParameters()
        self._events    = []  # timestamps of past order events

    def intensity(self, t: float) -> float:
        """
        Compute current order arrival intensity λ(t).
        λ(t) = µ_0 + Σ_{t_i < t} α × e^(-β × (t - t_i))
        """
        p      = self.params
        kernel = sum(
            p.alpha * np.exp(-p.beta * (t - ti))
            for ti in self._events
            if t > ti
        )
        return p.mu_0 + kernel

    def add_event(self, t: float):
        """Record a new order event (fill, arrival) at time t."""
        self._events.append(t)
        # Keep only recent events within window
        cutoff = t - self.params.window
        self._events = [ti for ti in self._events if ti > cutoff]

    def branching_ratio(self) -> float:
        """
        n = α/β — average number of events triggered per event.
        Must be < 1 for stationarity.
        """
        return self.params.alpha / self.params.beta

    def stationary_intensity(self) -> float:
        """
        E[λ] = µ_0 / (1 - n) — expected intensity in steady state.
        """
        n = self.branching_ratio()
        if n >= 1:
            return float('inf')
        return self.params.mu_0 / (1 - n)

    def calibrate_from_bars(self, bars: list[dict],
                             tick_size: float = 0.0025) -> dict:
        """
        Estimate Hawkes parameters from 1-second bar data.

        Method: moment matching
        - µ_0 estimated from quiet periods (low volume bars)
        - α estimated from autocorrelation of trade counts
        - β estimated from decay of autocorrelation

        Args:
            bars: list of 1-second bars with 'volume_delta' field
            tick_size: instrument tick size

        Returns:
            dict with calibrated mu_0, alpha, beta
        """
        if len(bars) < 60:
            return {'mu_0': self.params.mu_0,
                    'alpha': self.params.alpha,
                    'beta': self.params.beta}

        # Trade counts per second
        counts = np.array([
            1 if abs(b.get('volume_delta', 0)) > 0 else 0
            for b in bars
        ])

        # µ_0: baseline = rate in quiet periods (bottom 25%)
        mu_0 = max(0.01, np.percentile(counts, 25))

        # Autocorrelation at lag 1 and lag 2
        if counts.std() > 0:
            ac1 = np.corrcoef(counts[:-1], counts[1:])[0, 1]
            ac2 = np.corrcoef(counts[:-2], counts[2:])[0, 1] if len(counts) > 2 else 0
        else:
            ac1 = ac2 = 0

        # β from autocorrelation decay: ac2/ac1 = e^(-β)
        if ac1 > 0.01 and ac2 > 0:
            beta = max(0.1, -np.log(max(ac2/ac1, 1e-9)))
        else:
            beta = self.params.beta

        # α from branching ratio: n = ac1 / (1 - ac1)
        n     = max(0, min(0.9, ac1 / (1 - ac1 + 1e-9)))
        alpha = n * beta

        # Update params
        self.params.mu_0  = float(mu_0)
        self.params.alpha = float(alpha)
        self.params.beta  = float(beta)

        return {
            'mu_0':           float(mu_0),
            'alpha':          float(alpha),
            'beta':           float(beta),
            'branching_ratio':float(n),
            'stationary_rate':float(self.stationary_intensity()),
        }

    def effective_kappa(self, current_intensity: float,
                        base_kappa: float = 1.5) -> float:
        """
        Scale κ by ratio of current to stationary intensity.
        When order flow is clustered (high intensity):
            κ_eff increases → fills more likely → quotes can be tighter
        When order flow is quiet (low intensity):
            κ_eff decreases → fills less likely → quotes need to be wider

        κ_eff = κ_base × (λ(t) / E[λ])
        """
        stat = self.stationary_intensity()
        if stat <= 0 or np.isinf(stat):
            return base_kappa
        ratio = current_intensity / stat
        return float(base_kappa * np.clip(ratio, 0.1, 5.0))