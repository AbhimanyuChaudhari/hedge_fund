import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))

import numpy as np
from strategies.implementations.avellaneda_stoikov.cst.parameters import CSTParameters
from strategies.implementations.avellaneda_stoikov.cst.order_flow import OrderFlowRates


class DriftEstimator:
    def __init__(self, params: CSTParameters):
        self.params = params

    def estimate(self, bars: list[dict], flow: OrderFlowRates) -> float:
        """
        Estimate price drift µ from:
        1. OFI (order flow imbalance) — immediate pressure
        2. Price momentum              — trend component
        3. Cancel imbalance            — structural signal

        Returns µ in price units per unit time (normalized to session).

        TODO (notebook 07): calibrate kyle_lambda via regression
        TODO (notebook 09): calibrate ofi_weight, momentum_weight via IC analysis
        """
        if len(bars) < 10:
            return 0.0

        # ── OFI component ─────────────────────────────────────────
        imbalances = [
            float(b.get('imbalance_last', 0))
            for b in bars
            if b.get('imbalance_last') is not None
        ]
        ofi = float(np.mean(imbalances)) if imbalances else 0.0

        # ── Price momentum component ───────────────────────────────
        prices = [float(b.get('close', 0)) for b in bars if float(b.get('close', 0)) > 0]
        if len(prices) >= 10:
            momentum = (prices[-1] - prices[-10]) / (prices[-10] + 1e-9)
        else:
            momentum = 0.0

        # ── Cancel imbalance component ─────────────────────────────
        # Positive cancel_imbalance → asks canceling more → price going up
        cancel_signal = flow.cancel_imbalance * self.params.tick_size

        # ── Combine with Kyle's Lambda scaling ────────────────────
        mu = (
            self.params.kyle_lambda * (
                self.params.ofi_weight      * ofi +
                self.params.momentum_weight * momentum * 100  # scale momentum
            ) +
            cancel_signal * 0.1  # cancel impact (TODO: notebook 08)
        )

        # Clip to reasonable range — max 5 ticks of drift per second
        max_drift = self.params.tick_size * 5
        return float(np.clip(mu, -max_drift, max_drift))