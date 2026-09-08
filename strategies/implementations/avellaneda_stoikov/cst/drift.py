import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))

import numpy as np
from strategies.implementations.avellaneda_stoikov.cst.parameters import CSTParameters
from strategies.implementations.avellaneda_stoikov.cst.order_flow import OrderFlowRates


class DriftEstimator:
    def __init__(self, params: CSTParameters):
        self.params = params

    def _weighted_ofi(self, bars: list[dict]) -> float:
        """
        Cont-Kukanov-Stoikov (2014) weighted OFI across all 5 LOB levels.
        OFI_t = Σ_i w_i × (ΔBid_qi - ΔAsk_qi)
        w_i = 1 / distance_from_mid at level i

        Level 1 gets highest weight (closest to mid).
        Deeper levels get lower weight (further from mid).

        TODO (notebook 07): calibrate kyle_lambda via regression of
        Δprice on OFI_weighted to get β = kyle_lambda
        """
        if len(bars) < 2:
            return 0.0

        ofi_series = []
        for i in range(1, len(bars)):
            curr = bars[i]
            prev = bars[i - 1]
            mid  = (curr.get('bid_p1', 0) + curr.get('ask_p1', 0)) / 2
            if mid == 0:
                continue

            ofi = 0.0
            for level in range(1, 6):
                bid_p = curr.get(f'bid_p{level}', 0)
                ask_p = curr.get(f'ask_p{level}', 0)
                bid_q = curr.get(f'bid_q{level}', 0)
                ask_q = curr.get(f'ask_q{level}', 0)
                prev_bid_q = prev.get(f'bid_q{level}', 0)
                prev_ask_q = prev.get(f'ask_q{level}', 0)

                # Inverse distance weights
                w_bid = 1.0 / (mid - bid_p + 1e-9) if 0 < bid_p < mid else 0.0
                w_ask = 1.0 / (ask_p - mid + 1e-9) if ask_p > mid      else 0.0

                # Flow = change in resting quantity
                delta_bid = bid_q - prev_bid_q
                delta_ask = ask_q - prev_ask_q

                ofi += w_bid * delta_bid - w_ask * delta_ask

            ofi_series.append(ofi)

        return float(np.mean(ofi_series)) if ofi_series else 0.0

    def estimate(self, bars: list[dict], flow: OrderFlowRates) -> float:
        """
        Estimate price drift µ from:
        1. Weighted OFI (Cont-Kukanov-Stoikov 2014) — dominant signal
        2. Price momentum                            — trend component
        3. Cancel imbalance                          — structural signal

        Returns µ in price units per unit time (normalized to session).

        TODO (notebook 07): calibrate kyle_lambda via OFI regression
        TODO (notebook 09): calibrate ofi_weight, momentum_weight via IC
        """
        if len(bars) < 10:
            return 0.0

        # ── Signal 1: Weighted OFI (Cont-Kukanov-Stoikov) ────────────
        ofi_raw = self._weighted_ofi(bars)
        # Normalize to [-1, 1] range using sign + log scaling
        ofi_norm = float(np.sign(ofi_raw) * np.log1p(abs(ofi_raw)))
        ofi_norm = float(np.clip(ofi_norm, -1, 1))

        # ── Signal 2: Price momentum ──────────────────────────────────
        prices = [float(b.get('close', 0)) for b in bars
                  if float(b.get('close', 0)) > 0]
        if len(prices) >= 10:
            momentum = (prices[-1] - prices[-10]) / (prices[-10] + 1e-9)
        else:
            momentum = 0.0

        # ── Signal 3: Cancel imbalance (from CST order flow) ─────────
        # Positive cancel_imbalance → ask side canceling more → bullish
        cancel_signal = flow.cancel_imbalance * self.params.tick_size

        # ── Combined drift using Kyle's Lambda scaling ────────────────
        mu = (
            self.params.kyle_lambda * (
                self.params.ofi_weight      * ofi_norm +
                self.params.momentum_weight * momentum * 100
            ) +
            cancel_signal * 0.1  # TODO (notebook 08): calibrate cancel weight
        )

        # Clip to max 5 ticks of drift per second
        max_drift = self.params.tick_size * 5
        return float(np.clip(mu, -max_drift, max_drift))