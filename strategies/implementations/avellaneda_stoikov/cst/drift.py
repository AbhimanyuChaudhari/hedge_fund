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
        Raw imbalance — validated as best OFI signal for USDINR.
        Weighted OFI (CKS 2014) has NEGATIVE IC on USDINR due to
        tiny tick size (0.0025) — inverse distance weights amplify noise.

        For equity futures with larger tick sizes, switch to:
            ofi = Σ_i (1/distance_i) × (ΔBid_qi - ΔAsk_qi)

        Calibrated: IC = 0.051 at 120s, t-stat = 19.07 at 60s
        TODO (notebook 07): revalidate with 15+ days data
        """
        if len(bars) < 2:
            return 0.0
        imbalances = [b.get('imbalance_last', 0) for b in bars
                      if b.get('imbalance_last') is not None]
        return float(np.mean(imbalances)) if imbalances else 0.0

    def estimate(self, bars: list[dict], flow: OrderFlowRates) -> float:
        """
        Estimate price drift µ using calibrated weights from Ridge regression.

        Formula (calibrated Sep 2026, 4 days USDINR live data):
            µ = kyle_lambda × (ofi_weight×OFI - momentum_weight×momentum)
                - cancel_drift_weight × cancel_imb

        KEY FINDINGS from calibration:
        - momentum_weight is NEGATIVE — USDINR mean-reverts at 30-120s
        - cancel_drift_weight is NEGATIVE — ask cancels predict price DOWN
        - ofi_weight is small (0.11) — imbalance has modest impact
        - kyle_lambda = 0.00148 (from OLS: Δprice = λ×imbalance + ε)
        - IC of combined signal = 0.33 at 60s (R² = 0.11)

        Was hardcoded: kyle_lambda=0.01, ofi=0.6, mom=+0.4, cancel=0.1
        Now calibrated: kyle_lambda=0.00148, ofi=0.11, mom=-0.79, cancel=-0.11

        TODO (notebook 07): recalibrate with 15+ days of data
        TODO (notebook 09): validate IC stability across regimes
        """
        if len(bars) < 10:
            return 0.0

        p = self.params

        # ── Signal 1: OFI (raw imbalance — best for USDINR) ──────────
        ofi      = self._weighted_ofi(bars)
        ofi_norm = float(np.clip(ofi, -1, 1))

        # ── Signal 2: Price momentum ──────────────────────────────────
        prices = [float(b.get('close', 0)) for b in bars
                  if float(b.get('close', 0)) > 0]
        if len(prices) >= 10:
            momentum = (prices[-1] - prices[-10]) / (prices[-10] + 1e-9)
        else:
            momentum = 0.0

        # ── Signal 3: Cancel imbalance ────────────────────────────────
        cancel_signal = flow.cancel_imbalance * p.tick_size

        # ── Combined drift (calibrated weights) ───────────────────────
        # Momentum: NEGATIVE sign (mean reversion on USDINR)
        # Cancel:   NEGATIVE sign (ask cancels predict down, not up)
        mu = (
            p.kyle_lambda * (
                p.ofi_weight      *  ofi_norm           +
                p.momentum_weight * -momentum * 100      # NEGATIVE — mean reversion
            ) +
            p.cancel_drift_weight * -cancel_signal       # NEGATIVE
        )

        max_drift = p.tick_size * 5
        return float(np.clip(mu, -max_drift, max_drift))