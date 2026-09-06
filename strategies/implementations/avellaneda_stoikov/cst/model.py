import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))

import numpy as np
from strategies.implementations.avellaneda_stoikov.glft.model import GLFTModel
from strategies.implementations.avellaneda_stoikov.cst.parameters import CSTParameters
from strategies.implementations.avellaneda_stoikov.cst.order_flow import OrderFlowRates


class CSTModel(GLFTModel):
    """
    Cont-Stoikov-Talreja model.
    Extends GLFT with:
    - Price drift µ (from OFI + momentum + cancel imbalance)
    - Cancel-adjusted volatility σ_eff
    - Cancel-adjusted arrival rate κ_eff
    - Cancel drift impact on reservation price
    """

    def __init__(self, params: CSTParameters = None):
        self.cst_params = params or CSTParameters()
        # Pass compatible params to GLFT parent
        from strategies.implementations.avellaneda_stoikov.glft.parameters import GLFTParameters
        glft_params = GLFTParameters(
            gamma         = self.cst_params.gamma,
            sigma         = self.cst_params.sigma,
            kappa         = self.cst_params.kappa,
            A             = self.cst_params.A,
            T             = self.cst_params.T,
            max_inventory = self.cst_params.max_inventory,
            tick_size     = self.cst_params.tick_size,
        )
        super().__init__(glft_params)
        self.cst_params = params or CSTParameters()

    def effective_sigma(self, sigma: float, flow: OrderFlowRates) -> float:
        """
        σ²_eff = σ² + δ² × (λ_cancel_bid + λ_cancel_ask)
        Cancellations add extra realized volatility.
        """
        cancel_vol = (
            self.cst_params.tick_size ** 2 *
            (flow.lambda_cancel_bid + flow.lambda_cancel_ask) *
            self.cst_params.cancel_vol_weight
        )
        sigma_eff_sq = sigma ** 2 + cancel_vol
        return float(np.sqrt(sigma_eff_sq))

    def cancel_drift(self, flow: OrderFlowRates) -> float:
        """
        Price drift from cancel imbalance.
        cancel_drift = δ × (λ_cancel_ask - λ_cancel_bid)
        Positive → ask side canceling more → price going up
        """
        if not self.cst_params.cancel_drift_on:
            return 0.0
        return self.cst_params.tick_size * flow.cancel_imbalance

    def reservation_price(self, mid: float, q: int, sigma: float,
                          time_remaining: float, mu: float = 0.0,
                          flow: OrderFlowRates = None) -> float:
        """
        CST reservation price:
        r = mid + µ×(T-t) + cancel_drift×(T-t) - q×γ×σ²_eff×(T-t)
        """
        c_drift = self.cancel_drift(flow) if flow else 0.0
        sigma_e = self.effective_sigma(sigma, flow) if flow else sigma

        return (
            mid
            + mu       * time_remaining
            + c_drift  * time_remaining
            - q * self.cst_params.gamma * sigma_e ** 2 * time_remaining
        )

    def optimal_spread(self, sigma: float, time_remaining: float,
                       flow: OrderFlowRates = None) -> float:
        """
        CST spread uses σ_eff and κ_eff instead of raw values.
        """
        sigma_e = self.effective_sigma(sigma, flow) if flow else sigma
        kappa   = flow.kappa_eff if flow else self.cst_params.kappa
        gamma   = self.cst_params.gamma

        return (
            gamma * sigma_e ** 2 * time_remaining +
            (2 / gamma) * np.log(1 + gamma / kappa)
        )

    def optimal_quotes(self, mid: float, q: int, sigma: float,
                       time_remaining: float, mu: float = 0.0,
                       flow: OrderFlowRates = None) -> tuple[float, float]:
        """
        Full CST optimal quotes.
        """
        r      = self.reservation_price(mid, q, sigma, time_remaining, mu, flow)
        spread = self.optimal_spread(sigma, time_remaining, flow)
        bid    = round(r - spread / 2, 2)
        ask    = round(r + spread / 2, 2)
        return bid, ask