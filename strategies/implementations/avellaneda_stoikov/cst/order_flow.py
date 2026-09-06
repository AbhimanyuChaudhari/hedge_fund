import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))

import numpy as np
from dataclasses import dataclass
from strategies.implementations.avellaneda_stoikov.cst.parameters import CSTParameters


@dataclass
class OrderFlowRates:
    lambda_market:     float  # market order arrival rate
    lambda_limit:      float  # limit order arrival rate
    lambda_cancel_bid: float  # bid side cancellation rate
    lambda_cancel_ask: float  # ask side cancellation rate
    lambda_cancel_avg: float  # average cancellation rate
    kappa_eff:         float  # effective κ adjusted for cancels
    cancel_imbalance:  float  # ask_cancel - bid_cancel (positive = bullish)


class OrderFlowEstimator:
    def __init__(self, params: CSTParameters):
        self.params = params

    def estimate(self, bars: list[dict]) -> OrderFlowRates:
        """
        Estimate all order flow rates from 1-second bar history.

        bars: list of dicts with keys:
            volume_delta, total_bid_qty, total_ask_qty,
            bid_p1, ask_p1, close
        """
        n = len(bars)
        if n < 2:
            return self._defaults()

        lambda_market     = 0.0
        lambda_limit      = 0.0
        lambda_cancel_bid = 0.0
        lambda_cancel_ask = 0.0

        for i in range(1, n):
            curr = bars[i]
            prev = bars[i - 1]

            vol_delta      = float(curr.get('volume_delta', 0))
            bid_qty_curr   = float(curr.get('total_bid_qty', 0))
            bid_qty_prev   = float(prev.get('total_bid_qty', 0))
            ask_qty_curr   = float(curr.get('total_ask_qty', 0))
            ask_qty_prev   = float(prev.get('total_ask_qty', 0))
            price_curr     = float(curr.get('close', 0))
            price_prev     = float(prev.get('close', 0))
            price_unchanged = abs(price_curr - price_prev) < self.params.tick_size * 0.5

            # Market order — volume traded this second
            if vol_delta > 0:
                lambda_market += 1

            if price_unchanged:
                # Limit order arrival — depth grew without trade
                bid_delta = bid_qty_curr - bid_qty_prev
                ask_delta = ask_qty_curr - ask_qty_prev

                if bid_delta > 0:
                    lambda_limit += 1
                if ask_delta > 0:
                    lambda_limit += 0.5  # ask side limit

                # Cancellation — depth shrank without trade
                if bid_delta < 0:
                    lambda_cancel_bid += abs(bid_delta) / max(bid_qty_prev, 1)
                if ask_delta < 0:
                    lambda_cancel_ask += abs(ask_delta) / max(ask_qty_prev, 1)

        # Normalize to per-second rates
        lambda_market     /= n
        lambda_limit      /= n
        lambda_cancel_bid /= n
        lambda_cancel_ask /= n
        lambda_cancel_avg  = (lambda_cancel_bid + lambda_cancel_ask) / 2

        # Effective κ — adjusted for cancellations
        lambda_total = lambda_market + lambda_limit + 1e-9
        kappa_eff    = self.params.kappa * (1 - lambda_cancel_avg / lambda_total)
        kappa_eff    = max(0.1, kappa_eff)

        # Cancel imbalance — positive means ask side canceling more → bullish
        cancel_imbalance = lambda_cancel_ask - lambda_cancel_bid

        return OrderFlowRates(
            lambda_market     = lambda_market,
            lambda_limit      = lambda_limit,
            lambda_cancel_bid = lambda_cancel_bid,
            lambda_cancel_ask = lambda_cancel_ask,
            lambda_cancel_avg = lambda_cancel_avg,
            kappa_eff         = kappa_eff,
            cancel_imbalance  = cancel_imbalance,
        )

    def _defaults(self) -> OrderFlowRates:
        return OrderFlowRates(
            lambda_market     = 0.5,
            lambda_limit      = 0.5,
            lambda_cancel_bid = 0.1,
            lambda_cancel_ask = 0.1,
            lambda_cancel_avg = 0.1,
            kappa_eff         = self.params.kappa,
            cancel_imbalance  = 0.0,
        )