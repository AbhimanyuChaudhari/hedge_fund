import numpy as np
from strategies.implementations.avellaneda_stoikov.glft.parameters import GLFTParameters

class IntensityModel:
    def __init__(self, params:GLFTParameters):
        self.params = params

    def arrival_rate(self, delta:float) -> float:
        """λ(δ) = A × e^(-κδ) — probability of fill at depth δ"""
        return self.params.A * np.exp(-self.params.kappa*delta)

    def calibrate_from_depth(self, bid_prices: list[float], bid_qtys: list[float],
                             ask_prices: list[float], ask_qtys: list[float]) -> tuple:
        """
        Estimate A and κ from all 5 levels of order book depth.
        Fits exponential decay: q(δ) = A × e^(-κδ) using least squares.
        
        bid_prices: [bid_p1, bid_p2, bid_p3, bid_p4, bid_p5]
        bid_qtys:   [bid_q1, bid_q2, bid_q3, bid_q4, bid_q5]
        """
        def fit_side(prices, qtys, best_price):
            deltas = []
            qs = []
            for p, q in zip(prices, qtys):
                if q > 0 and p > 0:
                    delta = abs(p-best_price)
                    deltas.append(delta if delta > 0 else self.params.tick_size)
                    qs.append(q)
            if len(qs) < 2:
                return self.params.A, self.params.kappa
            log_q = np.log(np.array(qs))
            deltas = np.array(deltas)
            X = np.column_stack([np.ones(len(deltas)), deltas])
            try:
                coeffs = np.linalg.lstsq(X, log_q, rcond=None)[0]
                ln_A = coeffs[0]
                kappa = max(0.1, -coeffs[1])
                A = np.exp(ln_A)
            except Exception:
                return self.params.A, self.params.kappa
            return A, kappa
        best_bid = bid_prices[0] if bid_prices else 0
        best_ask = ask_prices[0] if ask_prices else 0
        A_bid, kappa_bid = fit_side(bid_prices, bid_qtys, best_bid)
        A_ask, kappa_ask = fit_side(ask_prices, ask_qtys, best_ask)
        A = (A_bid + A_ask)/2
        kappa = (kappa_bid+kappa_ask)/2
        return A, kappa

    def expected_fill_time(self, delta: float) -> float:
        """Expected time to fill at depth δ"""
        rate = self.arrival_rate(delta)
        return 1.0 / rate if rate > 0 else float('inf')
