import numpy as np
from strategies.implementations.avellaneda_stoikov.glft.parameters import GLFTParameters
from strategies.implementations.avellaneda_stoikov.glft.intensity import IntensityModel

class GLFTModel:
    def __init__(self, params: GLFTParameters = None):
        self.params    = params or GLFTParameters()
        self.intensity = IntensityModel(self.params)

    def solve_hjb(self) -> np.ndarray:
        """
        Solve HJB equation using finite differences (tridiagonal matrix).
        Returns value function w(t, q) for all inventory levels and times.
        """
        gamma = self.params.gamma
        sigma = self.params.sigma
        kappa = self.params.kappa
        A     = self.params.A
        N     = self.params.N
        M     = self.params.M
        q_max = self.params.max_inventory

        dt = self.params.T / M
        dq = 2 * q_max / N

        # Inventory grid from -q_max to +q_max
        q_grid = np.linspace(-q_max, q_max, N + 1)

        # Value function w[time, inventory]
        w = np.zeros((M + 1, N + 1))

        # Terminal condition: w(T, q) = 0 (no inventory penalty at terminal)
        # Some formulations add -alpha * q^2 here
        w[M, :] = 0.0

        # Backward induction
        for m in range(M - 1, -1, -1):
            # Build tridiagonal system
            diag  = np.ones(N + 1)
            lower = np.zeros(N + 1)
            upper = np.zeros(N + 1)
            rhs   = np.zeros(N + 1)

            for i in range(1, N):
                q = q_grid[i]

                # Diffusion term (sigma^2/2 * d^2w/dq^2)
                d2w = (w[m + 1, i + 1] - 2 * w[m + 1, i] + w[m + 1, i - 1]) / dq**2
                diff_term = 0.5 * sigma**2 * d2w

                # Optimal depths from asymptotic formula
                delta_b = (1/gamma) * np.log(1 + gamma/kappa) + (2*q + 1) * 0.5 * np.sqrt(gamma * sigma**2 / (2 * A * kappa))
                delta_a = (1/gamma) * np.log(1 + gamma/kappa) - (2*q - 1) * 0.5 * np.sqrt(gamma * sigma**2 / (2 * A * kappa))

                delta_b = max(0, delta_b)
                delta_a = max(0, delta_a)

                # Jump terms (fills)
                lambda_b = self.intensity.arrival_rate(delta_b)
                lambda_a = self.intensity.arrival_rate(delta_a)

                jump_b = lambda_b * (w[m + 1, min(i + 1, N)] - w[m + 1, i] + delta_b) if i < N else 0
                jump_a = lambda_a * (w[m + 1, max(i - 1, 0)] - w[m + 1, i] + delta_a) if i > 0 else 0

                rhs[i] = w[m + 1, i] + dt * (diff_term + jump_b + jump_a)
                diag[i] = 1.0

            # Boundary conditions (no inventory beyond limits)
            w[m, 0]  = w[m + 1, 0]
            w[m, N]  = w[m + 1, N]
            rhs[0]   = w[m, 0]
            rhs[N]   = w[m, N]

            # Thomas algorithm
            w[m, :] = self._thomas(diag, lower, upper, rhs)

        return w

    def _thomas(self, diag, lower, upper, rhs) -> np.ndarray:
        """Thomas algorithm for tridiagonal system."""
        n = len(diag)
        c = upper.copy()
        d = rhs.copy()
        x = np.zeros(n)

        c[0] /= diag[0]
        d[0] /= diag[0]

        for i in range(1, n):
            m = diag[i] - lower[i] * c[i - 1]
            c[i] /= m
            d[i] = (d[i] - lower[i] * d[i - 1]) / m

        x[-1] = d[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d[i] - c[i] * x[i + 1]

        return x

    def optimal_depths(self, q: int, time_remaining: float) -> tuple[float, float]:
        """
        Compute optimal bid and ask depths for current inventory and time.
        Uses asymptotic formula (fast, no need to solve full HJB each time).
        """
        gamma = self.params.gamma
        sigma = self.params.sigma
        kappa = self.params.kappa
        A     = self.params.A

        spread_component = (1/gamma) * np.log(1 + gamma/kappa)
        skew_component   = np.sqrt(gamma * sigma**2 / (2 * A * kappa)) * time_remaining

        delta_bid = spread_component + (2*q + 1) * skew_component / 2
        delta_ask = spread_component - (2*q - 1) * skew_component / 2

        return max(0.0, delta_bid), max(0.0, delta_ask)

    def reservation_price(self, mid: float, q: int,
                          sigma: float, time_remaining: float) -> float:
        """r = mid - q × γ × σ² × (T-t)"""
        return mid - q * self.params.gamma * sigma**2 * time_remaining

    def optimal_quotes(self, mid: float, q: int,
                       sigma: float, time_remaining: float) -> tuple[float, float]:
        """Returns (bid_price, ask_price)"""
        r              = self.reservation_price(mid, q, sigma, time_remaining)
        delta_b, delta_a = self.optimal_depths(q, time_remaining)
        return round(r - delta_b, 2), round(r + delta_a, 2)