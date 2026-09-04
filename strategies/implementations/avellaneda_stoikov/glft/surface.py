import numpy as np
from strategies.implementations.avellaneda_stoikov.glft.model import GLFTModel
from strategies.implementations.avellaneda_stoikov.glft.parameters import GLFTParameters

class QuoteSurface:
    """
    Precomputes full bid/ask quote surface at session start.
    Lookup is O(1) during live trading — no recomputation needed.
    """
    def __init__(self, params: GLFTParameters = None):
        self.params = params or GLFTParameters()
        self.model  = GLFTModel(self.params)
        self._bid_surface = None
        self._ask_surface = None
        self._q_grid      = None
        self._t_grid      = None

    def build(self):
        """Precompute full quote surface. Call once at session start."""
        N     = self.params.N
        M     = self.params.M
        q_max = self.params.max_inventory

        self._q_grid = np.linspace(-q_max, q_max, N + 1)
        self._t_grid = np.linspace(0, self.params.T, M + 1)

        self._bid_surface = np.zeros((M + 1, N + 1))
        self._ask_surface = np.zeros((M + 1, N + 1))

        for m, t in enumerate(self._t_grid):
            time_remaining = self.params.T - t
            for i, q in enumerate(self._q_grid):
                db, da = self.model.optimal_depths(int(q), time_remaining)
                self._bid_surface[m, i] = db
                self._ask_surface[m, i] = da

        print(f"Quote surface built: {N+1} inventory × {M+1} time steps")

    def lookup(self, q: int, time_remaining: float) -> tuple[float, float]:
        """O(1) lookup — returns (delta_bid, delta_ask)"""
        if self._bid_surface is None:
            self.build()

        q_max = self.params.max_inventory
        q_idx = int(np.clip(q + q_max, 0, self.params.N))
        t_idx = int(np.clip(
            (1 - time_remaining / self.params.T) * self.params.M,
            0, self.params.M
        ))

        return float(self._bid_surface[t_idx, q_idx]), \
               float(self._ask_surface[t_idx, q_idx])