from dataclasses import dataclass

@dataclass
class GLFTParameters:
    gamma: float = 0.1
    sigma: float = 2.0
    kappa: float = 0.5
    A: float = 1.0
    T: float = 1.0
    N: int = 20
    M: int = 100
    max_inventory: int = 5
    tick_size: float = 0.05