"""
execution/USDINR/live/config.py
Loads strategy_config.json + validates all params at startup.
"""

import json
import pathlib
from dataclasses import dataclass
from typing import Dict

ROOT        = pathlib.Path(__file__).resolve().parent.parent   # execution/USDINR/
CONFIG_PATH = ROOT / "models" / "strategy_config.json"
MODEL_DIR   = ROOT / "models"
LOG_DIR     = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class StrategyConfig:
    # thresholds
    base_thresh:    float
    state2_thresh:  float

    # costs / exits
    cost_bps:       float
    c_sl:           float

    # per-state dicts
    state_tp:       Dict[int, float]
    state_vol:      Dict[int, float]
    state_max_hold: Dict[int, int]
    state_size:     Dict[int, float]

    # time window (IST seconds-since-midnight)
    window_start:   int
    window_end:     int
    skip_start:     int
    skip_end:       int

    # Cartea-Jaimungal order logic
    cj_market_thresh: float = 0.90
    cj_limit_thresh:  float = 0.70
    cj_limit_timeout: int   = 1

    # risk
    daily_loss_limit_bps: float = 10.0
    max_open_positions:   int   = 1


def load_config(path: pathlib.Path = CONFIG_PATH) -> StrategyConfig:
    with open(path) as f:
        raw = json.load(f)

    def ikeys(d):
        return {int(k): v for k, v in d.items()}

    cfg = StrategyConfig(
        base_thresh    = float(raw["base_thresh"]),
        state2_thresh  = float(raw["state2_thresh"]),
        cost_bps       = float(raw["cost_bps"]),
        c_sl           = float(raw["c_sl"]),
        state_tp       = ikeys(raw["state_tp"]),
        state_vol      = ikeys(raw["state_vol"]),
        state_max_hold = ikeys(raw["state_max_hold"]),
        state_size     = ikeys(raw["state_size"]),
        window_start   = int(raw["window_start"]),
        window_end     = int(raw["window_end"]),
        skip_start     = int(raw["skip_start"]),
        skip_end       = int(raw["skip_end"]),
    )

    for attr in ("cj_market_thresh", "cj_limit_thresh", "cj_limit_timeout",
                 "daily_loss_limit_bps", "max_open_positions"):
        if attr in raw:
            object.__setattr__(cfg, attr, type(getattr(cfg, attr))(raw[attr]))

    _validate(cfg)
    return cfg


def _validate(cfg: StrategyConfig):
    assert 4 == len(cfg.state_tp) == len(cfg.state_vol) == \
           len(cfg.state_max_hold) == len(cfg.state_size), \
        "All per-state dicts must have exactly 4 entries (states 1-4)"
    for s in [1, 2, 3, 4]:
        assert s in cfg.state_tp, f"state_tp missing state {s}"
    assert cfg.window_start < cfg.skip_start < cfg.skip_end < cfg.window_end, \
        "Window / skip times are inconsistent"
    assert 0 < cfg.base_thresh <= 1.0
    assert 0 < cfg.state2_thresh <= 1.0