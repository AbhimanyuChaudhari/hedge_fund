"""
tests/test_risk.py
Unit tests for RiskManager.
Run from project root: pytest execution/USDINR/tests/test_risk.py
"""
import sys
import pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from execution.usdinr.live.risk    import RiskManager, RiskBreachError
from execution.usdinr.live.config  import StrategyConfig

CFG = StrategyConfig(
    base_thresh    = 0.70,
    state2_thresh  = 0.90,
    cost_bps       = 1.0,
    c_sl           = 1.2,
    state_tp       = {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.60},
    state_vol      = {1: 1.77, 2: 2.37, 3: 1.61, 4: 1.03},
    state_max_hold = {1: 60, 2: 30, 3: 90, 4: 180},
    state_size     = {1: 0.59, 2: 0.44, 3: 0.65, 4: 1.00},
    window_start   = 37800,
    window_end     = 52200,
    skip_start     = 46800,
    skip_end       = 50400,
    daily_loss_limit_bps = 10.0,
)


def test_allows_entry_normally():
    rm = RiskManager(CFG)
    rm.check_entry()   # should not raise


def test_blocks_after_loss_limit():
    rm = RiskManager(CFG)
    rm.record_trade(-11.0)
    with pytest.raises(RiskBreachError):
        rm.check_entry()


def test_kill_switch_blocks():
    rm = RiskManager(CFG)
    rm.trigger_kill_switch("test")
    with pytest.raises(RiskBreachError):
        rm.check_entry()


def test_reset_day_re_enables():
    rm = RiskManager(CFG)
    rm.record_trade(-11.0)
    rm.reset_day()
    rm.check_entry()   # should not raise after reset


def test_daily_pnl_accumulates():
    rm = RiskManager(CFG)
    rm.record_trade(2.0)
    rm.record_trade(-1.0)
    assert abs(rm.daily_pnl_bps - 1.0) < 1e-9