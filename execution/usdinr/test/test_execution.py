"""
tests/test_execution.py
Unit tests for ExecutionEngine — no real broker calls.
Run from project root: pytest execution/USDINR/tests/test_execution.py
"""
import sys
import pathlib
import pytest

# Make sure the live package is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from execution.usdinr.live.broker    import OrderType
from execution.usdinr.live.execution import ExecutionEngine, ClosedTrade
from execution.usdinr.live.config    import StrategyConfig

# ---------------------------------------------------------------------------
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
)


class FakeBroker:
    def __init__(self):
        self.orders = []

    def place_order(self, symbol, side, size, order_type, limit_price, lots=1):
        oid = f"O{len(self.orders)}"
        self.orders.append({"oid": oid, "side": side, "type": order_type})
        return oid

    def cancel_order(self, order_id):
        pass


def _eng():
    return ExecutionEngine(CFG, FakeBroker())


def test_no_entry_below_thresh():
    eng = _eng()
    entered = eng.try_entry("SYM", 1000, 83.0, pred_norm=0.69, state=1)
    assert not entered
    assert eng.is_flat


def test_entry_above_thresh():
    eng = _eng()
    entered = eng.try_entry("SYM", 1000, 83.0, pred_norm=0.75, state=1)
    assert entered
    assert not eng.is_flat


def test_state2_needs_higher_thresh():
    eng = _eng()
    entered = eng.try_entry("SYM", 1000, 83.0, pred_norm=0.80, state=2)
    assert not entered   # state2 needs 0.90

    entered = eng.try_entry("SYM", 1000, 83.0, pred_norm=0.91, state=2)
    assert entered


def test_tp_exit():
    eng = _eng()
    eng.try_entry("SYM", 1000, 83.0, pred_norm=0.75, state=4)
    pos = eng.position
    pos.fill_price = 83.0
    trade = eng.on_tick(1001, pos.tp_price + 0.001)
    assert trade is not None
    assert trade.exit_reason == "tp"
    assert eng.is_flat


def test_sl_exit():
    eng = _eng()
    eng.try_entry("SYM", 1000, 83.0, pred_norm=0.75, state=4)
    pos = eng.position
    pos.fill_price = 83.0
    trade = eng.on_tick(1001, pos.sl_price - 0.001)
    assert trade is not None
    assert trade.exit_reason == "sl"


def test_timeout_exit():
    eng = _eng()
    eng.try_entry("SYM", 1000, 83.0, pred_norm=0.75, state=4)
    pos = eng.position
    pos.fill_price = 83.0
    max_hold = CFG.state_max_hold[4]   # 180s
    trade = eng.on_tick(1000 + max_hold, 83.0)
    assert trade is not None
    assert trade.exit_reason == "timeout"


def test_net_pnl_includes_cost():
    eng = _eng()
    eng.try_entry("SYM", 1000, 83.0, pred_norm=0.75, state=4)
    pos = eng.position
    pos.fill_price = 83.0
    trade = eng.on_tick(1001, pos.tp_price + 0.001)
    assert trade is not None
    assert trade.net_ret_bps < trade.gross_ret_bps


def test_no_double_entry():
    eng = _eng()
    eng.try_entry("SYM", 1000, 83.0, pred_norm=0.75, state=1)
    second = eng.try_entry("SYM", 1001, 83.1, pred_norm=0.80, state=1)
    assert not second