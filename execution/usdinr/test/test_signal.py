"""
tests/test_signal.py
Unit tests for SignalEngine — stubs out hmmlearn and lightgbm,
no model files needed.
Run from project root: pytest execution/usdinr/test/test_signal.py
"""
import sys
import types
import pathlib
import numpy as np
import pytest

# --- stub hmmlearn ---
hmm_stub = types.ModuleType("hmmlearn")
hmm_stub.hmm = types.ModuleType("hmmlearn.hmm")


class _FakeHMM:
    def predict(self, obs):
        return np.zeros(len(obs), dtype=int)


hmm_stub.hmm.GaussianHMM = _FakeHMM
sys.modules.setdefault("hmmlearn", hmm_stub)
sys.modules.setdefault("hmmlearn.hmm", hmm_stub.hmm)

# --- stub lightgbm ---
lgb_stub = types.ModuleType("lightgbm")


class _FakeBooster:
    def predict(self, X):
        return np.full(len(X), 0.5)


lgb_stub.Booster = lambda **kw: _FakeBooster()
sys.modules.setdefault("lightgbm", lgb_stub)

# --- import SignalEngine and patch __init__ to skip disk I/O ---
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from execution.usdinr.live.signal import SignalEngine


def _fake_init(self):
    self._lgbm        = _FakeBooster()
    self._hmm         = _FakeHMM()
    self._daily_rets  = []
    self._daily_sigma = None
    self._hmm_obs     = []


SignalEngine.__init__ = _fake_init


def test_pred_norm_none_before_sigma():
    eng = SignalEngine()
    vec = np.zeros(50, dtype=np.float32)
    pn, state, raw = eng.predict(vec)
    assert pn is None


def test_pred_norm_after_sigma():
    eng = SignalEngine()
    # feed varied values so std > 0
    for i in range(10):
        eng.push_daily_ret(1.0 + i * 0.1)
    vec = np.zeros(50, dtype=np.float32)
    pn, state, raw = eng.predict(vec)
    assert pn is not None
    assert isinstance(pn, float)


def test_reset_day_clears_state():
    eng = SignalEngine()
    for i in range(10):
        eng.push_daily_ret(1.0 + i * 0.1)
    eng.reset_day()
    assert eng._daily_sigma is None
    assert eng._daily_rets == []


def test_hmm_state_none_before_10_obs():
    eng = SignalEngine()
    for _ in range(9):
        eng.push_hmm_obs(1.0)
    assert eng.current_hmm_state() is None


def test_hmm_state_1indexed_after_10_obs():
    eng = SignalEngine()
    for _ in range(10):
        eng.push_hmm_obs(1.0)
    s = eng.current_hmm_state()
    assert s in [1, 2, 3, 4], f"Expected 1-4, got {s}"