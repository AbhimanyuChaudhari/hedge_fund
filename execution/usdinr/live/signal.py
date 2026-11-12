"""
execution/USDINR/live/signal.py
HMM forward pass + LightGBM predict → pred_norm.
"""

import pickle
import pathlib
import numpy as np
import lightgbm as lgb

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None

ROOT      = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


class SignalEngine:
    """
    Loads lgbm_model.txt and hmm_model.pkl once at startup.
    Call .predict(feat_vec, hmm_obs) each second.
    Returns (pred_norm, hmm_state, raw_pred_bps).
    """

    def __init__(self):
        lgbm_path = MODEL_DIR / "lgbm_model.txt"
        hmm_path  = MODEL_DIR / "hmm_model.pkl"

        if not lgbm_path.exists():
            raise FileNotFoundError(f"LightGBM model not found: {lgbm_path}")
        if not hmm_path.exists():
            raise FileNotFoundError(f"HMM model not found: {hmm_path}")

        self._lgbm = lgb.Booster(model_file=str(lgbm_path))

        with open(hmm_path, "rb") as f:
            self._hmm = pickle.load(f)

        self._daily_rets  = []
        self._daily_sigma = None
        self._hmm_obs     = []

    # ------------------------------------------------------------------
    def push_daily_ret(self, ret_60s_bps: float) -> None:
        self._daily_rets.append(ret_60s_bps)
        if len(self._daily_rets) >= 5:
            sigma = float(np.std(self._daily_rets, ddof=1))
            self._daily_sigma = sigma if sigma > 1e-9 else None

    def reset_day(self) -> None:
        self._daily_rets  = []
        self._daily_sigma = None
        self._hmm_obs     = []

    # ------------------------------------------------------------------
    def push_hmm_obs(self, obs: float) -> None:
        self._hmm_obs.append([obs])

    def current_hmm_state(self):
        if len(self._hmm_obs) < 10:
            return None
        obs_arr = np.array(self._hmm_obs, dtype=np.float64)
        try:
            states = self._hmm.predict(obs_arr)
        except Exception:
            return None
        return int(states[-1]) + 1   # 0-indexed → 1-indexed

    # ------------------------------------------------------------------
    def predict(self, feat_vec: np.ndarray, hmm_obs: float = None):
        """
        Returns
        -------
        pred_norm : float | None
        hmm_state : int   | None
        raw_pred  : float
        """
        raw_pred = float(self._lgbm.predict(feat_vec.reshape(1, -1))[0])

        if hmm_obs is not None:
            self.push_hmm_obs(hmm_obs)
        hmm_state = self.current_hmm_state()

        if self._daily_sigma is None:
            return None, hmm_state, raw_pred

        pred_norm = raw_pred / self._daily_sigma
        return pred_norm, hmm_state, raw_pred