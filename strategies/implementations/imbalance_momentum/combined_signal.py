import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import numpy as np
from dataclasses import dataclass


@dataclass
class CombinedSignalParameters:
    # ── Score threshold ────────────────────────────────────────────
    score_threshold:     float = 0.35   # min score to trade (TODO: calibrate)

    # ── Signal weights (sum to 1.0) ────────────────────────────────
    w_imbalance:         float = 0.20   # rolling imbalance
    w_momentum:          float = 0.20   # price momentum
    w_deep_imbalance:    float = 0.15   # levels 2-5 depth
    w_weighted_mid:      float = 0.15   # weighted mid divergence
    w_cancel:            float = 0.10   # cancel imbalance
    w_volume:            float = 0.10   # volume surge
    w_cst_drift:         float = 0.10   # CST µ drift signal

    # ── Signal windows ─────────────────────────────────────────────
    imbalance_window:    int   = 30     # bars for rolling imbalance
    momentum_window:     int   = 30     # bars for price momentum
    volume_window:       int   = 60     # bars for avg volume

    # ── Position management ────────────────────────────────────────
    max_inventory:       int   = 5
    lot_size:            int   = 1000
    stop_loss_ticks:     int   = 8
    take_profit_ticks:   int   = 16
    max_hold_seconds:    int   = 60     # shorter because combined IC is higher
    tick_size:           float = 0.0025
    instrument_type:     str   = 'currency_futures'

    # ── Filters ───────────────────────────────────────────────────
    min_bid_qty:         float = 500
    min_ask_qty:         float = 500
    min_volume_ratio:    float = 1.0    # min volume surge


class CombinedSignalGenerator:
    """
    Combines 7 signals into a single score for USDINR directional trading.
    Each signal normalized to [-1, +1] range.
    Score = weighted sum. Trade when |score| > threshold.
    """

    def __init__(self, params: CombinedSignalParameters = None):
        self.params = params or CombinedSignalParameters()

    def compute_score(self, bars: list[dict]) -> dict:
        """
        bars: list of 1-second bar dicts with keys:
            close, imbalance_last, total_bid_qty, total_ask_qty,
            bid_p1, ask_p1, bid_q1, ask_q1,
            bid_q2, bid_q3, bid_q4, bid_q5,
            ask_q2, ask_q3, ask_q4, ask_q5,
            volume_delta, weighted_mid
        Returns score dict with individual signals and total score.
        """
        if len(bars) < self.params.imbalance_window:
            return {'score': 0.0, 'signals': {}, 'valid': False}

        p = self.params

        # ── Signal 1: Rolling imbalance ────────────────────────────
        recent_imb = [b.get('imbalance_last', 0) for b in bars[-p.imbalance_window:]]
        s_imbalance = float(np.mean(recent_imb))
        s_imbalance = np.clip(s_imbalance, -1, 1)

        # ── Signal 2: Price momentum ───────────────────────────────
        prices = [b.get('close', 0) for b in bars if b.get('close', 0) > 0]
        if len(prices) >= p.momentum_window:
            mom = (prices[-1] - prices[-p.momentum_window]) / (prices[-p.momentum_window] + 1e-9)
            s_momentum = np.clip(mom * 1000, -1, 1)  # scale paise move to [-1,1]
        else:
            s_momentum = 0.0

        # ── Signal 3: Deep imbalance (levels 2-5) ──────────────────
        last = bars[-1]
        deep_bid = (last.get('bid_q2', 0) + last.get('bid_q3', 0) +
                    last.get('bid_q4', 0) + last.get('bid_q5', 0))
        deep_ask = (last.get('ask_q2', 0) + last.get('ask_q3', 0) +
                    last.get('ask_q4', 0) + last.get('ask_q5', 0))
        denom = deep_bid + deep_ask + 1e-9
        s_deep = float((deep_bid - deep_ask) / denom)
        s_deep = np.clip(s_deep, -1, 1)

        # ── Signal 4: Weighted mid divergence ──────────────────────
        bid_p1  = last.get('bid_p1', 0)
        ask_p1  = last.get('ask_p1', 0)
        bid_q1  = last.get('bid_q1', 1)
        ask_q1  = last.get('ask_q1', 1)
        mid     = (bid_p1 + ask_p1) / 2
        w_mid   = last.get('weighted_mid', mid)
        if w_mid == 0:
            w_mid = (bid_p1 * ask_q1 + ask_p1 * bid_q1) / (bid_q1 + ask_q1 + 1e-9)
        spread  = ask_p1 - bid_p1 + 1e-9
        s_wmid  = float(np.clip((w_mid - mid) / spread, -1, 1))

        # ── Signal 5: Cancel imbalance ─────────────────────────────
        cancel_signals = []
        for i in range(1, min(len(bars), 11)):
            curr = bars[-i]
            prev = bars[-i-1] if i+1 < len(bars) else bars[-i]
            price_unchanged = abs(curr.get('close', 0) - prev.get('close', 0)) < p.tick_size * 0.5
            if price_unchanged:
                bid_chg = curr.get('total_bid_qty', 0) - prev.get('total_bid_qty', 0)
                ask_chg = curr.get('total_ask_qty', 0) - prev.get('total_ask_qty', 0)
                if bid_chg < 0:
                    cancel_signals.append(-1)  # bid cancel → bearish
                if ask_chg < 0:
                    cancel_signals.append(+1)  # ask cancel → bullish
        s_cancel = float(np.mean(cancel_signals)) if cancel_signals else 0.0
        s_cancel = np.clip(s_cancel, -1, 1)

        # ── Signal 6: Volume surge ─────────────────────────────────
        vols     = [b.get('volume_delta', 0) for b in bars[-p.volume_window:]]
        avg_vol  = np.mean(vols) + 1e-9
        cur_vol  = vols[-1] if vols else 0
        vol_ratio = cur_vol / avg_vol
        # Positive volume surge → direction of imbalance
        s_volume = float(np.clip((vol_ratio - 1.0) * s_imbalance, -1, 1))

        # ── Signal 7: CST drift (µ) ────────────────────────────────
        if len(prices) >= 10:
            log_returns = np.diff(np.log(np.array(prices[-30:]) + 1e-9))
            mu = float(np.mean(log_returns))
            s_drift = float(np.clip(mu * 10000, -1, 1))
        else:
            s_drift = 0.0

        # ── Combined score ─────────────────────────────────────────
        score = (
            p.w_imbalance      * s_imbalance +
            p.w_momentum       * s_momentum  +
            p.w_deep_imbalance * s_deep      +
            p.w_weighted_mid   * s_wmid      +
            p.w_cancel         * s_cancel    +
            p.w_volume         * s_volume    +
            p.w_cst_drift      * s_drift
        )

        return {
            'score':       round(float(score), 4),
            'valid':       True,
            'signals': {
                'imbalance':    round(s_imbalance, 4),
                'momentum':     round(s_momentum,  4),
                'deep_imb':     round(s_deep,      4),
                'weighted_mid': round(s_wmid,      4),
                'cancel':       round(s_cancel,    4),
                'volume':       round(s_volume,    4),
                'drift':        round(s_drift,     4),
            }
        }

    def action(self, score_dict: dict, inventory: int) -> str:
        if not score_dict['valid']:
            return 'HOLD'
        score = score_dict['score']
        p     = self.params
        if inventory == 0:
            if score > p.score_threshold:
                return 'LONG'
            elif score < -p.score_threshold:
                return 'SHORT'
        return 'HOLD'