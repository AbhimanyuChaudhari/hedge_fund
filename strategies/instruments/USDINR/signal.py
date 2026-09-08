"""
USDINR-specific signal generator.
Uses 3-signal combined scorer validated from IC analysis:
  1. Cancel imbalance (inverted, weight 0.50) — strongest signal
  2. Raw imbalance   (normal,   weight 0.30) — second
  3. Weighted mid divergence (weight 0.20)   — third

Market hours: 9:00am - 5:00pm IST (CDS exchange)
Avoid: 3pm IST (IC = -0.20, mean reversion dominates)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import time
import logging
import numpy as np
from collections import deque
from datetime import datetime, timezone
from data.store.redis_client import RedisClient
from execution.risk.transaction_costs import TransactionCosts
from strategies.instruments.USDINR.config import USDINRConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USDINRSignal:
    """
    USDINR directional signal using combined 3-signal scorer.
    Validated IC = 0.1345 at 60s horizon on 4 days of live data.
    """

    def __init__(self, token: int, lots: int = 20,
                 config: USDINRConfig = None):
        self.token   = str(token)
        self.lots    = lots
        self.config  = config or USDINRConfig()
        self.redis   = RedisClient()
        self.costs   = TransactionCosts(
            lot_size        = self.config.lot_size,
            instrument_type = self.config.instrument_type
        )
        self.inventory     = 0
        self._entry_price  = 0.0
        self._entry_time   = 0
        self._position_side = ''

        # Rolling buffers — deque for O(1) append/pop
        W = self.config.norm_window + self.config.signal_window + 10
        self._price_buf    = deque(maxlen=W)
        self._imb_buf      = deque(maxlen=W)
        self._bid_qty_buf  = deque(maxlen=W)
        self._ask_qty_buf  = deque(maxlen=W)
        self._wmid_buf     = deque(maxlen=W)
        self._cancel_buf   = deque(maxlen=W)

    def _ist_seconds(self) -> int:
        return (int(datetime.now(timezone.utc).timestamp()) + 19800) % 86400

    def _update_buffers(self, tick: dict, orderbook: dict):
        """Update rolling buffers from latest Redis data."""
        ltp      = float(tick.get('ltp', 0))
        bids     = orderbook.get('bids', [])
        asks     = orderbook.get('asks', [])
        if not bids or not asks or ltp == 0:
            return

        bid_p1   = bids[0].get('price',    ltp - self.config.tick_size)
        ask_p1   = asks[0].get('price',    ltp + self.config.tick_size)
        total_bid = sum(b.get('quantity', 0) for b in bids)
        total_ask = sum(a.get('quantity', 0) for a in asks)
        bid_q1    = bids[0].get('quantity', 0)
        ask_q1    = asks[0].get('quantity', 0)
        mid       = (bid_p1 + ask_p1) / 2
        w_mid     = (bid_p1 * ask_q1 + ask_p1 * bid_q1) / (bid_q1 + ask_q1 + 1e-9)

        imbalance = (total_bid - total_ask) / (total_bid + total_ask + 1e-9)

        # Cancel detection — depth decreased without price change
        cancel = 0.0
        if len(self._price_buf) > 0:
            prev_price = self._price_buf[-1]
            price_unch = abs(ltp - prev_price) < self.config.tick_size * 0.5
            if price_unch:
                prev_bid = self._bid_qty_buf[-1] if self._bid_qty_buf else total_bid
                prev_ask = self._ask_qty_buf[-1] if self._ask_qty_buf else total_ask
                if total_bid < prev_bid: cancel -= (prev_bid - total_bid)
                if total_ask < prev_ask: cancel += (prev_ask - total_ask)

        self._price_buf.append(ltp)
        self._imb_buf.append(imbalance)
        self._bid_qty_buf.append(total_bid)
        self._ask_qty_buf.append(total_ask)
        self._wmid_buf.append(w_mid - mid)
        self._cancel_buf.append(cancel)

    def _compute_score(self) -> float:
        """
        Combined 3-signal score normalized to [-1, +1].
        Positive = bullish, Negative = bearish.
        """
        W  = self.config.signal_window
        NW = self.config.norm_window

        if len(self._imb_buf) < W + 10:
            return 0.0

        imb_arr    = np.array(self._imb_buf)
        cancel_arr = np.array(self._cancel_buf)
        wmid_arr   = np.array(self._wmid_buf)

        # Raw signals
        s_imb    =  np.mean(imb_arr[-W:])
        s_cancel = -np.mean(cancel_arr[-W:])  # INVERTED — negative IC
        s_wmid   =  np.mean(wmid_arr[-W:])

        # Normalize by rolling std
        def norm(arr, val):
            std = np.std(arr[-NW:]) if len(arr) >= NW else np.std(arr)
            return float(np.clip(val / (std + 1e-9), -1, 1))

        s_imb    = norm(imb_arr,    s_imb)
        s_cancel = norm(cancel_arr, s_cancel)
        s_wmid   = norm(wmid_arr,   s_wmid)

        return (self.config.w_cancel    * s_cancel +
                self.config.w_imbalance * s_imb    +
                self.config.w_wmid      * s_wmid)

    def _check_exit(self, price: float, ist_sec: int) -> str | None:
        if self.inventory == 0:
            return None

        ticks = (price - self._entry_price) / self.config.tick_size

        if self._position_side == 'LONG':
            if ticks < -self.config.stop_loss_ticks:    return 'STOP_LOSS'
            if ticks >  self.config.take_profit_ticks:  return 'TAKE_PROFIT'
        elif self._position_side == 'SHORT':
            if ticks >  self.config.stop_loss_ticks:    return 'STOP_LOSS'
            if ticks < -self.config.take_profit_ticks:  return 'TAKE_PROFIT'

        if ist_sec - self._entry_time > self.config.max_hold_seconds:
            return 'TIME_STOP'

        return None

    def generate(self) -> dict | None:
        ist = self._ist_seconds()
        cfg = self.config

        # Market hours check
        if ist < cfg.market_open_ist or ist > cfg.market_close_ist:
            return None

        # Avoid bad hours
        hour = ist // 3600
        if hour in cfg.avoid_hours:
            return None

        tick      = self.redis.get_tick(self.token)
        orderbook = self.redis.get_orderbook(self.token)
        if not tick or not orderbook:
            return None

        self._update_buffers(tick, orderbook)

        bids   = orderbook.get('bids', [])
        asks   = orderbook.get('asks', [])
        ltp    = float(tick.get('ltp', 0))
        bid_p1 = bids[0].get('price', ltp - cfg.tick_size) if bids else ltp - cfg.tick_size
        ask_p1 = asks[0].get('price', ltp + cfg.tick_size) if asks else ltp + cfg.tick_size
        mid    = (bid_p1 + ask_p1) / 2
        score  = self._compute_score()

        # Force close near session end
        if ist >= cfg.market_close_ist - 120 and self.inventory != 0:
            return {
                'action':   'FORCE_CLOSE',
                'symbol':   cfg.symbol,
                'score':    score,
                'ltp':      ltp,
                'inventory':self.inventory,
                'side':     self._position_side,
            }

        # Check exit
        exit_reason = self._check_exit(ltp, ist)
        if exit_reason:
            return {
                'action':      f'EXIT_{exit_reason}',
                'symbol':      cfg.symbol,
                'score':       round(score, 4),
                'ltp':         ltp,
                'bid_p1':      bid_p1,
                'ask_p1':      ask_p1,
                'inventory':   self.inventory,
                'side':        self._position_side,
                'entry_price': self._entry_price,
                'hold_secs':   ist - self._entry_time,
            }

        # Check entry
        action = 'HOLD'
        if self.inventory == 0:
            if score > cfg.score_threshold:
                action = 'LONG'
            elif score < -cfg.score_threshold:
                action = 'SHORT'

        return {
            'action':   action,
            'symbol':   cfg.symbol,
            'score':    round(score, 4),
            'ltp':      ltp,
            'bid_p1':   bid_p1,
            'ask_p1':   ask_p1,
            'mid':      round(mid, 4),
            'inventory':self.inventory,
            'hour':     hour,
        }

    def update_fill(self, side: str, price: float, ist_sec: int):
        if side == 'LONG':
            self.inventory      = 1
            self._entry_price   = price
            self._entry_time    = ist_sec
            self._position_side = 'LONG'
        elif side == 'SHORT':
            self.inventory      = -1
            self._entry_price   = price
            self._entry_time    = ist_sec
            self._position_side = 'SHORT'
        elif side in ('EXIT', 'FORCE_CLOSE'):
            self.inventory      = 0
            self._position_side = ''
        logger.info(f"USDINR fill: {side} @ {price:.4f} inv={self.inventory}")