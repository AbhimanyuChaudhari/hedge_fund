import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import logging
import numpy as np
from datetime import datetime, timezone
from data.store.redis_client import RedisClient
from strategies.implementations.imbalance_momentum.parameters import ImbalanceMomentumParameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MARKET_OPEN  = 9 * 3600
MARKET_CLOSE = 17 * 3600


class ImbalanceMomentumSignal:
    def __init__(self, symbol: str, token: int,
                 params: ImbalanceMomentumParameters = None):
        self.symbol    = symbol
        self.token     = str(token)
        self.params    = params or ImbalanceMomentumParameters()
        self.redis     = RedisClient()
        self.inventory = 0

        # State tracking
        self._imbalance_history: list[float] = []
        self._entry_price:       float = 0.0
        self._entry_time:        int   = 0
        self._position_side:     str   = ''  # 'LONG' or 'SHORT'

    def _ist_seconds(self) -> int:
        utc_sec = int(datetime.now(timezone.utc).timestamp())
        return (utc_sec + 19800) % 86400

    def _get_market_data(self) -> dict | None:
        tick      = self.redis.get_tick(self.token)
        orderbook = self.redis.get_orderbook(self.token)
        if not tick or not orderbook:
            return None

        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        if not bids or not asks:
            return None

        ltp      = float(tick['ltp'])
        bid_p1   = bids[0].get('price',    ltp - self.params.tick_size)
        ask_p1   = asks[0].get('price',    ltp + self.params.tick_size)
        bid_q1   = bids[0].get('quantity', 0)
        ask_q1   = asks[0].get('quantity', 0)
        total_bid = sum(b.get('quantity', 0) for b in bids)
        total_ask = sum(a.get('quantity', 0) for a in asks)

        imbalance = (total_bid - total_ask) / (total_bid + total_ask + 1e-9)

        return {
            'ltp':       ltp,
            'bid_p1':    bid_p1,
            'ask_p1':    ask_p1,
            'bid_q1':    bid_q1,
            'ask_q1':    ask_q1,
            'spread':    ask_p1 - bid_p1,
            'imbalance': imbalance,
            'mid':       (bid_p1 + ask_p1) / 2,
        }

    def _check_exit(self, data: dict, ist_sec: int) -> str | None:
        if self.inventory == 0:
            return None

        price = data['ltp']
        ticks_moved = (price - self._entry_price) / self.params.tick_size

        # Stop loss
        if self._position_side == 'LONG' and ticks_moved < -self.params.stop_loss_ticks:
            return 'STOP_LOSS'
        if self._position_side == 'SHORT' and ticks_moved > self.params.stop_loss_ticks:
            return 'STOP_LOSS'

        # Take profit
        if self._position_side == 'LONG' and ticks_moved > self.params.take_profit_ticks:
            return 'TAKE_PROFIT'
        if self._position_side == 'SHORT' and ticks_moved < -self.params.take_profit_ticks:
            return 'TAKE_PROFIT'

        # Time stop
        if ist_sec - self._entry_time > self.params.max_hold_seconds:
            return 'TIME_STOP'

        # Imbalance reversal
        if len(self._imbalance_history) >= self.params.confirmation_bars:
            recent = self._imbalance_history[-self.params.confirmation_bars:]
            if self._position_side == 'LONG' and all(i < 0 for i in recent):
                return 'SIGNAL_REVERSAL'
            if self._position_side == 'SHORT' and all(i > 0 for i in recent):
                return 'SIGNAL_REVERSAL'

        return None

    def generate(self) -> dict | None:
        ist = self._ist_seconds()
        if ist < MARKET_OPEN or ist > MARKET_CLOSE:
            return None

        data = self._get_market_data()
        if not data:
            return None

        # Update imbalance history
        self._imbalance_history.append(data['imbalance'])
        if len(self._imbalance_history) > 20:
            self._imbalance_history.pop(0)

        # Check filters
        if data['spread'] < self.params.min_spread:
            return None
        if data['bid_q1'] < self.params.min_bid_qty:
            return None
        if data['ask_q1'] < self.params.min_ask_qty:
            return None

        action = 'HOLD'

        # Check exit first
        exit_reason = self._check_exit(data, ist)
        if exit_reason:
            action = f'EXIT_{exit_reason}'

        # Check entry only if flat
        elif self.inventory == 0 and len(self._imbalance_history) >= self.params.confirmation_bars:
            recent = self._imbalance_history[-self.params.confirmation_bars:]

            # Long signal — persistent buying pressure
            if all(i > self.params.imbalance_threshold for i in recent):
                action = 'LONG'

            # Short signal — persistent selling pressure
            elif all(i < -self.params.imbalance_threshold for i in recent):
                action = 'SHORT'

        return {
            'symbol':     self.symbol,
            'action':     action,
            'ltp':        data['ltp'],
            'bid_p1':     data['bid_p1'],
            'ask_p1':     data['ask_p1'],
            'imbalance':  round(data['imbalance'], 4),
            'spread':     round(data['spread'], 4),
            'inventory':  self.inventory,
        }

    def update_inventory(self, qty: int, side: str,
                         price: float, ist_sec: int):
        if side == 'BUY':
            self.inventory  += qty
            self._entry_price = price
            self._entry_time  = ist_sec
            self._position_side = 'LONG'
        elif side == 'SELL' and self.inventory > 0:
            self.inventory  -= qty
            if self.inventory == 0:
                self._position_side = ''
        elif side == 'SHORT':
            self.inventory   -= qty
            self._entry_price = price
            self._entry_time  = ist_sec
            self._position_side = 'SHORT'
        elif side == 'COVER' and self.inventory < 0:
            self.inventory  += qty
            if self.inventory == 0:
                self._position_side = ''