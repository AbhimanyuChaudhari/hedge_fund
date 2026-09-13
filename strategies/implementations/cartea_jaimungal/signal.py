import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import time
import logging
import numpy as np
from datetime import datetime, timezone
from data.store.redis_client import RedisClient
from execution.risk.transaction_costs import TransactionCosts
from strategies.implementations.cartea_jaimungal.parameters import CJParameters
from strategies.implementations.cartea_jaimungal.model import CJModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MARKET_OPEN  = 9 * 3600
MARKET_CLOSE = 17 * 3600
SESSION_SECS = MARKET_CLOSE - MARKET_OPEN


class CJSignalGenerator:
    """
    Cartea-Jaimungal live signal generator.

    Every second:
    1. Read Redis orderbook and tick data
    2. Compute fresh σ from price history
    3. Update alpha signal from OFI/imbalance
    4. Compute CJ optimal quotes
    5. Compare to market — generate BUY/SELL/HOLD
    """

    def __init__(self, symbol: str, token: int,
                 lot_size: int,
                 instrument_type: str = 'equity_futures',
                 params: CJParameters = None):
        self.symbol          = symbol
        self.token           = str(token)
        self.lot_size        = lot_size
        self.instrument_type = instrument_type
        self.params          = params or CJParameters()
        self.model           = CJModel(self.params)
        self.redis           = RedisClient()
        self.costs           = TransactionCosts(
                                   lot_size=lot_size,
                                   instrument_type=instrument_type
                               )
        self.inventory       = 0
        self._alpha          = 0.0      # current alpha signal
        self._last_calib     = 0.0

    def _ist_seconds(self) -> int:
        return (int(datetime.now(timezone.utc).timestamp()) + 19800) % 86400

    def _time_remaining(self) -> float:
        ist       = self._ist_seconds()
        remaining = max(MARKET_CLOSE - ist, 1)
        return remaining / SESSION_SECS

    def _get_stream_data(self, count: int = 60) -> list[dict]:
        raw  = self.redis.get_stream(self.token, count=count)
        bars = []
        for _, fields in raw:
            bars.append({
                'ltp':           float(fields.get('ltp',           0)),
                'total_bid_qty': float(fields.get('total_bid_qty', 0)),
                'total_ask_qty': float(fields.get('total_ask_qty', 0)),
                'bid_p1':        float(fields.get('bid_p1',        0)),
                'ask_p1':        float(fields.get('ask_p1',        0)),
                'bid_q1':        float(fields.get('bid_q1',        0)),
                'ask_q1':        float(fields.get('ask_q1',        0)),
            })
        return bars

    def _compute_sigma(self, bars: list[dict]) -> float:
        prices = [b['ltp'] for b in bars if b['ltp'] > 0]
        if len(prices) < 2:
            return self.params.sigma
        log_returns = np.diff(np.log(prices))
        return float(np.std(log_returns) * np.sqrt(len(prices)))

    def _compute_raw_signal(self, bars: list[dict]) -> float:
        """
        Compute raw alpha signal from order book.
        Uses validated signals from IC analysis:
        - USDINR: cancel_imb (inv) + imbalance + wmid
        - Returns normalized value in [-1, 1]
        """
        if not bars:
            return 0.0

        tot_bid = np.mean([b['total_bid_qty'] for b in bars])
        tot_ask = np.mean([b['total_ask_qty'] for b in bars])
        imb     = (tot_bid - tot_ask) / (tot_bid + tot_ask + 1e-9)
        return float(np.clip(imb, -1, 1))

    def _update_alpha(self, bars: list[dict]):
        """Update alpha using OU mean reversion + new signal."""
        raw_signal  = self._compute_raw_signal(bars)
        self._alpha = self.model.alpha_update(
            self._alpha, raw_signal * self.params.alpha_vol
        )

    def _min_edge(self, mid: float) -> float:
        return self.costs.breakeven_spread(mid, 1) / 2

    def generate(self) -> dict | None:
        ist = self._ist_seconds()
        if ist < MARKET_OPEN or ist > MARKET_CLOSE:
            return None

        tick      = self.redis.get_tick(self.token)
        orderbook = self.redis.get_orderbook(self.token)
        if not tick or not orderbook:
            return None

        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        if not bids or not asks:
            return None

        ltp        = float(tick['ltp'])
        market_bid = bids[0].get('price', ltp - self.params.tick_size)
        market_ask = asks[0].get('price', ltp + self.params.tick_size)
        mid        = (market_bid + market_ask) / 2

        bars  = self._get_stream_data(count=self.params.flow_window)
        sigma = self._compute_sigma(bars)
        self._update_alpha(bars)

        time_remaining = self._time_remaining()

        our_bid, our_ask = self.model.optimal_quotes(
            mid=mid, q=self.inventory,
            time_remaining=time_remaining,
            alpha=self._alpha,
            sigma=sigma
        )

        bid_edge = our_bid - market_bid
        ask_edge = market_ask - our_ask
        min_edge = self._min_edge(mid)

        action = 'HOLD'
        if bid_edge > min_edge and self.inventory < self.params.max_inventory:
            action = 'BUY'
        elif ask_edge > min_edge and self.inventory > -self.params.max_inventory:
            action = 'SELL'
        elif abs(self.inventory) >= self.params.max_inventory:
            action = 'REDUCE'
        if time_remaining < 0.02 and self.inventory != 0:
            action = 'FORCE_CLOSE'

        return {
            'symbol':         self.symbol,
            'action':         action,
            'our_bid':        our_bid,
            'our_ask':        our_ask,
            'market_bid':     market_bid,
            'market_ask':     market_ask,
            'mid':            mid,
            'bid_edge':       round(bid_edge, 4),
            'ask_edge':       round(ask_edge, 4),
            'min_edge':       round(min_edge, 4),
            'alpha':          round(self._alpha, 6),
            'sigma':          round(sigma, 6),
            'inventory':      self.inventory,
            'time_remaining': round(time_remaining, 4),
            'running_penalty':round(self.model.running_penalty(self.inventory), 6),
        }

    def update_inventory(self, qty: int, side: str):
        if side == 'BUY':
            self.inventory += qty
        elif side == 'SELL':
            self.inventory -= qty
        logger.info(f"{self.symbol} CJ inventory: {self.inventory} alpha: {self._alpha:.4f}")