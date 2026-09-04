import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import time
import logging
import numpy as np
from datetime import datetime, timezone
from data.store.redis_client import RedisClient
from execution.risk.transaction_costs import TransactionCosts
from strategies.implementations.avellaneda_stoikov.glft.model import GLFTModel
from strategies.implementations.avellaneda_stoikov.glft.intensity import IntensityModel
from strategies.implementations.avellaneda_stoikov.glft.parameters import GLFTParameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MARKET_OPEN  = 9 * 3600 + 15 * 60   # 9:15am IST in seconds
MARKET_CLOSE = 15 * 3600 + 30 * 60  # 3:30pm IST in seconds
SESSION_SECS = MARKET_CLOSE - MARKET_OPEN  # 375 minutes

class GLFTSignalGenerator:
    def __init__(self, symbol: str, token: int,
                 lot_size: int, instrument_type: str = 'equity_futures',
                 params: GLFTParameters = None):
        self.symbol          = symbol
        self.token           = str(token)
        self.lot_size        = lot_size
        self.instrument_type = instrument_type
        self.params          = params or GLFTParameters()
        self.model           = GLFTModel(self.params)
        self.intensity       = IntensityModel(self.params)
        self.redis           = RedisClient()
        self.costs           = TransactionCosts(lot_size=lot_size,
                                                instrument_type=instrument_type)
        self.inventory       = 0
        self._last_calib     = 0
        self._kappa          = self.params.kappa
        self._A              = self.params.A
        self._kappa_history  = []

    def _ist_seconds(self) -> int:
        utc_sec = int(datetime.now(timezone.utc).timestamp())
        return (utc_sec + 19800) % 86400

    def _time_remaining(self) -> float:
        ist = self._ist_seconds()
        remaining = max(MARKET_CLOSE - ist, 1)
        return remaining / SESSION_SECS

    def _get_market_data(self) -> dict | None:
        tick      = self.redis.get_tick(self.token)
        orderbook = self.redis.get_orderbook(self.token)
        stream    = self.redis.get_stream(self.token, count=60)

        if not tick or not orderbook:
            return None

        # Compute fresh σ from last 60 ticks
        prices = []
        for _, fields in stream:
            ltp = float(fields.get('ltp', 0))
            if ltp > 0:
                prices.append(ltp)

        if len(prices) < 2:
            return None

        log_returns = np.diff(np.log(prices))
        sigma       = float(np.std(log_returns) * np.sqrt(len(prices)))

        return {
            'ltp':      float(tick['ltp']),
            'bids':     orderbook.get('bids', []),
            'asks':     orderbook.get('asks', []),
            'sigma':    sigma,
        }

    def _recalibrate(self, bids: list, asks: list):
        """Recalibrate κ and A from 5-level depth every 10 seconds."""
        now = time.time()
        if now - self._last_calib < 10:
            return

        bid_prices = [b.get('price',    0) for b in bids[:5]]
        bid_qtys   = [b.get('quantity', 0) for b in bids[:5]]
        ask_prices = [a.get('price',    0) for a in asks[:5]]
        ask_qtys   = [a.get('quantity', 0) for a in asks[:5]]

        A, kappa = self.intensity.calibrate_from_depth(
            bid_prices, bid_qtys, ask_prices, ask_qtys
        )

        # Smooth κ with rolling average of last 10 estimates
        self._kappa_history.append(kappa)
        if len(self._kappa_history) > 10:
            self._kappa_history.pop(0)

        self._kappa          = float(np.mean(self._kappa_history))
        self._A              = A
        self._last_calib     = now

        # Update model params
        self.params.kappa = self._kappa
        self.params.A     = self._A

    def _compute_edge(self, our_bid: float, our_ask: float,
                      market_bid: float, market_ask: float) -> tuple[float, float]:
        """How much better are our quotes vs market?"""
        bid_edge = our_bid - market_bid   # positive = we bid higher = more likely to fill
        ask_edge = market_ask - our_ask   # positive = we ask lower = more likely to fill
        return bid_edge, ask_edge

    def _min_edge(self, mid: float) -> float:
        """Minimum edge needed to cover transaction costs."""
        breakeven = self.costs.breakeven_spread(mid, 1)
        return breakeven / 2  # per side

    def generate(self) -> dict | None:
        ist = self._ist_seconds()
        if ist < MARKET_OPEN or ist > MARKET_CLOSE:
            return None

        data = self._get_market_data()
        if not data:
            return None

        ltp   = data['ltp']
        bids  = data['bids']
        asks  = data['asks']
        sigma = data['sigma']

        if not bids or not asks:
            return None

        market_bid = bids[0].get('price', ltp - 0.05)
        market_ask = asks[0].get('price', ltp + 0.05)
        mid        = (market_bid + market_ask) / 2

        # Recalibrate κ every 10 seconds
        self._recalibrate(bids, asks)

        # Update model with fresh σ
        self.params.sigma = sigma
        time_remaining    = self._time_remaining()

        # Compute optimal quotes
        our_bid, our_ask = self.model.optimal_quotes(
            mid=mid, q=self.inventory,
            sigma=sigma, time_remaining=time_remaining
        )

        # Compute edge
        bid_edge, ask_edge = self._compute_edge(
            our_bid, our_ask, market_bid, market_ask
        )

        min_edge = self._min_edge(mid)

        # Determine action
        action = 'HOLD'
        if bid_edge > min_edge and self.inventory < self.params.max_inventory:
            action = 'BUY'
        elif ask_edge > min_edge and self.inventory > -self.params.max_inventory:
            action = 'SELL'
        elif abs(self.inventory) >= self.params.max_inventory:
            action = 'REDUCE'

        # Force close near end of session
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
            'sigma':          round(sigma, 6),
            'kappa':          round(self._kappa, 4),
            'A':              round(self._A, 4),
            'inventory':      self.inventory,
            'time_remaining': round(time_remaining, 4),
        }

    def update_inventory(self, qty: int, side: str):
        if side == 'BUY':
            self.inventory += qty
        elif side == 'SELL':
            self.inventory -= qty
        logger.info(f"{self.symbol} inventory: {self.inventory}")