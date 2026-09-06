import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))

import time
import logging
import numpy as np
from datetime import datetime, timezone
from data.store.redis_client import RedisClient
from execution.risk.transaction_costs import TransactionCosts
from strategies.implementations.avellaneda_stoikov.cst.parameters import CSTParameters
from strategies.implementations.avellaneda_stoikov.cst.model import CSTModel
from strategies.implementations.avellaneda_stoikov.cst.order_flow import OrderFlowEstimator
from strategies.implementations.avellaneda_stoikov.cst.drift import DriftEstimator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MARKET_OPEN  = 9 * 3600 + 15 * 60
MARKET_CLOSE = 15 * 3600 + 30 * 60
SESSION_SECS = MARKET_CLOSE - MARKET_OPEN


class CSTSignalGenerator:
    def __init__(self, symbol: str, token: int,
                 lot_size: int, instrument_type: str = 'equity_futures',
                 params: CSTParameters = None):
        self.symbol          = symbol
        self.token           = str(token)
        self.lot_size        = lot_size
        self.instrument_type = instrument_type
        self.params          = params or CSTParameters()
        self.model           = CSTModel(self.params)
        self.flow_est        = OrderFlowEstimator(self.params)
        self.drift_est       = DriftEstimator(self.params)
        self.redis           = RedisClient()
        self.costs           = TransactionCosts(
                                   lot_size=lot_size,
                                   instrument_type=instrument_type
                               )
        self.inventory       = 0
        self._last_calib     = 0
        self._cached_flow    = None
        self._cached_mu      = 0.0

    def _ist_seconds(self) -> int:
        utc_sec = int(datetime.now(timezone.utc).timestamp())
        return (utc_sec + 19800) % 86400

    def _time_remaining(self) -> float:
        ist = self._ist_seconds()
        remaining = max(MARKET_CLOSE - ist, 1)
        return remaining / SESSION_SECS

    def _get_bars(self, count: int = 60) -> list[dict]:
        """Get last N 1-second bars from Redis stream."""
        raw = self.redis.get_stream(self.token, count=count)
        bars = []
        for _, fields in raw:
            bars.append({
                'close':          float(fields.get('ltp',           0)),
                'volume_delta':   float(fields.get('volume',        0)),
                'total_bid_qty':  float(fields.get('total_bid_qty', 0)),
                'total_ask_qty':  float(fields.get('total_ask_qty', 0)),
                'imbalance_last': float(fields.get('total_bid_qty', 0) - 
                                        float(fields.get('total_ask_qty', 0))) /
                                  (float(fields.get('total_bid_qty', 0)) + 
                                   float(fields.get('total_ask_qty', 0)) + 1e-9),
            })
        return bars

    def _compute_sigma(self, bars: list[dict]) -> float:
        """Fresh σ from last 60 ticks."""
        prices = [b['close'] for b in bars if b['close'] > 0]
        if len(prices) < 2:
            return self.params.sigma
        log_returns = np.diff(np.log(prices))
        return float(np.std(log_returns) * np.sqrt(len(prices)))

    def _recalibrate(self, bars: list[dict]):
        """Recalibrate flow rates and drift every 10 seconds."""
        now = time.time()
        if now - self._last_calib < self.params.recalib_secs:
            return
        self._cached_flow = self.flow_est.estimate(bars)
        self._cached_mu   = self.drift_est.estimate(bars, self._cached_flow)
        self._last_calib  = now
        logger.debug(
            f"{self.symbol} | κ_eff={self._cached_flow.kappa_eff:.3f} "
            f"µ={self._cached_mu:.5f} "
            f"cancel_imb={self._cached_flow.cancel_imbalance:.3f}"
        )

    def _min_edge(self, mid: float) -> float:
        breakeven = self.costs.breakeven_spread(mid, 1)
        return breakeven / 2

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

        bars  = self._get_bars(count=self.params.flow_window)
        sigma = self._compute_sigma(bars)

        self._recalibrate(bars)

        flow = self._cached_flow
        mu   = self._cached_mu
        time_remaining = self._time_remaining()

        our_bid, our_ask = self.model.optimal_quotes(
            mid=mid, q=self.inventory,
            sigma=sigma, time_remaining=time_remaining,
            mu=mu, flow=flow
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
            'symbol':           self.symbol,
            'action':           action,
            'our_bid':          our_bid,
            'our_ask':          our_ask,
            'market_bid':       market_bid,
            'market_ask':       market_ask,
            'mid':              mid,
            'bid_edge':         round(bid_edge, 4),
            'ask_edge':         round(ask_edge, 4),
            'min_edge':         round(min_edge, 4),
            'sigma':            round(sigma, 6),
            'mu':               round(mu, 6),
            'kappa_eff':        round(flow.kappa_eff, 4) if flow else self.params.kappa,
            'lambda_cancel_bid':round(flow.lambda_cancel_bid, 4) if flow else 0,
            'lambda_cancel_ask':round(flow.lambda_cancel_ask, 4) if flow else 0,
            'cancel_imbalance': round(flow.cancel_imbalance, 4) if flow else 0,
            'inventory':        self.inventory,
            'time_remaining':   round(time_remaining, 4),
        }

    def update_inventory(self, qty: int, side: str):
        if side == 'BUY':
            self.inventory += qty
        elif side == 'SELL':
            self.inventory -= qty
        logger.info(f"{self.symbol} inventory: {self.inventory}")