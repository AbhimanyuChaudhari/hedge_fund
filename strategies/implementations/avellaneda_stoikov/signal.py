import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import logging
from datetime import datetime, time
from data.store.redis_client import RedisClient
from strategies.implementations.avellaneda_stoikov.model import AvellanedaStoikov
from strategies.implementations.avellaneda_stoikov.parameters import ASParameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MARKET_OPEN  = time(9, 15)
MARKET_CLOSE = time(15, 30)
SESSION_MINUTES = 375.0

class ASSignalGenerator:
    def __init__(self, symbol: str, params: ASParameters = None):
        self.symbol    = symbol
        self.model     = AvellanedaStoikov(params)
        self.redis     = RedisClient()
        self.inventory = 0
        self.price_history: list[float] = []

    def _time_remaining(self) -> float:
        now     = datetime.now().time()
        elapsed = (datetime.combine(datetime.today(), now) -
                   datetime.combine(datetime.today(), MARKET_OPEN)).seconds / 60
        return max(SESSION_MINUTES - elapsed, 1.0) / SESSION_MINUTES

    def _get_market_data(self) -> tuple[float, float, float] | None:
        tick      = self.redis.get_tick(self.symbol)
        orderbook = self.redis.get_orderbook(self.symbol)
        if not tick or not orderbook:
            return None
        ltp       = float(tick['ltp'])
        bids      = orderbook.get('bids', [])
        asks      = orderbook.get('asks', [])
        best_bid  = bids[0]['price'] if bids else ltp - 0.05
        best_ask  = asks[0]['price'] if asks else ltp + 0.05
        mid       = (best_bid + best_ask) / 2
        return mid, best_bid, best_ask

    def generate(self) -> dict | None:
        data = self._get_market_data()
        if not data:
            return None
        mid, best_bid, best_ask = data
        self.price_history.append(mid)
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]
        if len(self.price_history) < 20:
            return None
        time_remaining = self._time_remaining()
        signal = self.model.signal(
            mid=mid, best_bid=best_bid, best_ask=best_ask,
            inventory=self.inventory,
            prices=self.price_history,
            time_remaining=time_remaining
        )
        signal['symbol']    = self.symbol
        signal['mid']       = mid
        signal['best_bid']  = best_bid
        signal['best_ask']  = best_ask
        logger.info(f"{self.symbol} | {signal['action']} | mid={mid} | inv={self.inventory} | edge_bid={signal['bid_edge']} ask={signal['ask_edge']}")
        return signal

    def update_inventory(self, filled_qty: int, side: str):
        if side == 'BUY':
            self.inventory += filled_qty
        elif side == 'SELL':
            self.inventory -= filled_qty