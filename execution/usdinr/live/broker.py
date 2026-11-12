"""
execution/usdinr/live/broker.py
Zerodha Kite Connect broker wrapper for USDINR futures.
Credentials loaded from .env via python-dotenv (matches existing pipeline).
Paper mode: set PAPER_TRADE=1 or pass paper=True.
  - Paper mode = real KiteTicker ticks, fake order placement.
"""

import os
import logging
from enum import Enum
from typing import Optional, Callable, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; env vars may already be set

log = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"


class ZerodhaBroker:
    """
    Thin wrapper around KiteConnect.
    Uses ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN env vars
    (same names as the existing data pipeline in config/settings.py).

    Paper mode: orders are logged but never sent to exchange.
    Ticks always come from live KiteTicker regardless of paper mode.
    """

    def __init__(self, paper: bool = False):
        self.paper = paper or (os.getenv("PAPER_TRADE", "0") == "1")
        self._kite  = None
        self._kws   = None
        self._paper_order_counter = 0
        self._init_kite()

    def _init_kite(self):
        from kiteconnect import KiteConnect
        api_key      = os.environ["ZERODHA_API_KEY"]
        access_token = os.environ["ZERODHA_ACCESS_TOKEN"]
        self._kite = KiteConnect(api_key=api_key)
        self._kite.set_access_token(access_token)
        mode = "PAPER" if self.paper else "LIVE"
        log.info("KiteConnect initialised (%s mode)", mode)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,            # "BUY" or "SELL"
        size: int,            # number of lots
        order_type: OrderType,
        limit_price: Optional[float] = None,
        lots: int = 1,
    ) -> str:
        """Place order; returns order_id string."""
        if self.paper:
            self._paper_order_counter += 1
            oid = f"PAPER-{self._paper_order_counter}"
            log.info(
                "PAPER order | %s %s %d lots @ %s | id=%s",
                side, symbol, size, limit_price or "MKT", oid,
            )
            return oid

        kite_order_type = (
            self._kite.ORDER_TYPE_MARKET
            if order_type == OrderType.MARKET
            else self._kite.ORDER_TYPE_LIMIT
        )
        params = dict(
            tradingsymbol=symbol,
            exchange=self._kite.EXCHANGE_CDS,
            transaction_type=(
                self._kite.TRANSACTION_TYPE_BUY
                if side == "BUY"
                else self._kite.TRANSACTION_TYPE_SELL
            ),
            quantity=size,
            order_type=kite_order_type,
            product=self._kite.PRODUCT_MIS,
            validity=self._kite.VALIDITY_DAY,
        )
        if order_type == OrderType.LIMIT and limit_price is not None:
            params["price"] = limit_price

        oid = self._kite.place_order(variety=self._kite.VARIETY_REGULAR, **params)
        log.info("Live order placed | %s %s | id=%s", side, symbol, oid)
        return str(oid)

    def cancel_order(self, order_id: str) -> None:
        if self.paper:
            log.info("PAPER cancel | id=%s", order_id)
            return
        try:
            self._kite.cancel_order(
                variety=self._kite.VARIETY_REGULAR,
                order_id=order_id,
            )
        except Exception as exc:
            log.warning("cancel_order failed for %s: %s", order_id, exc)

    # ------------------------------------------------------------------
    # Market data helpers
    # ------------------------------------------------------------------

    def get_ltp(self, symbol: str) -> Optional[float]:
        """Last traded price."""
        try:
            data = self._kite.ltp(f"CDS:{symbol}")
            return data[f"CDS:{symbol}"]["last_price"]
        except Exception as exc:
            log.warning("get_ltp failed for %s: %s", symbol, exc)
            return None

    def get_instrument_token(self, symbol: str) -> Optional[int]:
        try:
            instruments = self._kite.instruments("CDS")
            for inst in instruments:
                if inst["tradingsymbol"] == symbol:
                    return inst["instrument_token"]
        except Exception as exc:
            log.warning("get_instrument_token failed: %s", exc)
        return None

    def stream_ticks(
        self,
        tokens: List[int],
        on_ticks: Callable,
    ) -> None:
        """
        Subscribe to live tick stream via KiteTicker.
        Calls on_ticks(ws, ticks) for every tick received.
        Always connects to live KiteTicker — paper mode only affects orders.
        Runs in a background thread; returns immediately.
        """
        from kiteconnect import KiteTicker
        api_key      = os.environ["ZERODHA_API_KEY"]
        access_token = os.environ["ZERODHA_ACCESS_TOKEN"]

        kws = KiteTicker(api_key, access_token)
        self._kws = kws

        def _on_connect(ws, response):
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            log.info("KiteTicker connected — subscribed to %s", tokens)

        def _on_error(ws, code, reason):
            log.error("KiteTicker error %s: %s", code, reason)

        def _on_close(ws, code, reason):
            log.warning("KiteTicker closed %s: %s", code, reason)

        kws.on_ticks   = on_ticks
        kws.on_connect = _on_connect
        kws.on_error   = _on_error
        kws.on_close   = _on_close

        # threaded=True returns immediately; tick callbacks fire in background
        kws.connect(threaded=True)
        log.info("KiteTicker started in background thread")

    def stop_stream(self) -> None:
        if self._kws is not None:
            try:
                self._kws.stop()
            except Exception as exc:
                log.warning("stop_stream error: %s", exc)