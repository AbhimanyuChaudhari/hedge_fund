"""
execution/USDINR/live/execution.py
Cartea-Jaimungal order logic + TP / SL / max-hold position manager.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto

# Import OrderType from broker so both modules share the same definition
from .broker import OrderType

log = logging.getLogger(__name__)


class ExitReason(Enum):
    TP      = "tp"
    SL      = "sl"
    TIMEOUT = "timeout"
    MANUAL  = "manual"


@dataclass
class OpenPosition:
    symbol:      str
    entry_time:  float
    entry_mid:   float
    state:       int
    pred_norm:   float
    size:        float
    tp_price:    float
    sl_price:    float
    expire_time: float
    fill_price:  Optional[float] = None
    order_id:    Optional[str]   = None


@dataclass
class ClosedTrade:
    symbol:        str
    entry_time:    float
    exit_time:     float
    entry_price:   float
    exit_price:    float
    state:         int
    pred_norm:     float
    size:          float
    exit_reason:   str
    gross_ret_bps: float
    net_ret_bps:   float


class ExecutionEngine:
    """
    Manages a single open position (long-only, 1 trade at a time).

    Per-second usage:
        eng.on_tick(ts_now, mid_now)   → ClosedTrade or None
        eng.try_entry(...)             → bool
    """

    def __init__(self, cfg, broker):
        self.cfg    = cfg
        self.broker = broker
        self.position: Optional[OpenPosition] = None
        self._pending_limit_order_id:  Optional[str]   = None
        self._pending_limit_placed_at: Optional[float] = None

    # ------------------------------------------------------------------
    def try_entry(
        self,
        symbol:    str,
        ts_now:    float,
        mid_now:   float,
        pred_norm: float,
        state:     int,
    ) -> bool:
        if self.position is not None:
            return False

        cfg    = self.cfg
        thresh = cfg.state2_thresh if state == 2 else cfg.base_thresh
        if pred_norm < thresh:
            return False

        vol      = cfg.state_vol[state]
        tp_dist  = cfg.state_tp[state] * vol
        sl_dist  = cfg.c_sl * vol
        max_hold = cfg.state_max_hold[state]
        size     = cfg.state_size[state]

        pos = OpenPosition(
            symbol      = symbol,
            entry_time  = ts_now,
            entry_mid   = mid_now,
            state       = state,
            pred_norm   = pred_norm,
            size        = size,
            tp_price    = mid_now * (1 + tp_dist * 1e-4),
            sl_price    = mid_now * (1 - sl_dist * 1e-4),
            expire_time = ts_now + max_hold,
        )

        if pred_norm >= cfg.cj_market_thresh:
            order_id       = self.broker.place_order(symbol, "BUY", size, OrderType.MARKET, None)
            pos.fill_price = mid_now
            pos.order_id   = order_id
            log.info("ENTRY MARKET  sym=%s mid=%.4f pred=%.3f state=%d",
                     symbol, mid_now, pred_norm, state)
        else:
            order_id     = self.broker.place_order(symbol, "BUY", size, OrderType.LIMIT, mid_now)
            pos.order_id = order_id
            self._pending_limit_order_id    = order_id
            self._pending_limit_placed_at   = ts_now
            log.info("ENTRY LIMIT   sym=%s mid=%.4f pred=%.3f state=%d",
                     symbol, mid_now, pred_norm, state)

        self.position = pos
        return True

    # ------------------------------------------------------------------
    def on_tick(self, ts_now: float, mid_now: float) -> Optional[ClosedTrade]:
        pos = self.position
        if pos is None:
            return None

        cfg = self.cfg

        # pending limit fallback → market after timeout
        if self._pending_limit_order_id is not None:
            age = ts_now - (self._pending_limit_placed_at or ts_now)
            if age >= cfg.cj_limit_timeout:
                self.broker.cancel_order(self._pending_limit_order_id)
                new_id         = self.broker.place_order(pos.symbol, "BUY", pos.size, OrderType.MARKET, None)
                pos.fill_price = mid_now
                pos.order_id   = new_id
                self._pending_limit_order_id  = None
                self._pending_limit_placed_at = None
                log.info("LIMIT TIMEOUT → MARKET  mid=%.4f", mid_now)

        if pos.fill_price is None:
            return None

        # exit checks
        exit_reason = None
        exit_price  = mid_now

        if mid_now >= pos.tp_price:
            exit_reason = ExitReason.TP
            exit_price  = pos.tp_price
        elif mid_now <= pos.sl_price:
            exit_reason = ExitReason.SL
            exit_price  = pos.sl_price
        elif ts_now >= pos.expire_time:
            exit_reason = ExitReason.TIMEOUT

        if exit_reason is None:
            return None

        self.broker.place_order(pos.symbol, "SELL", pos.size, OrderType.MARKET, None)
        self.position                 = None
        self._pending_limit_order_id  = None
        self._pending_limit_placed_at = None

        entry_p = pos.fill_price
        gross   = (exit_price / entry_p - 1) * 1e4
        net     = (gross - cfg.cost_bps) * pos.size

        trade = ClosedTrade(
            symbol        = pos.symbol,
            entry_time    = pos.entry_time,
            exit_time     = ts_now,
            entry_price   = entry_p,
            exit_price    = exit_price,
            state         = pos.state,
            pred_norm     = pos.pred_norm,
            size          = pos.size,
            exit_reason   = exit_reason.value,
            gross_ret_bps = round(gross, 4),
            net_ret_bps   = round(net,   4),
        )
        log.info("EXIT %-8s sym=%s net=%.3f bps  hold=%.0fs  state=%d",
                 exit_reason.value, pos.symbol, net, ts_now - pos.entry_time, pos.state)
        return trade

    # ------------------------------------------------------------------
    def confirm_fill(self, order_id: str, fill_price: float) -> None:
        pos = self.position
        if pos and pos.order_id == order_id:
            pos.fill_price               = fill_price
            self._pending_limit_order_id = None
            log.info("FILL confirmed  order=%s price=%.4f", order_id, fill_price)

    @property
    def is_flat(self) -> bool:
        return self.position is None