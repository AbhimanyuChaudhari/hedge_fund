"""
execution/USDINR/live/runner.py
Main loop — wires everything together.

Run from project root:
    python -m execution.USDINR.live.runner --symbol USDINR26NOVFUT [--paper]
"""

import os
import sys
import time
import logging
import argparse
import pathlib
import threading
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .config    import load_config
from .features  import LiveFeatureBuffer
from .signal    import SignalEngine
from .broker    import ZerodhaBroker
from .execution import ExecutionEngine
from .risk      import RiskManager, RiskBreachError
from .logger    import TradeLogger

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("runner")

IST_OFFSET = timedelta(hours=5, minutes=30)


def ist_now() -> datetime:
    return datetime.now(timezone.utc) + IST_OFFSET


def ist_seconds_since_midnight() -> int:
    t = ist_now()
    return t.hour * 3600 + t.minute * 60 + t.second


def in_trading_window(cfg) -> bool:
    s = ist_seconds_since_midnight()
    if s < cfg.window_start or s >= cfg.window_end:
        return False
    if cfg.skip_start <= s < cfg.skip_end:
        return False
    return True


def _tick_to_bar(tick: dict, ts_sec: int, mid: float) -> dict:
    """Convert a Kite tick dict to a minimal 1s bar for the feature builder."""
    depth = tick.get("depth", {})
    bids  = depth.get("buy",  [{}])
    asks  = depth.get("sell", [{}])
    return {
        "ts_sec":     ts_sec,
        "mid":        mid,
        "bid_p1":     bids[0].get("price", mid) if bids else mid,
        "ask_p1":     asks[0].get("price", mid) if asks else mid,
        "bid_q1":     bids[0].get("quantity", 0) if bids else 0,
        "ask_q1":     asks[0].get("quantity", 0) if asks else 0,
        "last_price": tick.get("last_price", mid),
        "volume":     tick.get("volume_traded", 0),
        "tick_count": tick.get("last_quantity", 0),
        "date":       datetime.utcfromtimestamp(ts_sec).date(),
        "ist_sec":    (ts_sec + 5 * 3600 + 30 * 60) % 86400,
    }


class LiveRunner:
    def __init__(self, symbol: str, paper: bool = False):
        self.symbol = symbol
        self.cfg    = load_config()
        log.info("Config loaded — State2 thresh=%.2f  skip Hour13", self.cfg.state2_thresh)

        self.broker   = ZerodhaBroker(paper=paper)
        self.feat_buf = LiveFeatureBuffer(symbol=symbol, window=600)
        self.signal   = SignalEngine()
        self.exec_eng = ExecutionEngine(self.cfg, self.broker)
        self.risk     = RiskManager(self.cfg)
        self.tlog     = TradeLogger()

        self._latest_mid   = 0.0
        self._latest_tick  = {}
        self._tick_lock    = threading.Lock()
        self._last_ts_proc = 0

    def _on_ticks(self, ws, ticks: list) -> None:
        for t in ticks:
            bid = t.get("depth", {}).get("buy",  [{}])[0].get("price", 0)
            ask = t.get("depth", {}).get("sell", [{}])[0].get("price", 0)
            mid = (bid + ask) / 2.0 if bid and ask else t.get("last_price", 0)
            with self._tick_lock:
                self._latest_mid  = mid
                self._latest_tick = t

    def run(self) -> None:
        self.signal.reset_day()
        self.risk.reset_day()

        token = self.broker.get_instrument_token(self.symbol)
        self.broker.stream_ticks([token], self._on_ticks)
        log.info("LiveRunner started  symbol=%s  paper=%s", self.symbol, self.broker.paper)

        while True:
            ts_now = int(time.time())
            if ts_now == self._last_ts_proc:
                time.sleep(0.05)
                continue
            self._last_ts_proc = ts_now

            with self._tick_lock:
                mid_now   = self._latest_mid
                tick_data = dict(self._latest_tick)

            if mid_now <= 0:
                continue

            bar = _tick_to_bar(tick_data, ts_now, mid_now)
            self.feat_buf.push(bar)

            # session end
            ist_s = ist_seconds_since_midnight()
            if ist_s >= self.cfg.window_end + 600:
                log.info("Session over — shutting down")
                self.broker.stop_stream()
                break

            # position management (runs even outside window)
            closed = self.exec_eng.on_tick(ts_now, mid_now)
            if closed is not None:
                self.risk.record_trade(closed.net_ret_bps)
                self.tlog.log_trade(closed)

            # entry logic
            if not in_trading_window(self.cfg):
                continue
            if not self.exec_eng.is_flat:
                continue

            try:
                feat_vec = self.feat_buf.latest_vector()
            except RuntimeError as e:
                log.warning("Feature build failed: %s", e)
                continue
            if feat_vec is None:
                continue

            tick_count = bar.get("tick_count", 0)
            pred_norm, hmm_state, raw_pred = self.signal.predict(feat_vec, tick_count)

            if pred_norm is None or hmm_state is None:
                continue

            try:
                self.risk.check_entry()
            except RiskBreachError as e:
                log.warning("Risk block: %s", e)
                continue

            self.exec_eng.try_entry(
                symbol    = self.symbol,
                ts_now    = ts_now,
                mid_now   = mid_now,
                pred_norm = pred_norm,
                state     = hmm_state,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="USDINR26NOVFUT")
    parser.add_argument("--paper",  action="store_true")
    args = parser.parse_args()

    if args.paper or os.environ.get("PAPER_TRADE") == "1":
        log.warning("=" * 50)
        log.warning("PAPER TRADE MODE — no real orders")
        log.warning("=" * 50)

    LiveRunner(symbol=args.symbol, paper=args.paper).run()


if __name__ == "__main__":
    main()