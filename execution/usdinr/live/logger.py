"""
execution/USDINR/live/logger.py
CSV trade log — appends one row per closed trade.
"""

import csv
import pathlib
import logging
from datetime import datetime, timezone

from .execution import ClosedTrade

log = logging.getLogger(__name__)

ROOT    = pathlib.Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "timestamp_utc", "symbol",
    "entry_time", "exit_time", "hold_sec",
    "entry_price", "exit_price",
    "state", "pred_norm", "size",
    "exit_reason", "gross_ret_bps", "net_ret_bps",
]


class TradeLogger:
    def __init__(self, date_str: str = None):
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._path = LOG_DIR / f"trades_{date_str}.csv"
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self._path.exists():
            with open(self._path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()
            log.info("Created trade log: %s", self._path)

    def log_trade(self, trade: ClosedTrade) -> None:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "symbol":        trade.symbol,
            "entry_time":    trade.entry_time,
            "exit_time":     trade.exit_time,
            "hold_sec":      round(trade.exit_time - trade.entry_time, 1),
            "entry_price":   trade.entry_price,
            "exit_price":    trade.exit_price,
            "state":         trade.state,
            "pred_norm":     round(trade.pred_norm, 4),
            "size":          trade.size,
            "exit_reason":   trade.exit_reason,
            "gross_ret_bps": trade.gross_ret_bps,
            "net_ret_bps":   trade.net_ret_bps,
        }
        with open(self._path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        log.info("TRADE LOGGED → %s", self._path.name)

    @property
    def path(self) -> pathlib.Path:
        return self._path