"""
execution/USDINR/live/risk.py
Daily loss limit, position cap, kill switch.
"""

import logging

log = logging.getLogger(__name__)


class RiskBreachError(RuntimeError):
    """Raised when a risk limit is breached — caller should halt trading."""


class RiskManager:
    """
    Tracks realised P&L for the session.
    Call .check_entry() before every new trade.
    Call .record_trade() after every exit.
    Call .reset_day() at session start.
    """

    def __init__(self, cfg):
        self.cfg                   = cfg
        self._daily_realised_bps   = 0.0
        self._n_trades_today       = 0
        self._kill_switch          = False

    def reset_day(self) -> None:
        self._daily_realised_bps = 0.0
        self._n_trades_today     = 0
        self._kill_switch        = False
        log.info("RiskManager: new day, counters reset")

    def check_entry(self) -> None:
        """Raise RiskBreachError if any pre-trade check fails."""
        if self._kill_switch:
            raise RiskBreachError("Kill switch is active — no new trades")

        if self._daily_realised_bps <= -abs(self.cfg.daily_loss_limit_bps):
            self._kill_switch = True
            raise RiskBreachError(
                f"Daily loss limit breached: {self._daily_realised_bps:.2f} bps "
                f"(<= -{self.cfg.daily_loss_limit_bps:.1f} bps)"
            )

    def record_trade(self, net_ret_bps: float) -> None:
        self._daily_realised_bps += net_ret_bps
        self._n_trades_today     += 1
        log.info(
            "RiskManager: trade #%d  net=%.3f bps  daily_total=%.3f bps",
            self._n_trades_today, net_ret_bps, self._daily_realised_bps,
        )

    def trigger_kill_switch(self, reason: str = "") -> None:
        self._kill_switch = True
        log.critical("KILL SWITCH triggered: %s", reason)

    @property
    def daily_pnl_bps(self) -> float:
        return self._daily_realised_bps

    @property
    def is_alive(self) -> bool:
        return not self._kill_switch