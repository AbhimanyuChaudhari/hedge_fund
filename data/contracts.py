"""
Auto-detection of active NSE currency futures contracts.
USDINR futures expire on the last business day of each month.
"""

from datetime import date, timedelta


def last_business_day(year: int, month: int) -> date:
    """Last Monday–Friday of the given month (no holiday calendar)."""
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    while last.weekday() > 4:
        last -= timedelta(days=1)
    return last


def contract_code(year: int, month: int) -> str:
    """Returns NSE contract suffix e.g. (2026, 10) → '26OCTFUT'"""
    month_abbr = date(year, month, 1).strftime('%b').upper()
    return f"{str(year)[-2:]}{month_abbr}FUT"


def active_contracts(as_of: date = None, n_near: int = 2) -> list:
    """
    Returns n_near nearest unexpired contract codes as of given date.
    Defaults to today.

    Example:
        active_contracts(date(2026, 10, 29))  →  ['26OCTFUT', '26NOVFUT']
        active_contracts(date(2026, 10, 30))  →  ['26NOVFUT', '26DECFUT']
        # Oct 30 is expiry day → near contract rolls to Nov
    """
    if as_of is None:
        as_of = date.today()

    contracts = []
    y, m = as_of.year, as_of.month
    checked = 0

    while len(contracts) < n_near and checked < 6:
        expiry = last_business_day(y, m)
        if expiry >= as_of:
            contracts.append(contract_code(y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
        checked += 1

    return contracts