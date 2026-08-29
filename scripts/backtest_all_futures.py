"""
Fast Batch Backtest — All Stock Futures
========================================
Supports both V1 (fast Numba engine) and V2 (Ricci Hawkes-Alpha).

Usage:
    python scripts/backtest_all_futures.py --start 2026-05-27 --end 2026-05-27
    python scripts/backtest_all_futures.py --start 2026-05-27 --end 2026-05-27 --use-optimal-params
    python scripts/backtest_all_futures.py --start 2026-05-27 --end 2026-05-27 --model v2

Contract roll handling:
    Expiry dates loaded dynamically from Zerodha API — no manual updates needed.
    Falls back to last-Thursday calculation if API unavailable.
"""

import os
import re
import calendar
import argparse
import functools
import time
import boto3
import pandas as pd
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

from strategies.backtesting.fill_simulator import run_fast_backtest
from strategies.backtesting.data_loader import load_day
from strategies.backtesting.param_loader import get_symbol_params

BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'hedge-fund-data-ac')
AWS_KEY     = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET  = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION  = os.getenv('AWS_REGION', 'ap-south-1')

INDEX_BASES = {
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
    "SENSEX", "NIFTYNXT50", "BANKEX",
}

_CONTRACT_RE = re.compile(
    r'\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)FUT$',
    re.IGNORECASE
)

_MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
}


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic expiry dates from Zerodha
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_nfo_expiries() -> dict:
    """
    Load all F&O expiry dates from Zerodha API.
    Returns {(year, month): expiry_date} for all active contracts.
    Cached in memory — only fetches once per process.
    """
    try:
        from data.ingest.zerodha_client import ZerodhaClient
        kite        = ZerodhaClient().kite
        instruments = pd.DataFrame(kite.instruments("NFO"))
        futures     = instruments[instruments['instrument_type'] == 'FUT'].copy()
        futures['expiry'] = pd.to_datetime(futures['expiry']).dt.date

        expiry_map = {}
        for _, row in futures.iterrows():
            exp = row['expiry']
            key = (exp.year, exp.month)
            if key not in expiry_map or exp > expiry_map[key]:
                expiry_map[key] = exp

        return expiry_map

    except Exception as e:
        print(f"Could not load expiries from Zerodha ({e}) — using fallback")
        return {}


def get_contract_expiry(year: int, month: int) -> date:
    """
    Returns NSE futures expiry date for a given month/year.
    Fetches from Zerodha API — accurate including holiday adjustments.
    Falls back to last-Thursday calculation if API unavailable.
    """
    expiry_map = _load_nfo_expiries()
    if (year, month) in expiry_map:
        return expiry_map[(year, month)]

    # Fallback: last Thursday of month
    last_day  = calendar.monthrange(year, month)[1]
    last_date = date(year, month, last_day)
    days_back = (last_date.weekday() - 3) % 7
    return last_date.replace(day=last_day - days_back)


# ─────────────────────────────────────────────────────────────────────────────
# Contract helpers
# ─────────────────────────────────────────────────────────────────────────────

def strip_contract_suffix(symbol: str) -> str:
    return _CONTRACT_RE.sub('', symbol)


def is_index_future(symbol: str) -> bool:
    return strip_contract_suffix(symbol) in INDEX_BASES


def is_contract_expired(symbol: str, start_date_str: str) -> bool:
    m = _CONTRACT_RE.search(symbol)
    if not m:
        return False
    suffix = m.group(0)
    yy     = int(suffix[:2])
    mon    = suffix[2:5].upper()
    month  = _MONTH_MAP.get(mon)
    if not month:
        return False
    expiry = get_contract_expiry(2000 + yy, month)
    return date.fromisoformat(start_date_str) > expiry


# ─────────────────────────────────────────────────────────────────────────────
# S3 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_s3():
    return boto3.client('s3',
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION
    )


def _s3_list(prefix: str) -> list[str]:
    try:
        paginator = _get_s3().get_paginator('list_objects_v2')
        keys = []
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])
        return keys
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_lot_sizes() -> dict:
    try:
        from data.ingest.zerodha_client import ZerodhaClient
        kite = ZerodhaClient().kite
        df   = pd.DataFrame(kite.instruments("NFO"))
        df   = df[df["instrument_type"] == "FUT"]
        return {row["name"]: int(row["lot_size"]) for _, row in df.iterrows()}
    except Exception as e:
        print(f"Could not load lot sizes ({e}) — using defaults")
        return {}


def get_symbols(start: str, end: str) -> list:
    all_keys = _s3_list("processed/features/")
    folders  = set()
    for key in all_keys:
        parts = key.split("/")
        if len(parts) >= 3:
            folders.add(parts[2])

    best: dict = {}
    for candidate in folders:
        if is_index_future(candidate):
            continue
        if not _CONTRACT_RE.search(candidate):
            continue
        if '.' in candidate or candidate.startswith('_'):
            continue
        if is_contract_expired(candidate, start):
            continue

        base     = strip_contract_suffix(candidate)
        files    = _s3_list(f"processed/features/{candidate}/")
        has_data = any(
            start <= k.split("/")[-1].replace(".parquet", "") <= end
            for k in files
        )
        if not has_data:
            continue
        if base not in best or candidate > best[base]:
            best[base] = candidate

    return sorted(best.values())


def resolve_params(symbol: str, global_params: dict,
                   use_optimal: bool, model: str) -> dict:
    if not use_optimal:
        return global_params
    opt    = get_symbol_params(symbol, model=model)
    merged = global_params.copy()
    merged['gamma']      = opt.get('gamma',      global_params['gamma'])
    merged['kappa']      = opt.get('kappa',      global_params['kappa'])
    merged['min_spread'] = opt.get('min_spread', global_params['min_spread'])
    merged['open_mult']  = opt.get('open_mult',  global_params['open_mult'])
    if model == 'v2':
        for k in ['phi', 'rho', 'beta', 'theta', 'eta', 'nu', 'zeta']:
            if k in opt:
                merged[k] = opt[k]
    return merged


def get_lot_size_for_symbol(symbol: str, lot_sizes: dict) -> int:
    base = strip_contract_suffix(symbol)
    if base in lot_sizes:
        return lot_sizes[base]
    if symbol in lot_sizes:
        return lot_sizes[symbol]
    return 75


# ─────────────────────────────────────────────────────────────────────────────
# V1 runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_v1(symbol: str, start: str, end: str,
               lot_sizes: dict, params: dict,
               min_bars: int = 100) -> dict:
    lot_size = get_lot_size_for_symbol(symbol, lot_sizes)
    inst     = 'currency_futures' if any(
        x in symbol.upper() for x in ['USDINR', 'EURINR']
    ) else 'equity_futures'

    try:
        frames  = []
        current = datetime.strptime(start, '%Y-%m-%d')
        end_dt  = datetime.strptime(end,   '%Y-%m-%d')
        while current <= end_dt:
            date_str = current.strftime('%Y-%m-%d')
            df = load_day(symbol, date_str, market_hours_only=True)
            if not df.empty:
                df['_date'] = date_str
                frames.append(df)
            current += timedelta(days=1)

        if not frames:
            return {'symbol': symbol, 'lot_size': lot_size,
                    'net_pnl': 0, 'bars': 0, 'ok': False, 'error': 'no data'}

        full_df = pd.concat(frames, ignore_index=True).sort_values('ts_sec')
        if len(full_df) < min_bars:
            return {'symbol': symbol, 'lot_size': lot_size,
                    'net_pnl': 0, 'bars': len(full_df), 'ok': False,
                    'error': f'only {len(full_df)} bars'}

        result = run_fast_backtest(
            df=full_df, gamma=params['gamma'], kappa=params['kappa'],
            min_spread=params['min_spread'], max_spread=params['max_spread'],
            open_mult=params.get('open_mult', 2.0), lot_size=lot_size,
            max_inventory=params.get('max_inv', 5),
            queue_aggression=params.get('queue_agg', 0.3),
            instrument_type=inst,
        )
        return {
            'symbol': symbol, 'lot_size': lot_size,
            'gross_pnl': round(result.gross_pnl, 2),
            'costs':     round(result.total_costs, 2),
            'net_pnl':   round(result.net_pnl, 2),
            'fills':     result.total_fills,
            'fill_rate': result.fill_rate,
            'win_rate':  result.win_rate,
            'sharpe':    result.sharpe_ratio,
            'max_dd':    result.max_drawdown,
            'bars':      result.bars_processed,
            'ok': True,
            'gamma': params['gamma'],
            'open_mult': params.get('open_mult', 2.0),
        }
    except Exception as e:
        return {'symbol': symbol, 'lot_size': lot_size,
                'net_pnl': 0, 'bars': 0, 'ok': False, 'error': str(e)[:80]}


# ─────────────────────────────────────────────────────────────────────────────
# V2 runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_v2(symbol: str, start: str, end: str,
               lot_sizes: dict, params: dict,
               min_bars: int = 100) -> dict:
    from strategies.backtesting.fill_simulator_v2 import run_fast_v2_backtest

    lot_size = get_lot_size_for_symbol(symbol, lot_sizes)
    inst     = 'currency_futures' if any(
        x in symbol.upper() for x in ['USDINR', 'EURINR']
    ) else 'equity_futures'

    try:
        frames  = []
        current = datetime.strptime(start, '%Y-%m-%d')
        end_dt  = datetime.strptime(end,   '%Y-%m-%d')
        while current <= end_dt:
            date_str = current.strftime('%Y-%m-%d')
            df = load_day(symbol, date_str, market_hours_only=True)
            if not df.empty:
                df['_date'] = date_str
                frames.append(df)
            current += timedelta(days=1)

        if not frames:
            return {'symbol': symbol, 'lot_size': lot_size,
                    'net_pnl': 0, 'bars': 0, 'ok': False, 'error': 'no data'}

        full_df = pd.concat(frames, ignore_index=True).sort_values('ts_sec')
        if len(full_df) < min_bars:
            return {'symbol': symbol, 'lot_size': lot_size,
                    'net_pnl': 0, 'bars': len(full_df), 'ok': False,
                    'error': f'only {len(full_df)} bars'}

        result = run_fast_v2_backtest(
            df=full_df,
            beta=params.get('beta', 1.0), theta=params.get('theta', 2.0),
            eta=params.get('eta', 0.5), nu=params.get('nu', 0.2),
            rho=params.get('rho', 0.30), zeta=params.get('zeta', 0.5),
            epsilon_plus=params.get('epsilon_plus', 0.002),
            epsilon_minus=params.get('epsilon_minus', 0.002),
            beta_kappa=params.get('beta_kappa', 0.5),
            theta_kappa=params.get('theta_kappa', 1.5),
            eta_kappa=params.get('eta_kappa', 0.3),
            nu_kappa=params.get('nu_kappa', 0.1),
            phi=params.get('phi', 0.001),
            min_spread=params['min_spread'], max_spread=params['max_spread'],
            open_mult=params.get('open_mult', 2.0), lot_size=lot_size,
            max_inventory=params.get('max_inv', 5),
            classifier_threshold=params.get('classifier_threshold', 0.50),
            instrument_type=inst,
        )
        return {
            'symbol': symbol, 'lot_size': lot_size,
            'gross_pnl': round(result.gross_pnl, 2),
            'costs':     round(result.total_costs, 2),
            'net_pnl':   round(result.net_pnl, 2),
            'fills':     result.total_fills,
            'fill_rate': result.fill_rate,
            'win_rate':  result.win_rate,
            'sharpe':    result.sharpe_ratio,
            'max_dd':    result.max_drawdown,
            'bars':      result.bars_processed,
            'ok': True,
        }
    except Exception as e:
        return {'symbol': symbol, 'lot_size': lot_size,
                'net_pnl': 0, 'bars': 0, 'ok': False, 'error': str(e)[:80]}


# ─────────────────────────────────────────────────────────────────────────────
# Results printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: list, errors: list,
                  total_symbols: int, model: str,
                  use_optimal: bool = False):
    if not results:
        print("\nNo results.")
        return

    results.sort(key=lambda x: x['net_pnl'], reverse=True)
    profitable = [r for r in results if r['net_pnl'] > 0]
    opt_tag    = " [OPTIMIZED PARAMS]" if use_optimal else ""

    print(f"\n{'='*80}")
    print(f"  FULL RANKING -- {len(results)} symbols  [Model: {model.upper()}{opt_tag}]")
    print(f"{'='*80}")
    print(f"\n  {'#':>3} {'Symbol':<25} {'Lot':>4} {'Gross':>11} {'Costs':>10} {'Net':>11} {'Win%':>6} {'Sharpe':>7}")
    print("  " + "-"*80)

    for i, r in enumerate(results, 1):
        sign   = "+" if r['net_pnl'] >= 0 else ""
        marker = " ✓" if r['net_pnl'] > 0 else ""
        print(f"  {i:>3} {r['symbol']:<25} {r['lot_size']:>4} "
              f"Rs.{r['gross_pnl']:>9,.0f} Rs.{r['costs']:>8,.0f} "
              f"Rs.{sign}{r['net_pnl']:>9,.0f} "
              f"{r['win_rate']:>5.1f}% {r['sharpe']:>7.2f}{marker}")

    print(f"\n{'='*80}")
    print(f"  PROFITABLE: {len(profitable)} / {len(results)} symbols")
    print(f"{'='*80}\n")

    if profitable:
        total = sum(r['net_pnl'] for r in profitable)
        print(f"  Combined net PnL if trading all: Rs.+{total:,.0f}\n")
        for r in profitable:
            print(f"  {r['symbol']:<25} {r['lot_size']:>4} "
                  f"Rs.+{r['net_pnl']:>10,.0f} "
                  f"{r['fills']:>7} {r['win_rate']:>5.1f}% "
                  f"-Rs.{abs(r['max_dd']):>9,.0f}")

    if errors:
        print(f"\n  {len(errors)} errors:")
        for e in errors[:5]:
            print(f"    {e['symbol']}: {e.get('error', 'unknown')}")

    skipped = total_symbols - len(results) - len(errors)
    print(f"\n  Tested: {len(results)} | Profitable: {len(profitable)} | "
          f"Skipped: {skipped} | Errors: {len(errors)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch backtest — all stock futures"
    )
    parser.add_argument("--start",              default="2026-05-27")
    parser.add_argument("--end",                default="2026-05-27")
    parser.add_argument("--model",              default="v1", choices=["v1", "v2"])
    parser.add_argument("--use-optimal-params", action="store_true")
    parser.add_argument("--min-bars",           type=int,   default=100)
    parser.add_argument("--workers",            type=int,   default=8)
    parser.add_argument("--gamma",              type=float, default=0.001)
    parser.add_argument("--kappa",              type=float, default=1.5)
    parser.add_argument("--min-spread",         type=float, default=0.10)
    parser.add_argument("--max-spread",         type=float, default=10.0)
    parser.add_argument("--open-mult",          type=float, default=2.0)
    parser.add_argument("--queue-aggression",   type=float, default=0.3)
    parser.add_argument("--max-inventory",      type=int,   default=5)
    parser.add_argument("--phi",   type=float, default=0.001)
    parser.add_argument("--rho",   type=float, default=0.30)
    parser.add_argument("--beta",  type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=2.0)
    parser.add_argument("--eta",   type=float, default=0.5)
    parser.add_argument("--nu",    type=float, default=0.2)
    parser.add_argument("--zeta",  type=float, default=0.5)
    args = parser.parse_args()

    global_params = {
        'gamma': args.gamma, 'kappa': args.kappa,
        'min_spread': args.min_spread, 'max_spread': args.max_spread,
        'open_mult': args.open_mult, 'queue_agg': args.queue_aggression,
        'max_inv': args.max_inventory, 'phi': args.phi, 'rho': args.rho,
        'beta': args.beta, 'theta': args.theta, 'eta': args.eta,
        'nu': args.nu, 'zeta': args.zeta,
    }

    t0      = time.perf_counter()
    opt_tag = " [OPTIMIZED PARAMS]" if args.use_optimal_params else ""

    print(f"\n{'='*70}")
    print(f"  Batch Backtest — All Stock Futures  [Model: {args.model.upper()}{opt_tag}]")
    print(f"  Period:  {args.start} -> {args.end}")
    print(f"  Workers: {args.workers}")

    # Dynamic expiry dates
    cur_month = datetime.now()
    exp1 = get_contract_expiry(cur_month.year, cur_month.month)
    next_month = cur_month.month % 12 + 1
    next_year  = cur_month.year + (1 if next_month == 1 else 0)
    exp2 = get_contract_expiry(next_year, next_month)
    print(f"  Expiries: {cur_month.strftime('%b').upper()}={exp1}  "
          f"{datetime(next_year, next_month, 1).strftime('%b').upper()}={exp2}")
    print(f"{'='*70}\n")

    print("Loading lot sizes...")
    lot_sizes = get_lot_sizes()

    print("Finding available symbols...")
    symbols = get_symbols(args.start, args.end)
    print(f"Found {len(symbols)} stock futures\n")

    if not symbols:
        print("No data found.")
        return

    results = []
    errors  = []
    counter = [0]
    run_one = run_one_v1 if args.model == 'v1' else run_one_v2

    print(f"{'#':>3} {'Symbol':<30} {'NetPnL':>12} {'Fills':>6} {'Win%':>6} {'Sharpe':>7}")
    print("-"*65)

    def handle_result(r):
        counter[0] += 1
        i = counter[0]
        if not r['ok']:
            errors.append(r)
            print(f"{i:>3} {r['symbol']:<30}  SKIP ({r.get('error', '')[:40]})")
            return
        results.append(r)
        sign = "+" if r['net_pnl'] >= 0 else ""
        print(f"{i:>3} {r['symbol']:<30}  "
              f"Rs.{sign}{r['net_pnl']:>9,.0f}  "
              f"{r['fills']:>5}  {r['win_rate']:>5.1f}%  {r['sharpe']:>6.2f}")

    def run_with_resolved_params(sym):
        p = resolve_params(sym, global_params, args.use_optimal_params, args.model)
        return run_one(sym, args.start, args.end, lot_sizes, p, args.min_bars)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures_map = {pool.submit(run_with_resolved_params, sym): sym for sym in symbols}
        for future in as_completed(futures_map):
            handle_result(future.result())

    elapsed = time.perf_counter() - t0
    print_results(results, errors, len(symbols), args.model, args.use_optimal_params)
    print(f"  Total time: {elapsed:.1f}s  ({elapsed/max(len(symbols),1):.2f}s per symbol)\n")


if __name__ == "__main__":
    main()


def run_backtest_all(
    model:              str  = 'v1',
    start_date:         str  = None,
    end_date:           str  = None,
    symbols:            list = None,
    use_optimal_params: bool = True,
    workers:            int  = 8,
) -> list:
    global_params = {
        'gamma': 0.001, 'kappa': 1.5, 'min_spread': 0.10,
        'max_spread': 10.0, 'open_mult': 2.0, 'queue_agg': 0.3,
        'max_inv': 5, 'phi': 0.001, 'rho': 0.30, 'beta': 1.0,
        'theta': 2.0, 'eta': 0.5, 'nu': 0.2, 'zeta': 0.5,
    }
    lot_sizes = get_lot_sizes()
    if symbols is None:
        symbols = get_symbols(start_date, end_date)
    if not symbols:
        return []

    run_one = run_one_v1 if model == 'v1' else run_one_v2
    results = []

    def run_sym(sym):
        p = resolve_params(sym, global_params, use_optimal_params, model)
        return run_one(sym, start_date, end_date, lot_sizes, p)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_sym, sym): sym for sym in symbols}
        for future in as_completed(futures):
            r = future.result()
            if r['ok']:
                results.append(r)

    return sorted(results, key=lambda x: x['net_pnl'], reverse=True)