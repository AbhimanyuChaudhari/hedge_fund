"""
Data Loader — Backtest
======================
Loads processed features for a symbol and date from AWS S3.
Uses boto3 for file discovery and DuckDB native S3 for reading.
No s3fs dependency.
"""

import os
import re
import boto3
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Iterator
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "hedge-fund-data-ac")
AWS_KEY     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET  = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION  = os.getenv("AWS_REGION", "ap-south-1")

MARKET_OPEN_IST  = 33300
MARKET_CLOSE_IST = 55800

_CONTRACT_RE = re.compile(
    r'\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)FUT$',
    re.IGNORECASE
)


def strip_contract_suffix(symbol: str) -> str:
    return _CONTRACT_RE.sub('', symbol)


def _get_s3():
    return boto3.client('s3',
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION
    )


def _get_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"""
        CREATE SECRET IF NOT EXISTS s3_secret (
            TYPE S3,
            KEY_ID '{AWS_KEY}',
            SECRET '{AWS_SECRET}',
            REGION '{AWS_REGION}'
        )
    """)
    return con


def _s3_exists(key: str) -> bool:
    try:
        _get_s3().head_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except Exception:
        return False


def _s3_glob(prefix: str) -> list[str]:
    try:
        s3       = _get_s3()
        paginator = s3.get_paginator('list_objects_v2')
        keys     = []
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])
        return keys
    except Exception:
        return []


def _market_filter(symbol: str = "") -> str:
    is_currency = any(x in symbol.upper()
                      for x in ["USDINR", "EURINR", "GBPINR", "JPYINR"])
    open_ist  = 9 * 3600 if is_currency else MARKET_OPEN_IST
    close_ist = 17 * 3600 if is_currency else MARKET_CLOSE_IST
    return (
        f"((ts_sec + 19800) % 86400) >= {open_ist} "
        f"AND ((ts_sec + 19800) % 86400) <= {close_ist}"
    )


def _find_s3_path(symbol: str, date: str) -> str | None:
    base  = strip_contract_suffix(symbol)
    year  = date[:4]
    month = date[5:7]

    # 1. New hierarchical structure
    key = f"processed/features/{base}/{year}/{month}/{date}.parquet"
    if _s3_exists(key):
        return f"s3://{BUCKET_NAME}/{key}"

    # 2. Old flat structure — exact symbol
    if symbol != base:
        key = f"processed/features/{symbol}/{date}.parquet"
        if _s3_exists(key):
            return f"s3://{BUCKET_NAME}/{key}"

    # 3. Scan for any contract suffix
    prefix  = f"processed/features/{base}"
    matches = [k for k in _s3_glob(prefix) if k.endswith(f"{date}.parquet")]
    if matches:
        return f"s3://{BUCKET_NAME}/{sorted(matches)[-1]}"

    return None


def load_day(symbol: str, date: str,
             market_hours_only: bool = True) -> pd.DataFrame:
    # DuckDB local cache
    try:
        from strategies.backtesting.duckdb_cache import LocalCache
        _df = LocalCache().load(symbol, date, market_hours_only)
        if not _df.empty:
            print(f"[data_loader] Cache: {len(_df):,} bars | {symbol} | {date}")
            return _df
    except Exception:
        pass

    s3_path = _find_s3_path(symbol, date)
    if s3_path is None:
        print(f"[data_loader] Loaded 0 bars | {symbol} | {date}")
        return pd.DataFrame()

    where = f"WHERE {_market_filter(symbol)}" if market_hours_only else ""
    con   = _get_con()

    try:
        df = con.execute(f"""
            SELECT * FROM read_parquet('{s3_path}')
            {where}
            ORDER BY ts_sec
        """).df()
    except Exception as e:
        print(f"[data_loader] Read error {symbol} | {date}: {e}")
        return pd.DataFrame()

    print(f"[data_loader] Loaded {len(df):,} bars | {symbol} | {date}")

    if not df.empty:
        try:
            from strategies.backtesting.duckdb_cache import LocalCache
            LocalCache().insert(symbol, date, df)
        except Exception:
            pass

    return df


def load_date_range(symbol: str, start: str, end: str,
                    market_hours_only: bool = True) -> pd.DataFrame:
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").date()

    base = strip_contract_suffix(symbol)

    candidate_keys = set()
    for prefix in [
        f"processed/features/{base}/",
        f"processed/features/{base}26",
    ]:
        candidate_keys.update(_s3_glob(prefix))

    frames     = []
    seen_dates = set()

    for key in sorted(candidate_keys):
        fname = Path(key).stem
        try:
            fdate = datetime.strptime(fname, "%Y-%m-%d").date()
        except ValueError:
            continue
        if not (start_dt <= fdate <= end_dt):
            continue
        if fname in seen_dates:
            continue
        seen_dates.add(fname)
        df = load_day(symbol, fname, market_hours_only)
        if not df.empty:
            frames.append(df)

    if not frames:
        print(f"[data_loader] No data: {symbol} | {start} → {end}")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True).sort_values("ts_sec").reset_index(drop=True)
    print(f"[data_loader] Total: {len(combined):,} bars | {symbol} | {start} → {end}")
    return combined


def get_active_symbols(start: str, end: str,
                       exclude_index: bool = True) -> list[str]:
    INDEX_BASES = {
        "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
        "SENSEX", "NIFTYNXT50", "BANKEX",
    }
    keys   = _s3_glob("processed/features/")
    active = set()
    for key in keys:
        parts = key.split("/")
        if len(parts) < 3:
            continue
        base = strip_contract_suffix(parts[2])
        if exclude_index and base in INDEX_BASES:
            continue
        active.add(base)
    return sorted(active)


def iter_bars(symbol: str, start: str, end: str,
              market_hours_only: bool = True) -> Iterator[pd.Series]:
    df = load_date_range(symbol, start, end, market_hours_only)
    if df.empty:
        return
    for _, row in df.iterrows():
        yield row


if __name__ == "__main__":
    print("=== Test: strip_contract_suffix ===")
    for sym in ["CHOLAFIN26JUNFUT", "NIFTY26JUNFUT", "USDINR26529FUT"]:
        print(f"  {sym} → {strip_contract_suffix(sym)}")