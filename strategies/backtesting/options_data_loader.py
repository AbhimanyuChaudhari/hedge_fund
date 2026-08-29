"""
Options Data Loader
===================
Loads processed options features from AWS S3.
"""

import os
import logging
import boto3
import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, Iterator

logger = logging.getLogger(__name__)

BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'hedge-fund-data-ac')
AWS_KEY     = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET  = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION  = os.getenv('AWS_REGION', 'ap-south-1')

MARKET_OPEN_IST  = 33300
MARKET_CLOSE_IST = 55800


def _get_s3():
    return boto3.client('s3',
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION
    )


def _s3_exists(key: str) -> bool:
    try:
        _get_s3().head_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except Exception:
        return False


def _s3_glob(prefix: str) -> list[str]:
    try:
        paginator = _get_s3().get_paginator('list_objects_v2')
        keys = []
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])
        return keys
    except Exception:
        return []


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


def load_options_day(underlying: str, date_str: str,
                     opt_type:          Optional[str]   = None,
                     strike_range:      Optional[tuple] = None,
                     market_hours_only: bool            = True,
                     min_premium:       float           = 1.0,
                     max_tte:           float           = 1.0) -> pd.DataFrame:
    key = f"processed/options/{underlying}/{date_str}.parquet"
    if not _s3_exists(key):
        logger.warning(f"No options data: {underlying} | {date_str}")
        return pd.DataFrame()

    s3_path    = f"s3://{BUCKET_NAME}/{key}"
    conditions = []

    if market_hours_only:
        conditions.append(
            f"((ts_sec + 19800) % 86400) BETWEEN {MARKET_OPEN_IST} AND {MARKET_CLOSE_IST}"
        )
    if opt_type:
        conditions.append(f"opt_type = '{opt_type}'")
    if strike_range:
        conditions.append(f"strike BETWEEN {strike_range[0]} AND {strike_range[1]}")
    if min_premium > 0:
        conditions.append(f"close >= {min_premium}")
    if max_tte < 1.0:
        conditions.append(f"tte <= {max_tte}")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    con   = _get_con()

    try:
        df = con.execute(f"""
            SELECT * FROM read_parquet('{s3_path}')
            {where}
            ORDER BY ts_sec, symbol
        """).df()
    except Exception as e:
        logger.error(f"Read error {underlying} | {date_str}: {e}")
        return pd.DataFrame()

    logger.info(
        f"Loaded {len(df):,} option bars | {underlying} | {date_str} | "
        f"symbols={df['symbol'].nunique() if not df.empty else 0}"
    )
    return df


def load_options_chain(underlying: str, date_str: str,
                       ts_sec: int, atm_price: float,
                       strikes_each_side: int = 5) -> pd.DataFrame:
    key = f"processed/options/{underlying}/{date_str}.parquet"
    if not _s3_exists(key):
        return pd.DataFrame()

    s3_path  = f"s3://{BUCKET_NAME}/{key}"
    interval = 50 if underlying == "NIFTY" else 100
    atm      = round(atm_price / interval) * interval
    min_s    = atm - strikes_each_side * interval
    max_s    = atm + strikes_each_side * interval
    con      = _get_con()

    try:
        return con.execute(f"""
            SELECT * FROM read_parquet('{s3_path}')
            WHERE ts_sec = {ts_sec}
              AND strike BETWEEN {min_s} AND {max_s}
              AND close > 0.5
            ORDER BY strike, opt_type
        """).df()
    except Exception as e:
        logger.error(f"Chain read error: {e}")
        return pd.DataFrame()


def iter_option_bars(underlying: str, date_str: str,
                     opt_type: Optional[str] = None,
                     strike_range: Optional[tuple] = None,
                     market_hours_only: bool = True,
                     min_premium: float = 5.0) -> Iterator[tuple]:
    df = load_options_day(
        underlying=underlying, date_str=date_str,
        opt_type=opt_type, strike_range=strike_range,
        market_hours_only=market_hours_only, min_premium=min_premium,
    )
    if df.empty:
        return
    for ts_sec, group in df.groupby("ts_sec"):
        yield int(ts_sec), group


if __name__ == "__main__":
    print("=== Options Data Loader Tests ===\n")
    keys  = _s3_glob(f"processed/options/NIFTY/")
    dates = sorted([Path(k).stem for k in keys if k.endswith('.parquet')])
    print(f"Available NIFTY options dates: {dates}")

    if dates:
        date_str = dates[-1]
        print(f"\nLoading {date_str}...")
        df = load_options_day("NIFTY", date_str, opt_type="CE", min_premium=1.0)
        if not df.empty:
            print(f"Rows: {len(df):,}")
            print(f"Symbols: {df['symbol'].nunique()}")
            print(df.head(5).to_string())
        else:
            print("No data found.")