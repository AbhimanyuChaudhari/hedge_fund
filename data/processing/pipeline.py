"""
Daily processing pipeline.
Reads raw S3 ticks → builds 1-second bars + features → saves processed → deletes raw.
Runs after market close (5:30pm IST for USDINR, 3:30pm IST for equity).
"""
import io
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import duckdb
import pandas as pd
import boto3
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

from data.store.s3_client import S3Client

s3     = S3Client()
BUCKET = s3.bucket


def get_raw_files(symbol: str, date: str) -> list[str]:
    """List all raw tick files for a symbol/date."""
    prefix = f'raw/orderbook/{symbol}/{date}/'
    resp   = s3.client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return [o['Key'] for o in resp.get('Contents', [])]


def read_raw_ticks(symbol: str, date: str) -> pd.DataFrame:
    """Download and concatenate all raw tick files for a symbol/date."""
    keys = get_raw_files(symbol, date)
    if not keys:
        return pd.DataFrame()

    dfs = []
    for key in sorted(keys):
        try:
            resp = s3.client.get_object(Bucket=BUCKET, Key=key)
            buf  = io.BytesIO(resp['Body'].read())
            dfs.append(pd.read_parquet(buf))
        except Exception as e:
            print(f'[READ ERROR] {key}: {e}')

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('ts_local_ns').reset_index(drop=True)
    return df


def build_1s_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Convert raw ticks to 1-second bars using DuckDB.
    Same logic as GCP duckdb_pipeline.py — ported to work on pandas DataFrame.
    """
    if df.empty:
        return pd.DataFrame()

    # Add ts_sec
    df['ts_sec'] = df['ts_local_ns'] // 1_000_000_000

    # Precompute weighted mid
    df['weighted_mid'] = (
        (df['bid_p1'] * df['ask_q1'] + df['ask_p1'] * df['bid_q1']) /
        (df['bid_q1'] + df['ask_q1'] + 1e-9)
    )
    df['spread_bps'] = df['spread'] / (df['mid_price'] + 1e-9) * 10000
    df['price_impact'] = (df['last_price'] - df['mid_price']).abs()

    con = duckdb.connect()
    con.register('ticks', df)

    result = con.execute("""
        WITH bars AS (
            SELECT
                symbol,
                ts_sec,
                FIRST(last_price)                        AS open,
                MAX(last_price)                          AS high,
                MIN(last_price)                          AS low,
                LAST(last_price)                         AS close,
                LAST(volume) - FIRST(volume)             AS volume_delta,
                COUNT(*)                                 AS tick_count,
                LAST(avg_price)                          AS vwap,
                LAST(oi)                                 AS oi,

                -- Spread
                AVG(spread)                              AS spread_mean,
                MAX(spread)                              AS spread_max,
                AVG(spread_bps)                          AS spread_bps,

                -- Imbalance
                AVG(book_imbalance)                      AS imbalance_mean,
                STDDEV(book_imbalance)                   AS imbalance_std,
                LAST(book_imbalance)                     AS imbalance_last,

                -- Depth
                LAST(total_bid_qty)                      AS total_bid_qty,
                LAST(total_ask_qty)                      AS total_ask_qty,
                LAST(weighted_mid)                       AS weighted_mid,
                AVG(price_impact)                        AS price_impact,

                -- L5 order book snapshot (last tick of second)
                LAST(bid_p1) AS bid_p1, LAST(bid_q1) AS bid_q1,
                LAST(bid_p2) AS bid_p2, LAST(bid_q2) AS bid_q2,
                LAST(bid_p3) AS bid_p3, LAST(bid_q3) AS bid_q3,
                LAST(bid_p4) AS bid_p4, LAST(bid_q4) AS bid_q4,
                LAST(bid_p5) AS bid_p5, LAST(bid_q5) AS bid_q5,
                LAST(ask_p1) AS ask_p1, LAST(ask_q1) AS ask_q1,
                LAST(ask_p2) AS ask_p2, LAST(ask_q2) AS ask_q2,
                LAST(ask_p3) AS ask_p3, LAST(ask_q3) AS ask_q3,
                LAST(ask_p4) AS ask_p4, LAST(ask_q4) AS ask_q4,
                LAST(ask_p5) AS ask_p5, LAST(ask_q5) AS ask_q5

            FROM ticks
            WHERE last_price > 0
            GROUP BY symbol, ts_sec
            ORDER BY ts_sec
        )

        SELECT
            symbol,
            ts_sec,
            open, high, low, close,
            volume_delta, tick_count, vwap, oi,
            spread_mean, spread_max, spread_bps,
            imbalance_mean, imbalance_std, imbalance_last,
            total_bid_qty, total_ask_qty,
            weighted_mid, price_impact,
            bid_p1, bid_q1, bid_p2, bid_q2, bid_p3, bid_q3,
            bid_p4, bid_q4, bid_p5, bid_q5,
            ask_p1, ask_q1, ask_p2, ask_q2, ask_p3, ask_q3,
            ask_p4, ask_q4, ask_p5, ask_q5,

            -- Rolling volatility
            STDDEV(close) OVER (
                ORDER BY ts_sec ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
            ) AS realized_vol_10s,
            STDDEV(close) OVER (
                ORDER BY ts_sec ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
            ) AS realized_vol_30s,
            STDDEV(close) OVER (
                ORDER BY ts_sec ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
            ) AS realized_vol_60s,
            STDDEV(close) OVER (
                ORDER BY ts_sec ROWS BETWEEN 299 PRECEDING AND CURRENT ROW
            ) AS realized_vol_300s,

            -- Imbalance MAs
            AVG(imbalance_last) OVER (
                ORDER BY ts_sec ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
            ) AS imbalance_ma_10s,
            AVG(imbalance_last) OVER (
                ORDER BY ts_sec ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
            ) AS imbalance_ma_30s,
            AVG(imbalance_last) OVER (
                ORDER BY ts_sec ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
            ) AS imbalance_ma_60s,

            -- Spread z-score
            (spread_mean - AVG(spread_mean) OVER (
                ORDER BY ts_sec ROWS BETWEEN 299 PRECEDING AND CURRENT ROW
            )) / NULLIF(STDDEV(spread_mean) OVER (
                ORDER BY ts_sec ROWS BETWEEN 299 PRECEDING AND CURRENT ROW
            ), 0) AS spread_zscore,

            -- Volume ratio
            tick_count / NULLIF(AVG(tick_count) OVER (
                ORDER BY ts_sec ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
            ), 0) AS volume_ratio,

            -- Price momentum
            (close - LAG(close, 10) OVER (ORDER BY ts_sec)) /
                NULLIF(LAG(close, 10) OVER (ORDER BY ts_sec), 0) AS price_mom_10s,
            (close - LAG(close, 30) OVER (ORDER BY ts_sec)) /
                NULLIF(LAG(close, 30) OVER (ORDER BY ts_sec), 0) AS price_mom_30s,
            (close - LAG(close, 60) OVER (ORDER BY ts_sec)) /
                NULLIF(LAG(close, 60) OVER (ORDER BY ts_sec), 0) AS price_mom_60s

        FROM bars
        ORDER BY ts_sec
    """).df()

    # Add readable IST timestamp
    result['ts_ist'] = (
        pd.to_datetime(result['ts_sec'], unit='s', utc=True)
        .dt.tz_convert('Asia/Kolkata')
        .dt.strftime('%Y-%m-%d %H:%M:%S')
    )

    # Market hours filter
    result['ist_sec'] = (result['ts_sec'] + 19800) % 86400
    if any(c in symbol for c in ['USDINR','EURINR','GBPINR','JPYINR']):
        result = result[
            (result['ist_sec'] >= 9*3600) &
            (result['ist_sec'] <= 17*3600)
        ]
    else:
        result = result[
            (result['ist_sec'] >= 9*3600 + 15*60) &
            (result['ist_sec'] <= 15*3600 + 30*60)
        ]

    con.close()
    print(f'Built {len(result):,} bars | {len(result.columns)} cols | {symbol}')
    return result.reset_index(drop=True)


def save_processed(df: pd.DataFrame, symbol: str, date: str):
    """Save processed 1-second bars to S3."""
    if df.empty:
        return
    key = f'processed/1s/{symbol}/{date}.parquet'
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression='zstd')
    buf.seek(0)
    s3.client.put_object(
        Bucket = BUCKET,
        Key    = key,
        Body   = buf.getvalue(),
    )
    print(f'Saved → s3://{BUCKET}/{key}')


def delete_raw(symbol: str, date: str):
    """Delete all raw tick files for a symbol/date after processing."""
    keys = get_raw_files(symbol, date)
    if not keys:
        return
    s3.client.delete_objects(
        Bucket = BUCKET,
        Delete = {'Objects': [{'Key': k} for k in keys]}
    )
    print(f'Deleted {len(keys)} raw files for {symbol}/{date}')


def validate(df: pd.DataFrame, symbol: str, date: str) -> bool:
    """Basic data quality checks before saving."""
    if df.empty:
        print(f'[SKIP] {symbol}/{date}: empty')
        return False

    # Minimum bars check
    min_bars = 10_000 if 'USDINR' not in symbol else 15_000
    if len(df) < min_bars:
        print(f'[WARN] {symbol}/{date}: only {len(df)} bars (min {min_bars})')
        return False

    # Price sanity check
    if df['close'].std() == 0:
        print(f'[SKIP] {symbol}/{date}: all prices identical — stale data')
        return False

    # Spread sanity check
    if (df['spread_mean'] == 0).mean() > 0.5:
        print(f'[WARN] {symbol}/{date}: >50% bars have zero spread')

    return True


def run_pipeline(symbol: str, date: str,
                 delete_after: bool = True) -> pd.DataFrame | None:
    """Full pipeline for one symbol/date."""
    print(f'\n{"="*50}')
    print(f'Processing: {symbol} | {date}')
    print(f'{"="*50}')

    # Read raw ticks
    df_raw = read_raw_ticks(symbol, date)
    if df_raw.empty:
        print(f'No raw data found')
        return None

    print(f'Raw ticks: {len(df_raw):,}')

    # Build 1-second bars
    df_bars = build_1s_bars(df_raw, symbol)

    # Validate
    if not validate(df_bars, symbol, date):
        return None

    # Save processed
    save_processed(df_bars, symbol, date)

    # Delete raw
    if delete_after:
        delete_raw(symbol, date)

    return df_bars


if __name__ == '__main__':
    # Test on USDINR
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'USDINR26SEPFUT'
    date   = sys.argv[2] if len(sys.argv) > 2 else '2026-09-17'
    run_pipeline(symbol, date, delete_after=False)  # don't delete on test run