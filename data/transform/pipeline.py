import os
import io
import boto3
import duckdb
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "hedge-fund-data-ac")
AWS_KEY     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET  = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION  = os.getenv("AWS_REGION", "ap-south-1")


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


def get_duckdb_con() -> duckdb.DuckDBPyConnection:
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


def build_features_duckdb(symbol: str, date: str) -> pd.DataFrame:
    prefix = f"raw/orderbook/{symbol}/{date}/"
    files  = _s3_list(prefix)
    if not files:
        print(f"No data: {symbol} | {date}")
        return pd.DataFrame()

    gcs_path = f"s3://{BUCKET_NAME}/{prefix}*.parquet"
    con      = get_duckdb_con()

    df = con.execute(f"""
        WITH raw AS (
            SELECT
                (ts_local_ns // 1000000000)::BIGINT AS ts_sec,
                symbol,
                last_price,
                volume,
                avg_price,
                oi,
                spread,
                mid_price,
                book_imbalance,
                total_bid_qty,
                total_ask_qty,
                bid_p1, bid_q1,
                bid_p2, bid_q2,
                bid_p3, bid_q3,
                bid_p4, bid_q4,
                bid_p5, bid_q5,
                ask_p1, ask_q1,
                ask_p2, ask_q2,
                ask_p3, ask_q3,
                ask_p4, ask_q4,
                ask_p5, ask_q5,
                (bid_p1 * ask_q1 + ask_p1 * bid_q1) /
                    NULLIF(bid_q1 + ask_q1, 0)        AS weighted_mid,
                ABS(last_price - mid_price)            AS price_impact,
                spread / NULLIF(mid_price, 0) * 10000 AS spread_bps
            FROM read_parquet('{gcs_path}')
            WHERE last_price > 0
        ),

        bars AS (
            SELECT
                symbol,
                ts_sec,
                FIRST(last_price)            AS open,
                MAX(last_price)              AS high,
                MIN(last_price)              AS low,
                LAST(last_price)             AS close,
                LAST(volume) - FIRST(volume) AS volume_delta,
                LAST(volume)                 AS volume_cumulative,
                COUNT(*)                     AS tick_count,
                LAST(avg_price)              AS vwap,
                LAST(oi)                     AS oi,
                AVG(spread)                  AS spread_mean,
                MAX(spread)                  AS spread_max,
                AVG(spread_bps)              AS spread_bps,
                AVG(book_imbalance)          AS imbalance_mean,
                STDDEV(book_imbalance)       AS imbalance_std,
                LAST(book_imbalance)         AS imbalance_last,
                LAST(total_bid_qty)          AS total_bid_qty,
                LAST(total_ask_qty)          AS total_ask_qty,
                LAST(weighted_mid)           AS weighted_mid,
                AVG(price_impact)            AS price_impact,
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
            FROM raw
            GROUP BY symbol, ts_sec
            ORDER BY ts_sec
        )

        SELECT
            symbol, ts_sec,
            open, high, low, close,
            volume_delta, volume_cumulative, tick_count, vwap, oi,
            spread_mean, spread_max, spread_bps,
            imbalance_mean, imbalance_std, imbalance_last,
            total_bid_qty, total_ask_qty, weighted_mid, price_impact,
            bid_p1, bid_q1, bid_p2, bid_q2, bid_p3, bid_q3,
            bid_p4, bid_q4, bid_p5, bid_q5,
            ask_p1, ask_q1, ask_p2, ask_q2, ask_p3, ask_q3,
            ask_p4, ask_q4, ask_p5, ask_q5,

            STDDEV(close) OVER (ORDER BY ts_sec ROWS BETWEEN 9  PRECEDING AND CURRENT ROW) AS realized_vol_10s,
            STDDEV(close) OVER (ORDER BY ts_sec ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS realized_vol_30s,
            STDDEV(close) OVER (ORDER BY ts_sec ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS realized_vol_60s,
            STDDEV(close) OVER (ORDER BY ts_sec ROWS BETWEEN 299 PRECEDING AND CURRENT ROW) AS realized_vol_300s,

            AVG(imbalance_last) OVER (ORDER BY ts_sec ROWS BETWEEN 9  PRECEDING AND CURRENT ROW) AS imbalance_ma_10s,
            AVG(imbalance_last) OVER (ORDER BY ts_sec ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS imbalance_ma_30s,
            AVG(imbalance_last) OVER (ORDER BY ts_sec ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS imbalance_ma_60s,

            (spread_mean - AVG(spread_mean) OVER (ORDER BY ts_sec ROWS BETWEEN 299 PRECEDING AND CURRENT ROW)) /
                NULLIF(STDDEV(spread_mean) OVER (ORDER BY ts_sec ROWS BETWEEN 299 PRECEDING AND CURRENT ROW), 0)
            AS spread_zscore,

            tick_count / NULLIF(AVG(tick_count) OVER (ORDER BY ts_sec ROWS BETWEEN 59 PRECEDING AND CURRENT ROW), 0) AS volume_ratio,

            (close - LAG(close, 10) OVER (ORDER BY ts_sec)) / NULLIF(LAG(close, 10) OVER (ORDER BY ts_sec), 0) AS price_mom_10s,
            (close - LAG(close, 30) OVER (ORDER BY ts_sec)) / NULLIF(LAG(close, 30) OVER (ORDER BY ts_sec), 0) AS price_mom_30s,
            (close - LAG(close, 60) OVER (ORDER BY ts_sec)) / NULLIF(LAG(close, 60) OVER (ORDER BY ts_sec), 0) AS price_mom_60s

        FROM bars ORDER BY ts_sec
    """).df()

    df["ts_ist"] = (
        pd.to_datetime(df["ts_sec"], unit="s", utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    print(f"Built {len(df):,} bars | {len(df.columns)} columns | {symbol} | {date}")
    return df


def save_processed(df: pd.DataFrame, symbol: str, date: str):
    if df.empty:
        return

    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    buf.seek(0)

    key = f"processed/features/{symbol}/{date}.parquet"
    _get_s3().put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=buf.getvalue(),
        ContentType="application/octet-stream"
    )
    print(f"Saved -> s3://{BUCKET_NAME}/{key}")


def run_pipeline(symbol: str, date: str):
    print(f"\n{'='*50}")
    print(f"Processing: {symbol} | {date}")
    print(f"{'='*50}")
    df = build_features_duckdb(symbol, date)
    if df.empty:
        return None
    save_processed(df, symbol, date)
    return df


if __name__ == "__main__":
    import time
    for date in ["2026-04-30"]:
        start   = time.time()
        df      = run_pipeline("NIFTY26MAYFUT", date)
        elapsed = time.time() - start
        if df is not None:
            print(f"\nCompleted in {elapsed:.1f}s")
            print(df[["ts_ist", "open", "close", "bid_p1", "bid_q1"]].head(5).to_string())