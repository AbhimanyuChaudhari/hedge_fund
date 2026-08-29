"""
Local DuckDB Cache — Fast Backtest Data Loader
===============================================
Syncs S3 parquets to local DuckDB. 500x faster than S3 reads for backtesting.

Usage:
    python strategies/backtesting/duckdb_cache.py --sync --start 2026-04-30 --end 2026-05-14
    python strategies/backtesting/duckdb_cache.py --status
"""

import os
import argparse
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import boto3
import duckdb
import pandas as pd

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'hedge-fund-data-ac')
AWS_KEY     = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET  = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION  = os.getenv('AWS_REGION', 'ap-south-1')

CACHE_DIR  = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_FILE = CACHE_DIR / "backtest.duckdb"

MARKET_OPEN_IST  = 33300
MARKET_CLOSE_IST = 55800
CDS_OPEN_IST     = 32400
CDS_CLOSE_IST    = 61200

INDEX_FUTURES = {
    "NIFTY26MAYFUT", "BANKNIFTY26MAYFUT",
    "FINNIFTY26MAYFUT", "MIDCPNIFTY26MAYFUT",
}


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


def _get_duckdb_con() -> duckdb.DuckDBPyConnection:
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


class LocalCache:
    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = cache_file
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._con: Optional[duckdb.DuckDBPyConnection] = None

    def _get_con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(str(self.cache_file))
        return self._con

    def _table_name(self, symbol: str, date: str) -> str:
        sym_clean  = symbol.replace("-", "_").replace("&", "n")
        date_clean = date.replace("-", "")
        return f"t_{sym_clean}_{date_clean}"

    def is_cached(self, symbol: str, date: str) -> bool:
        try:
            tbl    = self._table_name(symbol, date)
            result = self._get_con().execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [tbl]
            ).fetchone()
            return result[0] > 0
        except Exception:
            return False

    def insert(self, symbol: str, date: str, df: pd.DataFrame):
        if df.empty:
            return
        tbl = self._table_name(symbol, date)
        con = self._get_con()
        try:
            con.execute(f"DROP TABLE IF EXISTS {tbl}")
            con.execute(f"CREATE TABLE {tbl} AS SELECT * FROM df")
            con.commit()
        except Exception as e:
            log.warning(f"Insert failed {symbol}/{date}: {e}")

    def load(self, symbol: str, date: str,
             market_hours_only: bool = True) -> pd.DataFrame:
        try:
            tbl    = self._table_name(symbol, date)
            con    = self._get_con()
            exists = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [tbl]
            ).fetchone()[0]
            if not exists:
                return pd.DataFrame()

            is_currency = "USDINR" in symbol.upper()
            if market_hours_only:
                if is_currency:
                    mf = f"WHERE ((ts_sec+19800)%86400)>={CDS_OPEN_IST} AND ((ts_sec+19800)%86400)<={CDS_CLOSE_IST}"
                else:
                    mf = f"WHERE ((ts_sec+19800)%86400)>={MARKET_OPEN_IST} AND ((ts_sec+19800)%86400)<={MARKET_CLOSE_IST}"
            else:
                mf = ""

            return con.execute(f"SELECT * FROM {tbl} {mf} ORDER BY ts_sec").df()
        except Exception as e:
            log.debug(f"Cache load failed {symbol}/{date}: {e}")
            return pd.DataFrame()

    def load_date_range(self, symbol: str, start: str, end: str,
                        market_hours_only: bool = True) -> pd.DataFrame:
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt   = datetime.strptime(end,   "%Y-%m-%d").date()
        frames   = []
        current  = start_dt
        while current <= end_dt:
            if current.weekday() < 5:
                df = self.load(symbol, str(current), market_hours_only)
                if not df.empty:
                    frames.append(df)
            current += timedelta(days=1)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("ts_sec").reset_index(drop=True)

    def status(self) -> dict:
        try:
            con    = self._get_con()
            tables = con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 't_%'"
            ).df()
            if tables.empty:
                return {"tables": 0, "file_mb": 0}
            total_rows = 0
            dates      = set()
            symbols    = set()
            for tbl in tables["table_name"]:
                try:
                    n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                    total_rows += n
                    parts = tbl[2:].rsplit("_", 1)
                    if len(parts) == 2:
                        symbols.add(parts[0])
                        dates.add(parts[1])
                except Exception:
                    pass
            return {
                "tables":     len(tables),
                "symbols":    len(symbols),
                "dates":      len(dates),
                "total_bars": total_rows,
                "file_mb":    round(self.cache_file.stat().st_size / 1024 / 1024, 1) if self.cache_file.exists() else 0,
            }
        except Exception as e:
            return {"error": str(e)}

    def close(self):
        if self._con:
            self._con.close()
            self._con = None


class CacheSync:
    def __init__(self, cache: LocalCache = None):
        self.cache = cache or LocalCache()

    def sync_date(self, date: str, symbols: list = None,
                  force: bool = False) -> tuple:
        keys   = _s3_list(f"processed/features/")
        synced = skipped = 0

        for key in sorted(keys):
            parts = key.split("/")
            if len(parts) < 3:
                continue
            sym = parts[2]

            if not key.endswith(f"{date}.parquet"):
                continue
            if sym in INDEX_FUTURES:
                skipped += 1
                continue
            if sym.endswith(("CE", "PE", "_SPOT")):
                skipped += 1
                continue
            if symbols and sym not in symbols:
                skipped += 1
                continue
            if not force and self.cache.is_cached(sym, date):
                skipped += 1
                continue

            try:
                s3_path = f"s3://{BUCKET_NAME}/{key}"
                con     = _get_duckdb_con()
                df      = con.execute(f"SELECT * FROM read_parquet('{s3_path}')").df()
                con.close()
                if df.empty:
                    skipped += 1
                    continue
                self.cache.insert(sym, date, df)
                synced += 1
                print(f"  OK: {sym} | {date} | {len(df):,} bars")
            except Exception as e:
                log.warning(f"  FAIL: {sym}/{date}: {e}")
                skipped += 1

        return synced, skipped

    def sync_range(self, start: str, end: str, symbols: list = None,
                   force: bool = False) -> dict:
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt   = datetime.strptime(end,   "%Y-%m-%d").date()
        total_s = total_sk = 0
        current = start_dt
        while current <= end_dt:
            if current.weekday() < 5:
                date_str = str(current)
                print(f"\nSyncing {date_str}...")
                s, sk    = self.sync_date(date_str, symbols=symbols, force=force)
                total_s  += s
                total_sk += sk
                print(f"  {s} synced, {sk} skipped")
            current += timedelta(days=1)
        return {"synced": total_s, "skipped": total_sk}


class CachedDataLoader:
    def __init__(self):
        self.cache = LocalCache()

    def load_day(self, symbol: str, date: str,
                 market_hours_only: bool = True) -> pd.DataFrame:
        df = self.cache.load(symbol, date, market_hours_only)
        if not df.empty:
            return df
        from strategies.backtesting.data_loader import load_day as s3_load
        df = s3_load(symbol, date, market_hours_only)
        if not df.empty:
            self.cache.insert(symbol, date, df)
        return df

    def load_date_range(self, symbol: str, start: str, end: str,
                        market_hours_only: bool = True) -> pd.DataFrame:
        df = self.cache.load_date_range(symbol, start, end, market_hours_only)
        if not df.empty:
            return df
        from strategies.backtesting.data_loader import load_date_range as s3_load
        return s3_load(symbol, start, end, market_hours_only)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Local DuckDB cache")
    parser.add_argument("--sync",   action="store_true")
    parser.add_argument("--date",   type=str)
    parser.add_argument("--start",  type=str, default="2026-04-30")
    parser.add_argument("--end",    type=str)
    parser.add_argument("--force",  action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    cache = LocalCache()
    sync  = CacheSync(cache)

    if args.status:
        s = cache.status()
        print(f"\nCache: {CACHE_FILE}")
        print(f"  Tables:     {s.get('tables', 0)}")
        print(f"  Symbols:    {s.get('symbols', 0)}")
        print(f"  Dates:      {s.get('dates', 0)}")
        print(f"  Total bars: {s.get('total_bars', 0):,}")
        print(f"  Size:       {s.get('file_mb', 0)} MB\n")

    if args.sync:
        if args.date:
            print(f"\nSyncing {args.date}...")
            s, sk = sync.sync_date(args.date, force=args.force)
            print(f"Done: {s} synced, {sk} skipped")
        else:
            end = args.end or str(datetime.utcnow().date())
            print(f"\nSyncing {args.start} -> {end}...")
            r = sync.sync_range(args.start, end, force=args.force)
            print(f"\nTotal: {r['synced']} synced, {r['skipped']} skipped")
        s = cache.status()
        print(f"\nCache: {s.get('symbols',0)} symbols | {s.get('dates',0)} dates | {s.get('total_bars',0):,} bars | {s.get('file_mb',0)} MB")

    cache.close()


if __name__ == "__main__":
    main()