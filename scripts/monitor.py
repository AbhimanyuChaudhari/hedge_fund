"""
Live Trading Monitor — CLI Dashboard
Real-time view during market hours.

Usage:
    python scripts/monitor.py
    python scripts/monitor.py --interval 5
"""

import os
import sys
import time
import json
import boto3
import argparse
from datetime import datetime, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "hedge-fund-data-ac")
AWS_KEY     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET  = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION  = os.getenv("AWS_REGION", "ap-south-1")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def colored(text, color):
    return f"{color}{text}{RESET}"


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


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


def _count_files(today: str) -> dict:
    result = {"futures": 0, "options": 0, "currency": 0,
              "spot": 0, "latest_age": None}
    try:
        prefix  = f"raw/orderbook/"
        folders = set()
        for key in _s3_list(prefix):
            parts = key.split("/")
            if len(parts) > 2:
                folders.add(parts[2])

        now = time.time()
        for folder in folders:
            keys = _s3_list(f"raw/orderbook/{folder}/{today}/")
            if not keys:
                continue
            if folder.endswith("FUT") and "USDINR" not in folder:
                result["futures"] += len(keys)
            elif folder.endswith(("CE", "PE")):
                result["options"] += len(keys)
            elif "USDINR" in folder:
                result["currency"] += len(keys)
            elif folder.endswith("_SPOT"):
                result["spot"] += len(keys)

    except Exception as e:
        result["error"] = str(e)
    return result


def get_collector_status(today: str, timeout: int = 15) -> dict:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_count_files, today)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            return {"error": "Timeout — S3 slow to respond"}
        except Exception as e:
            return {"error": str(e)}


def count_processed(today: str) -> dict:
    try:
        futures = _s3_list(f"processed/features/")
        options = _s3_list(f"processed/options/")
        fut_count = sum(1 for k in futures if k.endswith(f"{today}.parquet"))
        opt_count = sum(1 for k in options if k.endswith(f"{today}.parquet"))
        return {"futures": fut_count, "options": opt_count}
    except Exception:
        return {"futures": 0, "options": 0}


def load_portfolio() -> dict:
    path = Path("logs/portfolio_state.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_recent_trades(n: int = 5) -> list:
    path = Path(f"logs/trades/{date.today()}.csv")
    if not path.exists():
        return []
    try:
        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))
        return rows[-n:]
    except Exception:
        return []


def load_recent_orders(n: int = 5) -> list:
    path = Path(f"logs/orders/{date.today()}.csv")
    if not path.exists():
        return []
    try:
        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))
        return rows[-n:]
    except Exception:
        return []


def get_ist_seconds() -> int:
    utc = (datetime.utcnow().hour * 3600 +
           datetime.utcnow().minute * 60 +
           datetime.utcnow().second)
    return (utc + 19800) % 86400


def format_ist(ist_sec: int) -> str:
    h = ist_sec // 3600
    m = (ist_sec % 3600) // 60
    s = ist_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def market_status() -> tuple[str, str]:
    ist       = get_ist_seconds()
    nse_open  = 9 * 3600 + 15 * 60
    nse_close = 15 * 3600 + 30 * 60
    cds_open  = 9 * 3600
    cds_close = 17 * 3600

    if nse_open <= ist <= nse_close:
        rem = nse_close - ist
        nse = f"OPEN ({rem//3600}h {(rem%3600)//60}m left)"
    else:
        to_open = (nse_open - ist) % 86400
        nse = f"CLOSED (opens in {to_open//3600}h {(to_open%3600)//60}m)"

    cds = f"OPEN ({(cds_close-ist)//3600}h left)" if cds_open <= ist <= cds_close else "CLOSED"
    return nse, cds


def render(refresh: int = 10):
    today   = date.today().strftime("%Y-%m-%d")
    now_str = datetime.utcnow().strftime("%H:%M:%S UTC")
    ist_str = format_ist(get_ist_seconds())

    clear_screen()
    print(colored("=" * 60, CYAN))
    print(colored(f"  HEDGE FUND MONITOR  |  {now_str}  |  IST {ist_str}", BOLD))
    print(colored(f"  {today}  |  refresh={refresh}s", CYAN))
    print(colored("=" * 60, CYAN))

    print(f"\n{colored('COLLECTOR', BOLD)}  (fetching...)", end="", flush=True)
    status = get_collector_status(today)
    print(f"\r{colored('COLLECTOR', BOLD)}              ")
    print("-" * 40)

    if "error" in status:
        print(f"  {colored(status['error'], RED)}")
    else:
        for cat in ["futures", "options", "currency", "spot"]:
            count = status.get(cat, 0)
            col   = colored("LIVE", GREEN) if count > 0 else colored("NO DATA", RED)
            print(f"  {cat:<12} {col:<20} {count:>5} files")

    print(f"\n{colored('PROCESSED TODAY', BOLD)}")
    print("-" * 40)
    proc = count_processed(today)
    print(f"  Futures:  {proc['futures']} symbols")
    print(f"  Options:  {proc['options']} underlyings")

    print(f"\n{colored('PORTFOLIO', BOLD)}")
    print("-" * 40)
    port = load_portfolio()
    if port and port.get("date") == today:
        pnl = port.get("realized_pnl", 0)
        col = GREEN if pnl >= 0 else RED
        print(f"  Realized PnL:  {colored(f'Rs.{pnl:+,.0f}', col)}")
        positions = {s: p for s, p in port.get("positions", {}).items()
                     if p.get("lots", 0) != 0}
        if positions:
            for sym, pos in positions.items():
                print(f"  Position: {sym} {pos['lots']:+d}L @ {pos['avg_price']:.2f}")
        else:
            print(f"  Position:      {colored('FLAT', GREEN)}")
    else:
        print(f"  No active session today")

    trades = load_recent_trades(5)
    if trades:
        print(f"\n{colored('RECENT TRADES', BOLD)}")
        print("-" * 40)
        for t in trades:
            pnl = float(t.get("pnl", 0))
            col = GREEN if pnl >= 0 else RED
            print(f"  {t.get('timestamp','')[-8:]}  "
                  f"{t.get('side',''):<5} {t.get('lots','')}L "
                  f"@ {float(t.get('price',0)):.2f}  "
                  f"{colored(f'Rs.{pnl:+,.0f}', col)}")

    print(f"\n{colored('MARKET STATUS', BOLD)}")
    print("-" * 40)
    nse, cds = market_status()
    print(f"  NSE (equity):   {colored(nse, GREEN if 'OPEN' in nse else RED)}")
    print(f"  CDS (currency): {colored(cds, GREEN if 'OPEN' in cds else RED)}")
    print(f"\n{colored('Ctrl+C to exit', YELLOW)}")
    print(colored("=" * 60, CYAN))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()
    print("Starting monitor...")
    try:
        while True:
            render(args.interval)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()