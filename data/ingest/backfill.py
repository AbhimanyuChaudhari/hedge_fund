import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datetime import datetime, timedelta
from data.ingest.historical import HistoricalDataDownloader
import time

INSTRUMENTS = [
    {"symbol": "NIFTY50",   "token": 256265},
    {"symbol": "BANKNIFTY", "token": 260105},
]

INTERVAL_MAX_DAYS = {
    "day":       2000,
    "60minute":  400,
    "15minute":  200,
    "5minute":   100,
}

def date_chunks(from_date, to_date, chunk_days):
    current = from_date
    while current < to_date:
        chunk_end = min(current + timedelta(days=chunk_days), to_date)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)

def backfill(from_date: datetime, to_date: datetime):
    downloader = HistoricalDataDownloader()

    for inst in INSTRUMENTS:
        for interval, max_days in INTERVAL_MAX_DAYS.items():
            for chunk_start, chunk_end in date_chunks(from_date, to_date, max_days - 5):
                try:
                    downloader.download_and_store(
                        instrument_token=inst["token"],
                        symbol=inst["symbol"],
                        from_date=chunk_start,
                        to_date=chunk_end,
                        interval=interval
                    )
                    time.sleep(0.4)
                except Exception as e:
                    print(f"Failed {inst['symbol']} {interval} {chunk_start.date()}: {e}")

if __name__ == '__main__':
    from_date = datetime(2023, 1, 1)
    to_date   = datetime(2024, 12, 31)
    print(f"Backfilling from {from_date.date()} to {to_date.date()}")
    backfill(from_date, to_date)
    print("Backfill complete.")