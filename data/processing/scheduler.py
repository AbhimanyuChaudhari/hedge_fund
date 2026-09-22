"""
Daily scheduler — runs after market close.
Processes all unprocessed raw files → 1-second bars → deletes raw.
Runs at 5:30pm IST (12:00pm UTC) via systemd timer.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from data.store.s3_client import S3Client
from data.processing.pipeline import run_pipeline

s3     = S3Client()
BUCKET = s3.bucket

MIN_BARS = 10_000


def get_pending_jobs() -> list[tuple]:
    """Find symbol/date combos with raw data but no processed file."""

    # Raw jobs
    resp     = s3.client.list_objects_v2(Bucket=BUCKET, Prefix='raw/orderbook/')
    raw_jobs = set()
    for obj in resp.get('Contents', []):
        parts = obj['Key'].split('/')
        if len(parts) >= 4:
            symbol = parts[2]
            date   = parts[3]
            if symbol and date and len(date) == 10:
                raw_jobs.add((symbol, date))

    # Processed jobs
    resp      = s3.client.list_objects_v2(Bucket=BUCKET, Prefix='processed/1s/')
    proc_jobs = set()
    for obj in resp.get('Contents', []):
        parts = obj['Key'].split('/')
        if len(parts) >= 4:
            symbol = parts[2]
            date   = parts[3].replace('.parquet', '')
            proc_jobs.add((symbol, date))

    pending = raw_jobs - proc_jobs
    print(f'Pending: {len(pending)} jobs '
          f'({len(raw_jobs)} raw - {len(proc_jobs)} processed)')
    return sorted(pending)


def run_all():
    print(f'\n{"="*55}')
    print(f'Daily processing — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*55}')

    pending = get_pending_jobs()
    if not pending:
        print('Nothing to process.')
        return

    print(f'Processing {len(pending)} jobs with 4 workers...\n')

    def process(job):
        symbol, date = job
        try:
            df = run_pipeline(symbol, date, delete_after=True)
            return f'OK: {symbol}/{date} ({len(df)} bars)' if df is not None else f'SKIP: {symbol}/{date}'
        except Exception as e:
            return f'ERROR: {symbol}/{date}: {e}'

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process, job): job for job in pending}
        for future in as_completed(futures):
            print(future.result())

    print(f'\nProcessing complete.')


if __name__ == '__main__':
    run_all()