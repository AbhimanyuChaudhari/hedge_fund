import sys
sys.path.insert(0, '.')
from data.store.s3_client import S3Client
from data.store.duckdb_client import DuckDBClient
import numpy as np

s  = S3Client()
db = DuckDBClient()

# Get all symbols
old_files = s.list_files('live/candles/1second/')
new_files = s.list_files('processed/1s/')

old_symbols = sorted(set([f.split('/')[3] for f in old_files if len(f.split('/')) > 3]))
new_symbols = sorted(set([f.split('/')[2] for f in new_files if len(f.split('/')) > 2 and '2026' in f]))
all_symbols = sorted(set(old_symbols + new_symbols))

print(f'Total symbols: {len(all_symbols)}')
print()
print(f'{"Symbol":<25} {"Days":>5} {"Total_bars":>12} {"Avg_bars/day":>14}')
print('-'*60)

for sym in all_symbols:
    try:
        old = [f.split('/')[-1].replace('.parquet','') for f in old_files if sym in f]
        new = [f.split('/')[-1].replace('.parquet','') for f in new_files if sym in f]
        dates = sorted(set(old + new))
        total = 0
        for d in dates:
            try:
                df = db.read_parquet(f'processed/1s/{sym}/{d}.parquet')
            except:
                try:
                    df = db.read_parquet(f'live/candles/1second/{sym}/{d}.parquet')
                except:
                    continue
            df['ist_sec'] = (df['ts_sec'] + 19800) % 86400
            mkt = df[(df['ist_sec'] >= 9*3600) & (df['ist_sec'] <= 17*3600)]
            total += len(mkt)
        avg = total // max(len(dates), 1)
        print(f'{sym:<25} {len(dates):>5} {total:>12,} {avg:>14,}')
    except Exception as e:
        print(f'{sym:<25} ERROR: {e}')

db.close()
