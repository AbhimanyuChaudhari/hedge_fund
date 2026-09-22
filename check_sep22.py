import sys
sys.path.insert(0, '.')
from data.store.s3_client import S3Client
from data.store.duckdb_client import DuckDBClient

s  = S3Client()
db = DuckDBClient()

files = s.list_files('processed/1s/')
sep22 = [f for f in files if '2026-09-22' in f]
print(f'Processed files for Sep 22: {len(sep22)}')

try:
    df = db.read_parquet('processed/1s/USDINR26SEPFUT/2026-09-22.parquet')
    df['ist_sec'] = (df['ts_sec'] + 19800) % 86400
    mkt = df[(df['ist_sec'] >= 9*3600) & (df['ist_sec'] <= 17*3600)]
    first = df.iloc[0]['ts_ist']
    last  = df.iloc[-1]['ts_ist']
    print(f'USDINR Sep 22:')
    print(f'  Total bars:      {len(df)}')
    print(f'  Market hours:    {len(mkt)}')
    print(f'  Columns:         {len(df.columns)}')
    print(f'  First bar:       {first}')
    print(f'  Last bar:        {last}')
    print(f'  Spread mean:     {mkt["spread_mean"].mean():.5f}')
    print(f'  Imbalance mean:  {mkt["imbalance_mean"].mean():.4f}')
    print(f'  bid_p1 non-zero: {(mkt["bid_p1"] > 0).sum()}')
    print(f'  Columns: {df.columns.tolist()[:10]}')
except Exception as e:
    print(f'USDINR error: {e}')

db.close()
