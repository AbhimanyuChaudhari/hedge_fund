import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from data.store.duckdb_client import DuckDBClient
from data.store.s3_client import S3Client

SYMBOL = 'USDINR26SEPFUT'
s      = S3Client()
files  = s.list_files(f'live/candles/1second/{SYMBOL}/')
dates  = sorted([f.split('/')[-1].replace('.parquet','') for f in files])
db     = DuckDBClient()
all_df = []

for date in dates:
    df = db.read_parquet(f'live/candles/1second/{SYMBOL}/{date}.parquet')
    df['ist_sec'] = (df['ts_sec'] + 19800) % 86400
    df = df[(df['ist_sec'] >= 9*3600) & (df['ist_sec'] <= 17*3600)].reset_index(drop=True)
    all_df.append(df)

db.close()
df = pd.concat(all_df, ignore_index=True)
print(f'Total bars: {len(df)} across {len(dates)} days')

WINDOW = 60
results = {'ofi_l1_only': [], 'ofi_equal_5lev': [], 'ofi_raw_imb': []}
idx_list = list(range(WINDOW, len(df), 10))

for i in idx_list:
    window = df.iloc[i-WINDOW:i]
    rows = [window.iloc[j].to_dict() for j in range(len(window))]

    # Signal 1: Level 1 only OFI
    ofi_l1 = np.mean([
        rows[j]['bid_q1'] - rows[j-1]['bid_q1'] -
        (rows[j]['ask_q1'] - rows[j-1]['ask_q1'])
        for j in range(1, len(rows))
    ])

    # Signal 2: Equal weight 5-level OFI
    ofi_5 = 0.0
    for j in range(1, len(rows)):
        for lev in range(1, 6):
            ofi_5 += (rows[j].get(f'bid_q{lev}',0) - rows[j-1].get(f'bid_q{lev}',0) -
                     (rows[j].get(f'ask_q{lev}',0) - rows[j-1].get(f'ask_q{lev}',0)))
    ofi_5 /= max(len(rows)-1, 1)

    # Signal 3: Raw imbalance
    ofi_raw = np.mean([r.get('imbalance_last', 0) for r in rows])

    results['ofi_l1_only'].append(ofi_l1)
    results['ofi_equal_5lev'].append(ofi_5)
    results['ofi_raw_imb'].append(ofi_raw)

df_sub = df.iloc[idx_list].reset_index(drop=True)

print(f'\n{"Signal":<20} {"IC_30s":>8} {"IC_60s":>8} {"IC_120s":>8} {"IC_300s":>8}')
print('-' * 55)

for name, vals in results.items():
    sig  = pd.Series(vals)
    ics  = []
    for h in [30, 60, 120, 300]:
        fwd   = df_sub['close'].shift(-h//10) - df_sub['close']
        valid = pd.concat([sig, fwd], axis=1).dropna()
        ic    = valid.iloc[:,0].corr(valid.iloc[:,1])
        ics.append(ic)
    print(f'{name:<20} {ics[0]:>8.4f} {ics[1]:>8.4f} {ics[2]:>8.4f} {ics[3]:>8.4f}')
