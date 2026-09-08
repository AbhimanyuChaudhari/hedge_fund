import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from data.store.duckdb_client import DuckDBClient
from data.store.s3_client import S3Client
from strategies.implementations.avellaneda_stoikov.cst.drift import DriftEstimator
from strategies.implementations.avellaneda_stoikov.cst.order_flow import OrderFlowEstimator
from strategies.implementations.avellaneda_stoikov.cst.parameters import CSTParameters

SYMBOL = 'USDINR26SEPFUT'
s      = S3Client()
files  = s.list_files(f'live/candles/1second/{SYMBOL}/')
dates  = sorted([f.split('/')[-1].replace('.parquet','') for f in files])

params    = CSTParameters()
flow_est  = OrderFlowEstimator(params)
drift_est = DriftEstimator(params)
db        = DuckDBClient()
all_df    = []

for date in dates:
    df = db.read_parquet(f'live/candles/1second/{SYMBOL}/{date}.parquet')
    df['ist_sec'] = (df['ts_sec'] + 19800) % 86400
    df = df[(df['ist_sec'] >= 9*3600) & (df['ist_sec'] <= 17*3600)].reset_index(drop=True)
    all_df.append(df)

db.close()
df = pd.concat(all_df, ignore_index=True)
print(f'Total bars: {len(df)} across {len(dates)} days')

WINDOW = 60
ofi_ck_vals = []
ofi_raw_vals = []

for i in range(WINDOW, len(df), 10):
    window = df.iloc[i-WINDOW:i]
    bars = []
    for _, r in window.iterrows():
        bar = {
            'close':         float(r.get('close', 0)),
            'volume_delta':  float(r.get('volume_delta', 0)),
            'total_bid_qty': float(r.get('total_bid_qty', 0)),
            'total_ask_qty': float(r.get('total_ask_qty', 0)),
            'imbalance_last':float(r.get('imbalance_last', 0)),
        }
        for level in range(1, 6):
            bar[f'bid_p{level}'] = float(r.get(f'bid_p{level}', 0))
            bar[f'bid_q{level}'] = float(r.get(f'bid_q{level}', 0))
            bar[f'ask_p{level}'] = float(r.get(f'ask_p{level}', 0))
            bar[f'ask_q{level}'] = float(r.get(f'ask_q{level}', 0))
        bars.append(bar)

    flow    = flow_est.estimate(bars)
    ofi_ck  = drift_est._weighted_ofi(bars)
    ofi_raw = np.mean([b['imbalance_last'] for b in bars])

    ofi_ck_vals.append(ofi_ck)
    ofi_raw_vals.append(ofi_raw)

# Compute IC at different horizons
idx = list(range(WINDOW, len(df), 10))
df_sub = df.iloc[idx].reset_index(drop=True)

print(f'\n{"Signal":<20} {"IC_30s":>8} {"IC_60s":>8} {"IC_120s":>8} {"IC_300s":>8}')
print('-' * 55)

signals = {
    'ofi_raw_imbalance': pd.Series(ofi_raw_vals),
    'ofi_cont_kukanov':  pd.Series(ofi_ck_vals),
}

for name, sig in signals.items():
    ics = []
    for h in [30, 60, 120, 300]:
        fwd  = df_sub['close'].shift(-h//10) - df_sub['close']
        valid = pd.concat([sig, fwd], axis=1).dropna()
        ic   = valid.iloc[:,0].corr(valid.iloc[:,1])
        ics.append(ic)
    print(f'{name:<20} {ics[0]:>8.4f} {ics[1]:>8.4f} {ics[2]:>8.4f} {ics[3]:>8.4f}')
