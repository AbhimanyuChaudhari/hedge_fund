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
df = pd.concat(all_df, ignore_index=True).reset_index(drop=True)
print(f'Total bars: {len(df)} across {len(dates)} days')

TICK = 0.0025

# ── OFI Definition (Cont-Kukanov-Stoikov) ─────────────────────────
# OFI_t = change in bid quantity - change in ask quantity at level 1
# Positive OFI = net buying pressure
df['ofi_l1'] = df['bid_q1'].diff() - df['ask_q1'].diff()

# Multi-level OFI (equal weights)
df['ofi_5lev'] = sum(
    df[f'bid_q{i}'].diff() - df[f'ask_q{i}'].diff()
    for i in range(1, 6)
)

# Raw imbalance (already shown to have highest IC)
tot_bid = df['total_bid_qty']
tot_ask = df['total_ask_qty']
df['imbalance'] = (tot_bid - tot_ask) / (tot_bid + tot_ask + 1e-9)

# ── Price change ───────────────────────────────────────────────────
# Δprice = close[t+1] - close[t]
df['dprice_1s']   = df['close'].diff(1).shift(-1)   # next second
df['dprice_5s']   = df['close'].diff(5).shift(-5)   # next 5 seconds
df['dprice_10s']  = df['close'].diff(10).shift(-10) # next 10 seconds
df['dprice_30s']  = df['close'].diff(30).shift(-30) # next 30 seconds
df['dprice_60s']  = df['close'].diff(60).shift(-60) # next 60 seconds

df = df.dropna().reset_index(drop=True)
print(f'After dropna: {len(df)} bars')

# ── OLS Regression: Δprice = λ × OFI + ε ─────────────────────────
from numpy.linalg import lstsq

print(f'\n=== Kyle Lambda Estimation ===')
print(f'Δprice = λ × OFI + ε')
print()
print(f'{"Signal":<15} {"horizon":>8} {"lambda":>12} {"R²":>8} {"t-stat":>8}')
print('-' * 55)

signals = {
    'ofi_l1':    df['ofi_l1'].values,
    'ofi_5lev':  df['ofi_5lev'].values,
    'imbalance': df['imbalance'].values,
}

for sig_name, ofi in signals.items():
    for h_col in ['dprice_1s','dprice_5s','dprice_10s','dprice_30s','dprice_60s']:
        y     = df[h_col].values
        X     = np.column_stack([ofi, np.ones(len(ofi))])
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_v   = X[valid]; y_v = y[valid]

        # OLS
        coeffs, _, _, _ = lstsq(X_v, y_v, rcond=None)
        lambda_est = coeffs[0]

        # R²
        y_pred = X_v @ coeffs
        ss_res = np.sum((y_v - y_pred)**2)
        ss_tot = np.sum((y_v - y_v.mean())**2)
        r2     = 1 - ss_res/ss_tot if ss_tot > 0 else 0

        # t-statistic
        n      = len(y_v)
        sigma2 = ss_res / (n - 2)
        XtX_inv = np.linalg.inv(X_v.T @ X_v)
        se     = np.sqrt(sigma2 * XtX_inv[0, 0])
        tstat  = lambda_est / (se + 1e-12)

        h = h_col.replace('dprice_','')
        print(f'{sig_name:<15} {h:>8} {lambda_est:>12.8f} {r2:>8.6f} {tstat:>8.2f}')

    print()

# ── Best estimate summary ──────────────────────────────────────────
print(f'=== Recommended kyle_lambda values ===')
for sig_name, ofi in signals.items():
    y     = df['dprice_60s'].values
    X     = np.column_stack([ofi, np.ones(len(ofi))])
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    coeffs, _, _, _ = lstsq(X[valid], y[valid], rcond=None)
    print(f'{sig_name:<15} kyle_lambda at 60s = {coeffs[0]:.8f}')
