import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
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

TICK = 0.0025

# ── Build signals ──────────────────────────────────────────────────
tot_bid = df['total_bid_qty']
tot_ask = df['total_ask_qty']
df['imbalance']  = (tot_bid - tot_ask) / (tot_bid + tot_ask + 1e-9)

# Price momentum
df['momentum_10s'] = df['close'].pct_change(10)
df['momentum_30s'] = df['close'].pct_change(30)

# Cancel imbalance
bid_chg = tot_bid.diff()
ask_chg = tot_ask.diff()
pu      = df['close'].diff().abs() < TICK * 0.5
cancel_bid = pd.Series(0.0, index=df.index)
cancel_ask = pd.Series(0.0, index=df.index)
cancel_bid[pu & (bid_chg < 0)] = bid_chg.abs()[pu & (bid_chg < 0)]
cancel_ask[pu & (ask_chg < 0)] = ask_chg.abs()[pu & (ask_chg < 0)]
df['cancel_imb'] = (cancel_ask - cancel_bid).rolling(10).mean()

# Forward returns
for h in [30, 60, 120]:
    df[f'fwd_{h}s'] = df['close'].shift(-h) - df['close']

df = df.dropna().reset_index(drop=True)
print(f'Bars: {len(df)}')

# ── Ridge regression for optimal drift weights ─────────────────────
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

print(f'\n=== Drift Weight Estimation (Ridge Regression) ===')
print(f'µ = β1×imbalance + β2×momentum + β3×cancel_imb')
print()
print(f'{"horizon":>8} {"β_imb":>12} {"β_mom":>12} {"β_cancel":>12} {"R²":>8} {"IC":>8}')
print('-' * 65)

features = ['imbalance', 'momentum_30s', 'cancel_imb']

for h in [30, 60, 120]:
    y     = df[f'fwd_{h}s'].values
    X_raw = df[features].values
    valid = ~(np.isnan(X_raw).any(axis=1) | np.isnan(y))
    X_v   = X_raw[valid]; y_v = y[valid]

    # Standardize features
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_v)

    # RidgeCV finds optimal regularization
    model  = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
    model.fit(X_sc, y_v)

    # Normalize weights to sum to 1
    w     = model.coef_
    w_norm = w / (np.abs(w).sum() + 1e-9)

    # R² and IC
    y_pred = model.predict(X_sc)
    ss_res = np.sum((y_v - y_pred)**2)
    ss_tot = np.sum((y_v - y_v.mean())**2)
    r2     = 1 - ss_res/ss_tot
    ic     = pd.Series(X_sc @ w).corr(pd.Series(y_v))

    print(f'{h:>6}s  {w_norm[0]:>12.4f} {w_norm[1]:>12.4f} {w_norm[2]:>12.4f} '
          f'{r2:>8.6f} {ic:>8.4f}')

# ── Cancel vol weight estimation ───────────────────────────────────
print(f'\n=== Cancel Volatility Weight Estimation ===')
print(f'σ²_eff = σ² + δ² × cancel_rate × weight')
print()

log_ret      = np.log(df['close'] / df['close'].shift(1))
df['vol_realized'] = log_ret.rolling(60).std() * np.sqrt(60) * df['close']
df['cancel_rate']  = (cancel_bid + cancel_ask).rolling(30).mean()
df = df.dropna().reset_index(drop=True)

# Regress realized vol on cancel rate
X_cancel = df['cancel_rate'].values.reshape(-1, 1)
y_vol    = df['vol_realized'].values
valid    = ~(np.isnan(X_cancel).any(axis=1) | np.isnan(y_vol))
X_v = X_cancel[valid]; y_v = y_vol[valid]

coeffs, _, _, _ = lstsq(
    np.column_stack([X_v, np.ones(len(X_v))]), y_v, rcond=None
)
cancel_vol_weight = coeffs[0]
print(f'cancel_vol_weight = {cancel_vol_weight:.6f}')
print(f'(replaces hardcoded 1.0)')

# ── Summary of all calibrated values ──────────────────────────────
print(f'\n=== Summary: Update these in CSTParameters and USDINRConfig ===')
print(f'kyle_lambda:       0.00148   (from OLS at 60s)')
print(f'cancel_vol_weight: {cancel_vol_weight:.6f}  (from vol regression)')
print()
print(f'Drift weights at 60s horizon (normalized):')
y     = df['fwd_60s'].values if 'fwd_60s' in df.columns else df['fwd_60s'].values
X_raw = df[features].values
valid = ~(np.isnan(X_raw).any(axis=1) | np.isnan(y))
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_raw[valid])
model  = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
model.fit(X_sc, y[valid])
w      = model.coef_
w_norm = w / (np.abs(w).sum() + 1e-9)
print(f'  ofi_weight (imbalance):  {w_norm[0]:.4f}  (was 0.6 hardcoded)')
print(f'  momentum_weight:         {w_norm[1]:.4f}  (was 0.4 hardcoded)')
print(f'  cancel_weight:           {w_norm[2]:.4f}  (was 0.0 — not included!)')
