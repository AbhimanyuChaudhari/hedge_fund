import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from data.store.duckdb_client import DuckDBClient
from data.store.s3_client import S3Client
from execution.risk.transaction_costs import TransactionCosts

def run_backtest(symbol, dates, lots=20, threshold=0.55,
                 sl_ticks=30, tp_ticks=90, hold_secs=1800,
                 w_cancel=0.40, w_imb=0.25, w_wmid=0.15, w_mom=0.20):
    db    = DuckDBClient()
    costs = TransactionCosts(lot_size=1000, instrument_type='currency_futures')
    TICK  = 0.0025
    cash  = 0.0
    all_trades = []

    for date in dates:
        df = db.read_parquet(f'live/candles/1second/{symbol}/{date}.parquet')
        df['ist_sec'] = (df['ts_sec'] + 19800) % 86400
        df = df[(df['ist_sec'] >= 9*3600) &
                (df['ist_sec'] <= 17*3600)].reset_index(drop=True)
        if len(df) < 300: continue

        W=30; NW=300
        df['bid_chg']    = df['total_bid_qty'].diff()
        df['ask_chg']    = df['total_ask_qty'].diff()
        pu               = df['close'].diff().abs() < TICK*0.5
        df['cancel_imb'] = 0.0
        df.loc[pu&(df['bid_chg']<0),'cancel_imb'] -= df['bid_chg'].abs()
        df.loc[pu&(df['ask_chg']<0),'cancel_imb'] += df['ask_chg'].abs()

        df['s_cancel']   = -df['cancel_imb'].rolling(W).mean()
        df['s_imb']      =  df['imbalance_last'].rolling(W).mean()
        df['mid']        = (df['bid_p1']+df['ask_p1'])/2
        df['s_wmid']     = (df['weighted_mid']-df['mid']).rolling(10).mean()
        df['s_momentum'] = -df['close'].pct_change(30).rolling(10).mean()

        for col in ['s_cancel','s_imb','s_wmid','s_momentum']:
            rs = df[col].rolling(NW).std()+1e-9
            df[col] = (df[col]/rs).clip(-1,1)

        df['score'] = (w_cancel * df['s_cancel'] +
                       w_imb    * df['s_imb']    +
                       w_wmid   * df['s_wmid']   +
                       w_mom    * df['s_momentum'])

        df['hour'] = df['ist_sec']//3600
        df = df.dropna().reset_index(drop=True)

        inv=0; ep=0.0; et=0; ps=''
        for i,row in df.iterrows():
            price=float(row['close'])
            b1=float(row.get('bid_p1',price-TICK))
            a1=float(row.get('ask_p1',price+TICK))
            ist=int(row['ist_sec'])
            sc=float(row['score'])

            if int(row['hour']) in [15]: continue
            if ist>=17*3600-120 and inv!=0:
                xp=b1 if ps=='LONG' else a1
                pnl=(xp-ep)*inv*lots*1000-costs.compute(xp,lots,'sell' if inv>0 else 'buy').total
                cash+=pnl; all_trades.append({'pnl':pnl,'hold':ist-et})
                inv=0; ps=''; continue
            if inv!=0:
                ticks=(price-ep)/TICK; reason=None
                if ps=='LONG':
                    if ticks<-sl_ticks: reason='SL'
                    elif ticks>tp_ticks: reason='TP'
                else:
                    if ticks>sl_ticks: reason='SL'
                    elif ticks<-tp_ticks: reason='TP'
                if ist-et>hold_secs: reason='TIME'
                if reason:
                    xp=b1 if ps=='LONG' else a1
                    pnl=(xp-ep)*inv*lots*1000-costs.compute(xp,lots,'sell' if inv>0 else 'buy').total
                    cash+=pnl; all_trades.append({'pnl':pnl,'hold':ist-et})
                    inv=0; ps=''
            if inv==0:
                if sc>threshold:
                    ep=a1;et=ist;inv=1;ps='LONG'
                    cash-=costs.compute(ep,lots,'buy').total
                elif sc<-threshold:
                    ep=b1;et=ist;inv=-1;ps='SHORT'
                    cash-=costs.compute(ep,lots,'sell').total

    db.close()
    pnls=[t['pnl'] for t in all_trades]
    wins=[p for p in pnls if p>0]
    return {
        'trades':    len(all_trades),
        'win_rate':  len(wins)/max(len(pnls),1),
        'total_pnl': round(cash,2),
        'avg_pnl':   round(np.mean(pnls) if pnls else 0,2),
    }

s     = S3Client()
files = s.list_files('live/candles/1second/USDINR26SEPFUT/')
dates = sorted([f.split('/')[-1].replace('.parquet','') for f in files])

print(f'Grid search — 4 signals combined')
print(f'{"w_cancel":>9} {"w_imb":>7} {"w_wmid":>7} {"w_mom":>7} {"trades":>7} {"win%":>6} {"total_pnl":>12} {"avg_pnl":>10}')
print('-'*75)

configs = [
    (0.50, 0.30, 0.20, 0.00),  # original
    (0.40, 0.25, 0.15, 0.20),  # add small momentum
    (0.35, 0.25, 0.15, 0.25),  # more momentum
    (0.30, 0.20, 0.10, 0.40),  # momentum heavy
    (0.40, 0.20, 0.10, 0.30),  # balanced
    (0.45, 0.25, 0.10, 0.20),  # cancel dominant + momentum
    (0.35, 0.30, 0.10, 0.25),  # imb + momentum
    (0.40, 0.30, 0.00, 0.30),  # no wmid
    (0.50, 0.20, 0.10, 0.20),  # cancel dominant
]

for wc,wi,ww,wm in configs:
    r = run_backtest('USDINR26SEPFUT', dates, lots=20,
                    threshold=0.55, sl_ticks=30,
                    tp_ticks=90, hold_secs=1800,
                    w_cancel=wc, w_imb=wi, w_wmid=ww, w_mom=wm)
    wr = r['win_rate']*100
    print(f'{wc:>9.2f} {wi:>7.2f} {ww:>7.2f} {wm:>7.2f} {r["trades"]:>7} '
          f'{wr:>5.1f}% Rs.{r["total_pnl"]:>10,.0f} Rs.{r["avg_pnl"]:>8,.0f}')
