import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from data.store.duckdb_client import DuckDBClient
from data.store.s3_client import S3Client
from execution.risk.transaction_costs import TransactionCosts

def run_backtest(symbol, dates, lots=20, threshold=0.55,
                 sl_ticks=10, tp_ticks=30, hold_secs=300):
    db    = DuckDBClient()
    costs = TransactionCosts(lot_size=1000, instrument_type='currency_futures')
    TICK  = 0.0025
    cash  = 0.0
    all_trades = []

    for date in dates:
        df = db.read_parquet(f'live/candles/1second/{symbol}/{date}.parquet')
        df['ist_sec'] = (df['ts_sec'] + 19800) % 86400
        df = df[(df['ist_sec']>=9*3600)&(df['ist_sec']<=17*3600)].reset_index(drop=True)
        if len(df)<300: continue

        df['bid_chg']    = df['total_bid_qty'].diff()
        df['ask_chg']    = df['total_ask_qty'].diff()
        pu               = df['close'].diff().abs()<TICK*0.5
        df['cancel_imb'] = 0.0
        df.loc[pu&(df['bid_chg']<0),'cancel_imb'] -= df['bid_chg'].abs()
        df.loc[pu&(df['ask_chg']<0),'cancel_imb'] += df['ask_chg'].abs()
        df['s_cancel'] = -df['cancel_imb'].rolling(30).mean()
        df['s_imb']    =  df['imbalance_last'].rolling(30).mean()
        df['mid']      = (df['bid_p1']+df['ask_p1'])/2
        df['s_wmid']   = (df['weighted_mid']-df['mid']).rolling(10).mean()
        for col in ['s_cancel','s_imb','s_wmid']:
            rs = df[col].rolling(300).std()+1e-9
            df[col] = (df[col]/rs).clip(-1,1)
        df['score'] = 0.50*df['s_cancel']+0.30*df['s_imb']+0.20*df['s_wmid']
        df = df.dropna().reset_index(drop=True)

        inv=0; ep=0.0; et=0; ps=''
        for i,row in df.iterrows():
            price=float(row['close'])
            b1=float(row.get('bid_p1',price-TICK))
            a1=float(row.get('ask_p1',price+TICK))
            ist=int(row['ist_sec'])
            sc=float(row['score'])

            if ist>=17*3600-60 and inv!=0:
                xp=b1 if ps=='LONG' else a1
                pnl=(xp-ep)*inv*lots*1000-costs.compute(xp,lots,'sell' if inv>0 else 'buy').total
                cash+=pnl
                all_trades.append({'pnl':pnl,'hold':ist-et})
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
                    cash+=pnl
                    all_trades.append({'pnl':pnl,'hold':ist-et})
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
        'trades':len(all_trades),
        'win_rate':len(wins)/max(len(pnls),1),
        'total_pnl':round(cash,2),
        'avg_pnl':round(np.mean(pnls) if pnls else 0,2),
        'max_win':round(max(pnls) if pnls else 0,2),
        'max_loss':round(min(pnls) if pnls else 0,2),
        'avg_hold':round(np.mean([t['hold'] for t in all_trades]) if all_trades else 0,1),
    }

s     = S3Client()
files = s.list_files('live/candles/1second/USDINR26SEPFUT/')
dates = sorted([f.split('/')[-1].replace('.parquet','') for f in files])

configs = [
    (0.55, 10, 30,  300),
    (0.55, 15, 45,  300),
    (0.55, 20, 60,  300),
    (0.55, 20, 60,  600),
    (0.55, 20, 60,  900),
    (0.55, 20, 60, 1200),
    (0.55, 20, 60, 1800),
    (0.55, 30, 90,  600),
    (0.55, 30, 90,  900),
    (0.55, 30, 90, 1200),
    (0.55, 30, 90, 1800),
    (0.55, 40,120,  900),
    (0.55, 40,120, 1800),
    (0.55, 50,150, 1800),
    (0.60, 20, 60,  900),
    (0.60, 30, 90,  900),
    (0.65, 30, 90,  900),
]

print(f'{"thr":>5} {"sl":>4} {"tp":>5} {"hold":>6} {"trades":>7} {"win%":>6} {"total_pnl":>12} {"avg_pnl":>9} {"max_win":>9} {"max_loss":>9} {"avg_hold":>9}')
print('-'*95)

for thr,sl,tp,hold in configs:
    r = run_backtest('USDINR26SEPFUT', dates, lots=20,
                    threshold=thr, sl_ticks=sl, tp_ticks=tp, hold_secs=hold)
    print(f'{thr:>5.2f} {sl:>4} {tp:>5} {hold:>6} {r["trades"]:>7} {r["win_rate"]*100:>5.1f}% '
          f'Rs.{r["total_pnl"]:>10,.0f} Rs.{r["avg_pnl"]:>7,.0f} '
          f'Rs.{r["max_win"]:>7,.0f} Rs.{r["max_loss"]:>7,.0f} {r["avg_hold"]:>7.0f}s')
