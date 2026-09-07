import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from data.store.duckdb_client import DuckDBClient
from data.store.s3_client import S3Client
from execution.risk.transaction_costs import TransactionCosts

def run_backtest(symbol, date, lots=20, threshold=0.55,
                 sl_ticks=10, tp_ticks=30, hold_secs=300,
                 trade_hours=None):
    db = DuckDBClient()
    df = db.read_parquet(f'live/candles/1second/{symbol}/{date}.parquet')
    db.close()
    df['ist_sec'] = (df['ts_sec'] + 19800) % 86400
    OPEN=9*3600; CLOSE=17*3600
    df = df[(df['ist_sec']>=OPEN)&(df['ist_sec']<=CLOSE)].reset_index(drop=True)
    if len(df)<300: return None
    TICK=0.0025
    costs=TransactionCosts(lot_size=1000,instrument_type='currency_futures')
    df['bid_chg']=df['total_bid_qty'].diff()
    df['ask_chg']=df['total_ask_qty'].diff()
    pu=df['close'].diff().abs()<TICK*0.5
    df['cancel_imb']=0.0
    df.loc[pu&(df['bid_chg']<0),'cancel_imb']-=df['bid_chg'].abs()
    df.loc[pu&(df['ask_chg']<0),'cancel_imb']+=df['ask_chg'].abs()
    df['s_cancel']=-df['cancel_imb'].rolling(30).mean()
    df['s_imb']=df['imbalance_last'].rolling(30).mean()
    df['mid']=(df['bid_p1']+df['ask_p1'])/2
    df['s_wmid']=(df['weighted_mid']-df['mid']).rolling(10).mean()
    for col in ['s_cancel','s_imb','s_wmid']:
        rs=df[col].rolling(300).std()+1e-9
        df[col]=(df[col]/rs).clip(-1,1)
    df['score']=0.50*df['s_cancel']+0.30*df['s_imb']+0.20*df['s_wmid']
    df['hour']=df['ist_sec']//3600
    df=df.dropna().reset_index(drop=True)

    inv=0; cash=0.0; trades=[]; ep=0.0; et=0; ps=''
    for i,row in df.iterrows():
        price=float(row['close'])
        b1=float(row.get('bid_p1',price-TICK))
        a1=float(row.get('ask_p1',price+TICK))
        ist=int(row['ist_sec'])
        hour=int(row['hour'])
        sc=float(row['score'])

        # Apply hour-based signal direction
        # 9am: normal, 11am+: inverted
        if hour == 9:
            effective_score = sc        # normal
        elif hour >= 11:
            effective_score = -sc       # inverted
        else:
            effective_score = sc * 0.5  # weak at 10am

        # Only trade during specified hours
        can_enter = True
        if trade_hours:
            can_enter = hour in trade_hours

        if ist>=CLOSE-60 and inv!=0:
            xp=b1 if ps=='LONG' else a1
            pnl=(xp-ep)*inv*lots*1000-costs.compute(xp,lots,'sell' if inv>0 else 'buy').total
            cash+=pnl; trades.append({'pnl':pnl,'hold':ist-et,'hour':hour})
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
                cash+=pnl; trades.append({'pnl':pnl,'hold':ist-et,'hour':hour})
                inv=0; ps=''

        if inv==0 and can_enter:
            if effective_score>threshold:
                ep=a1;et=ist;inv=1;ps='LONG'
                cash-=costs.compute(ep,lots,'buy').total
            elif effective_score<-threshold:
                ep=b1;et=ist;inv=-1;ps='SHORT'
                cash-=costs.compute(ep,lots,'sell').total

    final=float(df.iloc[-1]['close'])
    mtm=cash+inv*(final-ep)*lots*1000
    pnls=[t['pnl'] for t in trades]; wins=[p for p in pnls if p>0]
    return {'trades':len(trades),'win_rate':len(wins)/max(len(pnls),1),
            'total_pnl':round(mtm,2),'avg_pnl':round(np.mean(pnls) if pnls else 0,2),
            'max_win':round(max(pnls) if pnls else 0,2),
            'max_loss':round(min(pnls) if pnls else 0,2)}

s=S3Client()
files=s.list_files('live/candles/1second/USDINR26SEPFUT/')
dates=sorted([f.split('/')[-1].replace('.parquet','') for f in files])

print(f'{"hours":<20} {"sl":>4} {"tp":>4} {"hold":>6} {"trades":>7} {"win%":>6} {"total_pnl":>12} {"avg_pnl":>10}')
print('-'*75)

configs = [
    ([9],         10, 30, 300),
    ([9],         10, 30, 600),
    ([9],         15, 45, 300),
    ([9],         20, 60, 300),
    ([9,10],      10, 30, 300),
    ([11,12,13],  10, 30, 300),
    ([11,12,13],  15, 45, 300),
    ([9,11,12],   10, 30, 300),
    (None,        10, 30, 300),   # all hours regime-aware
]

for hours, sl, tp, hold in configs:
    tp_pnl=tt=tw=0
    for date in dates:
        r=run_backtest('USDINR26SEPFUT',date,lots=20,threshold=0.55,
                      sl_ticks=sl,tp_ticks=tp,hold_secs=hold,
                      trade_hours=hours)
        if r:
            tp_pnl+=r['total_pnl']; tt+=r['trades']; tw+=int(r['win_rate']*r['trades'])
    wr=tw/max(tt,1)*100; ap=tp_pnl/max(tt,1)
    h_str=str(hours) if hours else 'all+regime'
    print(f'{h_str:<20} {sl:>4} {tp:>4} {hold:>6} {tt:>7} {wr:>5.1f}% Rs.{tp_pnl:>10,.0f} Rs.{ap:>8,.0f}')
