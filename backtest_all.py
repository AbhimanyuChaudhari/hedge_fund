import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from data.store.duckdb_client import DuckDBClient
from data.store.s3_client import S3Client
from execution.risk.transaction_costs import TransactionCosts

def backtest_symbol(symbol, lots=1, threshold=0.55, sl=30, tp=90, hold=1800):
    s  = S3Client()
    db = DuckDBClient()
    costs = TransactionCosts(lot_size=25, instrument_type='equity_futures')
    TICK  = 0.05

    # USDINR specific
    if 'USDINR' in symbol:
        costs = TransactionCosts(lot_size=1000, instrument_type='currency_futures')
        TICK  = 0.0025
        lots  = 20

    old_files = s.list_files(f'live/candles/1second/{symbol}/')
    new_files = s.list_files(f'processed/1s/{symbol}/')
    old_dates = [f.split('/')[-1].replace('.parquet','') for f in old_files]
    new_dates = [f.split('/')[-1].replace('.parquet','') for f in new_files]
    dates     = sorted(set(old_dates + new_dates))

    all_trades = []
    total_cash = 0.0

    for d in dates:
        df = None
        try:
            df = db.read_parquet(f'processed/1s/{symbol}/{d}.parquet')
        except:
            try:
                df = db.read_parquet(f'live/candles/1second/{symbol}/{d}.parquet')
            except:
                continue

        if df is None or df.empty:
            continue

        df['ist_sec'] = (df['ts_sec'] + 19800) % 86400
        if 'USDINR' in symbol:
            df = df[(df['ist_sec'] >= 9*3600) & (df['ist_sec'] <= 17*3600)].reset_index(drop=True)
        else:
            df = df[(df['ist_sec'] >= 9*3600+15*60) & (df['ist_sec'] <= 15*3600+30*60)].reset_index(drop=True)

        if len(df) < 100:
            continue

        df['date'] = d
        W = 30; NW = 300

        df['bid_chg']    = df['total_bid_qty'].diff()
        df['ask_chg']    = df['total_ask_qty'].diff()
        pu               = df['close'].diff().abs() < TICK*0.5
        df['cancel_imb'] = 0.0
        df.loc[pu&(df['bid_chg']<0),'cancel_imb'] -= df['bid_chg'].abs()
        df.loc[pu&(df['ask_chg']<0),'cancel_imb'] += df['ask_chg'].abs()

        df['s_cancel'] = -df['cancel_imb'].rolling(W).mean()
        df['s_imb']    =  df['imbalance_last'].rolling(W).mean()

        if 'weighted_mid' in df.columns and 'bid_p1' in df.columns:
            df['mid']    = (df['bid_p1']+df['ask_p1'])/2
            df['s_wmid'] = (df['weighted_mid']-df['mid']).rolling(10).mean()
        else:
            df['s_wmid'] = 0.0

        for col in ['s_cancel','s_imb','s_wmid']:
            rs = df[col].rolling(NW).std()+1e-9
            df[col] = (df[col]/rs).clip(-1,1)

        df['score'] = 0.50*df['s_cancel'] + 0.30*df['s_imb'] + 0.20*df['s_wmid']
        df = df.dropna().reset_index(drop=True)

        inv=0; cash=0.0; ep=0.0; et=0; ps=''

        for i, row in df.iterrows():
            price  = float(row['close'])
            b1     = float(row.get('bid_p1', price-TICK))
            a1     = float(row.get('ask_p1', price+TICK))
            ist    = int(row['ist_sec'])
            score  = float(row['score'])

            close_ist = 17*3600 if 'USDINR' in symbol else 15*3600+30*60
            if ist >= close_ist-120 and inv != 0:
                xp   = b1 if ps=='LONG' else a1
                pnl  = (xp-ep)*inv*lots*costs.lot_size
                pnl -= costs.compute(xp,lots,'sell' if inv>0 else 'buy').total
                cash += pnl
                all_trades.append({'pnl':pnl,'date':d,'symbol':symbol})
                inv=0; ps=''; continue

            if inv != 0:
                ticks  = (price-ep)/TICK; reason=None
                if ps=='LONG':
                    if ticks<-sl: reason='SL'
                    elif ticks>tp: reason='TP'
                else:
                    if ticks>sl: reason='SL'
                    elif ticks<-tp: reason='TP'
                if ist-et>hold: reason='TIME'
                if reason:
                    xp   = b1 if ps=='LONG' else a1
                    pnl  = (xp-ep)*inv*lots*costs.lot_size
                    pnl -= costs.compute(xp,lots,'sell' if inv>0 else 'buy').total
                    cash += pnl
                    all_trades.append({'pnl':pnl,'date':d,'symbol':symbol})
                    inv=0; ps=''

            if inv==0:
                if score>threshold:
                    ep=a1;et=ist;inv=1;ps='LONG'
                    cash-=costs.compute(ep,lots,'buy').total
                elif score<-threshold:
                    ep=b1;et=ist;inv=-1;ps='SHORT'
                    cash-=costs.compute(ep,lots,'sell').total

        if inv!=0:
            final=float(df.iloc[-1]['close'])
            xp=float(df.iloc[-1].get('bid_p1',final)) if inv>0 else float(df.iloc[-1].get('ask_p1',final))
            pnl=(xp-ep)*inv*lots*costs.lot_size
            pnl-=costs.compute(xp,lots,'sell' if inv>0 else 'buy').total
            cash+=pnl
            all_trades.append({'pnl':pnl,'date':d,'symbol':symbol})
            inv=0

        total_cash += cash

    db.close()
    pnls = [t['pnl'] for t in all_trades]
    wins = [p for p in pnls if p>0]
    return {
        'symbol':   symbol,
        'trades':   len(all_trades),
        'win_rate': round(len(wins)/max(len(pnls),1)*100,1),
        'total_pnl':round(total_cash,2),
        'avg_pnl':  round(np.mean(pnls) if pnls else 0,2),
        'days':     len(dates)
    }

# Get all symbols
s = S3Client()
old = s.list_files('live/candles/1second/')
new = s.list_files('processed/1s/')
old_syms = sorted(set([f.split('/')[3] for f in old if len(f.split('/'))>3]))
new_syms = sorted(set([f.split('/')[2] for f in new if len(f.split('/'))>2 and '2026' in f]))
symbols  = sorted(set(old_syms + new_syms))

print(f'Running backtest on {len(symbols)} symbols...')
print()
print(f'{"Symbol":<25} {"Days":>5} {"Trades":>7} {"Win%":>6} {"Total_PnL":>12} {"Avg_PnL":>10}')
print('-'*70)

results = []
for sym in symbols:
    try:
        r = backtest_symbol(sym)
        results.append(r)
        print(f'{r["symbol"]:<25} {r["days"]:>5} {r["trades"]:>7} '
              f'{r["win_rate"]:>5.1f}% Rs.{r["total_pnl"]:>10,.0f} Rs.{r["avg_pnl"]:>8,.0f}')
    except Exception as e:
        print(f'{sym:<25} ERROR: {e}')

print()
print(f'=== Portfolio Summary ===')
total_pnl   = sum(r['total_pnl'] for r in results)
total_trades= sum(r['trades'] for r in results)
avg_wr      = np.mean([r['win_rate'] for r in results])
print(f'Total symbols:  {len(results)}')
print(f'Total trades:   {total_trades}')
print(f'Avg win rate:   {avg_wr:.1f}%')
print(f'Total PnL:      Rs.{total_pnl:,.0f}')
print(f'Best symbol:    {max(results, key=lambda x: x["total_pnl"])["symbol"]}')
print(f'Worst symbol:   {min(results, key=lambda x: x["total_pnl"])["symbol"]}')
