"""
USDINR instrument-specific backtest.
Reads from both old (live/candles/1second/) and new (processed/1s/) formats.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import numpy as np
import pandas as pd
from data.store.duckdb_client import DuckDBClient
from data.store.s3_client import S3Client
from execution.risk.transaction_costs import TransactionCosts
from strategies.instruments.USDINR.config import USDINRConfig


def run(date: str = None, lots: int = 20,
        config: USDINRConfig = None,
        verbose: bool = True) -> dict:

    cfg   = config or USDINRConfig()
    costs = TransactionCosts(lot_size=cfg.lot_size,
                             instrument_type=cfg.instrument_type)
    s     = S3Client()

    # Get dates from both old and new paths
    old_files = s.list_files(f'live/candles/1second/{cfg.symbol}/')
    new_files = s.list_files(f'processed/1s/{cfg.symbol}/')
    old_dates = [f.split('/')[-1].replace('.parquet','') for f in old_files]
    new_dates = [f.split('/')[-1].replace('.parquet','') for f in new_files]
    dates     = sorted(set(old_dates + new_dates))

    if date:
        dates = [d for d in dates if d == date]
    if not dates:
        return {'error': 'no data'}

    db     = DuckDBClient()
    all_df = []

    for d in dates:
        # Try new format first, fall back to old
        df = None
        try:
            df = db.read_parquet(f'processed/1s/{cfg.symbol}/{d}.parquet')
        except Exception:
            try:
                df = db.read_parquet(f'live/candles/1second/{cfg.symbol}/{d}.parquet')
            except Exception:
                print(f'No data for {d}')
                continue

        if df is None or df.empty:
            continue

        df['ist_sec'] = (df['ts_sec'] + 19800) % 86400
        df = df[(df['ist_sec'] >= cfg.market_open_ist) &
                (df['ist_sec'] <= cfg.market_close_ist)].reset_index(drop=True)
        if len(df) < 100:
            continue

        df['date'] = d

        TICK = cfg.tick_size
        W    = cfg.signal_window
        NW   = cfg.norm_window

        # Precompute signals per day
        df['bid_chg']    = df['total_bid_qty'].diff()
        df['ask_chg']    = df['total_ask_qty'].diff()
        pu               = df['close'].diff().abs() < TICK * 0.5
        df['cancel_imb'] = 0.0
        df.loc[pu & (df['bid_chg'] < 0), 'cancel_imb'] -= df['bid_chg'].abs()
        df.loc[pu & (df['ask_chg'] < 0), 'cancel_imb'] += df['ask_chg'].abs()

        df['s_cancel'] = -df['cancel_imb'].rolling(W).mean()
        df['s_imb']    =  df['imbalance_last'].rolling(W).mean()

        # Handle both old and new column names
        if 'weighted_mid' in df.columns:
            df['mid']    = (df['bid_p1'] + df['ask_p1']) / 2
            df['s_wmid'] = (df['weighted_mid'] - df['mid']).rolling(10).mean()
        else:
            df['s_wmid'] = 0.0

        for col in ['s_cancel', 's_imb', 's_wmid']:
            rs      = df[col].rolling(NW).std() + 1e-9
            df[col] = (df[col] / rs).clip(-1, 1)

        df['score'] = (cfg.w_cancel    * df['s_cancel'] +
                       cfg.w_imbalance * df['s_imb']    +
                       cfg.w_wmid      * df['s_wmid'])
        df['hour']  = df['ist_sec'] // 3600
        df = df.dropna().reset_index(drop=True)
        all_df.append(df)

    db.close()

    if not all_df:
        return {'error': 'no data after filter'}

    # ── Looser entry restrictions ──────────────────────────────────
    # Lower threshold → more trades
    # Shorter hold → more trades
    # No avoid hours — let data decide
    SCORE_THRESHOLD = 0.40   # was 0.55
    SL_TICKS        = 20     # was 30
    TP_TICKS        = 60     # was 90
    MAX_HOLD        = 1800   # keep 30 min
    AVOID_HOURS     = [15]   # only avoid 3pm

    all_trades = []
    total_cash = 0.0

    for df_day in all_df:
        if df_day.empty:
            continue

        inv = 0; cash = 0.0
        ep  = 0.0; et = 0; ps = ''

        for i, row in df_day.iterrows():
            price   = float(row['close'])
            bid_p1  = float(row.get('bid_p1', price - cfg.tick_size))
            ask_p1  = float(row.get('ask_p1', price + cfg.tick_size))
            ist_sec = int(row['ist_sec'])
            hour    = int(row['hour'])
            score   = float(row['score'])

            if hour in AVOID_HOURS:
                continue

            # Force close near session end
            if ist_sec >= cfg.market_close_ist - 120 and inv != 0:
                xp   = bid_p1 if ps == 'LONG' else ask_p1
                pnl  = (xp - ep) * inv * lots * cfg.lot_size
                pnl -= costs.compute(xp, lots,
                       'sell' if inv > 0 else 'buy').total
                cash += pnl
                all_trades.append({'pnl': pnl, 'reason': 'FORCE_CLOSE',
                                   'hold': ist_sec - et, 'hour': hour,
                                   'date': row['date']})
                inv = 0; ps = ''
                continue

            # Exit
            if inv != 0:
                ticks  = (price - ep) / cfg.tick_size
                reason = None
                if ps == 'LONG':
                    if ticks < -SL_TICKS:  reason = 'SL'
                    elif ticks > TP_TICKS: reason = 'TP'
                else:
                    if ticks >  SL_TICKS:  reason = 'SL'
                    elif ticks < -TP_TICKS: reason = 'TP'
                if ist_sec - et > MAX_HOLD:
                    reason = 'TIME'
                if reason:
                    xp   = bid_p1 if ps == 'LONG' else ask_p1
                    pnl  = (xp - ep) * inv * lots * cfg.lot_size
                    pnl -= costs.compute(xp, lots,
                           'sell' if inv > 0 else 'buy').total
                    cash += pnl
                    all_trades.append({'pnl': pnl, 'reason': reason,
                                       'hold': ist_sec - et, 'hour': hour,
                                       'date': row['date']})
                    inv = 0; ps = ''

            # Entry
            if inv == 0:
                if score > SCORE_THRESHOLD:
                    ep = ask_p1; et = ist_sec; inv = 1; ps = 'LONG'
                    cash -= costs.compute(ep, lots, 'buy').total
                elif score < -SCORE_THRESHOLD:
                    ep = bid_p1; et = ist_sec; inv = -1; ps = 'SHORT'
                    cash -= costs.compute(ep, lots, 'sell').total

        # End of day force close
        if inv != 0:
            final_price = float(df_day.iloc[-1]['close'])
            xp   = float(df_day.iloc[-1].get('bid_p1', final_price)) \
                   if inv > 0 else \
                   float(df_day.iloc[-1].get('ask_p1', final_price))
            pnl  = (xp - ep) * inv * lots * cfg.lot_size
            pnl -= costs.compute(xp, lots,
                   'sell' if inv > 0 else 'buy').total
            cash += pnl
            all_trades.append({'pnl': pnl, 'reason': 'EOD_CLOSE',
                               'hold': int(df_day.iloc[-1]['ist_sec']) - et,
                               'hour': int(df_day.iloc[-1]['hour']),
                               'date': df_day.iloc[-1]['date']})
            inv = 0; ps = ''

        total_cash += cash

    pnls = [t['pnl'] for t in all_trades]
    wins = [p for p in pnls if p > 0]

    # Per day breakdown
    by_date = {}
    for t in all_trades:
        d = t['date']
        if d not in by_date:
            by_date[d] = []
        by_date[d].append(t['pnl'])

    result = {
        'symbol':       cfg.symbol,
        'dates':        dates,
        'lots':         lots,
        'total_trades': len(all_trades),
        'win_rate':     round(len(wins) / max(len(pnls), 1), 4),
        'total_pnl':    round(total_cash, 2),
        'avg_pnl':      round(np.mean(pnls) if pnls else 0, 2),
        'max_win':      round(max(pnls) if pnls else 0, 2),
        'max_loss':     round(min(pnls) if pnls else 0, 2),
        'avg_hold':     round(np.mean([t['hold'] for t in all_trades
                              if t['hold'] > 0]) if all_trades else 0, 1),
        'trades':       all_trades,
        'by_date':      {d: {'trades': len(v), 'pnl': round(sum(v), 2)}
                         for d, v in by_date.items()},
    }

    if verbose:
        print(f'\n=== USDINR Backtest ===')
        print(f'Dates:        {dates}')
        print(f'Lots:         {lots}')
        print(f'Threshold:    {SCORE_THRESHOLD} (looser)')
        print(f'SL/TP ticks:  {SL_TICKS}/{TP_TICKS}')
        print(f'Max hold:     {MAX_HOLD}s')
        print(f'Trades:       {result["total_trades"]}')
        print(f'Win rate:     {result["win_rate"]*100:.1f}%')
        print(f'Total PnL:    Rs.{result["total_pnl"]:,.0f}')
        print(f'Avg PnL:      Rs.{result["avg_pnl"]:,.0f}')
        print(f'Max win:      Rs.{result["max_win"]:,.0f}')
        print(f'Max loss:     Rs.{result["max_loss"]:,.0f}')
        print(f'Avg hold:     {result["avg_hold"]:.0f}s')
        print(f'\nPer day:')
        for d, v in sorted(result['by_date'].items()):
            print(f'  {d}: {v["trades"]} trades  Rs.{v["pnl"]:,.0f}')

    return result


if __name__ == '__main__':
    result = run(lots=20, verbose=True)
    print(f'\nPer day avg: Rs.{result["total_pnl"]/max(len(result["dates"]),1):,.0f}')