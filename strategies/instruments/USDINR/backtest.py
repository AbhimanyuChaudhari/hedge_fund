"""
USDINR instrument-specific backtest.
Uses calibrated config and validated signal.
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

    # Get dates
    files = s.list_files(f'live/candles/1second/{cfg.symbol}/')
    dates = sorted([f.split('/')[-1].replace('.parquet','') for f in files])
    if date:
        dates = [d for d in dates if d == date]
    if not dates:
        return {'error': 'no data'}

    db      = DuckDBClient()
    all_df  = []
    for d in dates:
        df = db.read_parquet(f'live/candles/1second/{cfg.symbol}/{d}.parquet')
        df['ist_sec'] = (df['ts_sec'] + 19800) % 86400
        df = df[(df['ist_sec'] >= cfg.market_open_ist) &
                (df['ist_sec'] <= cfg.market_close_ist)].reset_index(drop=True)
        df['date'] = d
        all_df.append(df)
    db.close()

    df = pd.concat(all_df, ignore_index=True)
    if df.empty:
        return {'error': 'no data after filter'}

    TICK = cfg.tick_size
    W    = cfg.signal_window
    NW   = cfg.norm_window

    # Precompute signals
    df['bid_chg']    = df['total_bid_qty'].diff()
    df['ask_chg']    = df['total_ask_qty'].diff()
    pu               = df['close'].diff().abs() < TICK * 0.5
    df['cancel_imb'] = 0.0
    df.loc[pu & (df['bid_chg'] < 0), 'cancel_imb'] -= df['bid_chg'].abs()
    df.loc[pu & (df['ask_chg'] < 0), 'cancel_imb'] += df['ask_chg'].abs()

    df['s_cancel'] = -df['cancel_imb'].rolling(W).mean()
    df['s_imb']    =  df['imbalance_last'].rolling(W).mean()
    df['mid']      = (df['bid_p1'] + df['ask_p1']) / 2
    df['s_wmid']   = (df['weighted_mid'] - df['mid']).rolling(10).mean()

    for col in ['s_cancel', 's_imb', 's_wmid']:
        rs     = df[col].rolling(NW).std() + 1e-9
        df[col] = (df[col] / rs).clip(-1, 1)

    df['score'] = (cfg.w_cancel    * df['s_cancel'] +
                   cfg.w_imbalance * df['s_imb']    +
                   cfg.w_wmid      * df['s_wmid'])
    df['hour']  = df['ist_sec'] // 3600
    df = df.dropna().reset_index(drop=True)

    # Backtest
    inv   = 0; cash = 0.0; trades = []
    ep    = 0.0; et = 0; ps = ''

    for i, row in df.iterrows():
        price   = float(row['close'])
        bid_p1  = float(row.get('bid_p1', price - TICK))
        ask_p1  = float(row.get('ask_p1', price + TICK))
        ist_sec = int(row['ist_sec'])
        hour    = int(row['hour'])
        score   = float(row['score'])

        # Skip avoid hours
        if hour in cfg.avoid_hours:
            continue

        # Force close
        if ist_sec >= cfg.market_close_ist - 120 and inv != 0:
            xp   = bid_p1 if ps == 'LONG' else ask_p1
            pnl  = (xp - ep) * inv * lots * cfg.lot_size
            pnl -= costs.compute(xp, lots, 'sell' if inv > 0 else 'buy').total
            cash += pnl
            trades.append({'pnl': pnl, 'reason': 'FORCE_CLOSE',
                          'hold': ist_sec - et, 'hour': hour})
            inv = 0; ps = ''
            continue

        # Exit
        if inv != 0:
            ticks  = (price - ep) / TICK
            reason = None
            if ps == 'LONG':
                if ticks < -cfg.stop_loss_ticks:    reason = 'SL'
                elif ticks > cfg.take_profit_ticks: reason = 'TP'
            else:
                if ticks >  cfg.stop_loss_ticks:    reason = 'SL'
                elif ticks < -cfg.take_profit_ticks: reason = 'TP'
            if ist_sec - et > cfg.max_hold_seconds: reason = 'TIME'
            if reason:
                xp   = bid_p1 if ps == 'LONG' else ask_p1
                pnl  = (xp - ep) * inv * lots * cfg.lot_size
                pnl -= costs.compute(xp, lots,
                       'sell' if inv > 0 else 'buy').total
                cash += pnl
                trades.append({'pnl': pnl, 'reason': reason,
                              'hold': ist_sec - et, 'hour': hour})
                inv = 0; ps = ''

        # Entry
        if inv == 0:
            if score > cfg.score_threshold:
                ep = ask_p1; et = ist_sec; inv = 1; ps = 'LONG'
                cash -= costs.compute(ep, lots, 'buy').total
            elif score < -cfg.score_threshold:
                ep = bid_p1; et = ist_sec; inv = -1; ps = 'SHORT'
                cash -= costs.compute(ep, lots, 'sell').total

    final = float(df.iloc[-1]['close'])
    mtm   = cash + inv * (final - ep) * lots * cfg.lot_size
    pnls  = [t['pnl'] for t in trades]
    wins  = [p for p in pnls if p > 0]

    result = {
        'symbol':       cfg.symbol,
        'dates':        dates,
        'lots':         lots,
        'total_trades': len(trades),
        'win_rate':     round(len(wins) / max(len(pnls), 1), 4),
        'total_pnl':    round(mtm, 2),
        'avg_pnl':      round(np.mean(pnls) if pnls else 0, 2),
        'max_win':      round(max(pnls) if pnls else 0, 2),
        'max_loss':     round(min(pnls) if pnls else 0, 2),
        'avg_hold':     round(np.mean([t['hold'] for t in trades])
                              if trades else 0, 1),
        'trades':       trades,
    }

    if verbose:
        print(f'\n=== USDINR Backtest ===')
        print(f'Dates:        {dates}')
        print(f'Lots:         {lots}')
        print(f'Trades:       {result["total_trades"]}')
        print(f'Win rate:     {result["win_rate"]*100:.1f}%')
        print(f'Total PnL:    Rs.{result["total_pnl"]:,.0f}')
        print(f'Avg PnL:      Rs.{result["avg_pnl"]:,.0f}')
        print(f'Max win:      Rs.{result["max_win"]:,.0f}')
        print(f'Max loss:     Rs.{result["max_loss"]:,.0f}')
        print(f'Avg hold:     {result["avg_hold"]:.0f}s')

    return result


if __name__ == '__main__':
    result = run(lots=20, verbose=True)
    print(f'\nPer day avg: Rs.{result["total_pnl"]/max(len(result["dates"]),1):,.0f}')