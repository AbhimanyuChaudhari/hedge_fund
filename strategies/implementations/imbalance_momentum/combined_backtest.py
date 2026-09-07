import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import numpy as np
import pandas as pd
from data.store.duckdb_client import DuckDBClient
from execution.risk.transaction_costs import TransactionCosts
from strategies.implementations.imbalance_momentum.combined_signal import (
    CombinedSignalGenerator, CombinedSignalParameters
)


def run_combined_backtest(symbol: str, date: str,
                          params: CombinedSignalParameters = None,
                          lots: int = 20,
                          verbose: bool = True) -> dict:

    params = params or CombinedSignalParameters()
    gen    = CombinedSignalGenerator(params)
    costs  = TransactionCosts(
        lot_size=params.lot_size,
        instrument_type=params.instrument_type
    )

    db = DuckDBClient()
    df = db.read_parquet(f'live/candles/1second/{symbol}/{date}.parquet')
    db.close()

    df['ist_sec'] = (df['ts_sec'] + 19800) % 86400
    OPEN_SEC  = 9 * 3600
    CLOSE_SEC = 17 * 3600
    df = df[(df['ist_sec'] >= OPEN_SEC) &
            (df['ist_sec'] <= CLOSE_SEC)].reset_index(drop=True)

    if df.empty:
        return {'error': 'no data'}

    WINDOW        = params.imbalance_window + 5
    inventory     = 0
    cash          = 0.0
    trades        = []
    entry_price   = 0.0
    entry_time    = 0
    position_side = ''

    for i in range(WINDOW, len(df)):
        row    = df.iloc[i]
        window = df.iloc[i-WINDOW:i]
        price  = float(row['close'])
        bid_p1 = float(row.get('bid_p1', price - params.tick_size))
        ask_p1 = float(row.get('ask_p1', price + params.tick_size))
        ist_sec = int(row['ist_sec'])

        # Build bars
        bars = []
        for _, r in window.iterrows():
            bars.append({
                'close':          float(r.get('close', 0)),
                'imbalance_last': float(r.get('imbalance_last', 0)),
                'total_bid_qty':  float(r.get('total_bid_qty', 0)),
                'total_ask_qty':  float(r.get('total_ask_qty', 0)),
                'bid_p1':         float(r.get('bid_p1', 0)),
                'ask_p1':         float(r.get('ask_p1', 0)),
                'bid_q1':         float(r.get('bid_q1', 0)),
                'ask_q1':         float(r.get('ask_q1', 0)),
                'bid_q2':         float(r.get('bid_q2', 0)),
                'ask_q2':         float(r.get('ask_q2', 0)),
                'bid_q3':         float(r.get('bid_q3', 0)),
                'ask_q3':         float(r.get('ask_q3', 0)),
                'bid_q4':         float(r.get('bid_q4', 0)),
                'ask_q4':         float(r.get('ask_q4', 0)),
                'bid_q5':         float(r.get('bid_q5', 0)),
                'ask_q5':         float(r.get('ask_q5', 0)),
                'volume_delta':   float(r.get('volume_delta', 0)),
                'weighted_mid':   float(r.get('weighted_mid', 0)),
            })

        # Force close
        if ist_sec >= CLOSE_SEC - 60 and inventory != 0:
            pnl   = (price - entry_price) * inventory * lots * params.lot_size
            pnl  -= costs.compute(price, lots, 'sell' if inventory > 0 else 'buy').total
            cash += pnl
            trades.append({'side': position_side, 'entry': entry_price,
                          'exit': price, 'pnl': pnl, 'reason': 'FORCE_CLOSE',
                          'hold': ist_sec - entry_time})
            inventory = 0; position_side = ''
            continue

        # Exit
        if inventory != 0:
            ticks  = (price - entry_price) / params.tick_size
            reason = None
            if position_side == 'LONG':
                if ticks < -params.stop_loss_ticks:    reason = 'STOP_LOSS'
                elif ticks > params.take_profit_ticks: reason = 'TAKE_PROFIT'
            elif position_side == 'SHORT':
                if ticks > params.stop_loss_ticks:     reason = 'STOP_LOSS'
                elif ticks < -params.take_profit_ticks: reason = 'TAKE_PROFIT'
            if ist_sec - entry_time > params.max_hold_seconds:
                reason = 'TIME_STOP'

            if reason:
                exit_p = bid_p1 if position_side == 'LONG' else ask_p1
                pnl    = (exit_p - entry_price) * inventory * lots * params.lot_size
                pnl   -= costs.compute(exit_p, lots,
                         'sell' if position_side == 'LONG' else 'buy').total
                cash  += pnl
                trades.append({'side': position_side, 'entry': entry_price,
                               'exit': exit_p, 'pnl': pnl, 'reason': reason,
                               'hold': ist_sec - entry_time})
                inventory = 0; position_side = ''

        # Entry
        if inventory == 0:
            score_dict = gen.compute_score(bars)
            action     = gen.action(score_dict, inventory)

            if action == 'LONG':
                entry_price   = ask_p1
                entry_time    = ist_sec
                inventory     = 1
                position_side = 'LONG'
                cash -= costs.compute(entry_price, lots, 'buy').total

            elif action == 'SHORT':
                entry_price   = bid_p1
                entry_time    = ist_sec
                inventory     = -1
                position_side = 'SHORT'
                cash -= costs.compute(entry_price, lots, 'sell').total

    final_price = float(df.iloc[-1]['close'])
    mtm         = cash + inventory * (final_price - entry_price) * lots * params.lot_size
    trade_pnls  = [t['pnl'] for t in trades]
    wins        = [p for p in trade_pnls if p > 0]

    result = {
        'symbol':       symbol,
        'date':         date,
        'lots':         lots,
        'total_trades': len(trades),
        'win_rate':     len(wins) / max(len(trade_pnls), 1),
        'total_pnl':    round(mtm, 2),
        'avg_pnl':      round(np.mean(trade_pnls) if trade_pnls else 0, 2),
        'max_win':      round(max(trade_pnls) if trade_pnls else 0, 2),
        'max_loss':     round(min(trade_pnls) if trade_pnls else 0, 2),
        'avg_hold':     round(np.mean([t['hold'] for t in trades]) if trades else 0, 1),
        'trades':       trades,
    }

    if verbose:
        print(f'\n=== Combined Signal Backtest ===')
        print(f'Symbol:       {symbol}')
        print(f'Date:         {date}')
        print(f'Lots:         {lots}')
        print(f'Trades:       {result["total_trades"]}')
        print(f'Win rate:     {result["win_rate"]*100:.1f}%')
        print(f'Total PnL:    Rs.{result["total_pnl"]:,.0f}')
        print(f'Avg PnL:      Rs.{result["avg_pnl"]:,.0f}')
        print(f'Max win:      Rs.{result["max_win"]:,.0f}')
        print(f'Max loss:     Rs.{result["max_loss"]:,.0f}')
        print(f'Avg hold:     {result["avg_hold"]:.0f}s')
        if trades:
            print(f'\nFirst 5 trades:')
            for t in trades[:5]:
                print(f"  {t['side']:5s} "
                      f"entry={t['entry']:.4f} "
                      f"exit={t['exit']:.4f} "
                      f"pnl=Rs.{t['pnl']:,.0f} "
                      f"reason={t['reason']} "
                      f"hold={t['hold']}s")
    return result


if __name__ == '__main__':
    from data.store.s3_client import S3Client
    s     = S3Client()
    files = s.list_files('live/candles/1second/USDINR26SEPFUT/')
    dates = sorted([f.split('/')[-1].replace('.parquet', '') for f in files])

    params = CombinedSignalParameters(
        score_threshold    = 0.35,
        imbalance_window   = 30,
        momentum_window    = 30,
        stop_loss_ticks    = 8,
        take_profit_ticks  = 16,
        max_hold_seconds   = 60,
        min_bid_qty        = 500,
        min_ask_qty        = 500,
    )

    total_pnl = 0
    for date in dates:
        r          = run_combined_backtest('USDINR26SEPFUT', date, params, lots=20)
        total_pnl += r['total_pnl']

    print(f'\n=== TOTAL across {len(dates)} days ===')
    print(f'Total PnL: Rs.{total_pnl:,.0f}')
    print(f'Avg/day:   Rs.{total_pnl/max(len(dates),1):,.0f}')