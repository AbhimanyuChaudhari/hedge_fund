import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import numpy as np
import pandas as pd
from data.store.duckdb_client import DuckDBClient
from execution.risk.transaction_costs import TransactionCosts
from strategies.implementations.imbalance_momentum.parameters import ImbalanceMomentumParameters


def run_backtest(symbol: str, date: str,
                 params: ImbalanceMomentumParameters = None,
                 lots: int = 20,
                 verbose: bool = True) -> dict:

    params = params or ImbalanceMomentumParameters()
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

    # State
    inventory      = 0
    cash           = 0.0
    trades         = []
    entry_price    = 0.0
    entry_time     = 0
    position_side  = ''
    imb_history:   list[float] = []
    CONF           = params.confirmation_bars

    for i, row in df.iterrows():
        price   = float(row['close'])
        bid_p1  = float(row.get('bid_p1', price - params.tick_size))
        ask_p1  = float(row.get('ask_p1', price + params.tick_size))
        bid_q1  = float(row.get('bid_q1', 0))
        ask_q1  = float(row.get('ask_q1', 0))
        tot_bid = float(row.get('total_bid_qty', 0))
        tot_ask = float(row.get('total_ask_qty', 0))
        ist_sec = int(row['ist_sec'])
        spread  = ask_p1 - bid_p1

        imbalance = (tot_bid - tot_ask) / (tot_bid + tot_ask + 1e-9)
        imb_history.append(imbalance)
        if len(imb_history) > 20:
            imb_history.pop(0)

        # Force close at end of session
        if ist_sec >= CLOSE_SEC - 60 and inventory != 0:
            exit_price = price
            pnl = (exit_price - entry_price) * inventory * lots * params.lot_size
            pnl -= costs.compute(exit_price, lots, 'sell').total
            cash += pnl
            trades.append({
                'entry': entry_price, 'exit': exit_price,
                'side': position_side, 'pnl': pnl,
                'reason': 'FORCE_CLOSE', 'hold_secs': ist_sec - entry_time
            })
            inventory = 0
            position_side = ''
            continue

        # Filters
        if spread < params.min_spread:
            continue
        if bid_q1 < params.min_bid_qty or ask_q1 < params.min_ask_qty:
            continue

        # Exit logic
        if inventory != 0:
            ticks = (price - entry_price) / params.tick_size
            reason = None

            if position_side == 'LONG':
                if ticks < -params.stop_loss_ticks:   reason = 'STOP_LOSS'
                elif ticks > params.take_profit_ticks: reason = 'TAKE_PROFIT'
            elif position_side == 'SHORT':
                if ticks > params.stop_loss_ticks:    reason = 'STOP_LOSS'
                elif ticks < -params.take_profit_ticks: reason = 'TAKE_PROFIT'

            if ist_sec - entry_time > params.max_hold_seconds:
                reason = 'TIME_STOP'

            # if len(imb_history) >= CONF and reason is None:
            #     recent = imb_history[-CONF:]
            #     if position_side == 'LONG'  and all(i < 0 for i in recent):
            #         reason = 'SIGNAL_REVERSAL'
            #     if position_side == 'SHORT' and all(i > 0 for i in recent):
            #         reason = 'SIGNAL_REVERSAL'

            if reason:
                exit_price = bid_p1 if position_side == 'LONG' else ask_p1
                pnl = (exit_price - entry_price) * inventory * lots * params.lot_size
                pnl -= costs.compute(exit_price, lots,
                                     'sell' if position_side == 'LONG' else 'buy').total
                cash += pnl
                trades.append({
                    'entry': entry_price, 'exit': exit_price,
                    'side': position_side, 'pnl': pnl,
                    'reason': reason, 'hold_secs': ist_sec - entry_time
                })
                inventory = 0
                position_side = ''

        # Entry logic
        if inventory == 0 and len(imb_history) >= CONF:
            recent = imb_history[-CONF:]

            if all(i > params.imbalance_threshold for i in recent):
                entry_price   = ask_p1
                entry_time    = ist_sec
                inventory     = 1
                position_side = 'LONG'
                cash -= costs.compute(entry_price, lots, 'buy').total

            elif all(i < -params.imbalance_threshold for i in recent):
                entry_price   = bid_p1
                entry_time    = ist_sec
                inventory     = -1
                position_side = 'SHORT'
                cash -= costs.compute(entry_price, lots, 'sell').total

    # Final MTM
    final_price = float(df.iloc[-1]['close'])
    mtm = cash + inventory * (final_price - entry_price) * lots * params.lot_size

    trade_pnls = [t['pnl'] for t in trades]
    wins       = [p for p in trade_pnls if p > 0]

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
        'avg_hold':     round(np.mean([t['hold_secs'] for t in trades]) if trades else 0, 1),
        'trades':       trades,
    }

    if verbose:
        print(f'\n=== Imbalance Momentum Backtest ===')
        print(f'Symbol:       {symbol}')
        print(f'Date:         {date}')
        print(f'Lots:         {lots}')
        print(f'Trades:       {result["total_trades"]}')
        print(f'Win rate:     {result["win_rate"]*100:.1f}%')
        print(f'Total PnL:    Rs.{result["total_pnl"]:,.0f}')
        print(f'Avg PnL:      Rs.{result["avg_pnl"]:,.0f}')
        print(f'Max win:      Rs.{result["max_win"]:,.0f}')
        print(f'Max loss:     Rs.{result["max_loss"]:,.0f}')
        print(f'Avg hold:     {result["avg_hold"]:.0f} seconds')
        if trades:
            print(f'\nFirst 5 trades:')
            for t in trades[:5]:
                print(f"  {t['side']:5s} entry={t['entry']:.4f} "
                      f"exit={t['exit']:.4f} "
                      f"pnl=Rs.{t['pnl']:,.0f} "
                      f"reason={t['reason']} "
                      f"hold={t['hold_secs']}s")
    return result


if __name__ == '__main__':
    from data.store.s3_client import S3Client
    s     = S3Client()
    files = s.list_files('live/candles/1second/USDINR26SEPFUT/')
    dates = sorted([f.split('/')[-1].replace('.parquet', '') for f in files])

    params = ImbalanceMomentumParameters(
        imbalance_threshold = 0.30,
        confirmation_bars   = 3,
        stop_loss_ticks     = 10,
        take_profit_ticks   = 20,
        max_hold_seconds    = 300,
        min_bid_qty         = 500,
        min_ask_qty         = 500,
    )

    total_pnl = 0
    for date in dates:
        result     = run_backtest('USDINR26SEPFUT', date, params, lots=20)
        total_pnl += result['total_pnl']

    print(f'\n=== TOTAL across {len(dates)} days ===')
    print(f'Total PnL: Rs.{total_pnl:,.0f}')
    print(f'Avg/day:   Rs.{total_pnl/max(len(dates),1):,.0f}')