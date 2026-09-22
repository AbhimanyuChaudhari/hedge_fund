"""
Live tick collector — in-memory buffer → S3 raw parquet every 60s.
No Redis. No candle builder. Direct S3 write.
If WebSocket dies → buffer clears → no stale data.
"""
import io
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import threading
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

from config.settings import settings
from data.store.s3_client import S3Client
from data.ingest.instrument_manager import InstrumentManager

# ── Config ─────────────────────────────────────────────────────────
FLUSH_INTERVAL = 60  # seconds

# ── Global state ───────────────────────────────────────────────────
buffer          = []
buffer_lock     = threading.Lock()
TOKENS          = []
TOKEN_TO_SYMBOL = {}
s3              = S3Client()

# ── Market hours (IST seconds from midnight) ───────────────────────
MARKET_SESSIONS = {
    'equity':   (9*3600 + 15*60, 15*3600 + 30*60),
    'currency': (9*3600,          17*3600),
}
CURRENCY_SYMBOLS = {'USDINR', 'EURINR', 'GBPINR', 'JPYINR'}


def is_market_hours(symbol: str) -> bool:
    now_ist = (int(time.time()) + 19800) % 86400
    if any(c in symbol for c in CURRENCY_SYMBOLS):
        lo, hi = MARKET_SESSIONS['currency']
    else:
        lo, hi = MARKET_SESSIONS['equity']
    return lo <= now_ist <= hi


# ── Tick parser ────────────────────────────────────────────────────
def parse_tick(tick: dict) -> dict:
    depth = tick.get('depth', {})
    bids  = depth.get('buy',  [])
    asks  = depth.get('sell', [])

    while len(bids) < 5:
        bids.append({'price': 0, 'quantity': 0, 'orders': 0})
    while len(asks) < 5:
        asks.append({'price': 0, 'quantity': 0, 'orders': 0})

    best_bid      = bids[0].get('price',    0)
    best_ask      = asks[0].get('price',    0)
    bid_q1        = bids[0].get('quantity', 0)
    ask_q1        = asks[0].get('quantity', 0)
    mid           = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
    spread        = best_ask - best_bid if best_bid and best_ask else 0
    total_bid_qty = tick.get('total_buy_quantity',  0)
    total_ask_qty = tick.get('total_sell_quantity', 0)
    total_qty     = total_bid_qty + total_ask_qty
    imbalance     = (total_bid_qty - total_ask_qty) / total_qty if total_qty else 0
    weighted_mid  = (
        (best_bid * ask_q1 + best_ask * bid_q1) /
        (bid_q1 + ask_q1)
    ) if (bid_q1 + ask_q1) > 0 else mid

    return {
        'ts_local_ns':      time.time_ns(),
        'ts_exchange':      str(tick.get('exchange_timestamp', '')),
        'symbol':           TOKEN_TO_SYMBOL.get(tick['instrument_token'], ''),
        'instrument_token': tick['instrument_token'],
        'last_price':       tick.get('last_price',            0),
        'last_qty':         tick.get('last_traded_quantity',  0),
        'avg_price':        tick.get('average_traded_price',  0),
        'volume':           tick.get('volume_traded',         0),
        'open':             tick.get('ohlc', {}).get('open',  0),
        'high':             tick.get('ohlc', {}).get('high',  0),
        'low':              tick.get('ohlc', {}).get('low',   0),
        'close':            tick.get('ohlc', {}).get('close', 0),
        'oi':               tick.get('oi', 0),
        'total_bid_qty':    total_bid_qty,
        'total_ask_qty':    total_ask_qty,
        'mid_price':        round(mid,          4),
        'weighted_mid':     round(weighted_mid, 4),
        'spread':           round(spread,        4),
        'book_imbalance':   round(imbalance,     6),
        'bid_p1': bids[0].get('price', 0), 'bid_q1': bids[0].get('quantity', 0),
        'bid_p2': bids[1].get('price', 0), 'bid_q2': bids[1].get('quantity', 0),
        'bid_p3': bids[2].get('price', 0), 'bid_q3': bids[2].get('quantity', 0),
        'bid_p4': bids[3].get('price', 0), 'bid_q4': bids[3].get('quantity', 0),
        'bid_p5': bids[4].get('price', 0), 'bid_q5': bids[4].get('quantity', 0),
        'ask_p1': asks[0].get('price', 0), 'ask_q1': asks[0].get('quantity', 0),
        'ask_p2': asks[1].get('price', 0), 'ask_q2': asks[1].get('quantity', 0),
        'ask_p3': asks[2].get('price', 0), 'ask_q3': asks[2].get('quantity', 0),
        'ask_p4': asks[3].get('price', 0), 'ask_q4': asks[3].get('quantity', 0),
        'ask_p5': asks[4].get('price', 0), 'ask_q5': asks[4].get('quantity', 0),
    }


# ── S3 flush ───────────────────────────────────────────────────────
def flush_buffer():
    global buffer

    with buffer_lock:
        if not buffer:
            return
        rows   = buffer.copy()
        buffer = []

    df            = pd.DataFrame(rows)
    now           = datetime.now(timezone.utc)
    date_str      = now.strftime('%Y-%m-%d')
    timestamp_str = now.strftime('%H-%M-%S')

    flushed = 0
    for symbol, group in df.groupby('symbol'):
        if not is_market_hours(symbol):
            continue
        key = f'raw/orderbook/{symbol}/{date_str}/{timestamp_str}.parquet'
        try:
            buf = io.BytesIO()
            group.reset_index(drop=True).to_parquet(
                buf, index=False, compression='zstd'
            )
            buf.seek(0)
            s3.client.put_object(
                Bucket = s3.bucket,
                Key    = key,
                Body   = buf.getvalue(),
            )
            flushed += 1
        except Exception as e:
            print(f'[S3 ERROR] {symbol}: {e}')

    if flushed > 0:
        print(f'[{timestamp_str}] Flushed {flushed} symbols → S3 raw')


def flush_loop():
    while True:
        time.sleep(FLUSH_INTERVAL)
        flush_buffer()


# ── WebSocket handlers ─────────────────────────────────────────────
def on_ticks(ws, ticks):
    with buffer_lock:
        for tick in ticks:
            token = tick.get('instrument_token')
            if token in TOKENS:
                buffer.append(parse_tick(tick))


def on_connect(ws, response):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] WebSocket connected')
    ws.subscribe(TOKENS)
    ws.set_mode(ws.MODE_FULL, TOKENS)
    print(f'Subscribed: {len(TOKENS)} instruments')


def on_error(ws, code, reason):
    print(f'[WS ERROR] {code}: {reason}')


def on_close(ws, code, reason):
    print(f'[WS CLOSE] {code}: {reason}')
    flush_buffer()


def on_reconnect(ws, attempts):
    # KiteTicker's built-in reconnect handles resubscription automatically
    # via on_connect being called again on the same ws object.
    # DO NOT re-fetch instruments or re-init TOKENS here.
    print(f'[WS RECONNECT] Attempt {attempts}')


def on_noreconnect(ws):
    print('[WS NORECONNECT] Max attempts reached — flushing and exiting')
    flush_buffer()


# ── Entry point ────────────────────────────────────────────────────
def run_collector():
    global TOKENS, TOKEN_TO_SYMBOL

    from kiteconnect import KiteTicker

    # Fetch instruments ONCE at startup
    mgr = InstrumentManager()
    mgr.get_all_instruments()
    tokens, token_map = mgr.get_tokens_and_symbols()

    TOKEN_TO_SYMBOL = token_map
    TOKENS          = list(TOKEN_TO_SYMBOL.keys())

    print(f'=== Collector starting ===')
    print(f'Instruments: {len(TOKENS)}')
    print(f'Flush:       every {FLUSH_INTERVAL}s → S3 raw/orderbook/')
    print(f'No Redis. Market hours filter on flush.\n')

    threading.Thread(target=flush_loop, daemon=True).start()

    kws = KiteTicker(
        settings.zerodha_api_key,
        settings.zerodha_access_token
    )
    kws.on_ticks       = on_ticks
    kws.on_connect     = on_connect
    kws.on_error       = on_error
    kws.on_close       = on_close
    kws.on_reconnect   = on_reconnect
    kws.on_noreconnect = on_noreconnect

    # KiteTicker has built-in auto-reconnect — it will call on_connect
    # again automatically after a drop, using the same TOKENS list.
    print('Starting WebSocket (auto-reconnect enabled)...')
    kws.connect(threaded=False)


if __name__ == '__main__':
    run_collector()