import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import tempfile
import logging
import threading
import time
from datetime import datetime, timezone
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from data.store.redis_client import RedisClient
from data.store.s3_client import S3Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CandleBuilder:
    def __init__(self, token_map: dict, interval_seconds: int = 60):
        self.token_map        = token_map  # {token_int: symbol_name}
        self.interval_seconds = interval_seconds
        self.redis            = RedisClient()
        self.s3               = S3Client()
        self._running         = False

    def _build_candle(self, token: str) -> dict | None:
        raw = self.redis.get_stream(token, count=500)
        if not raw:
            return None

        rows = []
        for entry_id, fields in raw:
            rows.append({
                'ltp':           float(fields.get('ltp',           0)),
                'volume':        float(fields.get('volume',        0)),
                'oi':            float(fields.get('oi',            0)),
                'total_bid_qty': float(fields.get('total_bid_qty', 0)),
                'total_ask_qty': float(fields.get('total_ask_qty', 0)),
                'bid_p1':        float(fields.get('bid_p1',        0)),
                'ask_p1':        float(fields.get('ask_p1',        0)),
                'bid_q1':        float(fields.get('bid_q1',        0)),
                'ask_q1':        float(fields.get('ask_q1',        0)),
                'bid_q2':        float(fields.get('bid_q2',        0)),
                'ask_q2':        float(fields.get('ask_q2',        0)),
                'bid_q3':        float(fields.get('bid_q3',        0)),
                'ask_q3':        float(fields.get('ask_q3',        0)),
                'bid_q4':        float(fields.get('bid_q4',        0)),
                'ask_q4':        float(fields.get('ask_q4',        0)),
                'bid_q5':        float(fields.get('bid_q5',        0)),
                'ask_q5':        float(fields.get('ask_q5',        0)),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return None

        total_bid = df['total_bid_qty']
        total_ask = df['total_ask_qty']
        imbalance = (total_bid - total_ask) / (total_bid + total_ask + 1e-9)
        spread    = df['ask_p1'] - df['bid_p1']

        candle = {
            'timestamp':         datetime.now(timezone.utc).isoformat(),
            'open':              df['ltp'].iloc[-1],
            'high':              df['ltp'].max(),
            'low':               df['ltp'].min(),
            'close':             df['ltp'].iloc[0],
            'vwap':              (df['ltp'] * df['volume']).sum() / (df['volume'].sum() + 1e-9),
            'volume_open':       df['volume'].iloc[-1],
            'volume_close':      df['volume'].iloc[0],
            'volume_delta':      df['volume'].iloc[0] - df['volume'].iloc[-1],
            'oi_open':           df['oi'].iloc[-1],
            'oi_close':          df['oi'].iloc[0],
            'oi_delta':          df['oi'].iloc[0] - df['oi'].iloc[-1],
            'avg_imbalance':     imbalance.mean(),
            'max_imbalance':     imbalance.max(),
            'min_imbalance':     imbalance.min(),
            'avg_spread':        spread.mean(),
            'min_spread':        spread.min(),
            'avg_bid_qty_top':   df['bid_q1'].mean(),
            'avg_ask_qty_top':   df['ask_q1'].mean(),
            'avg_total_bid_qty': total_bid.mean(),
            'avg_total_ask_qty': total_ask.mean(),
            'avg_bid_qty_l2':    df['bid_q2'].mean(),
            'avg_ask_qty_l2':    df['ask_q2'].mean(),
            'avg_bid_qty_l3':    df['bid_q3'].mean(),
            'avg_ask_qty_l3':    df['ask_q3'].mean(),
            'avg_bid_qty_l4':    df['bid_q4'].mean(),
            'avg_ask_qty_l4':    df['ask_q4'].mean(),
            'avg_bid_qty_l5':    df['bid_q5'].mean(),
            'avg_ask_qty_l5':    df['ask_q5'].mean(),
        }
        return candle

    def _flush_to_s3(self, symbol: str, candle: dict):
        df       = pd.DataFrame([candle])
        date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        s3_key   = f"live/candles/1minute/{symbol}/{date_str}.parquet"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path    = os.path.join(tmpdir, 'candle.parquet')
            merged_path = os.path.join(tmpdir, 'merged.parquet')

            try:
                self.s3.download(s3_key, tmp_path)
                existing_df = pd.read_parquet(tmp_path)
                merged_df   = pd.concat([existing_df, df], ignore_index=True)
            except Exception:
                merged_df = df

            table = pa.Table.from_pandas(merged_df)
            pq.write_table(table, merged_path)
            self.s3.upload(merged_path, s3_key)
            logger.info(f"Flushed candle for {symbol} — {len(merged_df)} total today")

    def _run_loop(self):
        while self._running:
            time.sleep(self.interval_seconds)
            for token, symbol in self.token_map.items():
                try:
                    candle = self._build_candle(str(token))
                    if candle:
                        candle['symbol'] = symbol
                        self._flush_to_s3(symbol, candle)
                except Exception as e:
                    logger.error(f"Candle build failed for {symbol}: {e}")

    def start(self):
        self._running = True
        thread = threading.Thread(target=self._run_loop, daemon=True)
        thread.start()
        logger.info(f"Candle builder started for {len(self.token_map)} instruments")

    def stop(self):
        self._running = False
        logger.info("Candle builder stopped")