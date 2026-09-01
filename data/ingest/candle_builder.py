import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import tempfile
import logging
import threading
import time
from datetime import datetime, timezone
from collections import defaultdict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from data.store.redis_client import RedisClient
from data.store.s3_client import S3Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CandleBuilder:
    def __init__(self, token_map: dict, flush_interval: int = 60):
        self.token_map      = token_map  # {token_int: symbol_name}
        self.flush_interval = flush_interval
        self.redis          = RedisClient()
        self.s3             = S3Client()
        self._running       = False
        # In-memory buffer: {symbol: [candle_dict, ...]}
        self._buffer        = defaultdict(list)
        self._buffer_lock   = threading.Lock()

    def _build_1s_candle(self, token: str, symbol: str) -> dict | None:
        """Build one 1-second candle from current Redis stream snapshot."""
        raw = self.redis.get_stream(token, count=50)
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

        return {
            'symbol':            symbol,
            'ts_sec':            int(datetime.now(timezone.utc).timestamp()),
            'ts_ist':            datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'open':              float(df['ltp'].iloc[-1]),
            'high':              float(df['ltp'].max()),
            'low':               float(df['ltp'].min()),
            'close':             float(df['ltp'].iloc[0]),
            'vwap':              float((df['ltp'] * df['volume']).sum() / (df['volume'].sum() + 1e-9)),
            'volume_delta':      float(df['volume'].iloc[0] - df['volume'].iloc[-1]),
            'oi':                float(df['oi'].iloc[0]),
            'tick_count':        len(rows),
            'imbalance_last':    float(imbalance.iloc[0]),
            'imbalance_mean':    float(imbalance.mean()),
            'spread_mean':       float(spread.mean()),
            'weighted_mid':      float((df['bid_p1'].iloc[0] * df['ask_q1'].iloc[0] +
                                       df['ask_p1'].iloc[0] * df['bid_q1'].iloc[0]) /
                                      (df['bid_q1'].iloc[0] + df['ask_q1'].iloc[0] + 1e-9)),
            'total_bid_qty':     float(total_bid.iloc[0]),
            'total_ask_qty':     float(total_ask.iloc[0]),
            'bid_p1':            float(df['bid_p1'].iloc[0]),
            'bid_q1':            float(df['bid_q1'].iloc[0]),
            'bid_p2':            float(df['bid_p2' if 'bid_p2' in df else 'bid_p1'].iloc[0]) if 'bid_p2' in df.columns else 0.0,
            'bid_q2':            float(df['bid_q2'].iloc[0]) if 'bid_q2' in df.columns else 0.0,
            'bid_p3':            float(df['bid_p3'].iloc[0]) if 'bid_p3' in df.columns else 0.0,
            'bid_q3':            float(df['bid_q3'].iloc[0]) if 'bid_q3' in df.columns else 0.0,
            'bid_p4':            float(df['bid_p4'].iloc[0]) if 'bid_p4' in df.columns else 0.0,
            'bid_q4':            float(df['bid_q4'].iloc[0]) if 'bid_q4' in df.columns else 0.0,
            'bid_p5':            float(df['bid_p5'].iloc[0]) if 'bid_p5' in df.columns else 0.0,
            'bid_q5':            float(df['bid_q5'].iloc[0]) if 'bid_q5' in df.columns else 0.0,
            'ask_p1':            float(df['ask_p1'].iloc[0]),
            'ask_q1':            float(df['ask_q1'].iloc[0]),
            'ask_p2':            float(df['ask_p2'].iloc[0]) if 'ask_p2' in df.columns else 0.0,
            'ask_q2':            float(df['ask_q2'].iloc[0]) if 'ask_q2' in df.columns else 0.0,
            'ask_p3':            float(df['ask_p3'].iloc[0]) if 'ask_p3' in df.columns else 0.0,
            'ask_q3':            float(df['ask_q3'].iloc[0]) if 'ask_q3' in df.columns else 0.0,
            'ask_p4':            float(df['ask_p4'].iloc[0]) if 'ask_p4' in df.columns else 0.0,
            'ask_q4':            float(df['ask_q4'].iloc[0]) if 'ask_q4' in df.columns else 0.0,
            'ask_p5':            float(df['ask_p5'].iloc[0]) if 'ask_p5' in df.columns else 0.0,
            'ask_q5':            float(df['ask_q5'].iloc[0]) if 'ask_q5' in df.columns else 0.0,
        }

    def _flush_buffer_to_s3(self):
        """Flush all buffered 1-second candles to S3."""
        with self._buffer_lock:
            buffer_copy = dict(self._buffer)
            self._buffer.clear()

        if not buffer_copy:
            return

        date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        for symbol, candles in buffer_copy.items():
            if not candles:
                continue
            try:
                new_df  = pd.DataFrame(candles)
                s3_key  = f"live/candles/1second/{symbol}/{date_str}.parquet"

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path    = os.path.join(tmpdir, 'new.parquet')
                    merged_path = os.path.join(tmpdir, 'merged.parquet')

                    try:
                        self.s3.download(s3_key, tmp_path)
                        existing_df = pd.read_parquet(tmp_path)
                        merged_df   = pd.concat([existing_df, new_df], ignore_index=True)
                    except Exception:
                        merged_df = new_df

                    table = pa.Table.from_pandas(merged_df)
                    pq.write_table(table, merged_path)
                    self.s3.upload(merged_path, s3_key)

                logger.info(f"Flushed {len(candles)} 1s bars for {symbol} — {len(merged_df)} total today")
            except Exception as e:
                logger.error(f"Flush failed for {symbol}: {e}")

    def _collect_loop(self):
        """Every second — build 1s candle for all instruments and buffer it."""
        while self._running:
            start = time.time()
            for token, symbol in self.token_map.items():
                try:
                    candle = self._build_1s_candle(str(token), symbol)
                    if candle:
                        with self._buffer_lock:
                            self._buffer[symbol].append(candle)
                except Exception as e:
                    logger.error(f"1s candle build failed for {symbol}: {e}")

            elapsed = time.time() - start
            sleep_time = max(0, 1.0 - elapsed)
            time.sleep(sleep_time)

    def _flush_loop(self):
        """Every 60 seconds — flush buffer to S3."""
        while self._running:
            time.sleep(self.flush_interval)
            self._flush_buffer_to_s3()

    def start(self):
        self._running = True
        collect_thread = threading.Thread(target=self._collect_loop, daemon=True)
        flush_thread   = threading.Thread(target=self._flush_loop,   daemon=True)
        collect_thread.start()
        flush_thread.start()
        logger.info(f"Candle builder started — 1s bars, flush every {self.flush_interval}s")

    def stop(self):
        self._running = False
        self._flush_buffer_to_s3()  # final flush
        logger.info("Candle builder stopped")