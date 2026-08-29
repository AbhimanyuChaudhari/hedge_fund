import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
from datetime import datetime, timedelta
from data.ingest.zerodha_client import ZerodhaClient
from data.store.s3_client import S3Client

class HistoricalDataDownloader:
    def __init__(self):
        self.zerodha = ZerodhaClient()
        self.s3 = S3Client()

    def download_and_store(self, instrument_token: int, symbol: str, from_date: datetime, to_date: datetime, interval: str = "minute"):
        print(f"Downloading {symbol} from {from_date.date()} to {to_date.date()}...")
        
        records = self.zerodha.get_historical_data(instrument_token, from_date, to_date, interval)
        
        if not records:
            print(f"No data returned for {symbol}")
            return

        df = pd.DataFrame(records)
        df['symbol'] = symbol
        df['instrument_token'] = instrument_token

        s3_key = f"raw/historical/{interval}/{symbol}/{from_date.strftime('%Y-%m-%d')}_{to_date.strftime('%Y-%m-%d')}.parquet"

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, tmp_path)
            self.s3.upload(tmp_path, s3_key)
            print(f"Stored {len(df)} candles to s3://{self.s3.bucket}/{s3_key}")
        finally:
            os.unlink(tmp_path)

    def get_nifty_token(self) -> int:
        instruments = self.zerodha.get_instruments("NSE")
        for inst in instruments:
            if inst['tradingsymbol'] == 'NIFTY 50' and inst['instrument_type'] == 'EQ':
                return inst['instrument_token']
        raise ValueError("NIFTY 50 token not found")