import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kiteconnect import KiteConnect
from config.settings import settings
from datetime import datetime

class ZerodhaClient:
    def __init__(self):
        self.kite = KiteConnect(api_key=settings.zerodha_api_key)
        self.kite.set_access_token(settings.zerodha_access_token)

    def get_historical_data(self, instrument_token: int, from_date: datetime, to_date: datetime, interval: str) -> list:
        return self.kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            continuous=False
        )

    def get_instruments(self, exchange: str = "NFO") -> list:
        return self.kite.instruments(exchange)

    def get_quote(self, instruments: list) -> dict:
        return self.kite.quote(instruments)