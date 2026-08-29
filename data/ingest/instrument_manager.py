import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import logging
from datetime import datetime
from data.ingest.zerodha_client import ZerodhaClient
from data.store.redis_client import RedisClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOP_50_FUTURES = [
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
    "SUNPHARMA", "WIPRO", "ULTRACEMCO", "BAJFINANCE", "HCLTECH",
    "NESTLEIND", "POWERGRID", "NTPC", "ONGC", "COALINDIA",
    "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "HINDALCO", "ADANIENT",
    "ADANIPORTS", "BAJAJFINSV", "DIVISLAB", "DRREDDY", "EICHERMOT",
    "GRASIM", "HEROMOTOCO", "INDUSINDBK", "M&M", "BAJAJ-AUTO",
    "BRITANNIA", "CIPLA", "TECHM", "APOLLOHOSP", "BPCL", "SHRIRAMFIN"
]

class InstrumentManager:
    def __init__(self):
        self.zerodha = ZerodhaClient()
        self.redis   = RedisClient()

    def get_active_futures(self) -> list[dict]:
        logger.info("Fetching NFO instrument list from Zerodha...")
        instruments = self.zerodha.get_instruments("NFO")

        today = datetime.today().date()
        futures = []

        for inst in instruments:
            if inst['instrument_type'] != 'FUT':
                continue
            if inst['name'] not in TOP_50_FUTURES:
                continue
            expiry = inst['expiry']
            if isinstance(expiry, str):
                expiry = datetime.strptime(expiry, '%Y-%m-%d').date()
            if expiry < today:
                continue
            futures.append(inst)

        # Keep only front month (nearest expiry) per symbol
        front_month = {}
        for inst in futures:
            name   = inst['name']
            expiry = inst['expiry']
            if isinstance(expiry, str):
                expiry = datetime.strptime(expiry, '%Y-%m-%d').date()
            if name not in front_month or expiry < front_month[name]['expiry_date']:
                front_month[name] = {
                    'symbol':           inst['tradingsymbol'],
                    'name':             inst['name'],
                    'instrument_token': inst['instrument_token'],
                    'expiry':           inst['expiry'].strftime('%Y-%m-%d') if hasattr(inst['expiry'], 'strftime') else str(inst['expiry']),
                    'expiry_date':      expiry,
                    'lot_size':         inst['lot_size'],
                }

        result = list(front_month.values())
        logger.info(f"Found {len(result)} active front-month futures")

        # Cache in Redis
        self.redis.client.setex(
            'active_futures',
            86400,
            json.dumps([{k: v for k, v in r.items() if k != 'expiry_date'} for r in result])
        )
        return result

    def get_cached_futures(self) -> list[dict]:
        data = self.redis.client.get('active_futures')
        if data:
            return json.loads(data)
        return self.get_active_futures()

    def get_tokens_and_symbols(self) -> tuple[list[int], dict[int, str]]:
        futures  = self.get_cached_futures()
        tokens   = [f['instrument_token'] for f in futures]
        token_map = {f['instrument_token']: f['symbol'] for f in futures}
        return tokens, token_map