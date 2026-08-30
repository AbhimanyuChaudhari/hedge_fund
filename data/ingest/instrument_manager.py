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

# USDINR monthly contract suffix pattern
USDINR_MONTHLY = ["JANFUT", "FEBFUT", "MARFUT", "APRFUT", "MAYFUT",
                  "JUNFUT", "JULFUT", "AUGFUT", "SEPFUT", "OCTFUT",
                  "NOVFUT", "DECFUT"]


class InstrumentManager:
    def __init__(self):
        self.zerodha = ZerodhaClient()
        self.redis   = RedisClient()

    def get_active_futures(self) -> list[dict]:
        logger.info("Fetching NFO instrument list from Zerodha...")
        instruments = self.zerodha.get_instruments("NFO")

        today   = datetime.today().date()
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
                    'exchange':         'NFO',
                    'instrument_type':  'equity_futures',
                }

        result = list(front_month.values())
        logger.info(f"Found {len(result)} active front-month equity futures")
        return result

    def get_active_usdinr(self) -> list[dict]:
        """
        Get front month USDINR futures from CDS exchange.
        Uses monthly contracts only (not weekly) for liquidity.
        """
        logger.info("Fetching CDS instrument list for USDINR...")
        instruments = self.zerodha.get_instruments("CDS")

        today   = datetime.today().date()
        futures = []

        for inst in instruments:
            if inst['instrument_type'] != 'FUT':
                continue
            if inst['name'] != 'USDINR':
                continue
            # Only monthly contracts (not weekly)
            symbol = inst['tradingsymbol']
            if not any(symbol.endswith(m) for m in USDINR_MONTHLY):
                continue
            expiry = inst['expiry']
            if isinstance(expiry, str):
                expiry = datetime.strptime(expiry, '%Y-%m-%d').date()
            if expiry < today:
                continue
            futures.append(inst)

        if not futures:
            logger.warning("No USDINR futures found")
            return []

        # Front month only
        front = min(futures, key=lambda x: x['expiry'])
        expiry = front['expiry']
        if isinstance(expiry, str):
            expiry = datetime.strptime(expiry, '%Y-%m-%d').date()

        result = [{
            'symbol':           front['tradingsymbol'],
            'name':             'USDINR',
            'instrument_token': front['instrument_token'],
            'expiry':           front['expiry'].strftime('%Y-%m-%d') if hasattr(front['expiry'], 'strftime') else str(front['expiry']),
            'expiry_date':      expiry,
            'lot_size':         1000,
            'exchange':         'CDS',
            'instrument_type':  'currency_futures',
        }]

        logger.info(f"USDINR front month: {result[0]['symbol']} (token: {result[0]['instrument_token']})")
        return result

    def get_all_instruments(self) -> list[dict]:
        """Get equity futures + USDINR combined."""
        equity  = self.get_active_futures()
        usdinr  = self.get_active_usdinr()
        all_instruments = equity + usdinr
        logger.info(f"Total instruments: {len(all_instruments)} ({len(equity)} equity + {len(usdinr)} currency)")

        # Cache in Redis
        self.redis.client.setex(
            'active_futures',
            86400,
            json.dumps([{k: v for k, v in r.items() if k != 'expiry_date'} for r in all_instruments])
        )
        return all_instruments

    def get_cached_futures(self) -> list[dict]:
        data = self.redis.client.get('active_futures')
        if data:
            return json.loads(data)
        return self.get_all_instruments()

    def get_tokens_and_symbols(self) -> tuple[list[int], dict[int, str]]:
        instruments = self.get_cached_futures()
        tokens      = [f['instrument_token'] for f in instruments]
        token_map   = {f['instrument_token']: f['symbol'] for f in instruments}
        return tokens, token_map

    def get_instrument_type(self, token: int) -> str:
        """Returns 'currency_futures' or 'equity_futures' for a token."""
        instruments = self.get_cached_futures()
        for inst in instruments:
            if inst['instrument_token'] == token:
                return inst.get('instrument_type', 'equity_futures')
        return 'equity_futures'