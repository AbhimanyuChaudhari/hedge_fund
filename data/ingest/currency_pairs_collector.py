"""
Multi-currency pairs collector
Adds EURINR, GBPINR, JPYINR alongside USDINR
Runs separately — does NOT affect main pipeline
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from data.ingest.zerodha_client import ZerodhaClient
from data.store.s3_client import S3Client

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
IST = ZoneInfo('Asia/Kolkata')

# Currency pairs to collect
PAIRS = ['EURINR', 'GBPINR', 'JPYINR']


class CurrencyPairsManager:

    def __init__(self):
        self.zc = ZerodhaClient()
        log.info('CurrencyPairsManager initialized')

    def get_front_month_tokens(self):
        """Get front month token for each pair"""
        insts  = self.zc.kite.instruments('CDS')
        df     = pd.DataFrame(insts)
        fut    = df[df['instrument_type']=='FUT'].copy()
        fut['expiry'] = pd.to_datetime(fut['expiry'])
        today  = date.today()
        active = fut[fut['expiry'].dt.date >= today]

        tokens = {}
        for pair in PAIRS:
            pair_fut = active[active['name']==pair].copy()
            if pair_fut.empty:
                log.warning(f'{pair}: no active futures found')
                continue
            front = pair_fut.sort_values('expiry').iloc[0]
            tokens[pair] = {
                'token':  int(front['instrument_token']),
                'symbol': front['tradingsymbol'],
                'expiry': front['expiry'].date().isoformat(),
            }
            log.info(f'{pair}: {front["tradingsymbol"]} '
                     f'token={front["instrument_token"]}')
        return tokens


def update_instrument_manager_with_pairs():
    """
    Update instrument manager to also subscribe to
    EURINR, GBPINR, JPYINR WebSocket feeds
    
    Call this ONCE to add pairs to collection
    The main tick_collector.py handles the rest
    """
    mgr    = CurrencyPairsManager()
    tokens = mgr.get_front_month_tokens()

    print('=== Currency Pairs to Add ===')
    for pair, info in tokens.items():
        print(f'{pair}: {info["symbol"]}  '
              f'token={info["token"]}  '
              f'expiry={info["expiry"]}')

    return tokens


if __name__ == '__main__':
    tokens = update_instrument_manager_with_pairs()