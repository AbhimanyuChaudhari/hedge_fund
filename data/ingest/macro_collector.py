# data/ingest/macro_collector.py
"""
Macro data collector — runs separately from main pipeline
Fetches: India VIX, DXY, Spot rate, RBI dates
Saves snapshot every 15 minutes to S3
Does NOT affect tick_collector.py
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import requests
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from data.store.s3_client import S3Client

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)
IST = ZoneInfo('Asia/Kolkata')

# RBI MPC meeting dates 2026
RBI_MPC_DATES = [
    date(2026, 2, 7),  date(2026, 4, 9),
    date(2026, 6, 6),  date(2026, 8, 8),
    date(2026, 10, 8), date(2026, 12, 5),
]

# US FOMC dates 2026
FOMC_DATES = [
    date(2026, 1, 29), date(2026, 3, 19),
    date(2026, 5, 7),  date(2026, 6, 18),
    date(2026, 7, 30), date(2026, 9, 17),
    date(2026, 11, 5), date(2026, 12, 16),
]


class MacroCollector:

    def __init__(self):
        self.s3        = S3Client()
        self.snapshots = []
        self.session   = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        log.info('MacroCollector initialized')

    # ── Data fetchers ──────────────────────────────────────────────
    def fetch_india_vix(self) -> dict:
        try:
            nse_session = requests.Session()
            nse_session.headers.update({
                'User-Agent': 'Mozilla/5.0',
                'Accept':     'application/json',
                'Referer':    'https://www.nseindia.com',
            })
            nse_session.get('https://www.nseindia.com', timeout=10)
            resp = nse_session.get(
                'https://www.nseindia.com/api/allIndices',
                timeout=10
            )
            for idx in resp.json().get('data', []):
                if idx.get('index') == 'INDIA VIX':
                    vix = float(idx['last'])
                    return {
                        'vix':          vix,
                        'vix_chg':      float(idx['variation']),
                        'vix_chg_pct':  float(idx['percentChange']),
                        'vix_prev':     float(idx['previousDayVal']),
                        'vix_1w_ago':   float(idx['oneWeekAgoVal']),
                        'vix_1m_ago':   float(idx['oneMonthAgoVal']),
                        'vix_regime':   ('LOW'    if vix < 12 else
                                         'NORMAL' if vix < 18 else
                                         'HIGH'   if vix < 25 else
                                         'EXTREME'),
                    }
        except Exception as e:
            log.error(f'VIX fetch error: {e}')
        return {}

    def fetch_spot_rate(self) -> dict:
        try:
            resp = self.session.get(
                'https://api.exchangerate-api.com/v4/latest/USD',
                timeout=10
            )
            data = resp.json()
            return {
                'spot_usdinr': float(data['rates']['INR']),
                'spot_eurusd': float(data['rates']['EUR']),
                'spot_gbpusd': 1/float(data['rates']['GBP']),
                'spot_date':   data['date'],
            }
        except Exception as e:
            log.error(f'Spot rate error: {e}')
        return {}

    def fetch_dxy(self) -> dict:
        try:
            resp = self.session.get(
                'https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB',
                params={'interval':'1d','range':'5d'},
                timeout=10
            )
            result = resp.json()['chart']['result'][0]
            closes = result['indicators']['quote'][0]['close']
            valid  = [c for c in closes if c is not None]
            if len(valid) >= 2:
                dxy      = valid[-1]
                dxy_prev = valid[-2]
                return {
                    'dxy':         round(dxy, 2),
                    'dxy_prev':    round(dxy_prev, 2),
                    'dxy_chg':     round(dxy - dxy_prev, 3),
                    'dxy_chg_pct': round((dxy-dxy_prev)/dxy_prev*100, 3),
                    'dxy_regime':  ('STRONG' if dxy-dxy_prev > 0.3 else
                                    'WEAK'   if dxy-dxy_prev < -0.3 else
                                    'NEUTRAL'),
                }
        except Exception as e:
            log.error(f'DXY error: {e}')
        return {}

    def build_calendar_features(self) -> dict:
        today = date.today()
        
        days_to_rbi  = min([(d-today).days for d in RBI_MPC_DATES
                             if d >= today], default=999)
        days_to_fomc = min([(d-today).days for d in FOMC_DATES
                             if d >= today], default=999)

        # USDINR expiry (last business day of month)
        next_month   = today.replace(day=28) + timedelta(days=4)
        last_day     = next_month - timedelta(days=next_month.day)
        while last_day.weekday() > 4:
            last_day -= timedelta(days=1)
        days_to_expiry = (last_day - today).days

        return {
            'days_to_rbi':    days_to_rbi,
            'days_to_fomc':   days_to_fomc,
            'days_to_expiry': days_to_expiry,
            'near_rbi':       int(days_to_rbi  <= 2),
            'near_fomc':      int(days_to_fomc <= 2),
            'near_expiry':    int(days_to_expiry <= 5),
            'day_of_week':    today.weekday(),
            'week_of_month':  (today.day - 1) // 7 + 1,
        }

    def fetch_all(self) -> dict:
        """Fetch all macro data in one snapshot"""
        now     = datetime.now(IST)
        ist_sec = now.hour*3600 + now.minute*60 + now.second

        snapshot = {
            'ts_ist':  now.isoformat(),
            'ist_sec': ist_sec,
        }

        # Fetch all sources
        vix  = self.fetch_india_vix()
        spot = self.fetch_spot_rate()
        dxy  = self.fetch_dxy()
        cal  = self.build_calendar_features()

        snapshot.update(vix)
        snapshot.update(spot)
        snapshot.update(dxy)
        snapshot.update(cal)

        # Derived features
        if 'spot_usdinr' in snapshot and 'dxy' in snapshot:
            snapshot['usdinr_dxy_corr_proxy'] = (
                snapshot['spot_usdinr'] * snapshot['dxy'] / 100
            )

        log.info(f'Macro: VIX={snapshot.get("vix","N/A")} '
                 f'DXY={snapshot.get("dxy","N/A")} '
                 f'USDINR={snapshot.get("spot_usdinr","N/A")} '
                 f'days_to_RBI={snapshot.get("days_to_rbi","N/A")}')

        return snapshot

    def save_daily(self, date_str: str):
        if not self.snapshots:
            return
        df     = pd.DataFrame(self.snapshots)
        path   = f'/tmp/macro_{date_str}.parquet'
        os.makedirs('/tmp', exist_ok=True)
        df.to_parquet(path, index=False)
        s3_key = f'processed/macro/{date_str}.parquet'
        self.s3.upload_file(path, s3_key)
        log.info(f'Saved {len(df)} macro snapshots → {s3_key}')
        self.snapshots = []
        os.remove(path)

    def run(self):
        """
        Fetch macro data every 15 minutes during market hours
        Save daily to S3 at end of day
        """
        log.info('MacroCollector started')
        current_date = None

        while True:
            try:
                now     = datetime.now(IST)
                ist_sec = now.hour*3600 + now.minute*60 + now.second
                today   = now.date().isoformat()

                # Market hours
                if 8*3600 <= ist_sec <= 18*3600:
                    if current_date and current_date != today:
                        self.save_daily(current_date)
                    current_date = today

                    snap = self.fetch_all()
                    self.snapshots.append(snap)

                    # Every 15 minutes
                    time.sleep(900)

                elif ist_sec > 18*3600 and self.snapshots:
                    self.save_daily(today)
                    time.sleep(3600)

                else:
                    # Pre-market: fetch once and wait
                    snap = self.fetch_all()
                    self.snapshots.append(snap)
                    time.sleep(900)

            except Exception as e:
                log.error(f'MacroCollector error: {e}')
                time.sleep(60)


if __name__ == '__main__':
    collector = MacroCollector()
    collector.run()