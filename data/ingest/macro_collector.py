# data/ingest/macro_collector.py
"""
Macro data collector — runs separately from main pipeline
Fetches every 15 minutes during market hours:
  - India VIX, DXY, Spot rate (existing)
  - MCX commodities: CrudeOil, Gold, Silver, Copper, NatGas (NEW)
  - NSE bonds: India 10Y yield proxy (NEW)
  - NSE indices: Nifty50, BankNifty (NEW)
  - FII/DII flows: daily after 6pm (NEW)
  - Calendar features (existing)
Saves daily snapshot to S3
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
from data.ingest.zerodha_client import ZerodhaClient

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)
IST = ZoneInfo('Asia/Kolkata')

# ── Event calendars ────────────────────────────────────────────────
RBI_MPC_DATES = [
    date(2026, 2, 7),  date(2026, 4, 9),
    date(2026, 6, 6),  date(2026, 8, 8),
    date(2026, 10, 8), date(2026, 12, 5),
]
FOMC_DATES = [
    date(2026, 1, 29), date(2026, 3, 19),
    date(2026, 5, 7),  date(2026, 6, 18),
    date(2026, 7, 30), date(2026, 9, 17),
    date(2026, 11, 5), date(2026, 12, 16),
]
INDIA_CPI_DATES = [
    date(2026, 1, 13), date(2026, 2, 12),
    date(2026, 3, 12), date(2026, 4, 14),
    date(2026, 5, 13), date(2026, 6, 12),
    date(2026, 7, 14), date(2026, 8, 13),
    date(2026, 9, 14), date(2026, 10, 13),
    date(2026, 11, 12),date(2026, 12, 14),
]
US_NFP_DATES = [
    date(2026, 1, 9),  date(2026, 2, 6),
    date(2026, 3, 6),  date(2026, 4, 3),
    date(2026, 5, 8),  date(2026, 6, 5),
    date(2026, 7, 10), date(2026, 8, 7),
    date(2026, 9, 4),  date(2026, 10, 2),
    date(2026, 11, 6), date(2026, 12, 4),
]

# ── Zerodha instrument symbols ─────────────────────────────────────
# MCX front month (update monthly)
MCX_SYMBOLS = {
    'crudeoil': 'MCX:CRUDEOIL26OCTFUT',
    'gold':     'MCX:GOLD26DECFUT',
    'silver':   'MCX:SILVER26DECFUT',
    'copper':   'MCX:COPPER26OCTFUT',
    'natgas':   'MCX:NATURALGAS26OCTFUT',
}

# NSE bond price indices (NOT yield % — these are price indices)
BOND_SYMBOLS = {
    'india_10y': 'NSE:NIFTY GS 10YR',
    'india_mid': 'NSE:NIFTY GS 4 8YR',
    'india_lng': 'NSE:NIFTY GS 15YRPLUS',
}

# NSE equity indices
INDEX_SYMBOLS = {
    'nifty50':   'NSE:NIFTY 50',
    'banknifty': 'NSE:NIFTY BANK',
}

# Approximate USD/INR for MCX crude conversion
USD_INR_APPROX = 84.0


class MacroCollector:

    def __init__(self):
        self.s3             = S3Client()
        self.zc             = ZerodhaClient()
        self.snapshots      = []
        self.session        = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self._fii_cache     = {}   # cache FII data (fetched once per day)
        self._prev_snapshot = {}   # for computing changes
        log.info('MacroCollector initialized')

    # ── Existing fetchers ──────────────────────────────────────────
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
            data = resp.json().get('data', [])
            result = {}
            for idx in data:
                if idx.get('index') == 'INDIA VIX':
                    vix = float(idx['last'])
                    result.update({
                        'vix':         vix,
                        'vix_chg':     float(idx['variation']),
                        'vix_chg_pct': float(idx['percentChange']),
                        'vix_prev':    float(idx['previousDayVal']),
                        'vix_1w_ago':  float(idx['oneWeekAgoVal']),
                        'vix_1m_ago':  float(idx['oneMonthAgoVal']),
                        'vix_regime':  ('LOW'    if vix < 12 else
                                        'NORMAL' if vix < 18 else
                                        'HIGH'   if vix < 25 else
                                        'EXTREME'),
                    })
            return result
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
                'spot_eurusd': 1 / float(data['rates']['EUR']),
                'spot_gbpusd': 1 / float(data['rates']['GBP']),
                'spot_jpyusd': 1 / float(data['rates']['JPY']),
                'spot_date':   data['date'],
            }
        except Exception as e:
            log.error(f'Spot rate error: {e}')
        return {}

    def fetch_dxy(self) -> dict:
        try:
            resp = self.session.get(
                'https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB',
                params={'interval': '1d', 'range': '5d'},
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
                    'dxy_chg_pct': round((dxy - dxy_prev) / dxy_prev * 100, 3),
                    'dxy_regime':  ('STRONG'  if dxy - dxy_prev > 0.3 else
                                    'WEAK'    if dxy - dxy_prev < -0.3 else
                                    'NEUTRAL'),
                }
        except Exception as e:
            log.error(f'DXY error: {e}')
        return {}

    # ── New fetchers via Zerodha ───────────────────────────────────
    def fetch_mcx_commodities(self) -> dict:
        """Fetch real-time MCX commodity prices via Zerodha.
        All prices in INR. MCX crude is INR/barrel (not USD).
        """
        try:
            quotes = self.zc.kite.quote(list(MCX_SYMBOLS.values()))
            result = {}
            for name, sym in MCX_SYMBOLS.items():
                q    = quotes.get(sym, {})
                last = q.get('last_price', 0)
                prev = q.get('ohlc', {}).get('close', last)
                chg  = last - prev
                chgp = chg / prev * 100 if prev > 0 else 0

                result[f'{name}_price']   = last
                result[f'{name}_chg']     = round(chg, 2)
                result[f'{name}_chg_pct'] = round(chgp, 3)
                result[f'{name}_oi']      = q.get('oi', 0)

            # Derived commodity features
            crude = result.get('crudeoil_price', 0)
            gold  = result.get('gold_price', 0)
            cop   = result.get('copper_price', 0)

            # Oil regime — thresholds in INR/barrel (MCX quotes INR, NOT USD)
            result['oil_regime'] = (
                'LOW'     if crude < 5000 else
                'NORMAL'  if crude < 7000 else
                'HIGH'    if crude < 9000 else
                'EXTREME'
            )

            # FIX: approximate USD price for reporting/reference
            result['crudeoil_usd_approx'] = round(crude / USD_INR_APPROX, 1) if crude > 0 else 0

            result['gold_oil_ratio'] = round(gold / crude, 4) if crude > 0 else 0

            # Copper as global risk barometer (momentum vs prev snapshot)
            prev_cop = self._prev_snapshot.get('copper_price', cop)
            result['copper_momentum'] = (
                round((cop - prev_cop) / prev_cop * 100, 3)
                if prev_cop > 0 else 0
            )

            log.info(f'MCX: crude=Rs.{crude} '
                     f'(~${result["crudeoil_usd_approx"]}/bbl) '
                     f'gold=Rs.{gold} '
                     f'oil_regime={result["oil_regime"]}')
            return result

        except Exception as e:
            log.error(f'MCX fetch error: {e}')
        return {}

    def fetch_bonds_and_indices(self) -> dict:
        """Fetch NSE bond price indices and equity index levels via Zerodha.

        IMPORTANT: BOND_SYMBOLS return PRICE indices, not yield %.
          - NIFTY GS 10YR ≈ 2628 (price index, not 6.5% yield)
          - Price FALLING → yields RISING (inverse relationship)
          - We derive yield_direction from daily price change
        """
        result = {}

        # Bond price indices
        try:
            quotes = self.zc.kite.quote(list(BOND_SYMBOLS.values()))
            for name, sym in BOND_SYMBOLS.items():
                q    = quotes.get(sym, {})
                last = q.get('last_price', 0)
                prev = q.get('ohlc', {}).get('close', last)
                result[f'{name}_level'] = last
                result[f'{name}_chg']   = round(last - prev, 3)

            # Yield curve slope (long price - short price)
            # Higher long price relative to short = bull steepening
            lng = result.get('india_lng_level', 0)
            mid = result.get('india_mid_level', 0)
            sht = result.get('india_10y_level', 0)
            if lng > 0 and sht > 0:
                result['yield_curve_slope'] = round(lng - sht, 3)
                result['yield_curve_mid']   = round(lng - mid, 3)

            # FIX: yield direction from 10Y price change
            # Bond price FALLING (negative chg) → yields RISING
            # Bond price RISING  (positive chg) → yields FALLING
            india_10y_chg = result.get('india_10y_chg', 0)
            result['india_yield_direction'] = (
                'RISING'  if india_10y_chg < -2 else   # price fell → yield up
                'FALLING' if india_10y_chg >  2 else   # price rose → yield down
                'STABLE'
            )

        except Exception as e:
            log.error(f'Bond fetch error: {e}')

        # Equity indices
        try:
            quotes = self.zc.kite.quote(list(INDEX_SYMBOLS.values()))
            for name, sym in INDEX_SYMBOLS.items():
                q    = quotes.get(sym, {})
                last = q.get('last_price', 0)
                prev = q.get('ohlc', {}).get('close', last)
                chgp = (last - prev) / prev * 100 if prev > 0 else 0
                result[f'{name}_level']   = last
                result[f'{name}_chg_pct'] = round(chgp, 3)

            # BankNifty vs Nifty divergence
            bn = result.get('banknifty_chg_pct', 0)
            n  = result.get('nifty50_chg_pct', 0)
            result['banknifty_nifty_div'] = round(bn - n, 3)
            result['equity_risk_on']      = int(n > 0.3)
            result['bank_stress']         = int(bn < n - 0.5)

            log.info(f'Indices: Nifty={result.get("nifty50_level")} '
                     f'BankNifty={result.get("banknifty_level")} '
                     f'yield_dir={result.get("india_yield_direction")} '
                     f'div={result.get("banknifty_nifty_div")}')
        except Exception as e:
            log.error(f'Index fetch error: {e}')

        return result

    def fetch_fii_flows(self) -> dict:
        """
        Fetch FII/DII daily flows from NSE.
        Published after 6pm IST — cached for the day.
        """
        today = date.today().isoformat()
        if self._fii_cache.get('date') == today:
            return self._fii_cache.get('data', {})

        try:
            nse_session = requests.Session()
            nse_session.headers.update({
                'User-Agent': 'Mozilla/5.0',
                'Accept':     'application/json',
                'Referer':    'https://www.nseindia.com',
            })
            nse_session.get('https://www.nseindia.com', timeout=10)
            resp = nse_session.get(
                'https://www.nseindia.com/api/fiidiiTradeReact',
                timeout=10
            )
            data = resp.json()

            result = {}
            for entry in data:
                cat = entry.get('category', '')
                if 'FII' in cat or 'FPI' in cat:
                    result['fii_net_equity'] = float(
                        entry.get('netTurnover', 0) or 0
                    )
                elif 'DII' in cat:
                    result['dii_net_equity'] = float(
                        entry.get('netTurnover', 0) or 0
                    )

            fii = result.get('fii_net_equity', 0)
            dii = result.get('dii_net_equity', 0)
            result['fii_selling']       = int(fii < -1000)
            result['fii_buying']        = int(fii > 1000)
            result['institutional_net'] = round(fii + dii, 2)

            self._fii_cache = {'date': today, 'data': result}
            log.info(f'FII: net={fii} DII: net={dii}')
            return result

        except Exception as e:
            log.error(f'FII fetch error: {e}')
        return {}

    def build_calendar_features(self) -> dict:
        today = date.today()

        def days_to_next(dates):
            future = [d for d in dates if d >= today]
            return min([(d - today).days for d in future], default=999)

        days_to_rbi    = days_to_next(RBI_MPC_DATES)
        days_to_fomc   = days_to_next(FOMC_DATES)
        days_to_cpi    = days_to_next(INDIA_CPI_DATES)
        days_to_nfp    = days_to_next(US_NFP_DATES)

        # USDINR expiry (last business day of month)
        next_month     = today.replace(day=28) + timedelta(days=4)
        last_day       = next_month - timedelta(days=next_month.day)
        while last_day.weekday() > 4:
            last_day  -= timedelta(days=1)
        days_to_expiry = (last_day - today).days

        return {
            'days_to_rbi':    days_to_rbi,
            'days_to_fomc':   days_to_fomc,
            'days_to_cpi':    days_to_cpi,
            'days_to_nfp':    days_to_nfp,
            'days_to_expiry': days_to_expiry,
            'near_rbi':       int(days_to_rbi  <= 2),
            'near_fomc':      int(days_to_fomc <= 2),
            'near_cpi':       int(days_to_cpi  <= 1),
            'near_nfp':       int(days_to_nfp  <= 1),
            'near_expiry':    int(days_to_expiry <= 5),
            'expiry_week':    int(days_to_expiry <= 7),
            'day_of_week':    today.weekday(),
            'week_of_month':  (today.day - 1) // 7 + 1,
            'month':          today.month,
            'quarter_end':    int(today.month in [3, 6, 9, 12]
                                  and days_to_expiry <= 10),
        }

    def fetch_all(self) -> dict:
        """Fetch all macro data in one snapshot."""
        now     = datetime.now(IST)
        ist_sec = now.hour * 3600 + now.minute * 60 + now.second

        snapshot = {
            'ts_ist':  now.isoformat(),
            'ist_sec': ist_sec,
        }

        # Fetch all sources
        snapshot.update(self.fetch_india_vix())
        snapshot.update(self.fetch_spot_rate())
        snapshot.update(self.fetch_dxy())
        snapshot.update(self.fetch_mcx_commodities())
        snapshot.update(self.fetch_bonds_and_indices())
        snapshot.update(self.build_calendar_features())

        # FII flows only after 6pm IST (data published end of day)
        if ist_sec >= 18 * 3600:
            snapshot.update(self.fetch_fii_flows())

        # ── Derived composite features ─────────────────────────────
        vix    = snapshot.get('vix', 15)
        crude  = snapshot.get('crudeoil_chg_pct', 0)
        dxy    = snapshot.get('dxy_chg_pct', 0)
        fii    = snapshot.get('fii_net_equity', 0)
        bn_div = snapshot.get('banknifty_nifty_div', 0)

        # INR pressure index: positive = pressure on INR to weaken
        inr_pressure = (
            0.30 * (dxy / 0.5) +         # DXY strengthening → INR weak
            0.25 * (crude / 1.0) +        # oil rising → CAD widens → INR weak
            0.20 * (vix / 15 - 1) +       # high VIX → risk-off → EM outflow
            0.15 * (-fii / 1000) +        # FII selling → INR weak
            0.10 * (-bn_div / 0.5)        # bank stress → INR weak
        )
        snapshot['inr_pressure_index']  = round(inr_pressure, 4)
        snapshot['inr_pressure_regime'] = (
            'STRONG_PRESSURE' if inr_pressure >  0.5 else
            'MILD_PRESSURE'   if inr_pressure >  0.2 else
            'NEUTRAL'         if inr_pressure > -0.2 else
            'MILD_SUPPORT'    if inr_pressure > -0.5 else
            'STRONG_SUPPORT'
        )

        # FIX: rate differential — use yield direction, NOT price level as yield
        # india_10y_level is a price index (~2628), subtracting 4.5 is meaningless
        snapshot['rate_differential_direction'] = snapshot.get(
            'india_yield_direction', 'STABLE'
        )

        # DXY correlation proxy
        if 'spot_usdinr' in snapshot and 'dxy' in snapshot:
            snapshot['usdinr_dxy_corr_proxy'] = round(
                snapshot['spot_usdinr'] * snapshot['dxy'] / 100, 4
            )

        # Save for next snapshot's momentum calculations
        self._prev_snapshot = snapshot.copy()

        log.info(
            f'Macro: VIX={snapshot.get("vix","N/A")} '
            f'DXY={snapshot.get("dxy","N/A")} '
            f'Crude=Rs.{snapshot.get("crudeoil_price","N/A")} '
            f'(~${snapshot.get("crudeoil_usd_approx","N/A")}/bbl) '
            f'Gold=Rs.{snapshot.get("gold_price","N/A")} '
            f'Yield={snapshot.get("india_yield_direction","N/A")} '
            f'INR={snapshot.get("inr_pressure_regime","N/A")} '
            f'RBI={snapshot.get("days_to_rbi","N/A")}d'
        )
        return snapshot

    def save_daily(self, date_str: str):
        if not self.snapshots:
            return
        df     = pd.DataFrame(self.snapshots)
        path   = f'/tmp/macro_{date_str}.parquet'
        os.makedirs('/tmp', exist_ok=True)
        df.to_parquet(path, index=False)
        s3_key = f'processed/macro/{date_str}.parquet'
        self.s3.upload(path, s3_key)   # upload(), NOT upload_file()
        log.info(f'Saved {len(df)} macro snapshots → {s3_key}')
        self.snapshots = []
        os.remove(path)

    def run(self):
        """
        Fetch macro data every 15 minutes during market hours (8am–6pm IST).
        One final fetch after 6pm to capture FII flows, then save daily to S3.
        """
        log.info('MacroCollector started')
        current_date = None

        while True:
            try:
                now     = datetime.now(IST)
                ist_sec = now.hour * 3600 + now.minute * 60 + now.second
                today   = now.date().isoformat()

                # Market hours: 8am–6pm IST
                if 8 * 3600 <= ist_sec <= 18 * 3600:
                    if current_date and current_date != today:
                        self.save_daily(current_date)
                    current_date = today
                    snap = self.fetch_all()
                    self.snapshots.append(snap)
                    time.sleep(900)   # every 15 minutes

                # After market (6pm–8pm): one final fetch for FII, then save
                elif 18 * 3600 < ist_sec <= 20 * 3600:
                    if self.snapshots:
                        snap = self.fetch_all()   # includes FII flows
                        self.snapshots.append(snap)
                        self.save_daily(today)
                    time.sleep(3600)

                else:
                    time.sleep(900)

            except Exception as e:
                log.error(f'MacroCollector error: {e}')
                time.sleep(60)


if __name__ == '__main__':
    collector = MacroCollector()
    collector.run()