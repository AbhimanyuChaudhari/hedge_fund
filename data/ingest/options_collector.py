"""
USDINR Options IV Collector
Runs separately from main pipeline — does NOT affect tick_collector.py
Fetches ATM options every minute during market hours
Saves to S3: processed/options/USDINR/{DATE}.parquet
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

logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)
IST = ZoneInfo('Asia/Kolkata')


# ── Black-Scholes + IV ────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, option_type='CE'):
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(S-K, 0) if option_type=='CE' else max(K-S, 0)
    d1  = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2  = d1 - sigma*np.sqrt(T)
    if option_type == 'CE':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def compute_iv(price, S, K, T, r=0.065, option_type='CE',
               tol=1e-5, max_iter=100):
    """Newton-Raphson implied volatility"""
    from scipy.stats import norm
    if T <= 0 or price <= 0:
        return np.nan
    intrinsic = max(S-K,0) if option_type=='CE' else max(K-S,0)
    if price < intrinsic - 1e-6:
        return np.nan
    sigma = 0.10
    for _ in range(max_iter):
        f    = bs_price(S, K, T, r, sigma, option_type) - price
        d1   = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        if abs(vega) < 1e-10:
            break
        s_new = sigma - f/vega
        if s_new <= 0:
            sigma /= 2
            continue
        if abs(s_new - sigma) < tol:
            return s_new if 0.001 < s_new < 5.0 else np.nan
        sigma = s_new
    return sigma if 0.001 < sigma < 5.0 else np.nan


# ── Options Collector ─────────────────────────────────────────────
class OptionsCollector:

    def __init__(self):
        self.zc        = ZerodhaClient()
        self.s3        = S3Client()
        self.snapshots = []
        log.info('OptionsCollector initialized')

    def get_front_expiry(self):
        insts  = self.zc.kite.instruments('CDS')
        df     = pd.DataFrame(insts)
        opts   = df[(df['name']=='USDINR') &
                    (df['instrument_type']=='CE')].copy()
        opts['expiry'] = pd.to_datetime(opts['expiry'])
        today  = date.today()
        active = opts[opts['expiry'].dt.date >= today]
        return active['expiry'].min() if not active.empty else None

    def get_futures_price(self):
        try:
            insts  = self.zc.kite.instruments('CDS')
            df     = pd.DataFrame(insts)
            fut    = df[(df['name']=='USDINR') &
                        (df['instrument_type']=='FUT')].copy()
            fut['expiry'] = pd.to_datetime(fut['expiry'])
            today  = date.today()
            active = fut[fut['expiry'].dt.date >= today]
            front  = active.sort_values('expiry').iloc[0]
            sym    = f'CDS:{front["tradingsymbol"]}'
            quote  = self.zc.kite.quote([sym])
            return quote[sym]['last_price']
        except Exception as e:
            log.error(f'Futures price error: {e}')
            return None

    def fetch_snapshot(self, futures_price):
        try:
            expiry = self.get_front_expiry()
            if expiry is None:
                return None

            expiry_date = expiry.date()
            T           = max((expiry_date - date.today()).days/365, 1/365)
            expiry_str  = expiry.strftime('%y%b').upper()

            # ATM ± 4 strikes (0.125 intervals)
            atm     = round(futures_price * 8) / 8
            strikes = [atm + i*0.125 for i in range(-4, 5)]

            instruments = []
            for s in strikes:
                s_str = f'{s:.3f}'.rstrip('0').rstrip('.')
                for t in ['CE','PE']:
                    instruments.append(f'CDS:USDINR{expiry_str}{s_str}{t}')

            quotes = self.zc.kite.quote(instruments)

            now     = datetime.now(IST)
            ist_sec = now.hour*3600 + now.minute*60 + now.second

            snap = {
                'ts_ist':        now.isoformat(),
                'ist_sec':       ist_sec,
                'futures_price': futures_price,
                'expiry':        expiry_date.isoformat(),
                'T':             T,
                'atm_strike':    atm,
            }

            ivs_ce = {}
            ivs_pe = {}

            for sym, q in quotes.items():
                last = q.get('last_price', 0)
                bid  = q.get('depth',{}).get('buy', [{}])[0].get('price',0)
                ask  = q.get('depth',{}).get('sell',[{}])[0].get('price',0)
                mid  = (bid+ask)/2 if bid>0 and ask>0 else last
                oi   = q.get('oi', 0)

                # Parse
                opt_type = 'CE' if sym.endswith('CE') else 'PE'
                tmp      = sym.replace('CDS:USDINR','').replace(expiry_str,'')
                tmp      = tmp.replace('CE','').replace('PE','')
                try:
                    strike = float(tmp)
                except:
                    continue

                iv = compute_iv(mid, futures_price, strike, T,
                                option_type=opt_type) if mid > 0 else np.nan

                snap[f'{opt_type}_{strike:.3f}_iv']    = iv
                snap[f'{opt_type}_{strike:.3f}_price'] = mid
                snap[f'{opt_type}_{strike:.3f}_oi']    = oi

                if opt_type == 'CE' and not np.isnan(iv if iv else np.nan):
                    ivs_ce[strike] = iv
                elif opt_type == 'PE' and not np.isnan(iv if iv else np.nan):
                    ivs_pe[strike] = iv

            # ATM IV
            atm_ce = ivs_ce.get(atm, np.nan)
            atm_pe = ivs_pe.get(atm, np.nan)
            snap['atm_iv']    = np.nanmean([atm_ce, atm_pe])
            snap['atm_iv_ce'] = atm_ce
            snap['atm_iv_pe'] = atm_pe

            # IV skew (25d risk reversal proxy)
            # OTM put IV - OTM call IV
            if len(ivs_ce) >= 2 and len(ivs_pe) >= 2:
                otm_put  = ivs_pe.get(atm - 0.25, np.nan)
                otm_call = ivs_ce.get(atm + 0.25, np.nan)
                snap['iv_skew'] = (otm_put - otm_call
                                   if not np.isnan(otm_put)
                                   and not np.isnan(otm_call)
                                   else np.nan)
            else:
                snap['iv_skew'] = np.nan

            # IV change (vs previous snapshot)
            if self.snapshots:
                prev_iv         = self.snapshots[-1].get('atm_iv', np.nan)
                snap['iv_chg']  = snap['atm_iv'] - prev_iv
            else:
                snap['iv_chg']  = 0.0

            log.info(f'Options: F={futures_price:.4f} '
                     f'ATM_IV={snap["atm_iv"]:.4f} '
                     f'skew={snap["iv_skew"]:.4f}')
            return snap

        except Exception as e:
            log.error(f'Snapshot error: {e}')
            return None

    def save_daily(self, date_str):
        if not self.snapshots:
            return
        df      = pd.DataFrame(self.snapshots)
        path    = f'C:/tmp/usdinr_options_{date_str}.parquet'
        os.makedirs('C:/tmp', exist_ok=True)
        df.to_parquet(path, index=False)
        s3_key  = f'processed/options/USDINR/{date_str}.parquet'
        self.s3.upload_file(path, s3_key)
        log.info(f'Saved {len(df)} snapshots → {s3_key}')
        self.snapshots = []
        os.remove(path)

    def run(self):
        log.info('Options collector started')
        current_date = None
        while True:
            try:
                now     = datetime.now(IST)
                ist_sec = now.hour*3600 + now.minute*60 + now.second
                today   = now.date().isoformat()

                if 9*3600 <= ist_sec <= 17*3600:
                    if current_date and current_date != today:
                        self.save_daily(current_date)
                    current_date = today
                    fp = self.get_futures_price()
                    if fp:
                        snap = self.fetch_snapshot(fp)
                        if snap:
                            self.snapshots.append(snap)
                    time.sleep(60)

                elif ist_sec > 17*3600 and self.snapshots:
                    self.save_daily(today)
                    time.sleep(3600)
                else:
                    time.sleep(60)

            except Exception as e:
                log.error(f'Main loop error: {e}')
                time.sleep(60)


if __name__ == '__main__':
    collector = OptionsCollector()
    collector.run()