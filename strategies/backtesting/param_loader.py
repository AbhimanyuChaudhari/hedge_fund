"""
Parameter Loader
================
Loads per-symbol optimized parameters from AWS S3.

Lookup order (4 layers):
    1. Exact match:     CHOLAFIN26JUNFUT in params JSON
    2. Base name match: any CHOLAFIN* key (old contract fallback)
    3. Param transfer:  scale old contract params by vol ratio
    4. Defaults:        last resort hardcoded fallback

Param source priority:
    A. S3 bucket  — written nightly by rolling_optimizer.py on EC2
    B. Local JSON — fallback if S3 unavailable
"""

import re
import os
import json
import boto3
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

S3_BUCKET  = os.getenv('S3_BUCKET_NAME', 'hedge-fund-data-ac')
AWS_KEY    = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'ap-south-1')
S3_PARAMS_DIR = 'params'

_CONTRACT_RE = re.compile(
    r'\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)FUT$',
    re.IGNORECASE
)

_DEFAULTS = {
    'gamma':      0.001,
    'kappa':      1.5,
    'min_spread': 0.10,
    'open_mult':  2.0,
}

_params_cache:   dict = {}
_transfer_cache: dict = {}


def _get_s3():
    return boto3.client('s3',
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION
    )


def _s3_get_json(key: str) -> Optional[dict]:
    try:
        response = _get_s3().get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception:
        return None


def _s3_put_json(key: str, data: dict):
    _get_s3().put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(data, indent=2).encode('utf-8'),
        ContentType='application/json'
    )


def load_optimal_params(model: str = 'v1', force_refresh: bool = False) -> dict:
    global _params_cache
    if not force_refresh and model in _params_cache:
        return _params_cache[model]

    params = _load_from_s3(model)
    if params is None:
        params = _load_from_local(model)
    if params is None:
        logger.warning(f"No params found for model={model}")
        params = {}

    _params_cache[model] = params
    return params


def _load_from_s3(model: str) -> Optional[dict]:
    try:
        key    = f"{S3_PARAMS_DIR}/{model}_optimal_params.json"
        params = _s3_get_json(key)
        if params:
            logger.info(f"Loaded {len(params)} {model} params from s3://{S3_BUCKET}/{key}")
        return params
    except Exception as e:
        logger.warning(f"Could not load params from S3: {e}")
        return None


def _load_from_local(model: str) -> Optional[dict]:
    path = Path(f'research/findings/{model}_optimal_params.json')
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        logger.info(f"Loaded {len(params)} {model} params from {path}")
        return params
    return None


def clear_params_cache(model: Optional[str] = None):
    global _params_cache, _transfer_cache
    if model:
        _params_cache.pop(model, None)
        keys_to_remove = [k for k in _transfer_cache if k[1] == model]
        for k in keys_to_remove:
            del _transfer_cache[k]
    else:
        _params_cache   = {}
        _transfer_cache = {}


def strip_contract_suffix(symbol: str) -> str:
    return _CONTRACT_RE.sub('', symbol)


def get_symbol_params(
    symbol:       str,
    model:        str = 'v1',
    data_loader   = None,
    new_dates:    Optional[list] = None,
    old_dates:    Optional[list] = None,
    use_transfer: bool = True,
) -> dict:
    params = load_optimal_params(model)

    if symbol in params:
        return _extract(params[symbol])

    base = strip_contract_suffix(symbol)
    if base:
        matching = {k: v for k, v in params.items()
                    if strip_contract_suffix(k) == base}
        if matching:
            best_key   = sorted(matching.keys())[-1]
            old_params = _extract(params[best_key])

            if use_transfer and data_loader is not None:
                transferred = _get_transferred_params(
                    symbol=symbol, old_params=old_params,
                    old_key=best_key, data_loader=data_loader,
                    new_dates=new_dates, old_dates=old_dates,
                )
                if transferred:
                    return transferred

            return old_params

    logger.warning(f"{symbol}: no params found — using defaults")
    return _DEFAULTS.copy()


def _get_transferred_params(
    symbol, old_params, old_key, data_loader, new_dates, old_dates
) -> Optional[dict]:
    from strategies.backtesting.param_transfer import (
        transfer_params, validate_transfer,
        get_recent_trading_dates, _derive_old_symbol,
    )
    cache_key = (symbol, old_key)
    if cache_key in _transfer_cache:
        return _transfer_cache[cache_key]

    try:
        if new_dates is None or old_dates is None:
            all_recent = get_recent_trading_dates(n_days=10)
            new_dates  = new_dates or all_recent[-3:]
            old_dates  = old_dates or all_recent[:5]

        old_symbol  = _derive_old_symbol(symbol) or old_key
        transferred = transfer_params(
            symbol=symbol, old_params=old_params,
            data_loader=data_loader, new_dates=new_dates,
            old_dates=old_dates, old_symbol=old_symbol,
        )
        if not transferred:
            return None

        if not validate_transfer(old_params, transferred, symbol):
            _transfer_cache[cache_key] = old_params
            return old_params

        _transfer_cache[cache_key] = transferred
        return transferred

    except Exception as e:
        logger.error(f"Transfer failed for {symbol}: {e}")
        return None


def _extract(p: dict) -> dict:
    return {
        'gamma':      p.get('gamma',      _DEFAULTS['gamma']),
        'kappa':      p.get('kappa',      _DEFAULTS['kappa']),
        'min_spread': p.get('min_spread', _DEFAULTS['min_spread']),
        'open_mult':  p.get('open_mult',  _DEFAULTS['open_mult']),
    }


def get_all_symbols_with_params(model: str = 'v1') -> list:
    return list(load_optimal_params(model).keys())


def run_contract_roll_transfer(
    model:        str = 'v1',
    new_contract: str = 'JUNFUT',
    old_contract: str = 'MAYFUT',
    new_dates:    Optional[list] = None,
    old_dates:    Optional[list] = None,
):
    from strategies.backtesting.data_loader import load_day
    from strategies.backtesting.param_transfer import (
        transfer_all_params, get_recent_trading_dates,
    )

    old_params = load_optimal_params(model, force_refresh=True)
    if not old_params:
        print(f"No params found for model={model}")
        return

    if new_dates is None:
        new_dates = get_recent_trading_dates(n_days=3)
    if old_dates is None:
        old_dates = get_recent_trading_dates(n_days=5, before_date=new_dates[0])

    print(f"Transferring params: {old_contract} → {new_contract}")

    transferred = transfer_all_params(
        old_params_dict=old_params, new_contract=new_contract,
        old_contract=old_contract, data_loader=load_day,
        new_dates=new_dates, old_dates=old_dates, output_path=None,
    )

    old_params.update(transferred)

    try:
        key = f"{S3_PARAMS_DIR}/{model}_optimal_params.json"
        _s3_put_json(key, old_params)
        print(f"Saved to s3://{S3_BUCKET}/{key}")
    except Exception as e:
        local_path = Path(f'research/findings/{model}_optimal_params.json')
        with open(local_path, 'w') as f:
            json.dump(old_params, f, indent=2)
        print(f"S3 unavailable ({e}) — saved to {local_path}")

    clear_params_cache(model)
    print(f"Done. {len(transferred)} symbols transferred.")
    return transferred