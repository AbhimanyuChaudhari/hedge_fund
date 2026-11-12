"""
Triple-barrier labels for USDINR 1-second bar data.

Label convention:
  +1  — price hits the upper barrier first within the horizon
  -1  — price hits the lower barrier first within the horizon
   0  — horizon expires without hitting either barrier (time-out)

Default parameters (NSE CDS USDINR):
  horizon   = 1800 seconds  (30-minute forward window)
  threshold = 10 bps        (symmetric ±10 bps barriers)

Usage
-----
from data.processing.usdinr.labels import add_triple_barrier_labels

labeled = add_triple_barrier_labels(feature_df)
# adds columns: label, label_horizon, hit_time
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_KEYS    = ["symbol", "ts_sec"]
_SESSION = ["symbol", "date"]
_EPS     = 1e-12


def add_triple_barrier_labels(
    df: pd.DataFrame,
    *,
    horizon:   int   = 1800,
    threshold: float = 10.0,    # bps
    mid_col:   str   = None,    # auto-derived from bid_p1/ask_p1 if None
    price_col: str   = None,    # alias for mid_col
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    Compute triple-barrier labels and append to df.

    Parameters
    ----------
    df : DataFrame
        Must have ['symbol', 'ts_sec'] and either a mid-price column
        (named by mid_col/price_col) or ['bid_p1', 'ask_p1'].
    horizon : int
        Forward window in seconds. Default 1800 (30 min).
    threshold : float
        Barrier half-width in basis points. Default 10 bps.
    mid_col : str or None
        Name of an existing mid-price column.  If None, mid is computed
        as (bid_p1 + ask_p1) / 2.
    price_col : str or None
        Alias for mid_col (either name accepted).
    drop_incomplete : bool
        If True, set label=NaN for bars where fewer than `horizon` seconds
        remain in the session (the forward window is truncated).

    Returns
    -------
    DataFrame (copy) with three new columns:
        label          : int8  (+1, 0, -1) or NaN
        label_horizon  : int16  actual forward window used (seconds to boundary
                         or time-out), NaN when label is NaN
        hit_time       : float  ts_sec of the first barrier hit, NaN on time-out
    """
    mc = mid_col or price_col
    if mc is not None and mc not in df.columns:
        raise ValueError(f"mid_col '{mc}' not found in DataFrame")
    if mc is None and not {"bid_p1", "ask_p1"}.issubset(df.columns):
        raise ValueError("Provide mid_col= or ensure bid_p1 and ask_p1 are present")

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1 second, got {horizon}")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0 bps, got {threshold}")

    out = df.copy()
    out["symbol"] = out["symbol"].astype(str)
    out["ts_sec"] = pd.to_numeric(out["ts_sec"], errors="raise").astype("int64")

    if "date" not in out.columns:
        ist = pd.to_datetime(out["ts_sec"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
        out["date"] = ist.dt.date

    out = out.sort_values(["symbol", "date", "ts_sec"]).reset_index(drop=True)

    if mc is not None:
        mid = out[mc].to_numpy(dtype=np.float64)
    else:
        bp  = out["bid_p1"].to_numpy(dtype=np.float64)
        ap  = out["ask_p1"].to_numpy(dtype=np.float64)
        mid = (bp + ap) / 2.0

    ts   = out["ts_sec"].to_numpy(dtype=np.int64)
    N    = len(out)

    label        = np.full(N, np.nan, dtype=np.float32)
    label_horizon = np.full(N, np.nan, dtype=np.float32)
    hit_time     = np.full(N, np.nan, dtype=np.float64)

    session_groups = out.groupby(_SESSION, sort=False).indices

    for (symbol, day), idx in session_groups.items():
        idx       = np.sort(idx)
        ts_s      = ts[idx]
        mid_s     = mid[idx]
        n_s       = len(idx)

        valid     = np.isfinite(mid_s) & (mid_s > 0)

        for i in range(n_s):
            if not valid[i]:
                continue

            m0    = mid_s[i]
            t0    = ts_s[i]
            t_end = t0 + horizon

            # Find all bars within [t0+1, t0+horizon]
            right = np.searchsorted(ts_s, t_end,   side="right")
            left  = np.searchsorted(ts_s, t0 + 1,  side="left")

            fwd_idx = np.arange(left, right)

            if len(fwd_idx) == 0:
                # No forward bars — can't label
                continue

            if drop_incomplete:
                # Require at least `horizon` seconds of coverage
                actual_coverage = ts_s[fwd_idx[-1]] - t0
                if actual_coverage < horizon:
                    continue

            # Basis-point returns over the forward window
            fwd_mid  = mid_s[fwd_idx]
            valid_fwd = np.isfinite(fwd_mid) & (fwd_mid > 0)

            if not valid_fwd.any():
                continue

            # Clip invalid prices to last valid
            fwd_mid_clean = fwd_mid.copy()
            last_valid = m0
            for j in range(len(fwd_mid_clean)):
                if valid_fwd[j]:
                    last_valid = fwd_mid_clean[j]
                else:
                    fwd_mid_clean[j] = last_valid

            ret_bps = 1e4 * (fwd_mid_clean - m0) / m0

            # First upper hit
            up_hit = np.where(ret_bps >= threshold)[0]
            # First lower hit
            dn_hit = np.where(ret_bps <= -threshold)[0]

            first_up = up_hit[0] if len(up_hit) > 0 else len(fwd_idx)
            first_dn = dn_hit[0] if len(dn_hit) > 0 else len(fwd_idx)

            if first_up < first_dn:
                label[idx[i]]         = 1
                hit_ts                = ts_s[fwd_idx[first_up]]
                label_horizon[idx[i]] = int(hit_ts - t0)
                hit_time[idx[i]]      = float(hit_ts)
            elif first_dn < first_up:
                label[idx[i]]         = -1
                hit_ts                = ts_s[fwd_idx[first_dn]]
                label_horizon[idx[i]] = int(hit_ts - t0)
                hit_time[idx[i]]      = float(hit_ts)
            else:
                # Time-out (both hit at same bar or neither hit)
                label[idx[i]]         = 0
                label_horizon[idx[i]] = int(ts_s[fwd_idx[-1]] - t0)
                # hit_time stays NaN

    out["label"]         = label.astype(np.float32)
    out["label_horizon"] = label_horizon.astype(np.float32)
    out["hit_time"]      = hit_time.astype(np.float64)

    n_labeled   = int(np.isfinite(label).sum())
    n_up        = int((label == 1).sum())
    n_dn        = int((label == -1).sum())
    n_neutral   = int((label == 0).sum())
    n_unlabeled = N - n_labeled

    log.info(
        "add_triple_barrier_labels: %d rows | labeled=%d (+1=%d, 0=%d, -1=%d) | unlabeled=%d | "
        "horizon=%ds threshold=%.1fbps",
        N, n_labeled, n_up, n_neutral, n_dn, n_unlabeled, horizon, threshold,
    )

    return out


def label_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-symbol summary of label distribution.

    Parameters
    ----------
    df : DataFrame with a 'label' column (output of add_triple_barrier_labels)

    Returns
    -------
    DataFrame with columns: symbol, n_total, n_labeled, frac_up, frac_neutral,
                            frac_down, frac_unlabeled
    """
    if "label" not in df.columns:
        raise ValueError("DataFrame must have a 'label' column")

    rows = []
    for symbol, grp in df.groupby("symbol", sort=True):
        lbl     = grp["label"]
        n_tot   = len(lbl)
        labeled = lbl.dropna()
        n_lbl   = len(labeled)

        rows.append({
            "symbol":         symbol,
            "n_total":        n_tot,
            "n_labeled":      n_lbl,
            "frac_up":        (labeled == 1).sum()  / max(n_lbl, 1),
            "frac_neutral":   (labeled == 0).sum()  / max(n_lbl, 1),
            "frac_down":      (labeled == -1).sum() / max(n_lbl, 1),
            "frac_unlabeled": (n_tot - n_lbl)       / max(n_tot, 1),
        })

    return pd.DataFrame(rows)