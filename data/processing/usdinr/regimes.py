"""
HMM regime model for USDINR 1-second bar data.

Fits a 4-state Hidden Markov Model with a joint emission distribution:
  - Observation 1: order-book update count per second (categorical, 4 bins: 0/1/2/3+)
  - Observation 2: log mid-price return in bps (zero-inflated Gaussian)

Outputs four filtered state probabilities per row:
  P_state1_filtered ... P_state4_filtered

States are ordered by descending update activity so that the labelling is
consistent across re-fits (State 1 = most active, State 4 = quietest).

Usage
-----
# --- Fit (training time only) ---
from data.processing.usdinr.regimes import add_soft_hmm_features, save_hmm_model

df_with_probs, model = add_soft_hmm_features(
    df            = df,            # output of USDINRFeatureBuilder + tick_count column
    train_end_date= "2026-10-15",  # last training date (inclusive)
    K             = 4,
)
save_hmm_model(model, r"C:\\projects\\hedge_fund\\models\\hmm_usdinr.pkl")

# --- Inference (load pre-fitted model) ---
from data.processing.usdinr.regimes import load_hmm_model, apply_hmm_model

model        = load_hmm_model(r"C:\\projects\\hedge_fund\\models\\hmm_usdinr.pkl")
df_with_probs = apply_hmm_model(df, model)

# df_with_probs gains columns:
#   P_state1_filtered, P_state2_filtered, P_state3_filtered, P_state4_filtered
#   state_filtered        (hard label 1-4, argmax of soft probs)
#   confidence_filtered   (max soft prob)
"""

import logging
import pickle
import pathlib

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from numba import njit

log = logging.getLogger(__name__)

# Column names produced by this module
HMM_PROB_COLS = [f"P_state{i}_filtered" for i in range(1, 5)]
HMM_ALL_COLS  = HMM_PROB_COLS + ["state_filtered", "confidence_filtered"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@njit
def _e_step(logb, A, pi, starts, ends):
    """
    Forward-backward (Baum-Welch E-step) compiled with Numba.

    Parameters
    ----------
    logb   : (T, K) log emission probabilities
    A      : (K, K) transition matrix
    pi     : (K,)   initial state distribution
    starts : (S,)   start indices of each sequence (day × symbol)
    ends   : (S,)   end   indices of each sequence (exclusive)

    Returns
    -------
    gamma       : (T, K) posterior state probabilities
    transitions : (K, K) expected transition counts
    pi_counts   : (K,)   expected initial-state counts
    ll          : float  total log-likelihood
    """
    T, K = logb.shape
    gamma       = np.empty((T, K))
    transitions = np.zeros((K, K))
    pi_counts   = np.zeros(K)
    ll          = 0.0

    for seq in range(len(starts)):
        lo, hi = starts[seq], ends[seq]
        n      = hi - lo

        alpha = np.zeros((n, K))
        beta  = np.ones((n, K))
        b     = np.empty((n, K))
        scale = np.empty(n)

        # Forward pass
        for t in range(n):
            peak = np.max(logb[lo + t])
            for k in range(K):
                b[t, k] = np.exp(logb[lo + t, k] - peak)
            for j in range(K):
                if t == 0:
                    alpha[t, j] = pi[j] * b[t, j]
                else:
                    v = 0.0
                    for i in range(K):
                        v += alpha[t - 1, i] * A[i, j]
                    alpha[t, j] = b[t, j] * v
            s = max(np.sum(alpha[t]), 1e-300)
            scale[t] = s
            ll += np.log(s) + peak
            for j in range(K):
                alpha[t, j] /= s

        # Backward pass
        for t in range(n - 2, -1, -1):
            for i in range(K):
                v = 0.0
                for j in range(K):
                    v += A[i, j] * b[t + 1, j] * beta[t + 1, j]
                beta[t, i] = v / scale[t + 1]

        # Gamma (posterior)
        for t in range(n):
            z = 0.0
            for k in range(K):
                gamma[lo + t, k] = alpha[t, k] * beta[t, k]
                z += gamma[lo + t, k]
            for k in range(K):
                gamma[lo + t, k] /= max(z, 1e-300)

        # Accumulate sufficient statistics
        for k in range(K):
            pi_counts[k] += gamma[lo, k]
        for t in range(n - 1):
            for i in range(K):
                for j in range(K):
                    transitions[i, j] += (
                        alpha[t, i] * A[i, j] * b[t + 1, j] * beta[t + 1, j]
                        / scale[t + 1]
                    )

    return gamma, transitions, pi_counts, ll


def _log_emissions(counts, returns, q, p, sigma):
    """
    Joint log-emission: P(count_cat=c, x_bp=r | state=k).

    count_cat ~ Categorical(q[k])
    x_bp      ~ 0 with probability p[k, count_cat]
                Gaussian(0, sigma[k]) otherwise

    Parameters
    ----------
    counts  : (T,) int   — binned update count 0/1/2/3
    returns : (T,) float — log mid-price return in bps (NaN for first row of each day)
    q       : (K, 4)     — categorical probabilities over count bins
    p       : (K, 4)     — P(zero return | count_cat, state)
    sigma   : (K,)       — Gaussian std for non-zero returns

    Returns
    -------
    out : (T, K) log-emission matrix
    """
    if np.any((counts == 0) & (returns != 0)):
        raise ValueError(
            "count=0 with nonzero mid-price return; inspect forward-filling"
        )

    zero    = returns == 0
    nonzero = ~zero
    K       = len(sigma)
    out     = np.empty((len(counts), K))

    for k in range(K):
        out[:, k] = np.log(np.maximum(q[k, counts], 1e-300))

        out[zero, k] += np.log(
            np.maximum(p[k, counts[zero]], 1e-300)
        )

        x = returns[nonzero]
        out[nonzero, k] += (
            np.log(np.maximum(1 - p[k, counts[nonzero]], 1e-300))
            - np.log(sigma[k])
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * (x / sigma[k]) ** 2
        )

    return out


def _sequence_arrays(frame):
    """
    Build flat arrays and sequence boundary indices from a sorted DataFrame.

    Returns
    -------
    counts  : (T,) int64
    returns : (T,) float64
    starts  : (S,) int64 — start index of each (symbol, date) sequence
    ends    : (S,) int64 — end   index (exclusive)
    """
    lengths = (
        frame.groupby(["symbol", "date"], sort=False)
        .size()
        .to_numpy()
    )
    ends   = np.cumsum(lengths).astype(np.int64)
    starts = np.r_[0, ends[:-1]].astype(np.int64)
    return (
        frame["count_cat"].to_numpy(np.int64),
        frame["x_bp"].to_numpy(np.float64),
        starts,
        ends,
    )


def _fit_once(counts, returns, starts, ends, K, seed, max_iter, tol):
    """
    Single EM run with one random initialisation.

    Returns a dict with fitted parameters and final log-likelihood.
    """
    rng = np.random.default_rng(seed)

    # --- Initialise q (categorical over count bins) ---
    q       = rng.dirichlet(np.full(4, 2.0), size=K)
    centers = np.linspace(0, 3, K).astype(int)
    for k, c in enumerate(centers):
        q[k, c] += 2
        q[k]    /= q[k].sum()

    # --- Initialise p (zero-return probability) ---
    p       = np.empty((K, 4))
    p[:, 0] = 1.0   # zero ticks → always zero return
    raw_p   = np.array([
        1.0,
        np.mean(returns[counts == 1] == 0) if np.any(counts == 1) else 0.5,
        np.mean(returns[counts == 2] == 0) if np.any(counts == 2) else 0.5,
        np.mean(returns[counts == 3] == 0) if np.any(counts == 3) else 0.5,
    ])
    p[:, 1:] = np.clip(
        raw_p[1:] + rng.normal(0, 0.09, (K, 3)), 0.01, 0.999
    )

    # --- Initialise sigma ---
    nz    = returns[returns != 0]
    sigma = max(float(np.std(nz)), 0.03) * np.exp(rng.normal(0, 0.35, K))

    # --- Initialise A and pi ---
    A  = np.full((K, K), 0.12 / (K - 1))
    np.fill_diagonal(A, 0.88)
    pi = np.full(K, 1.0 / K)

    history = []
    for _ in range(max_iter):
        logb              = _log_emissions(counts, returns, q, p, sigma)
        gamma, trans, pi_counts, ll = _e_step(logb, A, pi, starts, ends)

        history.append(ll)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol * (1 + abs(ll)):
            break

        weights = gamma.sum(axis=0)
        nzmask  = returns != 0

        for k in range(K):
            cat_weights = np.bincount(counts, weights=gamma[:, k], minlength=4)
            q[k] = (cat_weights + 0.01) / (weights[k] + 0.04)

            for c in (1, 2, 3):
                zero_w    = gamma[(counts == c) & (returns == 0), k].sum()
                p[k, c]   = (zero_w + 0.1) / (cat_weights[c] + 0.2)
            p[k, 0] = 1.0

            w        = gamma[nzmask, k]
            sigma[k] = np.sqrt(
                max(np.sum(w * returns[nzmask] ** 2) / max(w.sum(), 1e-12), 1e-8)
            )

        A  = (trans + 1e-3) / (trans.sum(axis=1, keepdims=True) + K * 1e-3)
        pi = (pi_counts + 1e-3) / (pi_counts.sum() + K * 1e-3)

    final_ll = _e_step(
        _log_emissions(counts, returns, q, p, sigma), A, pi, starts, ends
    )[-1]

    return dict(
        q          = q,
        p          = p,
        sigma      = sigma,
        A          = A,
        pi         = pi,
        ll         = final_ll,
        iterations = len(history),
        history    = history,
    )


def _filter_probs(counts, returns, starts, ends, model):
    """
    Causal (online) filtering: P(Z_t = k | Y_1, ..., Y_t).

    Does NOT use future observations — safe for both training and test rows.
    """
    logb = _log_emissions(counts, returns, model["q"], model["p"], model["sigma"])
    K    = len(model["sigma"])
    out  = np.empty((len(counts), K))

    for lo, hi in zip(starts, ends):
        for t in range(lo, hi):
            prior = model["pi"] if t == lo else out[t - 1] @ model["A"]
            z     = np.log(np.maximum(prior, 1e-300)) + logb[t]
            out[t] = np.exp(z - logsumexp(z))

    return out


def _order_states_by_activity(model):
    """
    Re-order states so that State 1 = most active (highest P(count >= 1))
    and State 4 = least active. Makes labels stable across re-fits.
    """
    activity = 1 - model["q"][:, 0]   # P(at least one update)
    order    = np.argsort(-activity)   # descending

    model["q"]     = model["q"][order]
    model["p"]     = model["p"][order]
    model["sigma"] = model["sigma"][order]
    model["pi"]    = model["pi"][order]
    model["A"]     = model["A"][np.ix_(order, order)]
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_soft_hmm_features(
    df,
    train_end_date,
    *,
    K         = 4,
    restarts  = 3,
    max_iter  = 70,
    tol       = 1e-6,
    seed      = 42,
):
    """
    Fit a K-state HMM on training data and append soft state probabilities.

    Parameters
    ----------
    df             : DataFrame — output of USDINRFeatureBuilder with tick_count added
    train_end_date : str or date — last INCLUSIVE training date (HMM never sees val/test)
    K              : int — number of hidden states (default 4)
    restarts       : int — EM random restarts; best log-likelihood wins
    max_iter       : int — maximum EM iterations per restart
    tol            : float — convergence tolerance on log-likelihood
    seed           : int — base random seed (seed+r used for restart r)

    Returns
    -------
    df_out   : DataFrame (copy) — original columns plus HMM_ALL_COLS
    model    : dict — fitted HMM parameters (pass to save_hmm_model or apply_hmm_model)

    Notes
    -----
    - Requires columns: symbol, date, ts_sec, tick_count, bid_p1, ask_p1
    - tick_count = number of raw order-book snapshots in that 1-second bar
      (merge from df_raw.groupby([symbol,date,ts_sec]).size() if not present)
    - train_end_date must yield at least 3 training days and at least 1 future day
    """
    required = ["symbol", "date", "ts_sec", "tick_count", "bid_p1", "ask_p1"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d = df.copy()
    d["_orig_order_"] = np.arange(len(d))
    d["date"]         = pd.to_datetime(d["date"]).dt.date
    d = d.sort_values(["symbol", "date", "ts_sec"]).reset_index(drop=True)

    # Sanity checks
    if d.duplicated(["symbol", "date", "ts_sec"]).any():
        raise ValueError("Duplicate (symbol, date, ts_sec) rows found")

    delta = d.groupby(["symbol", "date"], sort=False)["ts_sec"].diff().dropna()
    if not delta.eq(1).all():
        raise ValueError(
            "Expected a complete 1-second grid within each session. "
            "Run USDINRFeatureBuilder.preprocess() first."
        )

    valid_book = (
        (d["bid_p1"] > 0)
        & (d["ask_p1"] >= d["bid_p1"])
        & d["ask_p1"].notna()
    )
    if not valid_book.all():
        raise ValueError("Invalid or missing bid_p1/ask_p1 prices")

    ticks = d["tick_count"]
    if ticks.isna().any() or (ticks < 0).any() or not np.allclose(ticks % 1, 0):
        raise ValueError("tick_count must be non-negative integers with no NaNs")

    # Build HMM observation columns
    d["mid"]   = (d["bid_p1"] + d["ask_p1"]) / 2.0
    d["x_bp"]  = d.groupby(["symbol", "date"], sort=False)["mid"].transform(
        lambda s: 10_000 * np.log(s).diff()
    )
    d["count_cat"] = np.minimum(ticks, 3).astype(np.int64)

    violated = d["count_cat"].eq(0) & d["x_bp"].notna() & d["x_bp"].ne(0)
    if violated.any():
        raise ValueError(
            f"{violated.sum()} rows have zero tick_count but nonzero mid-price return. "
            "Check forward-filling in preprocessing."
        )

    # Split into usable rows (have x_bp) and their train subset
    usable = d[d["x_bp"].notna()].copy()
    cutoff = pd.Timestamp(train_end_date).date()
    train  = usable[usable["date"] <= cutoff].copy()
    future = usable[usable["date"] >  cutoff]

    if train.empty or future.empty:
        raise ValueError(
            "train_end_date must yield a non-empty training period AND "
            "at least one future date."
        )
    if train["date"].nunique() < 3:
        raise ValueError("Need at least 3 training dates to fit the HMM.")

    log.info(
        "Fitting HMM | K=%d  restarts=%d  train_days=%d  train_rows=%d",
        K, restarts, train["date"].nunique(), len(train),
    )

    # Fit HMM on training data only
    tr_counts, tr_returns, tr_starts, tr_ends = _sequence_arrays(train)

    best = None
    for r in range(restarts):
        fitted = _fit_once(
            tr_counts, tr_returns, tr_starts, tr_ends,
            K, seed + r, max_iter, tol,
        )
        log.info("  restart %d/%d  ll=%.4f  iters=%d", r + 1, restarts,
                 fitted["ll"], fitted["iterations"])
        if best is None or fitted["ll"] > best["ll"]:
            best = fitted

    best = _order_states_by_activity(best)
    log.info("HMM fitted | best ll=%.4f", best["ll"])
    log.info("Transition matrix:\n%s", np.round(best["A"], 4))

    # Filter probabilities causally on ALL usable rows (train + val + test)
    all_counts, all_returns, all_starts, all_ends = _sequence_arrays(usable)
    probs = _filter_probs(all_counts, all_returns, all_starts, all_ends, best)

    if not np.allclose(probs.sum(axis=1), 1, atol=1e-8):
        raise RuntimeError("Posterior probabilities do not sum to 1 — numerical issue")

    # Write back to d
    for col in HMM_ALL_COLS:
        d[col] = np.nan

    d.loc[usable.index, HMM_PROB_COLS]          = probs
    d.loc[usable.index, "state_filtered"]        = np.argmax(probs, axis=1) + 1
    d.loc[usable.index, "confidence_filtered"]   = np.max(probs, axis=1)

    # State distribution summary
    state_counts = (
        d.loc[usable.index, "state_filtered"]
        .value_counts()
        .sort_index()
    )
    for state, count in state_counts.items():
        log.info(
            "  State %d: %d rows (%.1f%%)",
            state, count, 100 * count / len(usable),
        )

    # Restore original row order
    d = d.sort_values("_orig_order_").drop(columns=["_orig_order_"])

    return d, best


def apply_hmm_model(df, model):
    """
    Apply a pre-fitted HMM to new data (causal filtering, no re-fitting).

    Use this at inference time — load the saved model and call this instead
    of add_soft_hmm_features so the HMM parameters are never updated on
    new data.

    Parameters
    ----------
    df    : DataFrame — same requirements as add_soft_hmm_features
    model : dict — output of add_soft_hmm_features or load_hmm_model

    Returns
    -------
    DataFrame (copy) with HMM_ALL_COLS added
    """
    required = ["symbol", "date", "ts_sec", "tick_count", "bid_p1", "ask_p1"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d = df.copy()
    d["_orig_order_"] = np.arange(len(d))
    d["date"]         = pd.to_datetime(d["date"]).dt.date
    d = d.sort_values(["symbol", "date", "ts_sec"]).reset_index(drop=True)

    d["mid"]       = (d["bid_p1"] + d["ask_p1"]) / 2.0
    d["x_bp"]      = d.groupby(["symbol", "date"], sort=False)["mid"].transform(
        lambda s: 10_000 * np.log(s).diff()
    )
    d["count_cat"] = np.minimum(d["tick_count"], 3).astype(np.int64)

    usable = d[d["x_bp"].notna()].copy()

    all_counts, all_returns, all_starts, all_ends = _sequence_arrays(usable)
    probs = _filter_probs(all_counts, all_returns, all_starts, all_ends, model)

    for col in HMM_ALL_COLS:
        d[col] = np.nan

    d.loc[usable.index, HMM_PROB_COLS]        = probs
    d.loc[usable.index, "state_filtered"]      = np.argmax(probs, axis=1) + 1
    d.loc[usable.index, "confidence_filtered"] = np.max(probs, axis=1)

    d = d.sort_values("_orig_order_").drop(columns=["_orig_order_"])
    return d


def save_hmm_model(model, path):
    """
    Save fitted HMM parameters to disk using pickle.

    Parameters
    ----------
    model : dict — output of add_soft_hmm_features
    path  : str or Path — destination file (e.g. models/hmm_usdinr.pkl)
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info("HMM model saved to %s", path)


def load_hmm_model(path):
    """
    Load a previously saved HMM model from disk.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    model : dict
    """
    with open(pathlib.Path(path), "rb") as f:
        model = pickle.load(f)
    log.info("HMM model loaded from %s", path)
    return model