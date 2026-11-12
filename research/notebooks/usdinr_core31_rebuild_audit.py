"""USDINR original core31 formula reconstruction — PARITY TEST ONLY.
Extracted verbatim from original user's core_features_usdinr notebook.
Requires 1-second regularized market snapshots with is_observed/had_update,
book fields, and timestamp. Does not read holdout automatically.
Do not deploy until exact frozen 31-column parity is verified.
"""
import numpy as np
import pandas as pd
import contextlib
import io

CORE31 = [
    "ofi_l1_sum_60s_signed_log", "ofi_rate_diff_30_300_sl", "ofi_activity_60s", "ofi_activity_300s",
    "update_rate_300s", "price_move_rate_30s", "impact_lambda_1800s", "impact_r2_1800s",
    "ofi_response_gap_z", "queue_imbalance_l1", "depth_imbalance_l5", "queue_replenishment_pressure_60s",
    "queue_imbalance_delta", "depth_imbalance_delta", "spread_rank_1800s", "price_mom_10s_new",
    "price_mom_30s_new", "price_mom_60s_new", "log_rv_300s_new", "rv_ratio_60_300_new",
    "ret_acf_lag1_300s", "log_vr_5_300s", "log_vr_30_300s", "semivar_imbalance_300s",
    "semivar_imbalance_900s", "sign_persistence_900s", "jump_ratio_300s", "range_position_300s",
    "spread_depth_stress_900", "recovery_available_300", "ofi_disagreement_weighted_60"
]

def build_core31_from_regularized(df_1s, quiet=True):
    """Rebuild original core 31; return symbol, ts_sec, and 31 computed columns.

    Input already regularized one-second within each symbol/date session.
    Note: preserves historical notebook's exact roll and warmup conventions.
    """
    df_1s = df_1s.copy()
    needed = ['symbol','ts_sec','bid_p1','ask_p1','bid_q1','ask_q1','had_update']
    missing = sorted(set(needed)-set(df_1s.columns))
    if missing: raise ValueError(f'Missing required input: {missing}')
    ts_ist = pd.to_datetime(df_1s['ts_sec'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
    df_1s['date'] = ts_ist.dt.date
    if df_1s.duplicated(['symbol','ts_sec']).any(): raise ValueError('duplicate keys')
    df_1s = df_1s.sort_values(['symbol','date','ts_sec']).reset_index(drop=True)
    eps = 1e-12
    _stdout = io.StringIO()
    with contextlib.redirect_stdout(_stdout) if quiet else contextlib.nullcontext():

        # Source notebook cell 41
        eps = 1e-12
        # ---------------------------------------------------------
        # MID PRICE
        #---------------------------------------------------------
        df_1s["mid"] = (df_1s["bid_p1"]+ df_1s["ask_p1"]) / 2.0
        df_1s["log_mid"] = np.log(df_1s["mid"])
        g = df_1s.groupby(["symbol", "date"],group_keys=False)
        # ---------------------------------------------------------
        # 1-SECOND LOG RETURN
        # ---------------------------------------------------------
        df_1s["ret_1s"] = (df_1s["log_mid"]- g["log_mid"].shift(1))

        # Source notebook cell 43
        # =========================================================
        # L1 OFI
        # =========================================================

        bid_p_prev = g["bid_p1"].shift(1)
        bid_q_prev = g["bid_q1"].shift(1)

        ask_p_prev = g["ask_p1"].shift(1)
        ask_q_prev = g["ask_q1"].shift(1)

        bid_of = (
            (df_1s["bid_p1"] >= bid_p_prev).astype(float)
            * df_1s["bid_q1"]
            -
            (df_1s["bid_p1"] <= bid_p_prev).astype(float)
            * bid_q_prev
        )

        ask_of = (
            -
            (df_1s["ask_p1"] <= ask_p_prev).astype(float)
            * df_1s["ask_q1"]
            +
            (df_1s["ask_p1"] >= ask_p_prev).astype(float)
            * ask_q_prev
        )

        df_1s["ofi_l1"] = (
            bid_of + ask_of
        )

        # Source notebook cell 45
        def signed_log(x):
            return np.sign(x) * np.log1p(np.abs(x))

        # Source notebook cell 47
        for h in [30, 60, 300]:

            raw_col = f"_ofi_sum_{h}s"

            df_1s[raw_col] = (
                df_1s
                .groupby(["symbol", "date"])["ofi_l1"]
                .transform(
                    lambda x: x.rolling(
                        h,
                        min_periods=max(1, h // 3)
                    ).sum()
                )
            )

        # Source notebook cell 48
        df_1s["ofi_l1_sum_60s_signed_log"] = (
            signed_log(
                df_1s["_ofi_sum_60s"]
            )
        )

        # Source notebook cell 49
        ofi_rate_diff = (
            df_1s["_ofi_sum_30s"] / 30.0
            -
            df_1s["_ofi_sum_300s"] / 300.0
        )

        df_1s["ofi_rate_diff_30_300_sl"] = (
            signed_log(ofi_rate_diff)
        )

        # Source notebook cell 51
        df_1s["_ofi_active"] = (
            df_1s["ofi_l1"].fillna(0) != 0
        ).astype(float)

        for h in [60, 300]:

            df_1s[f"ofi_activity_{h}s"] = (
                df_1s
                .groupby(["symbol", "date"])["_ofi_active"]
                .transform(
                    lambda x: x.rolling(
                        h,
                        min_periods=max(1, h // 3)
                    ).mean()
                )
            )

        # Source notebook cell 52
        df_1s["update_rate_300s"] = (
            df_1s
            .groupby(["symbol", "date"])["had_update"]
            .transform(
                lambda x: x.rolling(
                    300,
                    min_periods=100
                ).mean()
            )
        )

        # Source notebook cell 53
        df_1s["_price_moved"] = (
            df_1s["ret_1s"].fillna(0) != 0
        ).astype(float)

        df_1s["price_move_rate_30s"] = (
            df_1s
            .groupby(["symbol", "date"])["_price_moved"]
            .transform(
                lambda x: x.rolling(
                    30,
                    min_periods=10
                ).mean()
            )
        )

        # Source notebook cell 54
        df_1s["_ret_30s"] = (
            df_1s["log_mid"]
            - g["log_mid"].shift(30)
        )

        df_1s["_ofi_30_sl"] = (
            signed_log(
                df_1s["_ofi_sum_30s"]
            )
        )

        # Source notebook cell 55
        impact_window = 1800
        impact_min = 600

        def build_impact_features(day):

            x = day["_ofi_30_sl"]
            y = day["_ret_30s"]

            mx = x.rolling(
                impact_window,
                min_periods=impact_min
            ).mean()

            my = y.rolling(
                impact_window,
                min_periods=impact_min
            ).mean()

            mxy = (x * y).rolling(
                impact_window,
                min_periods=impact_min
            ).mean()

            mx2 = (x * x).rolling(
                impact_window,
                min_periods=impact_min
            ).mean()

            my2 = (y * y).rolling(
                impact_window,
                min_periods=impact_min
            ).mean()

            cov_xy = mxy - mx * my
            var_x = mx2 - mx * mx
            var_y = my2 - my * my

            lam = cov_xy / (var_x + eps)

            alpha = (
                my
                - lam * mx
            )

            r2 = (
                cov_xy ** 2
                /
                (
                    (var_x + eps)
                    * (var_y + eps)
                )
            )

            # Parameters used at t must come from t-1
            day["impact_lambda_1800s"] = (
                lam.shift(1)
            )

            day["_impact_alpha_1800s"] = (
                alpha.shift(1)
            )

            day["impact_r2_1800s"] = (
                r2.shift(1)
            )

            predicted = (
                day["_impact_alpha_1800s"]
                +
                day["impact_lambda_1800s"]
                * x
            )

            resid = (
                y - predicted
            )

            # use historical residual volatility
            resid_sigma = (
                resid.shift(1)
                .rolling(
                    impact_window,
                    min_periods=impact_min
                )
                .std()
            )

            day["ofi_response_gap_z"] = (
                -np.sign(x)
                * resid
                /
                (resid_sigma + eps)
            )

            return day


        df_1s = (
            df_1s
            .groupby(
                ["symbol", "date"],
                group_keys=False
            )
            .apply(build_impact_features)
        )

        # Source notebook cell 57
        df_1s["queue_imbalance_l1"] = (
            (
                df_1s["bid_q1"]
                -
                df_1s["ask_q1"]
            )
            /
            (
                df_1s["bid_q1"]
                +
                df_1s["ask_q1"]
                +
                eps
            )
        )

        # Source notebook cell 58
        bid_depth_cols = [
            f"bid_q{i}"
            for i in range(1, 6)
        ]

        ask_depth_cols = [
            f"ask_q{i}"
            for i in range(1, 6)
        ]

        df_1s["_bid_depth_l5"] = (
            df_1s[bid_depth_cols].sum(axis=1)
        )

        df_1s["_ask_depth_l5"] = (
            df_1s[ask_depth_cols].sum(axis=1)
        )

        df_1s["depth_imbalance_l5"] = (
            (
                df_1s["_bid_depth_l5"]
                -
                df_1s["_ask_depth_l5"]
            )
            /
            (
                df_1s["_bid_depth_l5"]
                +
                df_1s["_ask_depth_l5"]
                +
                eps
            )
        )

        # Source notebook cell 59
        bid_same = (
            df_1s["bid_p1"]
            == bid_p_prev
        )

        ask_same = (
            df_1s["ask_p1"]
            == ask_p_prev
        )

        bid_replenish = np.where(
            bid_same,
            np.maximum(
                df_1s["bid_q1"]
                - bid_q_prev,
                0
            )
            /
            (bid_q_prev + eps),
            0.0
        )

        ask_replenish = np.where(
            ask_same,
            np.maximum(
                df_1s["ask_q1"]
                - ask_q_prev,
                0
            )
            /
            (ask_q_prev + eps),
            0.0
        )

        df_1s["_queue_replenishment_pressure"] = (
            np.log1p(bid_replenish)
            -
            np.log1p(ask_replenish)
        )

        df_1s["queue_replenishment_pressure_60s"] = (
            df_1s
            .groupby(["symbol", "date"])[
                "_queue_replenishment_pressure"
            ]
            .transform(
                lambda x: x.rolling(
                    60,
                    min_periods=20
                ).sum()
            )
        )

        # Source notebook cell 61
        df_1s["queue_imbalance_delta"] = (
            df_1s["queue_imbalance_l1"]
            -
            df_1s.groupby(
                ["symbol", "date"]
            )["queue_imbalance_l1"].shift(1)
        )

        df_1s["depth_imbalance_delta"] = (
            df_1s["depth_imbalance_l5"]
            -
            df_1s.groupby(
                ["symbol", "date"]
            )["depth_imbalance_l5"].shift(1)
        )

        # Source notebook cell 63
        df_1s["_spread_bps"] = (
            1e4
            *
            (
                df_1s["ask_p1"]
                -
                df_1s["bid_p1"]
            )
            /
            (df_1s["mid"] + eps)
        )

        # Source notebook cell 64
        df_1s["spread_rank_1800s"] = (
            df_1s
            .groupby(["symbol", "date"])["_spread_bps"]
            .transform(
                lambda x: x.rolling(
                    1800,
                    min_periods=300
                ).rank(
                    pct=True
                )
            )
        )

        # Source notebook cell 68
        for h in [10, 30, 60]:

            df_1s[f"price_mom_{h}s_new"] = (
                df_1s["log_mid"]
                -
                df_1s
                .groupby(["symbol", "date"])["log_mid"]
                .shift(h)
            )

        # Source notebook cell 70
        df_1s["_ret_sq"] = (
            df_1s["ret_1s"] ** 2
        )

        for h in [60, 300]:

            df_1s[f"_rv_{h}s"] = (
                df_1s
                .groupby(["symbol", "date"])["_ret_sq"]
                .transform(
                    lambda x: np.sqrt(
                        x.rolling(
                            h,
                            min_periods=max(
                                2,
                                h // 3
                            )
                        ).sum()
                    )
                )
            )

        # Source notebook cell 71
        df_1s["log_rv_300s_new"] = (
            np.log1p(
                df_1s["_rv_300s"]
            )
        )

        # Source notebook cell 72
        df_1s["rv_ratio_60_300_new"] = (
            df_1s["_rv_60s"]
            /
            (
                df_1s["_rv_300s"]
                + eps
            )
        )

        # Source notebook cell 74
        df_1s["_ret_lag1"] = (
            df_1s
            .groupby(["symbol", "date"])["ret_1s"]
            .shift(1)
        )

        df_1s["ret_acf_lag1_300s"] = (
            df_1s
            .groupby(
                ["symbol", "date"],
                group_keys=False
            )
            .apply(
                lambda day:
                    day["ret_1s"]
                    .rolling(
                        300,
                        min_periods=120
                    )
                    .corr(
                        day["_ret_lag1"]
                    )
            )
        )

        # Source notebook cell 76
        for q in [5, 30]:

            df_1s[f"_ret_sum_{q}"] = (
                df_1s
                .groupby(["symbol", "date"])["ret_1s"]
                .transform(
                    lambda x: x.rolling(
                        q,
                        min_periods=q
                    ).sum()
                )
            )

            var_1 = (
                df_1s
                .groupby(["symbol", "date"])["ret_1s"]
                .transform(
                    lambda x: x.rolling(
                        300,
                        min_periods=120
                    ).var()
                )
            )

            var_q = (
                df_1s
                .groupby(["symbol", "date"])[
                    f"_ret_sum_{q}"
                ]
                .transform(
                    lambda x: x.rolling(
                        300,
                        min_periods=120
                    ).var()
                )
            )

            vr = (
                var_q
                /
                (
                    q * var_1
                    + eps
                )
            )

            # Log compression.
            # No global quantile clipping here -> avoids future leakage.
            df_1s[f"log_vr_{q}_300s"] = (
                np.log(
                    np.maximum(
                        vr,
                        1e-6
                    )
                )
            )

        # Source notebook cell 77
        df_1s["_ret_sq_pos"] = np.where(
            df_1s["ret_1s"] > 0,
            df_1s["ret_1s"] ** 2,
            0.0
        )

        df_1s["_ret_sq_neg"] = np.where(
            df_1s["ret_1s"] < 0,
            df_1s["ret_1s"] ** 2,
            0.0
        )

        for w in [300, 900]:

            up = (
                df_1s
                .groupby(["symbol", "date"])["_ret_sq_pos"]
                .transform(
                    lambda x: x.rolling(
                        w,
                        min_periods=max(
                            120,
                            w // 3
                        )
                    ).sum()
                )
            )

            down = (
                df_1s
                .groupby(["symbol", "date"])["_ret_sq_neg"]
                .transform(
                    lambda x: x.rolling(
                        w,
                        min_periods=max(
                            120,
                            w // 3
                        )
                    ).sum()
                )
            )

            df_1s[
                f"semivar_imbalance_{w}s"
            ] = (
                (up - down)
                /
                (
                    up + down
                    + eps
                )
            )

        # Source notebook cell 81
        df_1s["_move_sign"] = np.nan

        df_1s.loc[
            df_1s["ret_1s"] > 0,
            "_move_sign"
        ] = 1.0

        df_1s.loc[
            df_1s["ret_1s"] < 0,
            "_move_sign"
        ] = -1.0


        def add_move_transition_features(day):

            move_sign = day["_move_sign"]

            # Most recent nonzero move sign
            previous_nonzero_sign = (
                move_sign
                .ffill()
                .shift(1)
            )

            valid_transition = (
                move_sign.notna()
                &
                previous_nonzero_sign.notna()
            )

            same_transition = (
                valid_transition
                &
                (
                    move_sign
                    == previous_nonzero_sign
                )
            )

            day["_valid_move_transition"] = (
                valid_transition.astype(float)
            )

            day["_same_move_transition"] = (
                same_transition.astype(float)
            )

            return day


        df_1s = (
            df_1s
            .groupby(
                ["symbol", "date"],
                group_keys=False
            )
            .apply(
                add_move_transition_features
            )
        )


        transition_count = (
            df_1s
            .groupby(
                ["symbol", "date"]
            )["_valid_move_transition"]
            .transform(
                lambda x: x.rolling(
                    900,
                    min_periods=300
                ).sum()
            )
        )

        same_count = (
            df_1s
            .groupby(
                ["symbol", "date"]
            )["_same_move_transition"]
            .transform(
                lambda x: x.rolling(
                    900,
                    min_periods=300
                ).sum()
            )
        )

        df_1s["sign_persistence_900s"] = (
            same_count
            /
            (
                transition_count
                + eps
            )
        )

        df_1s.loc[
            transition_count < 20,
            "sign_persistence_900s"
        ] = np.nan

        # Source notebook cell 83
        df_1s["_abs_ret"] = (
            np.abs(
                df_1s["ret_1s"]
            )
        )

        df_1s["_abs_ret_lag1"] = (
            df_1s
            .groupby(
                ["symbol", "date"]
            )["_abs_ret"]
            .shift(1)
        )

        df_1s["_bipower_component"] = (
            df_1s["_abs_ret"]
            *
            df_1s["_abs_ret_lag1"]
        )

        jump_rv = (
            df_1s
            .groupby(["symbol", "date"])["_ret_sq"]
            .transform(
                lambda x: x.rolling(
                    300,
                    min_periods=120
                ).sum()
            )
        )

        jump_bv = (
            np.pi / 2.0
            *
            df_1s
            .groupby(
                ["symbol", "date"]
            )["_bipower_component"]
            .transform(
                lambda x: x.rolling(
                    300,
                    min_periods=120
                ).sum()
            )
        )

        jump_var = (
            jump_rv - jump_bv
        ).clip(
            lower=0
        )

        df_1s["jump_ratio_300s"] = (
            jump_var
            /
            (
                jump_rv
                + eps
            )
        )

        # Source notebook cell 85
        rolling_high_300 = (
            df_1s
            .groupby(["symbol", "date"])["mid"]
            .transform(
                lambda x: x.rolling(
                    300,
                    min_periods=120
                ).max()
            )
        )

        rolling_low_300 = (
            df_1s
            .groupby(["symbol", "date"])["mid"]
            .transform(
                lambda x: x.rolling(
                    300,
                    min_periods=120
                ).min()
            )
        )

        df_1s["range_position_300s"] = (
            (
                df_1s["mid"]
                - rolling_low_300
            )
            /
            (
                rolling_high_300
                - rolling_low_300
                + eps
            )
        )

        # Source notebook cell 88
        import numpy as np
        import pandas as pd

        # ============================================================
        # FEATURE 29 — SPREAD-DEPTH STRESS (900)
        #
        # Research reference:
        #   Calendar experiments 21A–21G
        #
        # Selected specification:
        #   Lookback: 900 observations
        #   Minimum valid observations: 810
        #   Current best bid/ask spread in bps
        #   Trailing median five-level displayed depth
        #   Square-root depth normalization
        #   log1p transformation
        #
        # Output:
        #   df_1s["spread_depth_stress_900"]
        #
        # No forward-looking data.
        # No modification to existing engineered features.
        # ============================================================

        WINDOW = 900
        MIN_PERIODS = 810

        required = [
            "symbol",
            "date",
            "ts_sec",
            "bid_p1",
            "ask_p1",
            "bid_q1",
            "bid_q2",
            "bid_q3",
            "bid_q4",
            "bid_q5",
            "ask_q1",
            "ask_q2",
            "ask_q3",
            "ask_q4",
            "ask_q5"
        ]

        missing = [
            col for col in required
            if col not in df_1s.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        if not df_1s.columns.is_unique:
            raise ValueError("Duplicate column names detected.")

        if df_1s.duplicated(
            ["symbol", "ts_sec"]
        ).any():
            raise ValueError(
                "Duplicate symbol/timestamp records detected."
            )

        # ------------------------------------------------------------
        # 1. Build an independent calculation frame.
        #
        # This allows us to calculate the feature without
        # changing other columns in the original df_1s.
        # ------------------------------------------------------------

        x_stress = df_1s[required].copy()

        x_stress["ts_sec"] = pd.to_numeric(
            x_stress["ts_sec"], errors="raise"
        )

        x_stress = x_stress.sort_values(
            ["symbol", "date", "ts_sec"]
        )

        # ------------------------------------------------------------
        # 2. Reconstruct current quoted spread (bps).
        # ------------------------------------------------------------

        bid = pd.to_numeric(
            x_stress["bid_p1"], errors="coerce"
        )

        ask = pd.to_numeric(
            x_stress["ask_p1"], errors="coerce"
        )

        mid = (bid + ask) / 2.0

        valid_quote = (
            (bid > 0)
            & (ask > 0)
            & (ask >= bid)
            & np.isfinite(mid)
        )

        spread_bps = 10000.0 * (ask - bid) / mid

        x_stress["_stress_spread_bps"] = (
            spread_bps.where(valid_quote)
        )

        # ------------------------------------------------------------
        # 3. Calculate five-level displayed book depth.
        # ------------------------------------------------------------

        qty_cols = [
            "bid_q1",
            "bid_q2",
            "bid_q3",
            "bid_q4",
            "bid_q5",
            "ask_q1",
            "ask_q2",
            "ask_q3",
            "ask_q4",
            "ask_q5"
        ]

        quantities = x_stress[qty_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        valid_depth = (
            quantities.notna().all(axis=1)
            & np.isfinite(quantities).all(axis=1)
            & (quantities >= 0).all(axis=1)
        )

        x_stress["_stress_depth_l5"] = (
            quantities.sum(axis=1).where(valid_depth)
        )

        # ------------------------------------------------------------
        # 4. Trailing 900-observation median depth.
        #
        # Same symbol and same session only.
        # Current observation included.
        #
        # Matches the research construction.
        # ------------------------------------------------------------

        x_stress["_stress_depth_med_900"] = (
            x_stress
            .groupby(
                ["symbol", "date"],
                sort=False
            )["_stress_depth_l5"]
            .transform(
                lambda s: s.rolling(
                    window=WINDOW,
                    min_periods=MIN_PERIODS
                ).median()
            )
        )

        # ------------------------------------------------------------
        # 5. Final feature.
        # ------------------------------------------------------------

        safe_spread = (
            x_stress["_stress_spread_bps"]
            .clip(lower=0)
        )

        safe_depth = (
            x_stress["_stress_depth_med_900"]
            .clip(lower=0)
        )

        x_stress["spread_depth_stress_900"] = np.log1p(
            safe_spread / np.sqrt(safe_depth + 1.0)
        )

        # ------------------------------------------------------------
        # 6. Restore original row alignment.
        #
        # No master sorting or merge needed.
        # ------------------------------------------------------------

        new_feature = (
            x_stress["spread_depth_stress_900"]
            .reindex(df_1s.index)
        )

        if not df_1s.index.is_unique:
            raise ValueError(
                "Original df_1s index is not unique; "
                "use a key-based assignment instead."
            )

        # Add only the new feature.
        df_1s["spread_depth_stress_900"] = new_feature

        # ------------------------------------------------------------
        # 7. Validation.
        # ------------------------------------------------------------

        feature = df_1s["spread_depth_stress_900"]

        if np.isinf(feature.to_numpy(dtype=float)).any():
            raise ValueError(
                "Infinite values detected in new feature."
            )

        if (feature.dropna() < 0).any():
            raise ValueError(
                "Unexpected negative liquidity-stress values."
            )

        # Compare with calculation output.
        np.testing.assert_allclose(
            feature.to_numpy(dtype=float),
            new_feature.to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
            equal_nan=True
        )

        print("\n===== SPREAD-DEPTH STRESS 900 =====")

        print("Lookback observations:", WINDOW)
        print("Minimum valid observations:", MIN_PERIODS)
        print("Total rows:", len(feature))
        print("Valid rows:", feature.notna().sum())
        print(
            "Missing percentage:",
            round(100 * feature.isna().mean(), 4)
        )

        print("\n===== DISTRIBUTION =====")

        print(
            feature.describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99
                ]
            )
        )

        print("\n===== FEATURE VALIDATION =====")
        print("Infinite values: 0")
        print("Existing features: unchanged")
        print("Look-ahead leakage: none in construction")
        print("New feature: spread_depth_stress_900")
        print("Status: experimental / pending final OOS validation")

        # Source notebook cell 89
        import numpy as np
        import pandas as pd

        # ============================================================
        # FEATURE 30 — RECOVERY AVAILABILITY (300)
        #
        # Research reference:
        #   Experiments 22E–22I
        #
        # Definition:
        #   300-observation bid/ask recovery eligibility flag
        #
        # Parameters:
        #   Reference depth window = 300 observations
        #   Reference minimum observations = 270
        #   Depletion threshold = 0.75
        #   Availability window = 300 observations
        #   Minimum valid transitions = 270
        #   Minimum depleted transitions per side = 5
        #
        # Output:
        #   df_1s["recovery_available_300"]
        #
        # Uses current and historical observations only.
        # No change to existing feature columns.
        # ============================================================

        REFERENCE_WINDOW = 300
        REFERENCE_MIN = 270

        RECOVERY_WINDOW = 300
        MIN_VALID_TRANSITIONS = 270
        MIN_DEPLETED_EVENTS = 5
        DEPLETION_THRESHOLD = 0.75

        required = [
            "symbol",
            "ts_sec",
            "bid_q1", "bid_q2", "bid_q3", "bid_q4", "bid_q5",
            "ask_q1", "ask_q2", "ask_q3", "ask_q4", "ask_q5"
        ]

        missing = [
            c for c in required
            if c not in df_1s.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required fields: {missing}"
            )

        if not df_1s.columns.is_unique:
            raise ValueError("Duplicate column names found.")

        # Independent calculation frame.
        work = df_1s[required].copy()

        work["_original_position"] = np.arange(len(work))

        work["ts_sec"] = pd.to_numeric(
            work["ts_sec"], errors="raise"
        )

        if work.duplicated(["symbol", "ts_sec"]).any():
            raise ValueError(
                "Duplicate symbol/timestamp observations."
            )

        # Derive IST date directly from timestamps, matching 22E.
        ts_ist = pd.to_datetime(
            work["ts_sec"], unit="s", utc=True
        ).dt.tz_convert("Asia/Kolkata")

        work["_session_date"] = ts_ist.dt.date

        work = work.sort_values(
            ["symbol", "_session_date", "ts_sec"]
        ).reset_index(drop=True)

        GROUP = ["symbol", "_session_date"]

        # ------------------------------------------------------------
        # 1. Compute side-specific L5 displayed depth
        # ------------------------------------------------------------

        for side in ["bid", "ask"]:

            cols = [
                f"{side}_q{i}"
                for i in range(1, 6)
            ]

            q = work[cols].apply(
                pd.to_numeric, errors="coerce"
            ).replace(
                [np.inf, -np.inf], np.nan
            )

            valid_depth = (
                q.notna().all(axis=1)
                & q.ge(0).all(axis=1)
            )

            work[f"_{side}_depth"] = (
                q.sum(axis=1).where(valid_depth)
            )

        # ------------------------------------------------------------
        # 2. Causal previous-300-observation median reference
        # ------------------------------------------------------------

        for side in ["bid", "ask"]:

            depth_col = f"_{side}_depth"
            reference_col = f"_{side}_reference"
            relative_col = f"_{side}_relative"

            work[reference_col] = (
                work.groupby(GROUP, sort=False)[depth_col]
                .transform(
                    lambda s: s.shift(1).rolling(
                        window=REFERENCE_WINDOW,
                        min_periods=REFERENCE_MIN
                    ).median()
                )
            )

            work[relative_col] = (
                work[depth_col]
                / work[reference_col].where(
                    work[reference_col] > 0
                )
            )

        # ------------------------------------------------------------
        # 3. Identify valid consecutive-second transitions
        # ------------------------------------------------------------

        previous_ts = (
            work.groupby(GROUP, sort=False)["ts_sec"]
            .shift(1)
        )

        consecutive = (
            work["ts_sec"] - previous_ts
        ).eq(1)

        for side in ["bid", "ask"]:

            relative_col = f"_{side}_relative"

            current = work[relative_col]

            previous = (
                work.groupby(GROUP, sort=False)[relative_col]
                .shift(1)
            )

            valid_pair = (
                consecutive
                & np.isfinite(current)
                & np.isfinite(previous)
            )

            depleted_before = (
                previous < DEPLETION_THRESHOLD
            )

            work[f"_{side}_valid"] = (
                valid_pair.astype(np.float64)
            )

            work[f"_{side}_depleted"] = (
                (valid_pair & depleted_before)
                .astype(np.float64)
            )

        # ------------------------------------------------------------
        # 4. Count valid and depleted transitions over 300 rows
        # ------------------------------------------------------------

        for side in ["bid", "ask"]:

            grouped = work.groupby(GROUP, sort=False)

            work[f"_{side}_valid_count"] = (
                grouped[f"_{side}_valid"].transform(
                    lambda s: s.rolling(
                        window=RECOVERY_WINDOW,
                        min_periods=1
                    ).sum()
                )
            )

            work[f"_{side}_event_count"] = (
                grouped[f"_{side}_depleted"].transform(
                    lambda s: s.rolling(
                        window=RECOVERY_WINDOW,
                        min_periods=1
                    ).sum()
                )
            )

            work[f"_{side}_eligible"] = (
                work[f"_{side}_valid_count"].ge(
                    MIN_VALID_TRANSITIONS
                )
                &
                work[f"_{side}_event_count"].ge(
                    MIN_DEPLETED_EVENTS
                )
            )

        # ------------------------------------------------------------
        # 5. Final binary feature
        # ------------------------------------------------------------

        work["recovery_available_300"] = (
            work["_bid_eligible"]
            & work["_ask_eligible"]
        ).astype(np.int8)

        # ------------------------------------------------------------
        # 6. Restore original row order safely
        # ------------------------------------------------------------

        work = work.sort_values(
            "_original_position"
        )

        feature_values = (
            work["recovery_available_300"]
            .to_numpy(dtype=np.int8)
        )

        assert len(feature_values) == len(df_1s)

        # Assign positionally; works with non-unique dataframe index.
        df_1s["recovery_available_300"] = feature_values

        # ------------------------------------------------------------
        # 7. Health checks
        # ------------------------------------------------------------

        feature = df_1s["recovery_available_300"]

        if not feature.isin([0, 1]).all():
            raise ValueError(
                "Recovery availability must be binary."
            )

        print("\n===== RECOVERY AVAILABILITY 300 =====")

        print("Total rows:", len(feature))
        print("Flagged 1:", int((feature == 1).sum()))
        print("Flagged 0:", int((feature == 0).sum()))

        print("\n===== DISTRIBUTION =====")

        print(
            feature.value_counts(
                normalize=True
            ).mul(100).round(4)
        )

        print("\n===== FEATURE VALIDATION =====")

        print("Unique values:", sorted(feature.unique()))
        print("Missing values:", int(feature.isna().sum()))
        print("Current/future leakage: none in construction")
        print("Existing feature columns unchanged")
        print("Status: provisional / pending full OOS validation")

        # Source notebook cell 90
        import numpy as np
        import pandas as pd

        # ============================================================
        # FEATURE 31 — OFI / PRICE WEIGHTED DISAGREEMENT (60s)
        #
        # Research: Experiments 23A–23D
        #
        # Output:
        #   df_1s["ofi_disagreement_weighted_60"]
        #
        # Specification:
        #   Quote-based Level-1 OFI
        #   Lookback = 60 observations
        #   Minimum valid increments = 54
        #   Exact 60-second historical momentum
        #   Trailing 60-observation realized volatility
        #   Volatility floor = 0.1 bps
        #   Maximum relative-move weight = 3.0
        #
        # No future data.
        # No change to existing feature columns.
        # ============================================================

        WINDOW_31 = 60
        MIN_PERIODS_31 = 54
        VOL_FLOOR_BPS_31 = 0.1
        MAX_WEIGHT_31 = 3.0

        OUTPUT_31 = "ofi_disagreement_weighted_60"

        required_31 = [
            "symbol",
            "ts_sec",
            "bid_p1",
            "ask_p1",
            "bid_q1",
            "ask_q1",
        ]

        missing_31 = [
            c for c in required_31
            if c not in df_1s.columns
        ]

        if missing_31:
            raise ValueError(
                f"Missing columns: {missing_31}"
            )

        if not df_1s.columns.is_unique:
            raise ValueError("Duplicate dataframe columns.")

        # ------------------------------------------------------------
        # 1. Independent working frame, positional row identifier
        # ------------------------------------------------------------

        work_31 = df_1s[required_31].copy()

        work_31["_original_position"] = np.arange(
            len(work_31)
        )

        for c in [
            "ts_sec",
            "bid_p1",
            "ask_p1",
            "bid_q1",
            "ask_q1",
        ]:
            work_31[c] = pd.to_numeric(
                work_31[c],
                errors="coerce"
            ).replace(
                [np.inf, -np.inf], np.nan
            )

        if work_31["ts_sec"].isna().any():
            raise ValueError("Invalid ts_sec values.")

        if work_31.duplicated(
            ["symbol", "ts_sec"]
        ).any():
            raise ValueError(
                "Duplicate symbol/timestamp records."
            )

        ts_ist_31 = pd.to_datetime(
            work_31["ts_sec"],
            unit="s",
            utc=True
        ).dt.tz_convert("Asia/Kolkata")

        work_31["_session_date"] = ts_ist_31.dt.date

        work_31 = work_31.sort_values(
            ["symbol", "_session_date", "ts_sec"]
        ).reset_index(drop=True)

        GROUP_31 = ["symbol", "_session_date"]

        # ------------------------------------------------------------
        # 2. Validate current Level-1 quotes
        # ------------------------------------------------------------

        bid_p = work_31["bid_p1"]
        ask_p = work_31["ask_p1"]
        bid_q = work_31["bid_q1"]
        ask_q = work_31["ask_q1"]

        valid_quote_31 = (
            bid_p.gt(0)
            & ask_p.gt(0)
            & ask_p.ge(bid_p)
            & bid_q.ge(0)
            & ask_q.ge(0)
            & np.isfinite(bid_p)
            & np.isfinite(ask_p)
            & np.isfinite(bid_q)
            & np.isfinite(ask_q)
        )

        work_31["_mid"] = (
            (bid_p + ask_p) / 2
        ).where(valid_quote_31)

        # ------------------------------------------------------------
        # 3. Previous Level-1 quote
        # ------------------------------------------------------------

        grouped_31 = work_31.groupby(
            GROUP_31,
            sort=False
        )

        prev_ts = grouped_31["ts_sec"].shift(1)
        prev_bid_p = grouped_31["bid_p1"].shift(1)
        prev_ask_p = grouped_31["ask_p1"].shift(1)
        prev_bid_q = grouped_31["bid_q1"].shift(1)
        prev_ask_q = grouped_31["ask_q1"].shift(1)

        valid_prev_31 = (
            prev_bid_p.gt(0)
            & prev_ask_p.gt(0)
            & prev_ask_p.ge(prev_bid_p)
            & prev_bid_q.ge(0)
            & prev_ask_q.ge(0)
            & np.isfinite(prev_bid_p)
            & np.isfinite(prev_ask_p)
            & np.isfinite(prev_bid_q)
            & np.isfinite(prev_ask_q)
        )

        valid_pair_31 = (
            valid_quote_31
            & valid_prev_31
            & (work_31["ts_sec"] - prev_ts).eq(1)
        )

        # ------------------------------------------------------------
        # 4. Level-1 quote-based OFI increment
        # ------------------------------------------------------------

        bid_contribution_31 = (
            (bid_p >= prev_bid_p).astype(float) * bid_q
            - (bid_p <= prev_bid_p).astype(float) * prev_bid_q
        )

        ask_contribution_31 = (
            -(ask_p <= prev_ask_p).astype(float) * ask_q
            + (ask_p >= prev_ask_p).astype(float) * prev_ask_q
        )

        work_31["_ofi"] = (
            bid_contribution_31 + ask_contribution_31
        ).where(valid_pair_31)

        work_31["_abs_ofi"] = work_31["_ofi"].abs()

        # ------------------------------------------------------------
        # 5. Rolling 60-observation normalized OFI pressure
        # ------------------------------------------------------------

        grouped_31 = work_31.groupby(
            GROUP_31,
            sort=False
        )

        work_31["_ofi_sum_60"] = (
            grouped_31["_ofi"].transform(
                lambda s: s.rolling(
                    window=WINDOW_31,
                    min_periods=MIN_PERIODS_31
                ).sum()
            )
        )

        work_31["_ofi_gross_60"] = (
            grouped_31["_abs_ofi"].transform(
                lambda s: s.rolling(
                    window=WINDOW_31,
                    min_periods=MIN_PERIODS_31
                ).sum()
            )
        )

        gross = work_31["_ofi_gross_60"]
        net = work_31["_ofi_sum_60"]

        work_31["_pressure_60"] = (
            net / gross.where(gross > 0)
        )

        # Observed zero-flow window -> pressure = 0.
        # Insufficient history stays missing.
        zero_flow = gross.eq(0) & net.notna()

        work_31.loc[
            zero_flow,
            "_pressure_60"
        ] = 0.0

        # ------------------------------------------------------------
        # 6. Exact 60-second historical momentum
        # ------------------------------------------------------------

        grouped_31 = work_31.groupby(
            GROUP_31,
            sort=False
        )

        old_mid = grouped_31["_mid"].shift(WINDOW_31)
        old_ts = grouped_31["ts_sec"].shift(WINDOW_31)

        valid_historical_endpoint = (
            (work_31["ts_sec"] - old_ts).eq(WINDOW_31)
            & work_31["_mid"].gt(0)
            & old_mid.gt(0)
        )

        work_31["_momentum_60"] = (
            1e4 * np.log(
                work_31["_mid"] / old_mid
            )
        ).where(valid_historical_endpoint)

        # ------------------------------------------------------------
        # 7. Trailing 60-observation realized volatility
        # ------------------------------------------------------------

        previous_mid = grouped_31["_mid"].shift(1)

        one_sec_log_return = np.log(
            work_31["_mid"] / previous_mid
        ).where(valid_pair_31)

        work_31["_ret_sq"] = (
            one_sec_log_return ** 2
        )

        work_31["_rv_60"] = (
            work_31.groupby(
                GROUP_31,
                sort=False
            )["_ret_sq"]
            .transform(
                lambda s: s.rolling(
                    window=WINDOW_31,
                    min_periods=MIN_PERIODS_31
                ).sum()
            )
            .clip(lower=0)
            .pow(0.5)
            * 1e4
        )

        # ------------------------------------------------------------
        # 8. Construct directional disagreement
        # ------------------------------------------------------------

        pressure = work_31["_pressure_60"]
        momentum = work_31["_momentum_60"]
        rv = work_31["_rv_60"]

        valid_components = (
            np.isfinite(pressure)
            & np.isfinite(momentum)
            & np.isfinite(rv)
        )

        disagreement = (
            (pressure * momentum) < 0
        )

        relative_move_weight = (
            momentum.abs()
            / rv.clip(lower=VOL_FLOOR_BPS_31)
        ).clip(upper=MAX_WEIGHT_31)

        work_31[OUTPUT_31] = (
            (pressure * relative_move_weight)
            .where(disagreement, 0.0)
            .where(valid_components)
        )

        # ------------------------------------------------------------
        # 9. Restore original row order
        # ------------------------------------------------------------

        work_31 = work_31.sort_values(
            "_original_position"
        )

        feature_values_31 = (
            work_31[OUTPUT_31]
            .to_numpy(dtype=float)
        )

        if len(feature_values_31) != len(df_1s):
            raise ValueError("Unexpected feature length.")

        # Only new output column is assigned to df_1s.
        df_1s[OUTPUT_31] = feature_values_31

        # ------------------------------------------------------------
        # 10. Feature validation
        # ------------------------------------------------------------

        feature_31 = df_1s[OUTPUT_31]

        if np.isinf(feature_31.to_numpy()).any():
            raise ValueError("Infinite values detected.")

        if (feature_31.abs().dropna() > MAX_WEIGHT_31 + 1e-9).any():
            raise ValueError(
                "Unexpected value outside theoretical range."
            )

        print("\n===== FEATURE 31 — OFI DISAGREEMENT =====")

        print("Feature:", OUTPUT_31)
        print("Total rows:", len(feature_31))
        print("Valid rows:", int(feature_31.notna().sum()))
        print(
            "Valid percentage:",
            round(feature_31.notna().mean() * 100, 4)
        )
        print(
            "Zero percentage:",
            round(feature_31.eq(0).mean() * 100, 4)
        )

        print("\n===== DISTRIBUTION =====")

        print(
            feature_31.describe(
                percentiles=[
                    0.01, 0.05, 0.25,
                    0.50, 0.75, 0.95, 0.99
                ]
            )
        )

        print("\n===== VALIDATION =====")
        print("Infinite values: 0")
        print("Theoretical range: [-3, 3]")
        print("Session boundaries: respected")
        print("Exact historical momentum endpoints: checked")
        print("Original features: unchanged")
        print("Status: experimental, pending future validation")

    missing_outputs = sorted(set(CORE31)-set(df_1s.columns))
    if missing_outputs: raise ValueError(f'Unbuilt outputs: {missing_outputs}')
    return df_1s[['symbol','ts_sec'] + CORE31].copy()
