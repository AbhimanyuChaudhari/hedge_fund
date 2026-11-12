"""
Canonical feature lists for USDINR feature pipeline.
All other files import from here — never define these lists elsewhere.
Bump FEATURE_VERSION when any formula changes.
"""

FEATURE_VERSION = "v1"

CORE31 = [
    "ofi_l1_sum_60s_signed_log",
    "ofi_rate_diff_30_300_sl",
    "ofi_activity_60s",
    "ofi_activity_300s",
    "update_rate_300s",
    "price_move_rate_30s",
    "impact_lambda_1800s",
    "impact_r2_1800s",
    "ofi_response_gap_z",
    "queue_imbalance_l1",
    "depth_imbalance_l5",
    "queue_replenishment_pressure_60s",
    "queue_imbalance_delta",
    "depth_imbalance_delta",
    "spread_rank_1800s",
    "price_mom_10s_new",
    "price_mom_30s_new",
    "price_mom_60s_new",
    "log_rv_300s_new",
    "rv_ratio_60_300_new",
    "ret_acf_lag1_300s",
    "log_vr_5_300s",
    "log_vr_30_300s",
    "semivar_imbalance_300s",
    "semivar_imbalance_900s",
    "sign_persistence_900s",
    "jump_ratio_300s",
    "range_position_300s",
    "spread_depth_stress_900",
    "recovery_available_300",
    "ofi_disagreement_weighted_60",
]

ADDED6 = [
    "quote_retreat_imbalance_30",
    "slope_asymmetry",
    "curvature_asymmetry",
    "weighted_gap_asymmetry",
    "fragility_asymmetry",
    "relative_migration_60",
]

CONTROLS9 = [
    "queue_imbalance_ctrl",
    "depth_imbalance_ctrl",
    "log_total_depth_ctrl",
    "spread_bps_ctrl",
    "bid_l1_concentration_ctrl",
    "ask_l1_concentration_ctrl",
    "momentum_30_ctrl",
    "momentum_60_ctrl",
    "clock_seconds",
]

CHALLENGERS4 = [
    "recovery_pressure",
    "ask_shock_age",
    "side_update_imbalance_60",
    "migration_activity_asym_60",
]

BASE46     = CORE31 + ADDED6 + CONTROLS9
CHALLENGER50 = BASE46 + CHALLENGERS4

# Required raw columns for the builder
REQUIRED_RAW = (
    ["symbol", "ts_sec"]
    + [f"bid_p{i}" for i in range(1, 6)]
    + [f"ask_p{i}" for i in range(1, 6)]
    + [f"bid_q{i}" for i in range(1, 6)]
    + [f"ask_q{i}" for i in range(1, 6)]
    + ["is_observed", "had_update"]
)