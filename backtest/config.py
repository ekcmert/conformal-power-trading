from __future__ import annotations

from pathlib import Path

from base_models.point_estimation.common import REPO_ROOT, target_artifact_stem


DATE_COLUMN = "date"
REFERENCE_PREDICTION_COLUMN = "initial_y_pred_center"
REFERENCE_PREDICTION_FALLBACK_COLUMNS = (
    "initial_y_pred_center",
    "initial_prediction",
    "base_y_pred_center",
    "y_pred_center",
)
CALIBRATED_LOWER_COLUMN = "y_pred_lower"
CALIBRATED_UPPER_COLUMN = "y_pred_upper"
INTERVAL_WIDTH_COLUMN = "interval_width"

Y_PATH = REPO_ROOT / "data" / "final" / "y.csv"
BACKTEST_RESULTS_ROOT = REPO_ROOT / "results" / "backtest"
REGIME_AWARE_RESULTS_ROOT = REPO_ROOT / "results" / "regime_aware"
REGIME_FREE_RESULTS_ROOT = REPO_ROOT / "results" / "regime_free"

DA_PRICE_COLUMN = "DE Price Spot EUR/MWh EPEX H Actual"
CLOSE_PRICE_COLUMNS = {
    "ID": "DE Price Intraday VWAP EUR/MWh EPEX H Actual",
    "ID3": "DE Price Intraday VWAP ID3 EUR/MWh EPEX H Actual",
    "ID1": "DE Price Intraday VWAP ID1 EUR/MWh EPEX H Actual",
    "IMB": "DE Price Imbalance Single EUR/MWh 15min Actual",
}
ALL_PRICE_COLUMNS = {"DA": DA_PRICE_COLUMN, **CLOSE_PRICE_COLUMNS}

DEFAULT_TARGET_STEM = target_artifact_stem(DA_PRICE_COLUMN)

REQUIRED_PREDICTION_COLUMNS = (
    DATE_COLUMN,
    CALIBRATED_LOWER_COLUMN,
    CALIBRATED_UPPER_COLUMN,
)

DEFAULT_SIZE_MAP = {
    -100.0: 50.0,
    -75.0: 45.0,
    -50.0: 40.0,
    -40.0: 35.0,
    -30.0: 30.0,
    -25.0: 25.0,
    -20.0: 20.0,
    -15.0: 10.0,
    -10.0: 0.0,
    -5.0: 0.0,
    0.0: 0.0,
    5.0: 0.0,
    10.0: 0.0,
    15.0: -10.0,
    20.0: -20.0,
    25.0: -25.0,
    30.0: -30.0,
    40.0: -35.0,
    50.0: -40.0,
    75.0: -45.0,
    100.0: -50.0,
}

DEFAULT_POSITION_METHOD = "interval_band"
DEFAULT_ENTRY_BAND_FRACTION = 0.40
DEFAULT_EXIT_BAND_FRACTION = 0.65
DEFAULT_SIGNAL_FAMILY = "tanh"
DEFAULT_SIGNAL_SCALE = 6.0
DEFAULT_SIGNAL_POWER = 1.0
DEFAULT_POSITION_CAP = 20.0

SCENARIO_LABELS = {
    "strategy_close_id": "Strategy | Close ID",
    "strategy_close_id3": "Strategy | Close ID3",
    "strategy_close_id1": "Strategy | Close ID1",
    "strategy_close_imb": "Strategy | Close IMB",
    "strategy_perfect_close": "Strategy | Perfect Close",
    "perfect_direction_close_id": "Perfect Direction | Close ID",
    "perfect_direction_close_id3": "Perfect Direction | Close ID3",
    "perfect_direction_close_id1": "Perfect Direction | Close ID1",
    "perfect_direction_close_imb": "Perfect Direction | Close IMB",
    "perfect_direction_perfect_close": "Perfect Direction | Perfect Close",
}

SCENARIO_ORDER = list(SCENARIO_LABELS)
