from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.config import (
    ALL_PRICE_COLUMNS,
    BACKTEST_RESULTS_ROOT,
    CALIBRATED_LOWER_COLUMN,
    CALIBRATED_UPPER_COLUMN,
    DATE_COLUMN,
    DEFAULT_SIZE_MAP,
    DEFAULT_TARGET_STEM,
    INTERVAL_WIDTH_COLUMN,
    REFERENCE_PREDICTION_COLUMN,
    REFERENCE_PREDICTION_FALLBACK_COLUMNS,
    REGIME_AWARE_RESULTS_ROOT,
    REGIME_FREE_RESULTS_ROOT,
    REQUIRED_PREDICTION_COLUMNS,
    Y_PATH,
)


def normalize_datetime(series: pd.Series) -> pd.Series:
    normalized = pd.to_datetime(series, utc=True, errors="coerce")
    if normalized.isna().any():
        invalid_count = int(normalized.isna().sum())
        raise ValueError(f"Found {invalid_count} invalid datetime values in {series.name!r}.")
    return normalized


def resolve_prediction_csv(pred_path: str | Path, *, target_stem: str = DEFAULT_TARGET_STEM) -> Path:
    candidate = Path(pred_path).expanduser().resolve()
    if candidate.is_file():
        if candidate.suffix.lower() != ".csv":
            raise ValueError(f"Prediction path must point to a CSV file, got {candidate}.")
        return candidate

    if not candidate.exists():
        raise FileNotFoundError(f"Prediction path does not exist: {candidate}")
    if not candidate.is_dir():
        raise ValueError(f"Prediction path must be a directory or CSV file, got {candidate}.")

    prediction_files = sorted(candidate.glob("*_predictions.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No *_predictions.csv file found under {candidate}.")

    preferred_name = f"{target_stem}_predictions.csv"
    preferred_matches = [path for path in prediction_files if path.name == preferred_name]
    if preferred_matches:
        return preferred_matches[0]

    if len(prediction_files) == 1:
        return prediction_files[0]

    discovered = ", ".join(path.name for path in prediction_files)
    raise ValueError(
        f"Multiple prediction CSV files were found under {candidate}, but none matched {preferred_name!r}: {discovered}"
    )


def infer_output_dir(
    pred_path: str | Path,
    *,
    results_root: Path = BACKTEST_RESULTS_ROOT,
) -> Path:
    candidate = Path(pred_path).expanduser().resolve()
    source_dir = candidate.parent if candidate.is_file() else candidate

    for source_root in (REGIME_AWARE_RESULTS_ROOT.resolve(), REGIME_FREE_RESULTS_ROOT.resolve()):
        try:
            relative_path = source_dir.relative_to(source_root)
        except ValueError:
            continue
        return (results_root.resolve() / relative_path).resolve()

    raise ValueError(
        "Could not infer the backtest output directory. "
        f"Expected the prediction path to live under {REGIME_AWARE_RESULTS_ROOT} or {REGIME_FREE_RESULTS_ROOT}."
    )


def parse_size_map(size_map_input: str | Path | None) -> dict[float, float]:
    if size_map_input is None:
        return dict(DEFAULT_SIZE_MAP)

    raw_input = str(size_map_input).strip()
    candidate_path = Path(raw_input).expanduser()
    if candidate_path.exists() and candidate_path.is_file():
        raw_text = candidate_path.read_text(encoding="utf-8")
    else:
        raw_text = raw_input

    parsed: Any
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(raw_text)

    if not isinstance(parsed, dict):
        raise ValueError("Size map must be a dictionary of margin thresholds to position sizes.")

    normalized_map = {float(key): float(value) for key, value in parsed.items()}
    if not normalized_map:
        raise ValueError("Size map cannot be empty.")

    return dict(sorted(normalized_map.items(), key=lambda item: item[0]))


def load_prediction_frame(prediction_csv: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(prediction_csv)
    missing_columns = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Missing required prediction columns in {prediction_csv}: {missing_columns}")

    reference_column = next(
        (column for column in REFERENCE_PREDICTION_FALLBACK_COLUMNS if column in frame.columns),
        None,
    )
    if reference_column is None:
        raise KeyError(
            "Could not find a reference point-forecast column. "
            f"Checked: {list(REFERENCE_PREDICTION_FALLBACK_COLUMNS)}"
        )

    working = frame.copy()
    working[DATE_COLUMN] = normalize_datetime(working[DATE_COLUMN])
    working[REFERENCE_PREDICTION_COLUMN] = working[reference_column]
    if INTERVAL_WIDTH_COLUMN not in working.columns:
        working[INTERVAL_WIDTH_COLUMN] = working[CALIBRATED_UPPER_COLUMN] - working[CALIBRATED_LOWER_COLUMN]

    working = (
        working.dropna(
            subset=[
                DATE_COLUMN,
                CALIBRATED_LOWER_COLUMN,
                CALIBRATED_UPPER_COLUMN,
                REFERENCE_PREDICTION_COLUMN,
                INTERVAL_WIDTH_COLUMN,
            ]
        )
        .sort_values(DATE_COLUMN)
        .drop_duplicates(subset=[DATE_COLUMN], keep="last")
        .reset_index(drop=True)
    )

    return working


def load_price_frame(y_path: str | Path = Y_PATH) -> pd.DataFrame:
    frame = pd.read_csv(y_path)
    required_columns = [DATE_COLUMN, *ALL_PRICE_COLUMNS.values()]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Missing required price columns in {y_path}: {missing_columns}")

    working = frame.loc[:, required_columns].copy()
    working[DATE_COLUMN] = normalize_datetime(working[DATE_COLUMN])
    working = (
        working.sort_values(DATE_COLUMN)
        .drop_duplicates(subset=[DATE_COLUMN], keep="last")
        .reset_index(drop=True)
    )
    return working

