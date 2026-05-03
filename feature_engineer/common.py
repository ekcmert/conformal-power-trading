from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
EDA_DIR = REPO_ROOT / "results" / "EDA"

DATASET_FILENAMES = {
    "timeseries": "timeseries_data.csv",
    "scenario": "scenario_data.csv",
    "ohlc": "ohlc_data.csv",
    "instance": "instance_data.csv",
}

INDEX_LABELS = {
    "timeseries": "date",
    "scenario": "date",
    "ohlc": "traded",
    "instance": "date",
}

TARGET_COLUMNS = [
    "DE Price Spot EUR/MWh EPEX 15min Actual",
    "DE Price Spot EUR/MWh EPEX H Actual",
    "DE Price Intraday VWAP EUR/MWh EPEX 15min Actual",
    "DE Price Intraday VWAP EUR/MWh EPEX 30min Actual",
    "DE Price Intraday VWAP EUR/MWh EPEX H Actual",
    "DE Price Intraday VWAP ID1 EUR/MWh EPEX 15min Actual",
    "DE Price Intraday VWAP ID1 EUR/MWh EPEX 30min Actual",
    "DE Price Intraday VWAP ID1 EUR/MWh EPEX H Actual",
    "DE Price Intraday VWAP ID3 EUR/MWh EPEX 15min Actual",
    "DE Price Intraday VWAP ID3 EUR/MWh EPEX 30min Actual",
    "DE Price Intraday VWAP ID3 EUR/MWh EPEX H Actual",
    "DE Price Imbalance Single EUR/MWh 15min Actual",
    "DE Volume Imbalance Net MWh 15min Actual",
]


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _parse_datetime_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True)


def ensure_datetime_index(dataframe: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = dataframe.copy()

    if isinstance(df.index, pd.DatetimeIndex):
        index = pd.to_datetime(df.index, errors="coerce", utc=True)
        df = df.loc[index.notna()].copy()
        df.index = index[index.notna()]
    else:
        datetime_column: str | None = None
        parsed_index: pd.Series | None = None

        for candidate in ("date", "datetime", "timestamp", "traded"):
            if candidate in df.columns:
                candidate_index = _parse_datetime_series(df[candidate])
                if candidate_index.notna().any():
                    datetime_column = candidate
                    parsed_index = candidate_index
                    break

        if parsed_index is None:
            first_column = str(df.columns[0])
            first_values = df.iloc[:, 0]
            first_column_name = first_column.lower()
            looks_like_saved_index = (
                first_column_name.startswith("unnamed")
                or "date" in first_column_name
                or "time" in first_column_name
                or "traded" in first_column_name
            )
            is_text_like = is_object_dtype(first_values) or is_string_dtype(first_values)

            if looks_like_saved_index or (is_text_like and not is_numeric_dtype(first_values)):
                candidate_index = _parse_datetime_series(first_values)
                if candidate_index.notna().mean() >= 0.95:
                    datetime_column = first_column
                    parsed_index = candidate_index

        if parsed_index is None:
            raise ValueError(
                f"{dataset_name} dataset does not contain a readable datetime index or column."
            )

        valid_rows = parsed_index.notna()
        df = df.loc[valid_rows].copy()
        df.index = parsed_index.loc[valid_rows]

        if datetime_column in df.columns:
            df = df.drop(columns=[datetime_column])

    df.index.name = INDEX_LABELS[dataset_name]
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def coerce_numeric_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()

    df = dataframe.copy()
    bool_columns = df.select_dtypes(include=["bool"]).columns
    if len(bool_columns):
        df[bool_columns] = df[bool_columns].astype(int)

    return df.apply(pd.to_numeric, errors="coerce")


def drop_all_null_columns(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if dataframe.empty:
        return dataframe.copy(), []

    all_null_columns = dataframe.columns[dataframe.isna().all()].tolist()
    cleaned = dataframe.drop(columns=all_null_columns)
    return cleaned, all_null_columns


def merge_feature_frames(
    frames: Iterable[pd.DataFrame],
    how: str = "inner",
) -> pd.DataFrame:
    valid_frames = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame()

    merged = pd.concat(valid_frames, axis=1, join=how)
    merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


def align_and_drop_nulls(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    X = coerce_numeric_frame(features)
    y = coerce_numeric_frame(targets)

    X, dropped_x_columns = drop_all_null_columns(X)
    y, dropped_y_columns = drop_all_null_columns(y)

    common_index = X.index.intersection(y.index)
    X = X.loc[common_index].sort_index()
    y = y.loc[common_index].sort_index()

    if X.empty or y.empty:
        return X, y, dropped_x_columns, dropped_y_columns

    mask = X.notna().all(axis=1) & y.notna().all(axis=1)
    return X.loc[mask], y.loc[mask], dropped_x_columns, dropped_y_columns


def save_dataframe(
    dataframe: pd.DataFrame,
    path: str | Path,
    index_label: str = "date",
) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    dataframe.to_csv(output_path, index_label=index_label)
    return output_path


def save_named_frames(
    frames: dict[str, pd.DataFrame],
    output_dir: str | Path = PROCESSED_DIR,
) -> dict[str, Path]:
    directory = ensure_directory(output_dir)
    saved_paths: dict[str, Path] = {}

    for name, dataframe in frames.items():
        path = directory / DATASET_FILENAMES[name]
        dataframe.to_csv(path, index_label=INDEX_LABELS[name])
        saved_paths[name] = path

    return saved_paths


def rebuild_merged_datasets(
    cache_to_processed: bool = True,
) -> dict[str, pd.DataFrame]:
    from data_preprocessor.data_prep import preprocess_all_from_curves_folder

    results = preprocess_all_from_curves_folder()
    frames = {
        name: ensure_datetime_index(dataframe, name)
        for name, dataframe in results["merged"].items()
    }

    if cache_to_processed:
        save_named_frames(frames, output_dir=PROCESSED_DIR)

    return frames


def load_merged_datasets(
    cache_rebuilt: bool = True,
) -> tuple[dict[str, pd.DataFrame], str]:
    frames: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}

    for name, filename in DATASET_FILENAMES.items():
        path = PROCESSED_DIR / filename
        if not path.exists():
            failures[name] = f"missing file: {path}"
            continue

        try:
            raw = pd.read_csv(path)
            frames[name] = ensure_datetime_index(raw, dataset_name=name)
        except Exception as exc:  # pragma: no cover - defensive branch
            failures[name] = str(exc)

    if not failures:
        return frames, "data/processed"

    rebuilt = rebuild_merged_datasets(cache_to_processed=cache_rebuilt)
    return rebuilt, "rebuilt_from_raw_curves"


__all__ = [
    "DATASET_FILENAMES",
    "EDA_DIR",
    "FINAL_DIR",
    "INDEX_LABELS",
    "PROCESSED_DIR",
    "REPO_ROOT",
    "TARGET_COLUMNS",
    "align_and_drop_nulls",
    "coerce_numeric_frame",
    "drop_all_null_columns",
    "ensure_datetime_index",
    "ensure_directory",
    "load_merged_datasets",
    "merge_feature_frames",
    "rebuild_merged_datasets",
    "save_dataframe",
    "save_named_frames",
]
