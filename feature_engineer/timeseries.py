from __future__ import annotations

import pandas as pd

from .common import TARGET_COLUMNS, coerce_numeric_frame


ACTUAL_LAG_COLUMNS = (
    "DE Wind Power Production MWh/h 15min Actual",
    "DE Wind Power Production Offshore MWh/h 15min Actual",
    "DE Wind Power Production Onshore MWh/h 15min Actual",
    "DE Solar Photovoltaic Production MWh/h 15min Actual",
    "DE Residual Load MWh/h 15min Actual",
    "DE Consumption MWh/h 15min Actual",
)
ACTUAL_LAGS_HOURS = (48, 168)
TARGET_LAG_COLUMN = "DE Price Spot EUR/MWh EPEX H Actual"
TARGET_LAGS_HOURS = (24, 48, 72, 168)


def _build_lagged_feature_columns(
    dataframe: pd.DataFrame,
    source_columns: list[str] | tuple[str, ...],
    lag_hours: list[int] | tuple[int, ...],
) -> dict[str, pd.Series]:
    feature_columns: dict[str, pd.Series] = {}

    for column in source_columns:
        if column not in dataframe.columns:
            continue

        series = dataframe[column]
        for lag in lag_hours:
            feature_columns[f"{column}__lag_{lag}h"] = series.shift(lag)

    return feature_columns


def build_target_frame(
    dataframe: pd.DataFrame,
    target_columns: list[str] | tuple[str, ...] = tuple(TARGET_COLUMNS),
) -> pd.DataFrame:
    missing_columns = [column for column in target_columns if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Missing target columns: {missing_columns}")

    targets = dataframe.loc[:, list(target_columns)].copy()
    targets["DAID"] = (
        targets["DE Price Spot EUR/MWh EPEX H Actual"]
        - targets["DE Price Intraday VWAP EUR/MWh EPEX H Actual"]
    )
    targets["DAID3"] = (
        targets["DE Price Spot EUR/MWh EPEX H Actual"]
        - targets["DE Price Intraday VWAP ID3 EUR/MWh EPEX H Actual"]
    )
    targets["DAID1"] = (
        targets["DE Price Spot EUR/MWh EPEX H Actual"]
        - targets["DE Price Intraday VWAP ID1 EUR/MWh EPEX H Actual"]
    )
    targets["DAIMB"] = (
        targets["DE Price Spot EUR/MWh EPEX H Actual"]
        - targets["DE Price Imbalance Single EUR/MWh 15min Actual"]
    )
    return coerce_numeric_frame(targets)


def engineer_timeseries_features(
    dataframe: pd.DataFrame,
    target_columns: list[str] | tuple[str, ...] = tuple(TARGET_COLUMNS),
) -> pd.DataFrame:
    raw = coerce_numeric_frame(dataframe)
    excluded_columns = set(target_columns).union(ACTUAL_LAG_COLUMNS)
    features = raw.drop(columns=list(excluded_columns), errors="ignore").copy()

    import_columns = [
        column
        for column in features.columns
        if ">DE Exchange Net Transfer Capacity" in column and not column.startswith("DE>")
    ]
    export_columns = [
        column
        for column in features.columns
        if column.startswith("DE>") and "Exchange Net Transfer Capacity" in column
    ]

    if import_columns:
        features["DE Total Import Capacity MW REMIT"] = features[import_columns].sum(
            axis=1,
            min_count=1,
        )
    if export_columns:
        features["DE Total Export Capacity MW REMIT"] = features[export_columns].sum(
            axis=1,
            min_count=1,
        )
    if import_columns and export_columns:
        features["DE Net Transfer Capacity Balance MW REMIT"] = (
            features["DE Total Import Capacity MW REMIT"]
            - features["DE Total Export Capacity MW REMIT"]
        )

    lagged_features = {}
    lagged_features.update(
        _build_lagged_feature_columns(
            dataframe=raw,
            source_columns=ACTUAL_LAG_COLUMNS,
            lag_hours=ACTUAL_LAGS_HOURS,
        )
    )
    lagged_features.update(
        _build_lagged_feature_columns(
            dataframe=raw,
            source_columns=(TARGET_LAG_COLUMN,),
            lag_hours=TARGET_LAGS_HOURS,
        )
    )
    if lagged_features:
        features = pd.concat(
            [features, pd.DataFrame(lagged_features, index=raw.index)],
            axis=1,
        )

    return features


__all__ = [
    "build_target_frame",
    "engineer_timeseries_features",
]
