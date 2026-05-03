from __future__ import annotations

import pandas as pd

from .common import coerce_numeric_frame


DE_LOCAL_TIMEZONE = "Europe/Berlin"
DE_WIND_MEAN_COLUMN = "DE Wind Power Production MWh/h 15min Forecast__mean"
DE_SOLAR_MEAN_COLUMN = "DE Solar Photovoltaic Production MWh/h 15min Forecast__mean"
DE_RESIDUAL_LOAD_MEAN_COLUMN = "DE Residual Load MWh/h 15min Forecast__mean"
DE_CONSUMPTION_MEAN_COLUMN = "DE Consumption MWh/h 15min Forecast__mean"
def _local_day_labels(
    index: pd.Index,
    timezone: str = DE_LOCAL_TIMEZONE,
) -> pd.Series:
    dt_index = pd.DatetimeIndex(index)
    if dt_index.tz is None:
        dt_index = dt_index.tz_localize("UTC")
    local_index = dt_index.tz_convert(timezone)
    return pd.Series(local_index.date, index=dt_index)


def _daily_extreme_difference(
    series: pd.Series,
    reducer: str,
    timezone: str = DE_LOCAL_TIMEZONE,
) -> pd.Series:
    local_days = _local_day_labels(series.index, timezone=timezone)
    daily_extreme = series.groupby(local_days).transform(reducer)
    return series - daily_extreme


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.where(denominator.ne(0))
    return numerator.divide(safe_denominator)


def extract_column_groups_and_types(columns: pd.Index | list[str]) -> tuple[dict[str, list[str]], pd.DataFrame]:
    series = pd.Index(columns).to_series().astype(str)
    parts = series.str.split("|", n=1, expand=True, regex=False)

    groups = parts[0].str.strip()
    types = parts[1].str.strip() if parts.shape[1] > 1 else pd.Series([None] * len(series), index=series.index)
    types = types.where(types.notna(), None)

    mapping = pd.DataFrame(
        {
            "column": series.values,
            "group": groups.values,
            "type": types.values,
        }
    )

    grouped_types = (
        mapping.dropna(subset=["type"])
        .groupby("group")["type"]
        .apply(lambda values: sorted(set(values)))
        .to_dict()
    )
    return grouped_types, mapping


def engineer_instance_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()

    raw = coerce_numeric_frame(dataframe)
    _, mapping = extract_column_groups_and_types(raw.columns)
    feature_columns: dict[str, pd.Series] = {}

    for group_name, column_names in mapping.groupby("group")["column"]:
        block = raw.loc[:, list(column_names)].copy()
        block = block.dropna(axis=1, how="all")
        if block.empty:
            continue

        mean_values = block.mean(axis=1, skipna=True)
        min_values = block.min(axis=1, skipna=True)
        max_values = block.max(axis=1, skipna=True)

        feature_columns[f"{group_name}__mean"] = mean_values
        feature_columns[f"{group_name}__min"] = min_values
        feature_columns[f"{group_name}__max"] = max_values
        feature_columns[f"{group_name}__std"] = block.std(axis=1, skipna=True, ddof=0)
        feature_columns[f"{group_name}__range"] = max_values - min_values

    if DE_WIND_MEAN_COLUMN in feature_columns:
        feature_columns[f"{DE_WIND_MEAN_COLUMN}__minus_cet_day_max"] = (
            _daily_extreme_difference(feature_columns[DE_WIND_MEAN_COLUMN], reducer="max")
        )
        feature_columns[f"{DE_WIND_MEAN_COLUMN}__minus_cet_day_min"] = (
            _daily_extreme_difference(feature_columns[DE_WIND_MEAN_COLUMN], reducer="min")
        )

    if DE_SOLAR_MEAN_COLUMN in feature_columns:
        feature_columns[f"{DE_SOLAR_MEAN_COLUMN}__minus_cet_day_max"] = (
            _daily_extreme_difference(feature_columns[DE_SOLAR_MEAN_COLUMN], reducer="max")
        )

    if DE_RESIDUAL_LOAD_MEAN_COLUMN in feature_columns:
        feature_columns[f"{DE_RESIDUAL_LOAD_MEAN_COLUMN}__minus_cet_day_max"] = (
            _daily_extreme_difference(feature_columns[DE_RESIDUAL_LOAD_MEAN_COLUMN], reducer="max")
        )
        feature_columns[f"{DE_RESIDUAL_LOAD_MEAN_COLUMN}__minus_cet_day_min"] = (
            _daily_extreme_difference(feature_columns[DE_RESIDUAL_LOAD_MEAN_COLUMN], reducer="min")
        )

    if DE_CONSUMPTION_MEAN_COLUMN in feature_columns:
        feature_columns[f"{DE_CONSUMPTION_MEAN_COLUMN}__minus_cet_day_max"] = (
            _daily_extreme_difference(feature_columns[DE_CONSUMPTION_MEAN_COLUMN], reducer="max")
        )
        feature_columns[f"{DE_CONSUMPTION_MEAN_COLUMN}__minus_cet_day_min"] = (
            _daily_extreme_difference(feature_columns[DE_CONSUMPTION_MEAN_COLUMN], reducer="min")
        )

    if (
        DE_RESIDUAL_LOAD_MEAN_COLUMN in feature_columns
        and DE_CONSUMPTION_MEAN_COLUMN in feature_columns
    ):
        feature_columns["DE Residual Load to Consumption Forecast__ratio"] = _safe_ratio(
            feature_columns[DE_RESIDUAL_LOAD_MEAN_COLUMN],
            feature_columns[DE_CONSUMPTION_MEAN_COLUMN],
        )

    if (
        DE_WIND_MEAN_COLUMN in feature_columns
        and DE_CONSUMPTION_MEAN_COLUMN in feature_columns
    ):
        feature_columns["DE Wind to Consumption Forecast__ratio"] = _safe_ratio(
            feature_columns[DE_WIND_MEAN_COLUMN],
            feature_columns[DE_CONSUMPTION_MEAN_COLUMN],
        )

    if (
        DE_SOLAR_MEAN_COLUMN in feature_columns
        and DE_CONSUMPTION_MEAN_COLUMN in feature_columns
    ):
        feature_columns["DE Solar to Consumption Forecast__ratio"] = _safe_ratio(
            feature_columns[DE_SOLAR_MEAN_COLUMN],
            feature_columns[DE_CONSUMPTION_MEAN_COLUMN],
        )

    return pd.DataFrame(feature_columns, index=raw.index)


__all__ = [
    "engineer_instance_features",
    "extract_column_groups_and_types",
]
