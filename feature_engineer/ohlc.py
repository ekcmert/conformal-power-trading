from __future__ import annotations

import numpy as np
import pandas as pd

from .common import coerce_numeric_frame, ensure_datetime_index


def _expand_daily_frame_to_hourly(
    dataframe: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    daily = dataframe.copy().sort_index()
    expanded = daily.reindex(target_index.floor("D"), method="ffill")
    expanded.index = target_index
    return expanded


def engineer_ohlc_features(
    dataframe: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    availability_lag_days: int = 1,
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame(index=target_index)

    daily = ensure_datetime_index(dataframe, dataset_name="ohlc")
    daily = coerce_numeric_frame(daily)
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    daily.index = daily.index - pd.Timedelta(days=availability_lag_days)

    diff_features = daily.diff().rename(columns=lambda column: f"{column}__diff_1d")
    return_features = (
        daily.pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .rename(columns=lambda column: f"{column}__return_1d")
    )

    expanded_level = _expand_daily_frame_to_hourly(daily, target_index)
    expanded_diff = _expand_daily_frame_to_hourly(diff_features, target_index)
    expanded_return = _expand_daily_frame_to_hourly(return_features, target_index)

    return pd.concat([expanded_level, expanded_diff, expanded_return], axis=1)


__all__ = ["engineer_ohlc_features"]
