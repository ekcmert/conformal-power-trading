from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    import holidays  # type: ignore

    HOLIDAYS_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import fallback
    HOLIDAYS_AVAILABLE = False


def add_time_features_de(
    dataframe: pd.DataFrame,
    tz: str = "Europe/Berlin",
    holiday_country: str = "DE",
    holiday_prov: Optional[str] = None,
    holiday_state: Optional[str] = None,
    clip_holiday_dist_days: int = 7,
    add_dow_hour_onehot: bool = False,
    drop_original_time_cols: bool = False,
) -> pd.DataFrame:
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise TypeError("dataframe.index must be a pandas DatetimeIndex.")

    features = dataframe.copy()
    index = features.index
    if index.tz is None:
        index = index.tz_localize("UTC")

    local_index = index.tz_convert(tz)
    local_day = local_index.normalize()

    features["hour"] = local_index.hour
    features["dow"] = local_index.dayofweek
    features["doy"] = local_index.dayofyear
    features["month"] = local_index.month
    features["quarter"] = local_index.quarter
    features["woy"] = local_index.isocalendar().week.astype(int).to_numpy()

    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24.0)
    features["dow_sin"] = np.sin(2 * np.pi * features["dow"] / 7.0)
    features["dow_cos"] = np.cos(2 * np.pi * features["dow"] / 7.0)
    features["doy_sin"] = np.sin(2 * np.pi * features["doy"] / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * features["doy"] / 365.25)
    features["woy_sin"] = np.sin(2 * np.pi * features["woy"] / 52.18)
    features["woy_cos"] = np.cos(2 * np.pi * features["woy"] / 52.18)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12.0)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12.0)

    features["is_weekend"] = features["dow"].isin([5, 6]).astype(int)
    features["is_night"] = features["hour"].between(0, 6).astype(int)
    features["is_peak_hour"] = features["hour"].between(8, 20).astype(int)
    features["is_offpeak"] = (1 - features["is_peak_hour"]).astype(int)
    features["is_month_start"] = local_index.is_month_start.astype(int)
    features["is_month_end"] = local_index.is_month_end.astype(int)
    features["is_quarter_start"] = local_index.is_quarter_start.astype(int)
    features["is_quarter_end"] = local_index.is_quarter_end.astype(int)

    features["is_winter"] = features["month"].isin([12, 1, 2]).astype(int)
    features["is_spring"] = features["month"].isin([3, 4, 5]).astype(int)
    features["is_summer"] = features["month"].isin([6, 7, 8]).astype(int)
    features["is_fall"] = features["month"].isin([9, 10, 11]).astype(int)

    local_dates = pd.Index(local_day.date, name="date")
    unique_dates = pd.Index(pd.unique(local_dates), name="date")

    if HOLIDAYS_AVAILABLE:
        try:
            de_holidays = holidays.country_holidays(
                holiday_country,
                prov=holiday_prov,
                state=holiday_state,
            )
        except TypeError:
            de_holidays = holidays.country_holidays(holiday_country)

        is_holiday_map = {date_value: int(date_value in de_holidays) for date_value in unique_dates}
    else:
        is_holiday_map = {date_value: 0 for date_value in unique_dates}

    holiday_dates = sorted(date_value for date_value, flag in is_holiday_map.items() if flag == 1)

    def days_to_next_holiday(date_value: object) -> int:
        for holiday_date in holiday_dates:
            if holiday_date >= date_value:
                return min((holiday_date - date_value).days, clip_holiday_dist_days)
        return clip_holiday_dist_days

    def days_since_previous_holiday(date_value: object) -> int:
        for holiday_date in reversed(holiday_dates):
            if holiday_date <= date_value:
                return min((date_value - holiday_date).days, clip_holiday_dist_days)
        return clip_holiday_dist_days

    features["is_holiday_de"] = pd.Series(
        [is_holiday_map[date_value] for date_value in local_dates],
        index=features.index,
    ).astype(int)
    features["days_to_next_holiday_de"] = pd.Series(
        [days_to_next_holiday(date_value) for date_value in local_dates],
        index=features.index,
    ).astype(int)
    features["days_since_prev_holiday_de"] = pd.Series(
        [days_since_previous_holiday(date_value) for date_value in local_dates],
        index=features.index,
    ).astype(int)

    bridge_map: dict[object, int] = {}
    for date_value in unique_dates:
        weekday = pd.Timestamp(date_value).dayofweek
        is_bridge_day = 0
        if is_holiday_map.get(date_value, 0) == 0:
            if weekday == 0 and is_holiday_map.get((pd.Timestamp(date_value) + pd.Timedelta(days=1)).date(), 0) == 1:
                is_bridge_day = 1
            elif weekday == 4 and is_holiday_map.get((pd.Timestamp(date_value) - pd.Timedelta(days=1)).date(), 0) == 1:
                is_bridge_day = 1
        bridge_map[date_value] = is_bridge_day

    features["is_bridge_day"] = pd.Series(
        [bridge_map[date_value] for date_value in local_dates],
        index=features.index,
    ).astype(int)
    features["is_business_day"] = (
        (features["dow"] <= 4) & (features["is_holiday_de"] == 0)
    ).astype(int)

    features["is_dst"] = local_index.map(
        lambda timestamp: int(bool(timestamp.dst() and timestamp.dst().total_seconds() != 0))
    )

    utc_offsets = pd.Series(
        local_index.map(lambda timestamp: int(timestamp.utcoffset().total_seconds() // 60)),
        index=features.index,
    )
    features["is_dst_transition_day"] = utc_offsets.groupby(local_day).transform(
        lambda values: int(values.nunique() > 1)
    )

    local_wall_clock = pd.Index(local_index.strftime("%Y-%m-%d %H"), name="local_wall_clock")
    features["is_repeated_hour"] = pd.Series(
        local_wall_clock.duplicated(keep=False),
        index=features.index,
    ).astype(int)

    rows_per_day = pd.Series(1, index=features.index).groupby(local_day).transform("sum")
    first_offset = utc_offsets.groupby(local_day).transform("first")
    last_offset = utc_offsets.groupby(local_day).transform("last")
    offset_delta = last_offset - first_offset
    expected_rows = np.where(offset_delta == 60, 23, np.where(offset_delta == -60, 25, 24))
    features["is_dst_day_with_non24_hours"] = (rows_per_day != expected_rows).astype(int)

    features["hour_x_weekend"] = features["hour"] * features["is_weekend"]
    features["hour_x_holiday"] = features["hour"] * features["is_holiday_de"]
    features["dow_hour"] = (features["dow"] * 24 + features["hour"]).astype(int)

    if add_dow_hour_onehot:
        dow_hour_dummies = pd.get_dummies(features["dow_hour"], prefix="dow_hour", dtype=int)
        features = pd.concat([features, dow_hour_dummies], axis=1)

    if drop_original_time_cols:
        features = features.drop(columns=["hour", "dow", "doy", "woy"], errors="ignore")

    return features


__all__ = ["add_time_features_de"]
