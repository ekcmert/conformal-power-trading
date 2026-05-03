import numpy as np
import pandas as pd


DEFAULT_HOUR_COLUMN = "hour"
DEFAULT_MONTH_COLUMN = "month"
DEFAULT_RESIDUAL_LOAD_COLUMN = "DE Residual Load MWh/h 15min Forecast__mean"
DEFAULT_WIND_COLUMN = "DE Wind Power Production MWh/h 15min Forecast__mean"
DEFAULT_SOLAR_COLUMN = "DE Solar Photovoltaic Production MWh/h 15min Forecast__mean"
DEFAULT_WEEKEND_COLUMN = "is_weekend"
HEURISTIC_REQUIRED_COLUMNS = [
    DEFAULT_HOUR_COLUMN,
    DEFAULT_MONTH_COLUMN,
    DEFAULT_RESIDUAL_LOAD_COLUMN,
    DEFAULT_WIND_COLUMN,
    DEFAULT_SOLAR_COLUMN,
    DEFAULT_WEEKEND_COLUMN,
]


def _weekend_indicator(
    df: pd.DataFrame,
    *,
    weekend_col: str | None,
    dayofweek_col: str | None,
) -> pd.Series:
    if weekend_col is not None:
        return df[weekend_col].astype(bool)
    if dayofweek_col is not None:
        return df[dayofweek_col].isin([5, 6])  # Sat, Sun
    return pd.Series(False, index=df.index)


def assign_german_da_regimes(
    df_sample: pd.DataFrame,
    df_pred: pd.DataFrame,
    hour_col: str = DEFAULT_HOUR_COLUMN,
    month_col: str = DEFAULT_MONTH_COLUMN,
    residual_load_col: str = DEFAULT_RESIDUAL_LOAD_COLUMN,
    wind_col: str = DEFAULT_WIND_COLUMN,
    solar_col: str = DEFAULT_SOLAR_COLUMN,
    weekend_col: str | None = DEFAULT_WEEKEND_COLUMN,
    dayofweek_col: str | None = None,
    output_col: str = "regime",
) -> pd.DataFrame:
    """
    Assign simple heuristic regimes for German day-ahead power price modeling.

    Quantile thresholds are estimated from ``df_sample`` and then applied to
    the separate ``df_pred`` dataframe. This keeps the regime assignment
    out-of-sample relative to the calibration window.

    Regimes:
        1. scarcity_dunkelflaute
        2. solar_surplus_midday
        3. evening_ramp
        4. windy_night_surplus
        5. weekend_res_oversupply
        6. normal_balanced

    Parameters
    ----------
    df_sample : pd.DataFrame
        Historical sample dataframe used to estimate heuristic thresholds.
    df_pred : pd.DataFrame
        Prediction dataframe that receives regime labels.
    hour_col : str
        Column name for hour of day (0-23).
    month_col : str
        Column name for month (1-12).
    residual_load_col : str
        Column name for residual load.
    wind_col : str
        Column name for wind generation.
    solar_col : str
        Column name for solar generation.
    weekend_col : str, optional
        Boolean/int column indicating weekend (1=True, 0=False).
    dayofweek_col : str, optional
        Column with day of week, assumed Monday=0, ..., Sunday=6.
        Used only if weekend_col is not given.
    output_col : str, default="regime"
        Name of the output regime label column.

    Returns
    -------
    pd.DataFrame
        Copy of ``df_pred`` with an added regime label column.
    """

    if df_sample.empty:
        raise ValueError("df_sample must contain at least one row to estimate regime thresholds.")

    out = df_pred.copy()

    # --- weekend indicator ---
    weekend = _weekend_indicator(
        out,
        weekend_col=weekend_col,
        dayofweek_col=dayofweek_col,
    )

    # --- quantile thresholds from the historical sample dataframe ---
    rl_q30 = df_sample[residual_load_col].quantile(0.30)
    rl_q35 = df_sample[residual_load_col].quantile(0.35)
    rl_q70 = df_sample[residual_load_col].quantile(0.70)
    rl_q80 = df_sample[residual_load_col].quantile(0.80)

    wind_q30 = df_sample[wind_col].quantile(0.30)
    wind_q70 = df_sample[wind_col].quantile(0.70)

    solar_q20 = df_sample[solar_col].quantile(0.20)
    solar_q30 = df_sample[solar_col].quantile(0.30)
    solar_q70 = df_sample[solar_col].quantile(0.70)

    total_res_sample = df_sample[wind_col] + df_sample[solar_col]
    total_res_q70 = total_res_sample.quantile(0.70)

    hour = out[hour_col]
    month = out[month_col]
    rl = out[residual_load_col]
    wind = out[wind_col]
    solar = out[solar_col]

    # ------------------------------------------------------------------
    # Priority order matters: first matching condition gets the regime.
    # This makes the regimes mutually exclusive and exhaustive.
    # ------------------------------------------------------------------

    cond_scarcity = (
        (rl >= rl_q80) &
        (wind <= wind_q30) &
        (solar <= solar_q20)
    )

    cond_solar_surplus_midday = (
        (hour.between(11, 15)) &
        (month.between(4, 9)) &
        (solar >= solar_q70) &
        (rl <= rl_q30)
    )

    cond_evening_ramp = (
        (hour.between(17, 21)) &
        (solar <= solar_q30) &
        (rl >= rl_q70)
    )

    cond_windy_night_surplus = (
        (hour.between(0, 5)) &
        (wind >= wind_q70) &
        (rl <= rl_q30)
    )

    cond_weekend_res_oversupply = (
        weekend &
        ((wind + solar) >= total_res_q70) &
        (rl <= rl_q35)
    )

    cond_normal_balanced = np.ones(len(out), dtype=bool)  # fallback, covers all remaining cases

    conditions = [
        cond_scarcity,
        cond_solar_surplus_midday,
        cond_evening_ramp,
        cond_windy_night_surplus,
        cond_weekend_res_oversupply,
        cond_normal_balanced
    ]

    labels = [
        "scarcity_dunkelflaute",
        "solar_surplus_midday",
        "evening_ramp",
        "windy_night_surplus",
        "weekend_res_oversupply",
        "normal_balanced"
    ]

    out[output_col] = np.select(conditions, labels, default="normal_balanced")

    return out
