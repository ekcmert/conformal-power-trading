from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

try:
    from energy_quantified.eq_curves import (
        INSTANCE_CURVES,
        OHLC_CURVES,
        SCENARIO_TIMESERIES_CURVES,
        TIMESERIES_CURVES,
    )
    from energy_quantified.eq_helper import REPO_ROOT, curve_name_to_filename
except ImportError:
    from eq_curves import (
        INSTANCE_CURVES,
        OHLC_CURVES,
        SCENARIO_TIMESERIES_CURVES,
        TIMESERIES_CURVES,
    )
    from eq_helper import REPO_ROOT, curve_name_to_filename


CurveFrameMap = dict[str, pd.DataFrame]
InstanceFrameMap = dict[str, dict[str, pd.DataFrame]]

DEFAULT_CURVES_DIR = REPO_ROOT / "data" / "curves"
DEFAULT_CUTOFF: pd.Timestamp | None = pd.Timestamp("2022-12-01", tz="UTC")
_YEAR_TOKEN_RE = re.compile(r"\by(19|20)\d{2}\b", re.IGNORECASE)
_OHLC_NUMERIC_COLUMNS = (
    "front",
    "open",
    "high",
    "low",
    "close",
    "settlement",
    "volume",
    "open_interest",
)


def _detect_data_start_after_date_header(
    path: Path,
    max_scan_lines: int = 120,
) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle):
            if line.strip().lower().startswith("date"):
                return i + 1
            if i >= max_scan_lines:
                break
    return 0


def _find_line_index(
    path: Path,
    predicate: Callable[[str], bool],
    max_scan_lines: int = 250,
) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle):
            if predicate(line):
                return i
            if i >= max_scan_lines:
                break
    return -1


def _ensure_datetime_index(
    dataframe: pd.DataFrame,
    date_col: str = "date",
    utc: bool = True,
    errors: str = "raise",
) -> pd.DataFrame:
    df = dataframe.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(
                df[date_col],
                errors="coerce" if errors == "raise" else errors,
                cache=True,
                utc=utc,
            )
            df = df.dropna(subset=[date_col]).set_index(date_col)
        else:
            df.index = pd.to_datetime(df.index, errors=errors, utc=utc)

    df.index.name = "date"
    return df.sort_index()


def _normalize_datetime_index_timezone(
    dataframe: pd.DataFrame,
    tz: str | None = "UTC",
) -> pd.DataFrame:
    if tz is None:
        return dataframe

    df = dataframe.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df = df.tz_convert(tz)
    return df


def read_timeseries_like_csv(
    path: str | Path,
    value_col_name: str,
    read_csv_kwargs: dict | None = None,
    utc: bool = True,
) -> pd.DataFrame:
    csv_path = Path(path)
    kwargs = read_csv_kwargs or {}
    skiprows = _detect_data_start_after_date_header(csv_path)

    df = pd.read_csv(
        csv_path,
        skiprows=skiprows,
        header=None,
        names=["date", value_col_name],
        usecols=[0, 1],
        engine="c",
        **kwargs,
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce", cache=True, utc=utc)
    df[value_col_name] = pd.to_numeric(df[value_col_name], errors="coerce")
    return df.dropna(subset=["date"]).set_index("date").sort_index()


def read_scenario_csv(
    path: str | Path,
    read_csv_kwargs: dict | None = None,
    utc: bool = True,
) -> pd.DataFrame:
    csv_path = Path(path)
    kwargs = read_csv_kwargs or {}

    header_idx = _find_line_index(
        csv_path,
        lambda line: bool(_YEAR_TOKEN_RE.search(line)),
    )
    if header_idx < 0:
        raise ValueError(f"Could not find scenario header (y#### columns) in {csv_path}")

    df = pd.read_csv(
        csv_path,
        skiprows=header_idx,
        header=0,
        engine="c",
        **kwargs,
    )

    date_col = "date" if "date" in df.columns else df.columns[0]
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
        date_col = "date"

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
        format="ISO8601",
        utc=utc,
    )
    df = df.dropna(subset=[date_col])

    for column in df.columns:
        if column != date_col:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.set_index(date_col).sort_index()


def read_ohlc_csv(
    path: str | Path,
    read_csv_kwargs: dict | None = None,
) -> pd.DataFrame:
    csv_path = Path(path)
    kwargs = read_csv_kwargs or {}
    df = pd.read_csv(csv_path, engine="c", **kwargs)

    if len(df.columns) and str(df.columns[0]).lower().startswith("unnamed"):
        df = df.drop(columns=[df.columns[0]])

    if "traded" in df.columns:
        df["traded"] = pd.to_datetime(df["traded"], errors="coerce", cache=True)
    if "delivery" in df.columns:
        df["delivery"] = pd.to_datetime(df["delivery"], errors="coerce", cache=True)

    for column in _OHLC_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "traded" in df.columns:
        sort_cols = ["traded"] + (["period"] if "period" in df.columns else [])
        df = df.sort_values(sort_cols)

    return df


def infer_instance_tags_from_folder(
    curve_names: Iterable[str],
    data_folder: str | Path = DEFAULT_CURVES_DIR,
    ext: str = ".csv",
) -> dict[str, list[str]]:
    folder = Path(data_folder)
    out: dict[str, list[str]] = {}

    for curve in curve_names:
        curve_file = curve_name_to_filename(curve, ext=ext)
        matched_tags: list[str] = []

        for path in folder.glob(f"*_{curve_file}"):
            prefix = path.name[: -len(curve_file) - 1]
            if prefix:
                matched_tags.append(prefix)

        out[curve] = sorted(set(matched_tags))

    return out


def load_curve_csvs_timeseries(
    curve_names: Iterable[str],
    data_folder: str | Path = DEFAULT_CURVES_DIR,
    ext: str = ".csv",
    strict: bool = False,
    read_csv_kwargs: dict | None = None,
    utc: bool = True,
) -> CurveFrameMap:
    folder = Path(data_folder)
    kwargs = read_csv_kwargs or {}
    out: CurveFrameMap = {}
    missing: list[tuple[str, Path]] = []

    for curve in curve_names:
        path = folder / curve_name_to_filename(curve, ext=ext)
        if not path.exists():
            missing.append((curve, path))
            if strict:
                raise FileNotFoundError(f"Missing timeseries file for curve '{curve}': {path}")
            continue

        out[curve] = read_timeseries_like_csv(
            path,
            value_col_name=curve,
            read_csv_kwargs=kwargs,
            utc=utc,
        )

    if missing and not strict:
        print(f"Skipped {len(missing)} missing timeseries files. Example:", missing[0])

    return out


def load_curve_csvs_scenario(
    curve_names: Iterable[str],
    data_folder: str | Path = DEFAULT_CURVES_DIR,
    ext: str = ".csv",
    strict: bool = False,
    read_csv_kwargs: dict | None = None,
    utc: bool = True,
) -> CurveFrameMap:
    folder = Path(data_folder)
    kwargs = read_csv_kwargs or {}
    out: CurveFrameMap = {}
    missing: list[tuple[str, Path]] = []

    for curve in curve_names:
        path = folder / curve_name_to_filename(curve, ext=ext)
        if not path.exists():
            missing.append((curve, path))
            if strict:
                raise FileNotFoundError(f"Missing scenario file for curve '{curve}': {path}")
            continue

        out[curve] = read_scenario_csv(path, read_csv_kwargs=kwargs, utc=utc)

    if missing and not strict:
        print(f"Skipped {len(missing)} missing scenario files. Example:", missing[0])

    return out


def load_curve_csvs_ohlc(
    curve_names: Iterable[str],
    data_folder: str | Path = DEFAULT_CURVES_DIR,
    ext: str = ".csv",
    strict: bool = False,
    read_csv_kwargs: dict | None = None,
) -> CurveFrameMap:
    folder = Path(data_folder)
    kwargs = read_csv_kwargs or {}
    out: CurveFrameMap = {}
    missing: list[tuple[str, Path]] = []

    for curve in curve_names:
        path = folder / curve_name_to_filename(curve, ext=ext)
        if not path.exists():
            missing.append((curve, path))
            if strict:
                raise FileNotFoundError(f"Missing OHLC file for curve '{curve}': {path}")
            continue

        out[curve] = read_ohlc_csv(path, read_csv_kwargs=kwargs)

    if missing and not strict:
        print(f"Skipped {len(missing)} missing OHLC files. Example:", missing[0])

    return out


def load_curve_csvs_instance(
    curve_names: Iterable[str],
    curve_tags: dict[str, list[str]],
    data_folder: str | Path = DEFAULT_CURVES_DIR,
    ext: str = ".csv",
    strict: bool = False,
    read_csv_kwargs: dict | None = None,
    utc: bool = True,
) -> InstanceFrameMap:
    folder = Path(data_folder)
    kwargs = read_csv_kwargs or {}
    out: InstanceFrameMap = {}
    missing: list[tuple[str, str, Path]] = []

    for curve in curve_names:
        out[curve] = {}
        curve_file = curve_name_to_filename(curve, ext=ext)

        for tag in curve_tags.get(curve, []):
            path = folder / f"{tag}_{curve_file}"
            if not path.exists():
                missing.append((curve, tag, path))
                if strict:
                    raise FileNotFoundError(
                        f"Missing instance file curve='{curve}', tag='{tag}': {path}"
                    )
                continue

            out[curve][tag] = read_timeseries_like_csv(
                path,
                value_col_name=curve,
                read_csv_kwargs=kwargs,
                utc=utc,
            )

    if missing and not strict:
        print(f"Skipped {len(missing)} missing instance files. Example:", missing[0])

    return out


def load_all_curve_csvs(
    data_folder: str | Path = DEFAULT_CURVES_DIR,
    curve_tags: dict[str, list[str]] | None = None,
    ext: str = ".csv",
    strict: bool = False,
    read_csv_kwargs: dict | None = None,
    utc: bool = True,
) -> dict[str, CurveFrameMap | InstanceFrameMap]:
    resolved_curve_tags = curve_tags or infer_instance_tags_from_folder(
        INSTANCE_CURVES,
        data_folder=data_folder,
        ext=ext,
    )

    return {
        "timeseries": load_curve_csvs_timeseries(
            TIMESERIES_CURVES,
            data_folder=data_folder,
            ext=ext,
            strict=strict,
            read_csv_kwargs=read_csv_kwargs,
            utc=utc,
        ),
        "scenario": load_curve_csvs_scenario(
            SCENARIO_TIMESERIES_CURVES,
            data_folder=data_folder,
            ext=ext,
            strict=strict,
            read_csv_kwargs=read_csv_kwargs,
            utc=utc,
        ),
        "ohlc": load_curve_csvs_ohlc(
            OHLC_CURVES,
            data_folder=data_folder,
            ext=ext,
            strict=strict,
            read_csv_kwargs=read_csv_kwargs,
        ),
        "instance": load_curve_csvs_instance(
            INSTANCE_CURVES,
            curve_tags=resolved_curve_tags,
            data_folder=data_folder,
            ext=ext,
            strict=strict,
            read_csv_kwargs=read_csv_kwargs,
            utc=utc,
        ),
    }


def first_date_in_obj(obj: pd.DataFrame | None) -> pd.Timestamp | None:
    if obj is None:
        return None

    if isinstance(obj, pd.DataFrame) and isinstance(obj.index, pd.DatetimeIndex):
        if obj.index.size == 0:
            return None
        first = obj.index.min()
        if first.tzinfo is None:
            first = first.tz_localize("UTC")
        else:
            first = first.tz_convert("UTC")
        return first

    if isinstance(obj, pd.DataFrame):
        for column in ("traded", "date"):
            if column in obj.columns:
                first = pd.to_datetime(obj[column], errors="coerce", utc=True).min()
                return None if pd.isna(first) else first

    return None


def filter_dict_by_start_date(
    data: CurveFrameMap | InstanceFrameMap,
    cutoff_utc: pd.Timestamp | None = DEFAULT_CUTOFF,
) -> tuple[CurveFrameMap | InstanceFrameMap, list[str]]:
    if cutoff_utc is None:
        return data, []

    removed: list[str] = []

    if data and isinstance(next(iter(data.values())), dict):
        out: InstanceFrameMap = {}
        for curve, tag_map in data.items():
            kept_tag_map: dict[str, pd.DataFrame] = {}
            for tag, dataframe in tag_map.items():
                first_dt = first_date_in_obj(dataframe)
                if first_dt is not None and first_dt > cutoff_utc:
                    removed.append(f"{curve} | {tag}")
                else:
                    kept_tag_map[tag] = dataframe
            if kept_tag_map:
                out[curve] = kept_tag_map
        return out, removed

    out_flat: CurveFrameMap = {}
    for curve, dataframe in data.items():
        first_dt = first_date_in_obj(dataframe)
        if first_dt is not None and first_dt > cutoff_utc:
            removed.append(curve)
        else:
            out_flat[curve] = dataframe
    return out_flat, removed


def resample_dict(
    dfs: CurveFrameMap,
    key_substr: str = "15min",
    freq: str = "15min",
    keep_others: bool = True,
    utc: bool = True,
) -> CurveFrameMap:
    out: CurveFrameMap = {}

    for key, dataframe in dfs.items():
        if key_substr.lower() in key.lower() and dataframe is not None and not dataframe.empty:
            df = _ensure_datetime_index(dataframe, utc=utc)
            out[key] = df.resample(freq, label="left", closed="left").mean(numeric_only=True)
        elif keep_others:
            out[key] = dataframe

    return out


def resample_instance_dict_by_tag(
    dfs_in: InstanceFrameMap,
    key_substr: str = "15min",
    freq: str = "15min",
    keep_others: bool = True,
    tz: str | None = "UTC",
) -> InstanceFrameMap:
    out: InstanceFrameMap = {}

    for curve, tag_map in dfs_in.items():
        out[curve] = {}
        do_resample = key_substr.lower() in curve.lower()

        for tag, dataframe in (tag_map or {}).items():
            if dataframe is None or dataframe.empty:
                if keep_others:
                    out[curve][tag] = dataframe
                continue

            if not do_resample:
                if keep_others:
                    out[curve][tag] = dataframe
                continue

            df = _ensure_datetime_index(dataframe, utc=True)
            df = _normalize_datetime_index_timezone(df, tz=tz)
            out[curve][tag] = df.resample(
                freq,
                label="left",
                closed="left",
            ).mean(numeric_only=True)

    return out


def merge_instance_curves(
    dfs_in: InstanceFrameMap,
    sep: str = " | ",
    tz: str = "UTC",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    freq: str | None = None,
    how: str = "outer",
) -> pd.DataFrame:
    if not dfs_in:
        return pd.DataFrame()

    prepared: list[pd.DataFrame] = []
    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []

    for curve, tag_map in dfs_in.items():
        for tag, dataframe in (tag_map or {}).items():
            if dataframe is None or dataframe.empty:
                continue

            df = _ensure_datetime_index(dataframe, utc=True)
            df = _normalize_datetime_index_timezone(df, tz=tz)

            column_name = f"{curve}{sep}{tag}"
            if curve in df.columns:
                series_df = df[[curve]].rename(columns={curve: column_name})
            elif df.shape[1] == 1:
                source_column = df.columns[0]
                series_df = df[[source_column]].rename(columns={source_column: column_name})
            else:
                raise ValueError(
                    f"Ambiguous columns for curve={curve}, tag={tag}: {df.columns.tolist()}"
                )

            prepared.append(series_df)
            mins.append(series_df.index.min())
            maxs.append(series_df.index.max())

    if not prepared:
        return pd.DataFrame()

    global_start = min(mins) if start is None else pd.to_datetime(start)
    global_end = max(maxs) if end is None else pd.to_datetime(end)

    if freq is not None:
        tzinfo = global_start.tz
        if tzinfo is not None:
            if global_end.tz is None:
                global_end = global_end.tz_localize(tzinfo)
            else:
                global_end = global_end.tz_convert(tzinfo)
        target_index = pd.date_range(
            start=global_start,
            end=global_end,
            freq=freq,
            tz=tzinfo,
        )
        merged = pd.DataFrame(index=target_index)
        join_how = "left"
    else:
        target_index = None
        merged = None
        join_how = how

    for series_df in prepared:
        trimmed = series_df.loc[
            (series_df.index >= global_start) & (series_df.index <= global_end)
        ]
        if target_index is not None:
            trimmed = trimmed.reindex(target_index)
        merged = trimmed if merged is None else merged.merge(
            trimmed,
            how=join_how,
            left_index=True,
            right_index=True,
        )

    return pd.DataFrame() if merged is None else merged.sort_index()


def merge_timeseries_curves(
    dfs: CurveFrameMap,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    freq: str | None = None,
    how: str = "outer",
    prefix_sep: str = " | ",
) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()

    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []
    for dataframe in dfs.values():
        if dataframe is None or dataframe.empty:
            continue
        df = _ensure_datetime_index(dataframe, utc=True, errors="raise")
        mins.append(df.index.min())
        maxs.append(df.index.max())

    if not mins:
        return pd.DataFrame()

    global_start = min(mins) if start is None else pd.to_datetime(start)
    global_end = max(maxs) if end is None else pd.to_datetime(end)

    if freq is not None:
        tzinfo = global_start.tz
        if tzinfo is not None:
            if global_end.tz is None:
                global_end = global_end.tz_localize(tzinfo)
            else:
                global_end = global_end.tz_convert(tzinfo)
        target_index = pd.date_range(
            start=global_start,
            end=global_end,
            freq=freq,
            tz=tzinfo,
        )
        merged = pd.DataFrame(index=target_index)
        join_how = "left"
    else:
        target_index = None
        merged = None
        join_how = how

    def _rename_cols(columns: list[str], key: str) -> list[str]:
        column_names = [str(column) for column in columns]
        if len(column_names) == 1 and column_names[0] == key:
            return [column_names[0]]
        if len(column_names) == 1:
            return [key]
        return [f"{key}{prefix_sep}{column}" for column in column_names]

    for key, dataframe in dfs.items():
        if dataframe is None or dataframe.empty:
            continue

        df = _ensure_datetime_index(dataframe, utc=True, errors="raise")
        df.columns = _rename_cols(list(df.columns), str(key))
        if target_index is not None:
            df = df.reindex(target_index)
        merged = df if merged is None else merged.join(df, how=join_how)

    return pd.DataFrame() if merged is None else merged.sort_index()


def merge_scenario_curves(
    dfs_sc: CurveFrameMap,
    tz: str | None = "UTC",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    freq: str | None = None,
    how: str = "outer",
) -> pd.DataFrame:
    if not dfs_sc:
        return pd.DataFrame()

    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []
    normalized: CurveFrameMap = {}

    for curve, dataframe in dfs_sc.items():
        if dataframe is None or dataframe.empty:
            continue

        df = _ensure_datetime_index(dataframe, utc=True, errors="raise")
        df = _normalize_datetime_index_timezone(df, tz=tz)
        normalized[curve] = df
        mins.append(df.index.min())
        maxs.append(df.index.max())

    if not mins:
        return pd.DataFrame()

    global_start = min(mins) if start is None else pd.to_datetime(start)
    global_end = max(maxs) if end is None else pd.to_datetime(end)

    if freq is not None:
        tzinfo = global_start.tz
        if tzinfo is not None:
            if global_end.tz is None:
                global_end = global_end.tz_localize(tzinfo)
            else:
                global_end = global_end.tz_convert(tzinfo)
        target_index = pd.date_range(
            start=global_start,
            end=global_end,
            freq=freq,
            tz=tzinfo,
        )
        merged = pd.DataFrame(index=target_index)
        join_how = "left"
    else:
        target_index = None
        merged = None
        join_how = how

    for curve, dataframe in normalized.items():
        year_cols = [column for column in dataframe.columns if str(column).lower().startswith("y")]
        if not year_cols:
            raise ValueError(
                f"No year columns found for scenario curve '{curve}'. "
                f"Example cols: {list(dataframe.columns)[:10]}"
            )

        values = dataframe[year_cols]
        stats_df = pd.DataFrame(index=dataframe.index)
        stats_df[f"{curve}__min"] = values.min(axis=1, skipna=True)
        stats_df[f"{curve}__max"] = values.max(axis=1, skipna=True)
        stats_df[f"{curve}__mean"] = values.mean(axis=1, skipna=True)
        stats_df[f"{curve}__median"] = values.median(axis=1, skipna=True)
        stats_df[f"{curve}__std"] = values.std(axis=1, skipna=True)

        stats_df = stats_df.loc[
            (stats_df.index >= global_start) & (stats_df.index <= global_end)
        ]
        if target_index is not None:
            stats_df = stats_df.reindex(target_index)

        merged = stats_df if merged is None else merged.join(stats_df, how=join_how)

    if merged is None:
        return pd.DataFrame()

    merged = merged.sort_index()
    merged.index.name = "date"
    return merged


def merge_ohlc_curves(
    dfs_oh: CurveFrameMap,
    value: str = "settlement",
    traded_col: str = "traded",
    keep_global_minmax: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    if not dfs_oh:
        merged = pd.DataFrame()
        return (merged, None, None) if keep_global_minmax else merged

    series_list: list[pd.Series] = []
    global_min: pd.Timestamp | None = None
    global_max: pd.Timestamp | None = None

    for key, dataframe in dfs_oh.items():
        if dataframe is None or dataframe.empty:
            continue
        if traded_col not in dataframe.columns:
            raise KeyError(f"'{traded_col}' not found in df for key: {key}")
        if value not in dataframe.columns:
            raise KeyError(f"'{value}' not found in df for key: {key}")

        df = dataframe[[traded_col, value]].copy()
        df[traded_col] = pd.to_datetime(df[traded_col], errors="coerce")
        df[value] = pd.to_numeric(df[value], errors="coerce")
        df = df.dropna(subset=[traded_col])

        if not df.empty:
            df_min = df[traded_col].min()
            df_max = df[traded_col].max()
            global_min = df_min if global_min is None else min(global_min, df_min)
            global_max = df_max if global_max is None else max(global_max, df_max)

        series_list.append(df.groupby(traded_col)[value].mean().rename(key))

    if not series_list:
        merged = pd.DataFrame()
        return (merged, global_min, global_max) if keep_global_minmax else merged

    merged = pd.concat(series_list, axis=1).sort_index()
    merged.index.name = traded_col

    if keep_global_minmax:
        return merged, global_min, global_max
    return merged


def preprocess_all_from_curves_folder(
    data_folder: str | Path = DEFAULT_CURVES_DIR,
    curve_tags: dict[str, list[str]] | None = None,
    cutoff_utc: pd.Timestamp | None = DEFAULT_CUTOFF,
    read_csv_kwargs: dict | None = None,
    ext: str = ".csv",
) -> dict[str, pd.DataFrame | CurveFrameMap | InstanceFrameMap | list[str]]:
    loaded = load_all_curve_csvs(
        data_folder=data_folder,
        curve_tags=curve_tags,
        ext=ext,
        read_csv_kwargs=read_csv_kwargs,
    )

    removed: dict[str, list[str]] = {
        "timeseries": [],
        "scenario": [],
        "ohlc": [],
        "instance": [],
    }

    if cutoff_utc is not None:
        loaded["timeseries"], removed["timeseries"] = filter_dict_by_start_date(
            loaded["timeseries"],
            cutoff_utc=cutoff_utc,
        )
        loaded["scenario"], removed["scenario"] = filter_dict_by_start_date(
            loaded["scenario"],
            cutoff_utc=cutoff_utc,
        )
        loaded["ohlc"], removed["ohlc"] = filter_dict_by_start_date(
            loaded["ohlc"],
            cutoff_utc=cutoff_utc,
        )
        loaded["instance"], removed["instance"] = filter_dict_by_start_date(
            loaded["instance"],
            cutoff_utc=cutoff_utc,
        )

    timeseries_resampled = resample_dict(loaded["timeseries"], freq="h")
    scenario_resampled = resample_dict(loaded["scenario"], freq="h")
    instance_resampled = resample_instance_dict_by_tag(loaded["instance"], freq="h")

    merged = {
        "timeseries": merge_timeseries_curves(timeseries_resampled, freq="h"),
        "scenario": merge_scenario_curves(scenario_resampled, freq="h"),
        "ohlc": merge_ohlc_curves(loaded["ohlc"]),
        "instance": merge_instance_curves(instance_resampled, freq="h"),
    }

    return {
        "loaded": loaded,
        "removed": removed,
        "resampled": {
            "timeseries": timeseries_resampled,
            "scenario": scenario_resampled,
            "instance": instance_resampled,
        },
        "merged": merged,
    }


__all__ = [
    "DEFAULT_CUTOFF",
    "DEFAULT_CURVES_DIR",
    "CurveFrameMap",
    "InstanceFrameMap",
    "filter_dict_by_start_date",
    "first_date_in_obj",
    "infer_instance_tags_from_folder",
    "load_all_curve_csvs",
    "load_curve_csvs_instance",
    "load_curve_csvs_ohlc",
    "load_curve_csvs_scenario",
    "load_curve_csvs_timeseries",
    "merge_instance_curves",
    "merge_ohlc_curves",
    "merge_scenario_curves",
    "merge_timeseries_curves",
    "preprocess_all_from_curves_folder",
    "read_ohlc_csv",
    "read_scenario_csv",
    "read_timeseries_like_csv",
    "resample_dict",
    "resample_instance_dict_by_tag",
]
