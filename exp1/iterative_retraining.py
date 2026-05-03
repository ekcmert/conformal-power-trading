from __future__ import annotations

import argparse
import ast
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from base_models.interval_estimation.common import (
    DEFAULT_LOWER_QUANTILE,
    DEFAULT_UPPER_QUANTILE,
    build_prediction_frame,
    compute_interval_metrics,
)
from base_models.interval_estimation.model_registry import (
    ExperimentSpec as IntervalExperimentSpec,
    get_experiment_spec_groups as get_interval_experiment_spec_groups,
)
from base_models.point_estimation.common import (
    MCP_TARGETS,
    REPO_ROOT,
    compute_metrics,
    load_final_datasets,
    output_dir_for,
    standardize_features,
    target_artifact_stem,
)
from base_models.point_estimation.model_registry import (
    ExperimentSpec as PointExperimentSpec,
    get_experiment_spec_groups as get_point_experiment_spec_groups,
)
from base_models.tuning_grids import canonicalize_params


RANK_INTERVAL = 4
RANK_POINT = 8
TRAINING_RANGE = 24
TRAINING_FREQUENCY = 1
DATE_COLUMN = "date"

LOWER_QUANTILE = DEFAULT_LOWER_QUANTILE
UPPER_QUANTILE = DEFAULT_UPPER_QUANTILE

PREDICTIONS_ROOT = REPO_ROOT / "data" / "predictions"

POINT_RESULTS_DIR = REPO_ROOT / "results" / "point_estimation" / "mcp_models"
POINT_TUNING_RESULTS_DIR = REPO_ROOT / "results" / "tuning" / "point_estimation" / "mcp_models"
INTERVAL_RESULTS_DIR = REPO_ROOT / "results" / "interval_estimation" / "mcp_models"
INTERVAL_TUNING_RESULTS_DIR = REPO_ROOT / "results" / "tuning" / "interval_estimation" / "mcp_models"


@dataclass(frozen=True)
class ResultSource:
    label: str
    results_dir: Path
    tuning_active: bool


@dataclass(frozen=True)
class RollingWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    prediction_start: pd.Timestamp
    prediction_end: pd.Timestamp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MCP iterative retraining and prediction pipeline.",
    )
    parser.add_argument("--rank-point", type=int, default=RANK_POINT)
    parser.add_argument("--rank-interval", type=int, default=RANK_INTERVAL)
    parser.add_argument("--training-range", type=int, default=TRAINING_RANGE)
    parser.add_argument("--training-frequency", type=int, default=TRAINING_FREQUENCY)
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing point/interval prediction outputs before running.",
    )
    return parser.parse_args()


def _log(message: str) -> None:
    tqdm.write(message)


def _normalize_objective_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value or "")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    return df


def _pair_key(model_name: object, objective_name: object) -> tuple[str, str]:
    return str(model_name), _normalize_objective_name(objective_name)


def _result_key(row: pd.Series) -> tuple[str, str, str]:
    return (
        str(row["model_name"]),
        _normalize_objective_name(row["objective_name"]),
        str(row["target"]),
    )


def _month_start(timestamp: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(
        year=timestamp.year,
        month=timestamp.month,
        day=1,
        tz=timestamp.tz,
    )


def _results_sources(estimation_kind: str) -> list[ResultSource]:
    if estimation_kind == "point":
        return [
            ResultSource(label="standard", results_dir=POINT_RESULTS_DIR, tuning_active=False),
            ResultSource(label="tuning", results_dir=POINT_TUNING_RESULTS_DIR, tuning_active=True),
        ]
    if estimation_kind == "interval":
        return [
            ResultSource(label="standard", results_dir=INTERVAL_RESULTS_DIR, tuning_active=False),
            ResultSource(label="tuning", results_dir=INTERVAL_TUNING_RESULTS_DIR, tuning_active=True),
        ]
    raise ValueError(f"Unsupported estimation_kind={estimation_kind!r}.")


def _load_results_for_source(source: ResultSource) -> pd.DataFrame:
    results_path = source.results_dir / "all_model_results.csv"
    results_df = _safe_read_csv(results_path)
    if results_df.empty:
        return results_df

    results_df = results_df.copy()
    results_df["objective_name"] = results_df["objective_name"].apply(_normalize_objective_name)
    results_df["source_label"] = source.label
    results_df["source_results_dir"] = str(source.results_dir)
    results_df["source_results_csv"] = str(results_path)
    results_df["source_tuning_active"] = source.tuning_active
    results_df["source_preference"] = int(source.tuning_active)

    best_params_path = source.results_dir / "best_params.csv"
    best_params_df = _safe_read_csv(best_params_path)
    if not best_params_df.empty:
        best_params_df = best_params_df.copy()
        best_params_df["objective_name"] = best_params_df["objective_name"].apply(_normalize_objective_name)
        merge_columns = [
            "model_name",
            "objective_name",
            "best_params_json",
            "best_params_json_path",
            "selected_params_summary",
        ]
        available_merge_columns = [
            column_name for column_name in merge_columns if column_name in best_params_df.columns
        ]
        results_df = results_df.merge(
            best_params_df[available_merge_columns].drop_duplicates(subset=["model_name", "objective_name"]),
            on=["model_name", "objective_name"],
            how="left",
        )

    return results_df


def _rank_metric(
    series: pd.Series,
    *,
    ascending: bool,
) -> pd.Series:
    return series.rank(method="dense", ascending=ascending)


def _select_best_results(estimation_kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [
        _load_results_for_source(source)
        for source in _results_sources(estimation_kind)
    ]
    available_frames = [frame for frame in frames if not frame.empty]
    if not available_frames:
        raise FileNotFoundError(f"No MCP result CSVs were found for {estimation_kind} estimation.")

    combined_df = pd.concat(available_frames, ignore_index=True)
    combined_df = combined_df.copy()
    combined_df["objective_name"] = combined_df["objective_name"].apply(_normalize_objective_name)

    if estimation_kind == "point":
        combined_df = combined_df.sort_values(
            by=["mae", "rmse", "r2", "source_preference"],
            ascending=[True, True, False, False],
            kind="mergesort",
        )
    else:
        combined_df = combined_df.sort_values(
            by=["mean_pinball_loss", "mean_interval_width", "empirical_coverage", "source_preference"],
            ascending=[True, True, False, False],
            kind="mergesort",
        )

    best_by_pair_df = (
        combined_df.groupby(["model_name", "objective_name", "target"], as_index=False, sort=False)
        .first()
        .copy()
    )

    if estimation_kind == "point":
        best_by_pair_df["rmse_rank"] = _rank_metric(best_by_pair_df["rmse"], ascending=True)
        best_by_pair_df["mae_rank"] = _rank_metric(best_by_pair_df["mae"], ascending=True)
        best_by_pair_df["r2_rank"] = _rank_metric(best_by_pair_df["r2"], ascending=False)
        best_by_pair_df["average_rank"] = best_by_pair_df[["rmse_rank", "mae_rank", "r2_rank"]].mean(axis=1)
        best_by_pair_df = best_by_pair_df.sort_values(
            by=["average_rank", "mae", "rmse", "r2"],
            ascending=[True, True, True, False],
            kind="mergesort",
        ).reset_index(drop=True)
        selected_df = best_by_pair_df.head(RANK_POINT).copy()
    else:
        best_by_pair_df["pinball_rank"] = _rank_metric(best_by_pair_df["mean_pinball_loss"], ascending=True)
        best_by_pair_df["mean_interval_width_rank"] = _rank_metric(
            best_by_pair_df["mean_interval_width"],
            ascending=True,
        )
        best_by_pair_df["empirical_coverage_rank"] = _rank_metric(
            best_by_pair_df["empirical_coverage"],
            ascending=False,
        )
        best_by_pair_df["coverage_gap_to_nominal"] = (
            best_by_pair_df["empirical_coverage"] - (UPPER_QUANTILE - LOWER_QUANTILE)
        ).abs()
        best_by_pair_df["average_rank"] = best_by_pair_df[
            ["pinball_rank", "mean_interval_width_rank", "empirical_coverage_rank"]
        ].mean(axis=1)
        best_by_pair_df = best_by_pair_df.sort_values(
            by=["average_rank", "mean_pinball_loss", "mean_interval_width", "empirical_coverage"],
            ascending=[True, True, True, False],
            kind="mergesort",
        ).reset_index(drop=True)
        selected_df = best_by_pair_df.head(RANK_INTERVAL).copy()

    selected_df["selection_order"] = range(1, len(selected_df) + 1)
    selected_keys = {_result_key(selected_row) for _, selected_row in selected_df.iterrows()}
    best_by_pair_df["selected_for_prediction"] = best_by_pair_df.apply(
        lambda row: _result_key(row) in selected_keys,
        axis=1,
    )

    return best_by_pair_df, selected_df


def _params_from_row(row: pd.Series) -> dict[str, object]:
    best_params_json = row.get("best_params_json", "")
    if isinstance(best_params_json, str) and best_params_json.strip():
        return json.loads(best_params_json)

    for column_name in ("selected_params_summary", "params_summary"):
        raw_value = row.get(column_name, "")
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue
        try:
            parsed = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, dict):
            return parsed

    raise KeyError(
        "Could not resolve a parameter dictionary from the selected row for "
        f"{row['model_name']} / {_normalize_objective_name(row['objective_name'])}."
    )


def _build_point_spec_lookups() -> tuple[
    dict[tuple[str, str], PointExperimentSpec],
    dict[tuple[str, str], dict[tuple[tuple[str, object], ...], PointExperimentSpec]],
]:
    default_lookup: dict[tuple[str, str], PointExperimentSpec] = {}
    tuned_lookup: dict[tuple[str, str], dict[tuple[tuple[str, object], ...], PointExperimentSpec]] = {}

    for group in get_point_experiment_spec_groups(tuning_active=False):
        default_lookup[_pair_key(group.model_name, group.objective_name)] = group.default_spec

    for group in get_point_experiment_spec_groups(tuning_active=True):
        pair = _pair_key(group.model_name, group.objective_name)
        tuned_lookup[pair] = {
            canonicalize_params(spec.params): spec
            for spec in group.candidates
        }

    return default_lookup, tuned_lookup


def _build_interval_spec_lookups() -> tuple[
    dict[tuple[str, str], IntervalExperimentSpec],
    dict[tuple[str, str], dict[tuple[tuple[str, object], ...], IntervalExperimentSpec]],
]:
    default_lookup: dict[tuple[str, str], IntervalExperimentSpec] = {}
    tuned_lookup: dict[tuple[str, str], dict[tuple[tuple[str, object], ...], IntervalExperimentSpec]] = {}

    for group in get_interval_experiment_spec_groups(
        LOWER_QUANTILE,
        UPPER_QUANTILE,
        tuning_active=False,
    ):
        default_lookup[_pair_key(group.model_name, group.objective_name)] = group.default_spec

    for group in get_interval_experiment_spec_groups(
        LOWER_QUANTILE,
        UPPER_QUANTILE,
        tuning_active=True,
    ):
        pair = _pair_key(group.model_name, group.objective_name)
        tuned_lookup[pair] = {
            canonicalize_params(spec.params): spec
            for spec in group.candidates
        }

    return default_lookup, tuned_lookup


def _resolve_point_spec(
    row: pd.Series,
    default_lookup: dict[tuple[str, str], PointExperimentSpec],
    tuned_lookup: dict[tuple[str, str], dict[tuple[tuple[str, object], ...], PointExperimentSpec]],
) -> PointExperimentSpec:
    pair = _pair_key(row["model_name"], row["objective_name"])
    if bool(row["source_tuning_active"]):
        params = _params_from_row(row)
        return tuned_lookup[pair][canonicalize_params(params)]
    return default_lookup[pair]


def _resolve_interval_spec(
    row: pd.Series,
    default_lookup: dict[tuple[str, str], IntervalExperimentSpec],
    tuned_lookup: dict[tuple[str, str], dict[tuple[tuple[str, object], ...], IntervalExperimentSpec]],
) -> IntervalExperimentSpec:
    pair = _pair_key(row["model_name"], row["objective_name"])
    if bool(row["source_tuning_active"]):
        params = _params_from_row(row)
        return tuned_lookup[pair][canonicalize_params(params)]
    return default_lookup[pair]


def _build_rolling_windows(index: pd.DatetimeIndex) -> list[RollingWindow]:
    if index.empty:
        return []

    first_available = index.min()
    last_available = index.max()
    first_month = _month_start(first_available)
    last_month = _month_start(last_available)
    has_full_first_month = first_available == first_month

    prediction_start = first_month + pd.DateOffset(
        months=TRAINING_RANGE + (0 if has_full_first_month else 1)
    )
    windows: list[RollingWindow] = []

    while prediction_start <= last_month:
        prediction_end = prediction_start + pd.DateOffset(months=TRAINING_FREQUENCY)
        train_start = prediction_start - pd.DateOffset(months=TRAINING_RANGE)
        windows.append(
            RollingWindow(
                train_start=train_start,
                train_end=prediction_start,
                prediction_start=prediction_start,
                prediction_end=prediction_end,
            )
        )
        prediction_start = prediction_end

    return windows


def _write_selection_artifacts(
    estimation_kind: str,
    ranking_df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> None:
    base_dir = PREDICTIONS_ROOT / f"{estimation_kind}_estimation"
    base_dir.mkdir(parents=True, exist_ok=True)
    ranking_df.to_csv(base_dir / "model_ranking_summary.csv", index=False)
    selected_df.to_csv(base_dir / "selected_models.csv", index=False)


def _write_metadata(
    output_dir: Path,
    *,
    estimation_kind: str,
    row: pd.Series,
    spec_params: dict[str, object],
    window_count: int,
) -> None:
    metadata = {
        "pipeline": "iterative_retraining",
        "estimation_kind": estimation_kind,
        "targets": list(MCP_TARGETS),
        "model_name": row["model_name"],
        "objective_name": _normalize_objective_name(row["objective_name"]),
        "display_name": row["display_name"],
        "source_label": row["source_label"],
        "source_tuning_active": bool(row["source_tuning_active"]),
        "source_results_dir": row["source_results_dir"],
        "training_range_months": TRAINING_RANGE,
        "training_frequency_months": TRAINING_FREQUENCY,
        "window_count": window_count,
        "params": spec_params,
    }
    if estimation_kind == "point":
        metadata["ranking"] = {
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "r2": float(row["r2"]),
            "mae_rank": float(row["mae_rank"]),
            "rmse_rank": float(row["rmse_rank"]),
            "r2_rank": float(row["r2_rank"]),
            "average_rank": float(row["average_rank"]),
            "selection_order": int(row["selection_order"]),
        }
    else:
        metadata["ranking"] = {
            "mean_pinball_loss": float(row["mean_pinball_loss"]),
            "empirical_coverage": float(row["empirical_coverage"]),
            "mean_interval_width": float(row["mean_interval_width"]),
            "pinball_rank": float(row["pinball_rank"]),
            "mean_interval_width_rank": float(row["mean_interval_width_rank"]),
            "empirical_coverage_rank": float(row["empirical_coverage_rank"]),
            "average_rank": float(row["average_rank"]),
            "selection_order": int(row["selection_order"]),
            "lower_quantile": LOWER_QUANTILE,
            "upper_quantile": UPPER_QUANTILE,
        }

    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _prediction_iteration_path(output_dir: Path, window_index: int) -> Path:
    return output_dir / f"pred_{window_index + 1}.csv"


def _checkpoint_date_columns() -> list[str]:
    return [
        DATE_COLUMN,
        "train_start",
        "train_end_exclusive",
        "prediction_start",
        "prediction_end_exclusive",
    ]


def _load_prediction_checkpoint(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=_checkpoint_date_columns())


def _checkpoint_matches_window(path: Path, window: RollingWindow) -> bool:
    if not path.exists():
        return False

    try:
        checkpoint_df = pd.read_csv(path, nrows=1, parse_dates=_checkpoint_date_columns())
    except (pd.errors.EmptyDataError, ValueError):
        return False

    required_columns = {
        "train_start",
        "train_end_exclusive",
        "prediction_start",
        "prediction_end_exclusive",
    }
    if checkpoint_df.empty or not required_columns.issubset(checkpoint_df.columns):
        return False

    checkpoint_row = checkpoint_df.iloc[0]
    return (
        pd.Timestamp(checkpoint_row["train_start"]) == window.train_start
        and pd.Timestamp(checkpoint_row["train_end_exclusive"]) == window.train_end
        and pd.Timestamp(checkpoint_row["prediction_start"]) == window.prediction_start
        and pd.Timestamp(checkpoint_row["prediction_end_exclusive"]) == window.prediction_end
    )


def _window_has_data(X: pd.DataFrame, window: RollingWindow) -> bool:
    train_mask = (X.index >= window.train_start) & (X.index < window.train_end)
    prediction_mask = (X.index >= window.prediction_start) & (X.index < window.prediction_end)
    return bool(train_mask.any() and prediction_mask.any())


def _expected_window_indices(
    X: pd.DataFrame,
    windows: list[RollingWindow],
) -> list[int]:
    return [
        window_index
        for window_index, window in enumerate(windows)
        if _window_has_data(X, window)
    ]


def _point_metrics_from_checkpoint(
    X: pd.DataFrame,
    checkpoint_df: pd.DataFrame,
    window_index: int,
    window: RollingWindow,
) -> dict[str, object]:
    train_mask = (X.index >= window.train_start) & (X.index < window.train_end)
    metrics = compute_metrics(
        checkpoint_df["y_true"],
        checkpoint_df["y_pred"],
    )
    return {
        "window_index": window_index,
        "prediction_start": window.prediction_start,
        "prediction_end_exclusive": window.prediction_end,
        "train_start": window.train_start,
        "train_end_exclusive": window.train_end,
        "train_rows": int(train_mask.sum()),
        "prediction_rows": int(len(checkpoint_df)),
        **metrics,
    }


def _interval_metrics_from_checkpoint(
    X: pd.DataFrame,
    checkpoint_df: pd.DataFrame,
    window_index: int,
    window: RollingWindow,
) -> dict[str, object]:
    train_mask = (X.index >= window.train_start) & (X.index < window.train_end)
    metrics = compute_interval_metrics(
        checkpoint_df["y_true"],
        checkpoint_df["y_pred_lower"],
        checkpoint_df["y_pred_upper"],
        lower_quantile=LOWER_QUANTILE,
        upper_quantile=UPPER_QUANTILE,
    )
    return {
        "window_index": window_index,
        "prediction_start": window.prediction_start,
        "prediction_end_exclusive": window.prediction_end,
        "train_start": window.train_start,
        "train_end_exclusive": window.train_end,
        "train_rows": int(train_mask.sum()),
        "prediction_rows": int(len(checkpoint_df)),
        **metrics,
    }


def _run_point_pipeline(
    X: pd.DataFrame,
    y: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    default_lookup, tuned_lookup = _build_point_spec_lookups()
    windows = _build_rolling_windows(X.index)
    expected_window_indices = _expected_window_indices(X, windows)
    base_dir = PREDICTIONS_ROOT / "point_estimation"
    base_dir.mkdir(parents=True, exist_ok=True)

    _log("")
    _log("=" * 80)
    _log("Selected Point Models")
    _log("=" * 80)
    if not selected_df.empty:
        printable_columns = ["selection_order", "display_name", "source_label", "mae", "rmse", "r2", "average_rank"]
        _log(selected_df[printable_columns].to_string(index=False))
    _log("=" * 80)

    with tqdm(
        total=len(selected_df),
        desc="Point Models",
        unit="model",
        dynamic_ncols=True,
    ) as model_progress:
        for _, row in selected_df.iterrows():
            model_progress.set_postfix_str(str(row["display_name"]))
            try:
                spec = _resolve_point_spec(row, default_lookup, tuned_lookup)
                pair_output_dir = output_dir_for(
                    base_dir,
                    str(row["model_name"]),
                    _normalize_objective_name(row["objective_name"]),
                )
                pair_output_dir.mkdir(parents=True, exist_ok=True)

                _log(
                    f"[Point {int(row['selection_order'])}/{len(selected_df)}] "
                    f"{row['display_name']} | source={row['source_label']} | avg_rank={row['average_rank']:.3f}"
                )

                combined_predictions: list[pd.DataFrame] = []
                monthly_metrics: list[dict[str, object]] = []
                matching_checkpoint_indices = {
                    window_index
                    for window_index in expected_window_indices
                    if _checkpoint_matches_window(
                        _prediction_iteration_path(pair_output_dir, window_index),
                        windows[window_index],
                    )
                }

                if expected_window_indices and len(matching_checkpoint_indices) == len(expected_window_indices):
                    _log(
                        f"Skipping training for {row['display_name']}: "
                        f"found {len(matching_checkpoint_indices)}/{len(expected_window_indices)} matching prediction checkpoints."
                    )
                    for window_index in expected_window_indices:
                        window = windows[window_index]
                        checkpoint_df = _load_prediction_checkpoint(
                            _prediction_iteration_path(pair_output_dir, window_index)
                        )
                        combined_predictions.append(checkpoint_df)
                        monthly_metrics.append(
                            _point_metrics_from_checkpoint(X, checkpoint_df, window_index, window)
                        )
                else:
                    with tqdm(
                        total=len(expected_window_indices),
                        desc=f"Rolling | {row['display_name']}",
                        unit="month",
                        dynamic_ncols=True,
                        leave=False,
                    ) as window_progress:
                        for window_index in expected_window_indices:
                            window = windows[window_index]
                            checkpoint_path = _prediction_iteration_path(pair_output_dir, window_index)
                            window_progress.set_postfix_str(window.prediction_start.strftime("%Y-%m"))

                            if _checkpoint_matches_window(checkpoint_path, window):
                                checkpoint_df = _load_prediction_checkpoint(checkpoint_path)
                                combined_predictions.append(checkpoint_df)
                                monthly_metrics.append(
                                    _point_metrics_from_checkpoint(X, checkpoint_df, window_index, window)
                                )
                                window_progress.update(1)
                                continue

                            train_mask = (X.index >= window.train_start) & (X.index < window.train_end)
                            prediction_mask = (X.index >= window.prediction_start) & (X.index < window.prediction_end)

                            X_train_raw = X.loc[train_mask].copy()
                            X_prediction_raw = X.loc[prediction_mask].copy()
                            y_train = y.loc[train_mask].copy()
                            y_prediction = y.loc[prediction_mask].copy()

                            if X_train_raw.empty or X_prediction_raw.empty:
                                window_progress.update(1)
                                continue

                            X_train, X_prediction, _ = standardize_features(X_train_raw, X_prediction_raw)
                            model = spec.builder()
                            model.fit(X_train, y_train[MCP_TARGETS[0]])
                            prediction_values = model.predict(X_prediction)

                            prediction_frame = pd.DataFrame(
                                {
                                    DATE_COLUMN: X_prediction.index,
                                    "y_true": y_prediction[MCP_TARGETS[0]].values,
                                    "y_pred": prediction_values,
                                    "train_start": window.train_start,
                                    "train_end_exclusive": window.train_end,
                                    "prediction_start": window.prediction_start,
                                    "prediction_end_exclusive": window.prediction_end,
                                }
                            )
                            prediction_frame.to_csv(checkpoint_path, index=False)
                            combined_predictions.append(prediction_frame)

                            metrics = compute_metrics(
                                y_prediction[MCP_TARGETS[0]],
                                pd.Series(prediction_values, index=X_prediction.index),
                            )
                            monthly_metrics.append(
                                {
                                    "window_index": window_index,
                                    "prediction_start": window.prediction_start,
                                    "prediction_end_exclusive": window.prediction_end,
                                    "train_start": window.train_start,
                                    "train_end_exclusive": window.train_end,
                                    "train_rows": int(len(X_train_raw)),
                                    "prediction_rows": int(len(X_prediction_raw)),
                                    **metrics,
                                }
                            )
                            window_progress.update(1)

                if not combined_predictions:
                    raise RuntimeError("No rolling prediction windows produced any rows.")

                all_predictions_df = (
                    pd.concat(combined_predictions, ignore_index=True)
                    .sort_values(by=[DATE_COLUMN], kind="mergesort")
                    .reset_index(drop=True)
                )
                monthly_metrics_df = (
                    pd.DataFrame(monthly_metrics)
                    .sort_values(by=["window_index"], kind="mergesort")
                    .reset_index(drop=True)
                )
                overall_metrics = compute_metrics(
                    all_predictions_df["y_true"],
                    all_predictions_df["y_pred"],
                )
                overall_metrics_df = pd.DataFrame(
                    [
                        {
                            "train_range_months": TRAINING_RANGE,
                            "training_frequency_months": TRAINING_FREQUENCY,
                            "prediction_rows": int(len(all_predictions_df)),
                            "window_count": int(len(monthly_metrics_df)),
                            **overall_metrics,
                        }
                    ]
                )

                prediction_path = pair_output_dir / f"{target_artifact_stem(MCP_TARGETS[0])}_predictions.csv"
                all_predictions_df.to_csv(prediction_path, index=False)
                monthly_metrics_df.to_csv(pair_output_dir / "monthly_metrics.csv", index=False)
                overall_metrics_df.to_csv(pair_output_dir / "overall_metrics.csv", index=False)
                _write_metadata(
                    pair_output_dir,
                    estimation_kind="point",
                    row=row,
                    spec_params=dict(spec.params),
                    window_count=len(monthly_metrics_df),
                )

                _log(
                    f"Saved point predictions to {prediction_path} | "
                    f"rows={len(all_predictions_df):,} | MAE={overall_metrics['mae']:.4f} | "
                    f"RMSE={overall_metrics['rmse']:.4f} | R2={overall_metrics['r2']:.4f}"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append((_pair_key(row["model_name"], row["objective_name"])[0], repr(exc)))
                _log(f"FAILED point model {row['display_name']}: {exc}")
            finally:
                model_progress.update(1)

    return failures


def _run_interval_pipeline(
    X: pd.DataFrame,
    y: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    default_lookup, tuned_lookup = _build_interval_spec_lookups()
    windows = _build_rolling_windows(X.index)
    expected_window_indices = _expected_window_indices(X, windows)
    base_dir = PREDICTIONS_ROOT / "interval_estimation"
    base_dir.mkdir(parents=True, exist_ok=True)

    _log("")
    _log("=" * 80)
    _log("Selected Interval Models")
    _log("=" * 80)
    if not selected_df.empty:
        printable_columns = [
            "selection_order",
            "display_name",
            "source_label",
            "mean_pinball_loss",
            "empirical_coverage",
            "mean_interval_width",
            "average_rank",
        ]
        _log(selected_df[printable_columns].to_string(index=False))
    _log("=" * 80)

    with tqdm(
        total=len(selected_df),
        desc="Interval Models",
        unit="model",
        dynamic_ncols=True,
    ) as model_progress:
        for _, row in selected_df.iterrows():
            model_progress.set_postfix_str(str(row["display_name"]))
            try:
                spec = _resolve_interval_spec(row, default_lookup, tuned_lookup)
                pair_output_dir = output_dir_for(
                    base_dir,
                    str(row["model_name"]),
                    _normalize_objective_name(row["objective_name"]),
                )
                pair_output_dir.mkdir(parents=True, exist_ok=True)

                _log(
                    f"[Interval {int(row['selection_order'])}/{len(selected_df)}] "
                    f"{row['display_name']} | source={row['source_label']} | avg_rank={row['average_rank']:.3f}"
                )

                combined_predictions: list[pd.DataFrame] = []
                monthly_metrics: list[dict[str, object]] = []
                matching_checkpoint_indices = {
                    window_index
                    for window_index in expected_window_indices
                    if _checkpoint_matches_window(
                        _prediction_iteration_path(pair_output_dir, window_index),
                        windows[window_index],
                    )
                }

                if expected_window_indices and len(matching_checkpoint_indices) == len(expected_window_indices):
                    _log(
                        f"Skipping training for {row['display_name']}: "
                        f"found {len(matching_checkpoint_indices)}/{len(expected_window_indices)} matching prediction checkpoints."
                    )
                    for window_index in expected_window_indices:
                        window = windows[window_index]
                        checkpoint_df = _load_prediction_checkpoint(
                            _prediction_iteration_path(pair_output_dir, window_index)
                        )
                        combined_predictions.append(checkpoint_df)
                        monthly_metrics.append(
                            _interval_metrics_from_checkpoint(X, checkpoint_df, window_index, window)
                        )
                else:
                    with tqdm(
                        total=len(expected_window_indices),
                        desc=f"Rolling | {row['display_name']}",
                        unit="month",
                        dynamic_ncols=True,
                        leave=False,
                    ) as window_progress:
                        for window_index in expected_window_indices:
                            window = windows[window_index]
                            checkpoint_path = _prediction_iteration_path(pair_output_dir, window_index)
                            window_progress.set_postfix_str(window.prediction_start.strftime("%Y-%m"))

                            if _checkpoint_matches_window(checkpoint_path, window):
                                checkpoint_df = _load_prediction_checkpoint(checkpoint_path)
                                combined_predictions.append(checkpoint_df)
                                monthly_metrics.append(
                                    _interval_metrics_from_checkpoint(X, checkpoint_df, window_index, window)
                                )
                                window_progress.update(1)
                                continue

                            train_mask = (X.index >= window.train_start) & (X.index < window.train_end)
                            prediction_mask = (X.index >= window.prediction_start) & (X.index < window.prediction_end)

                            X_train_raw = X.loc[train_mask].copy()
                            X_prediction_raw = X.loc[prediction_mask].copy()
                            y_train = y.loc[train_mask].copy()
                            y_prediction = y.loc[prediction_mask].copy()

                            if X_train_raw.empty or X_prediction_raw.empty:
                                window_progress.update(1)
                                continue

                            X_train, X_prediction, _ = standardize_features(X_train_raw, X_prediction_raw)
                            model = spec.builder()
                            model.fit(X_train, y_train[MCP_TARGETS[0]])
                            if not hasattr(model, "predict_interval"):
                                raise TypeError(f"{row['display_name']} does not implement predict_interval().")

                            lower_values, upper_values = model.predict_interval(X_prediction)
                            prediction_frame = build_prediction_frame(
                                y_prediction[MCP_TARGETS[0]],
                                pd.Series(lower_values, index=X_prediction.index),
                                pd.Series(upper_values, index=X_prediction.index),
                            ).reset_index()
                            prediction_frame.columns = [DATE_COLUMN, *prediction_frame.columns.tolist()[1:]]
                            prediction_frame["train_start"] = window.train_start
                            prediction_frame["train_end_exclusive"] = window.train_end
                            prediction_frame["prediction_start"] = window.prediction_start
                            prediction_frame["prediction_end_exclusive"] = window.prediction_end
                            prediction_frame.to_csv(checkpoint_path, index=False)
                            combined_predictions.append(prediction_frame)

                            metrics = compute_interval_metrics(
                                y_prediction[MCP_TARGETS[0]],
                                pd.Series(lower_values, index=X_prediction.index),
                                pd.Series(upper_values, index=X_prediction.index),
                                lower_quantile=LOWER_QUANTILE,
                                upper_quantile=UPPER_QUANTILE,
                            )
                            monthly_metrics.append(
                                {
                                    "window_index": window_index,
                                    "prediction_start": window.prediction_start,
                                    "prediction_end_exclusive": window.prediction_end,
                                    "train_start": window.train_start,
                                    "train_end_exclusive": window.train_end,
                                    "train_rows": int(len(X_train_raw)),
                                    "prediction_rows": int(len(X_prediction_raw)),
                                    **metrics,
                                }
                            )
                            window_progress.update(1)

                if not combined_predictions:
                    raise RuntimeError("No rolling prediction windows produced any rows.")

                all_predictions_df = (
                    pd.concat(combined_predictions, ignore_index=True)
                    .sort_values(by=[DATE_COLUMN], kind="mergesort")
                    .reset_index(drop=True)
                )
                monthly_metrics_df = (
                    pd.DataFrame(monthly_metrics)
                    .sort_values(by=["window_index"], kind="mergesort")
                    .reset_index(drop=True)
                )
                overall_metrics = compute_interval_metrics(
                    all_predictions_df["y_true"],
                    all_predictions_df["y_pred_lower"],
                    all_predictions_df["y_pred_upper"],
                    lower_quantile=LOWER_QUANTILE,
                    upper_quantile=UPPER_QUANTILE,
                )
                overall_metrics_df = pd.DataFrame(
                    [
                        {
                            "train_range_months": TRAINING_RANGE,
                            "training_frequency_months": TRAINING_FREQUENCY,
                            "prediction_rows": int(len(all_predictions_df)),
                            "window_count": int(len(monthly_metrics_df)),
                            **overall_metrics,
                        }
                    ]
                )

                prediction_path = pair_output_dir / f"{target_artifact_stem(MCP_TARGETS[0])}_predictions.csv"
                all_predictions_df.to_csv(prediction_path, index=False)
                monthly_metrics_df.to_csv(pair_output_dir / "monthly_metrics.csv", index=False)
                overall_metrics_df.to_csv(pair_output_dir / "overall_metrics.csv", index=False)
                _write_metadata(
                    pair_output_dir,
                    estimation_kind="interval",
                    row=row,
                    spec_params=dict(spec.params),
                    window_count=len(monthly_metrics_df),
                )

                _log(
                    f"Saved interval predictions to {prediction_path} | "
                    f"rows={len(all_predictions_df):,} | Pinball={overall_metrics['mean_pinball_loss']:.4f} | "
                    f"Coverage={overall_metrics['empirical_coverage']:.4f} | "
                    f"Width={overall_metrics['mean_interval_width']:.4f}"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append((_pair_key(row["model_name"], row["objective_name"])[0], repr(exc)))
                _log(f"FAILED interval model {row['display_name']}: {exc}")
            finally:
                model_progress.update(1)

    return failures


def _write_failures(estimation_kind: str, failures: list[tuple[str, str]]) -> None:
    base_dir = PREDICTIONS_ROOT / f"{estimation_kind}_estimation"
    failures_path = base_dir / "failures.csv"
    if failures:
        pd.DataFrame(failures, columns=["model_name", "error"]).to_csv(failures_path, index=False)
    else:
        failures_path.unlink(missing_ok=True)


def main() -> int:
    global RANK_INTERVAL, RANK_POINT, TRAINING_RANGE, TRAINING_FREQUENCY

    args = _parse_args()
    if args.rank_point <= 0:
        raise ValueError(f"--rank-point must be positive, got {args.rank_point}.")
    if args.rank_interval <= 0:
        raise ValueError(f"--rank-interval must be positive, got {args.rank_interval}.")
    if args.training_range <= 0:
        raise ValueError(f"--training-range must be positive, got {args.training_range}.")
    if args.training_frequency <= 0:
        raise ValueError(f"--training-frequency must be positive, got {args.training_frequency}.")

    RANK_POINT = args.rank_point
    RANK_INTERVAL = args.rank_interval
    TRAINING_RANGE = args.training_range
    TRAINING_FREQUENCY = args.training_frequency

    if args.clean_output:
        shutil.rmtree(PREDICTIONS_ROOT / "point_estimation", ignore_errors=True)
        shutil.rmtree(PREDICTIONS_ROOT / "interval_estimation", ignore_errors=True)

    PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    X, y = load_final_datasets(MCP_TARGETS)
    rolling_windows = _build_rolling_windows(X.index)
    if not rolling_windows:
        raise RuntimeError("No rolling monthly windows could be built from the available MCP data.")

    _log("")
    _log("=" * 80)
    _log("Iterative Retraining Pipeline")
    _log("=" * 80)
    _log(f"Targets              : {list(MCP_TARGETS)}")
    _log(f"Training range       : {TRAINING_RANGE} month(s)")
    _log(f"Training frequency   : {TRAINING_FREQUENCY} month(s)")
    _log(f"Point model cap      : top {RANK_POINT}")
    _log(f"Interval model cap   : top {RANK_INTERVAL}")
    _log(f"Quantiles            : ({LOWER_QUANTILE:.2f}, {UPPER_QUANTILE:.2f})")
    _log(f"Prediction root      : {PREDICTIONS_ROOT}")
    _log(
        f"Rolling windows      : {len(rolling_windows)} month(s) "
        f"from {rolling_windows[0].prediction_start} to {rolling_windows[-1].prediction_start}"
    )
    _log("=" * 80)

    point_ranking_df, point_selected_df = _select_best_results("point")
    interval_ranking_df, interval_selected_df = _select_best_results("interval")

    _write_selection_artifacts("point", point_ranking_df, point_selected_df)
    _write_selection_artifacts("interval", interval_ranking_df, interval_selected_df)

    point_failures = _run_point_pipeline(X, y, point_selected_df)
    interval_failures = _run_interval_pipeline(X, y, interval_selected_df)

    _write_failures("point", point_failures)
    _write_failures("interval", interval_failures)

    total_failures = len(point_failures) + len(interval_failures)
    _log("")
    _log("=" * 80)
    if total_failures:
        _log(f"Pipeline completed with {total_failures} failure(s).")
        _log(f"Point failures    : {len(point_failures)}")
        _log(f"Interval failures : {len(interval_failures)}")
        _log("=" * 80)
        return 1

    _log("Pipeline completed successfully.")
    _log("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
