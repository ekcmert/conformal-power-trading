from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from base_models.interval_estimation.common import (
    DEFAULT_LOWER_QUANTILE,
    DEFAULT_UPPER_QUANTILE,
)
from base_models.point_estimation.common import REPO_ROOT, output_dir_for, target_artifact_stem
from conformal_prediction import ConformalPrediction, ConformalizedQuantileRegression
from exp2 import regime_free_experiment as rfe
from exp3.mondrian_conformal_prediction import (
    METHOD_MODEL_LOSS_LIST as BASE_METHOD_MODEL_LOSS_LIST,
    MethodSelection,
    resolve_method_selections,
    _filter_artifacts,
    _method_is_selected,
)


CALIBRATION_RANGE = 24
CALIBRATION_FREQUENCY = 1
TARGET_LIST = ["DA", "ID", "ID3", "ID1", "IMB"]

# TARGET_LIST = ["DA"]

DATE_COLUMN = "date"
LOWER_QUANTILE = DEFAULT_LOWER_QUANTILE
UPPER_QUANTILE = DEFAULT_UPPER_QUANTILE
ALPHA = 1.0 - (UPPER_QUANTILE - LOWER_QUANTILE)

PREDICTIONS_ROOT = REPO_ROOT / "data" / "predictions"
RESIDUALS_ROOT = REPO_ROOT / "data" / "residuals"
SCALES_ROOT = REPO_ROOT / "data" / "scales"
Y_PATH = REPO_ROOT / "data" / "final" / "y.csv"
RESULTS_ROOT = REPO_ROOT / "results" / "regime_aware" / "lacp"

SCALE_FILENAME = "scales.csv"
SCALE_COLUMN_NAME = "target_column"
METHOD_MODEL_LOSS_LIST: tuple[object, ...] = BASE_METHOD_MODEL_LOSS_LIST
SCALE_NAME_LIST: tuple[str, ...] = ()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local adaptive conformal recalibration on saved predictions and residuals.",
    )
    parser.add_argument("--calibration-range", type=int, default=CALIBRATION_RANGE)
    parser.add_argument("--calibration-frequency", type=int, default=CALIBRATION_FREQUENCY)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=TARGET_LIST,
        help="Subset of targets to run. Available: DA ID ID3 ID1 IMB",
    )
    parser.add_argument(
        "--method-spec",
        action="append",
        default=[],
        help=(
            "Optional method selection using regime-free folder naming, for example "
            "'cp_symmetric_lightgbm/regression' or 'cqr_lightgbm/quantile'. "
            "Repeat to run multiple combinations."
        ),
    )
    parser.add_argument(
        "--scale-names",
        nargs="+",
        default=list(SCALE_NAME_LIST),
        help="Optional subset of scale subfolders under data/scales.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing results/regime_aware/lacp outputs before running.",
    )
    return parser.parse_args()


def _load_scale_series(csv_path: Path) -> tuple[pd.Series, str]:
    scale_frame = pd.read_csv(csv_path, parse_dates=[DATE_COLUMN])
    if DATE_COLUMN not in scale_frame.columns:
        raise ValueError(f"Expected a {DATE_COLUMN!r} column in {csv_path}.")

    non_date_columns = [column for column in scale_frame.columns if column != DATE_COLUMN]
    if not non_date_columns:
        raise KeyError(f"No scale column found in {csv_path}.")

    if SCALE_COLUMN_NAME in non_date_columns:
        scale_column = SCALE_COLUMN_NAME
    elif len(non_date_columns) == 1:
        scale_column = non_date_columns[0]
    else:
        raise ValueError(
            f"Expected a single scale column in {csv_path}, found {non_date_columns}."
        )

    scale_series = (
        scale_frame.set_index(DATE_COLUMN)[scale_column]
        .astype(float)
        .sort_index()
    )
    return scale_series, scale_column


def _discover_scale_csvs(
    scales_root: Path,
    requested_scale_names: Iterable[str] | None = None,
) -> list[tuple[str, Path]]:
    scales_root = scales_root.resolve()
    requested = {str(value).strip() for value in (requested_scale_names or []) if str(value).strip()}
    discovered: list[tuple[str, Path]] = []

    if not scales_root.exists():
        raise FileNotFoundError(f"Scales root does not exist: {scales_root}")

    for scale_dir in sorted(path for path in scales_root.iterdir() if path.is_dir()):
        if requested and scale_dir.name not in requested:
            continue
        csv_path = scale_dir / SCALE_FILENAME
        if csv_path.exists():
            discovered.append((scale_dir.name, csv_path))

    if requested:
        found_names = {name for name, _ in discovered}
        missing = sorted(requested - found_names)
        if missing:
            raise FileNotFoundError(
                f"Requested scales were not found under {scales_root}: {missing}"
            )

    if not discovered:
        raise FileNotFoundError(
            f"No {SCALE_FILENAME!r} files were found directly under subfolders of {scales_root}."
        )

    return discovered


def _validate_scale_alignment(
    scale_series: pd.Series,
    index: pd.DatetimeIndex,
    *,
    label: str,
) -> pd.Series:
    aligned = scale_series.reindex(index)
    if aligned.isna().any():
        missing_dates = aligned[aligned.isna()].index.astype(str).tolist()[:5]
        raise KeyError(
            f"Missing scale values for {label} on dates {missing_dates}. "
            "Ensure scales.csv spans the full prediction history."
        )
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError(f"Scale values for {label} contain non-finite values.")
    if (aligned <= 0.0).any():
        raise ValueError(f"Scale values for {label} must all be positive.")
    return aligned.astype(float)


def _write_pipeline_metadata(
    *,
    output_dir: Path,
    target_spec: rfe.TargetSpec,
    artifacts: rfe.ModelArtifacts,
    method_name: str,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
    window_count: int,
    residual_paths: list[Path],
    extra_metadata: dict[str, object],
) -> None:
    metadata = {
        "pipeline": "local_adaptive_conformal_prediction",
        "target_code": target_spec.code,
        "target_slug": target_spec.slug,
        "target_name": target_spec.y_column,
        "method_name": method_name,
        "estimation_kind": artifacts.estimation_kind,
        "model_name": artifacts.model_name,
        "objective_name": artifacts.objective_name,
        "base_display_name": artifacts.display_name,
        "prediction_path": str(artifacts.prediction_path),
        "residual_paths": [str(path) for path in residual_paths],
        "calibration_range_weeks": calibration_range_weeks,
        "calibration_frequency_weeks": calibration_frequency_weeks,
        "alpha": ALPHA,
        "lower_quantile": LOWER_QUANTILE,
        "upper_quantile": UPPER_QUANTILE,
        "window_count": window_count,
        "source_metadata": artifacts.metadata,
        **extra_metadata,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_lacp_result_frame(
    *,
    y_true: pd.Series,
    lower_pred: pd.Series,
    upper_pred: pd.Series,
    base_lower_pred: pd.Series,
    base_upper_pred: pd.Series,
    margin_lower: pd.Series,
    margin_upper: pd.Series,
    scaled_margin_lower: pd.Series,
    scaled_margin_upper: pd.Series,
    scale_values: pd.Series,
    window_index: int,
    window: rfe.RollingWindow,
    calibration_rows: int,
    scale_name: str,
) -> pd.DataFrame:
    prediction_frame = rfe._build_result_prediction_frame(
        y_true=y_true,
        lower_pred=lower_pred,
        upper_pred=upper_pred,
        base_lower_pred=base_lower_pred,
        base_upper_pred=base_upper_pred,
        margin_lower=margin_lower,
        margin_upper=margin_upper,
        window_index=window_index,
        window=window,
        calibration_rows=calibration_rows,
    )
    prediction_frame["scale_name"] = scale_name
    prediction_frame["scale_value"] = scale_values.to_numpy()
    prediction_frame["scaled_margin_lower"] = scaled_margin_lower.to_numpy()
    prediction_frame["scaled_margin_upper"] = scaled_margin_upper.to_numpy()
    return prediction_frame


def _run_scaled_point_method(
    *,
    artifacts: rfe.ModelArtifacts,
    target_spec: rfe.TargetSpec,
    prediction_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    residual_series: pd.Series,
    scale_series: pd.Series,
    scale_name: str,
    symmetric: bool,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if "y_pred" not in prediction_frame.columns:
        raise KeyError(f"Missing y_pred column in {artifacts.prediction_path}.")

    y_series = y_frame[target_spec.y_column].reindex(prediction_frame.index)
    if y_series.isna().any():
        raise KeyError(f"Missing y values for target {target_spec.y_column}.")

    windows = rfe._build_rolling_windows(
        prediction_frame.index,
        calibration_range_weeks=calibration_range_weeks,
        calibration_frequency_weeks=calibration_frequency_weeks,
    )
    if not windows:
        raise RuntimeError("No rolling windows were created for local adaptive point recalibration.")

    method_name = "cp_symmetric" if symmetric else "cp_asymmetric"
    combined_predictions: list[pd.DataFrame] = []
    weekly_metrics: list[dict[str, object]] = []

    with tqdm(
        total=len(windows),
        desc=f"Weeks | {target_spec.code} | {method_name} | {artifacts.model_name} | {scale_name}",
        unit="week",
        dynamic_ncols=True,
        leave=False,
    ) as week_progress:
        for window_index, window in enumerate(windows):
            calibration_residuals = rfe._window_slice(
                residual_series,
                start=window.calibration_start,
                end=window.calibration_end,
            )
            prediction_slice = rfe._window_slice(
                prediction_frame,
                start=window.prediction_start,
                end=window.prediction_end,
            )
            if calibration_residuals.empty or prediction_slice.empty:
                week_progress.update(1)
                continue

            calibration_scales = _validate_scale_alignment(
                scale_series,
                calibration_residuals.index,
                label=f"{scale_name} calibration window {window_index}",
            )
            prediction_scales = _validate_scale_alignment(
                scale_series,
                prediction_slice.index,
                label=f"{scale_name} prediction window {window_index}",
            )
            scaled_residuals = calibration_residuals.astype(float) / calibration_scales
            y_true = y_series.reindex(prediction_slice.index).astype(float)
            base_prediction = prediction_slice["y_pred"].astype(float)
            zero_predictions = np.zeros(len(prediction_slice), dtype=float)

            if symmetric:
                conformal = ConformalPrediction(
                    residuals_upper=scaled_residuals.to_numpy(),
                    residuals_lower=scaled_residuals.to_numpy(),
                    predictions_upper=zero_predictions,
                    predictions_lower=zero_predictions,
                    alpha=ALPHA,
                    symmetric=True,
                )
            else:
                conformal = ConformalPrediction(
                    residuals_upper=scaled_residuals[scaled_residuals >= 0.0].to_numpy(),
                    residuals_lower=scaled_residuals[scaled_residuals < 0.0].to_numpy(),
                    predictions_upper=zero_predictions,
                    predictions_lower=zero_predictions,
                    alpha=ALPHA,
                    symmetric=False,
                )

            result = conformal.get_result()
            scaled_margins_lower = pd.Series(result.margin_lower, index=prediction_slice.index)
            scaled_margins_upper = pd.Series(result.margin_upper, index=prediction_slice.index)
            margins_lower = scaled_margins_lower * prediction_scales
            margins_upper = scaled_margins_upper * prediction_scales
            lower_pred = base_prediction - margins_lower
            upper_pred = base_prediction + margins_upper

            calibrated_frame = _build_lacp_result_frame(
                y_true=y_true,
                lower_pred=lower_pred,
                upper_pred=upper_pred,
                base_lower_pred=base_prediction,
                base_upper_pred=base_prediction,
                margin_lower=margins_lower,
                margin_upper=margins_upper,
                scaled_margin_lower=scaled_margins_lower,
                scaled_margin_upper=scaled_margins_upper,
                scale_values=prediction_scales,
                window_index=window_index,
                window=window,
                calibration_rows=int(len(calibration_residuals)),
                scale_name=scale_name,
            )
            combined_predictions.append(calibrated_frame)

            metrics = rfe._compute_interval_metrics_for_frame(calibrated_frame)
            weekly_metrics.append(
                {
                    "window_index": window_index,
                    "calibration_start": window.calibration_start,
                    "calibration_end_exclusive": window.calibration_end,
                    "prediction_start": window.prediction_start,
                    "prediction_end_exclusive": window.prediction_end,
                    "calibration_rows": int(len(calibration_residuals)),
                    "prediction_rows": int(len(prediction_slice)),
                    "scale_name": scale_name,
                    "mean_calibration_scale": float(calibration_scales.mean()),
                    "mean_prediction_scale": float(prediction_scales.mean()),
                    **metrics,
                }
            )
            week_progress.update(1)

    if not combined_predictions:
        raise RuntimeError("No local adaptive point windows produced any prediction rows.")

    all_predictions = (
        pd.concat(combined_predictions, ignore_index=True)
        .sort_values(by=[DATE_COLUMN], kind="mergesort")
        .reset_index(drop=True)
    )
    weekly_metrics_df = (
        pd.DataFrame(weekly_metrics)
        .sort_values(by=["window_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    overall_metrics = rfe._compute_interval_metrics_for_frame(all_predictions)
    return all_predictions, weekly_metrics_df, overall_metrics


def _run_scaled_interval_method(
    *,
    artifacts: rfe.ModelArtifacts,
    target_spec: rfe.TargetSpec,
    prediction_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    residual_low: pd.Series,
    residual_up: pd.Series,
    scale_series: pd.Series,
    scale_name: str,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    required_columns = {"y_pred_lower", "y_pred_upper"}
    if not required_columns.issubset(prediction_frame.columns):
        raise KeyError(
            f"Missing required interval prediction columns in {artifacts.prediction_path}: {required_columns}"
        )

    y_series = y_frame[target_spec.y_column].reindex(prediction_frame.index)
    if y_series.isna().any():
        raise KeyError(f"Missing y values for target {target_spec.y_column}.")

    windows = rfe._build_rolling_windows(
        prediction_frame.index,
        calibration_range_weeks=calibration_range_weeks,
        calibration_frequency_weeks=calibration_frequency_weeks,
    )
    if not windows:
        raise RuntimeError("No rolling windows were created for local adaptive CQR recalibration.")

    combined_predictions: list[pd.DataFrame] = []
    weekly_metrics: list[dict[str, object]] = []

    with tqdm(
        total=len(windows),
        desc=f"Weeks | {target_spec.code} | cqr | {artifacts.model_name} | {scale_name}",
        unit="week",
        dynamic_ncols=True,
        leave=False,
    ) as week_progress:
        for window_index, window in enumerate(windows):
            calibration_low = rfe._window_slice(
                residual_low,
                start=window.calibration_start,
                end=window.calibration_end,
            )
            calibration_up = rfe._window_slice(
                residual_up,
                start=window.calibration_start,
                end=window.calibration_end,
            )
            prediction_slice = rfe._window_slice(
                prediction_frame,
                start=window.prediction_start,
                end=window.prediction_end,
            )
            if calibration_low.empty or calibration_up.empty or prediction_slice.empty:
                week_progress.update(1)
                continue

            calibration_scales = _validate_scale_alignment(
                scale_series,
                calibration_low.index,
                label=f"{scale_name} interval calibration window {window_index}",
            )
            prediction_scales = _validate_scale_alignment(
                scale_series,
                prediction_slice.index,
                label=f"{scale_name} interval prediction window {window_index}",
            )
            scaled_low = calibration_low.astype(float) / calibration_scales
            scaled_up = calibration_up.astype(float) / calibration_scales
            y_true = y_series.reindex(prediction_slice.index).astype(float)
            base_lower = prediction_slice["y_pred_lower"].astype(float)
            base_upper = prediction_slice["y_pred_upper"].astype(float)
            zero_predictions = np.zeros(len(prediction_slice), dtype=float)

            conformal = ConformalizedQuantileRegression(
                residuals_lower=scaled_low.to_numpy(),
                residuals_upper=scaled_up.to_numpy(),
                predictions_lower=zero_predictions,
                predictions_upper=zero_predictions,
                alpha=ALPHA,
            )
            result = conformal.get_result()

            scaled_margins_lower = pd.Series(result.margin_lower, index=prediction_slice.index)
            scaled_margins_upper = pd.Series(result.margin_upper, index=prediction_slice.index)
            margins_lower = scaled_margins_lower * prediction_scales
            margins_upper = scaled_margins_upper * prediction_scales
            lower_pred = base_lower - margins_lower
            upper_pred = base_upper + margins_upper

            calibrated_frame = _build_lacp_result_frame(
                y_true=y_true,
                lower_pred=lower_pred,
                upper_pred=upper_pred,
                base_lower_pred=base_lower,
                base_upper_pred=base_upper,
                margin_lower=margins_lower,
                margin_upper=margins_upper,
                scaled_margin_lower=scaled_margins_lower,
                scaled_margin_upper=scaled_margins_upper,
                scale_values=prediction_scales,
                window_index=window_index,
                window=window,
                calibration_rows=int(len(calibration_low)),
                scale_name=scale_name,
            )
            combined_predictions.append(calibrated_frame)

            metrics = rfe._compute_interval_metrics_for_frame(calibrated_frame)
            weekly_metrics.append(
                {
                    "window_index": window_index,
                    "calibration_start": window.calibration_start,
                    "calibration_end_exclusive": window.calibration_end,
                    "prediction_start": window.prediction_start,
                    "prediction_end_exclusive": window.prediction_end,
                    "calibration_rows": int(len(calibration_low)),
                    "prediction_rows": int(len(prediction_slice)),
                    "scale_name": scale_name,
                    "mean_calibration_scale": float(calibration_scales.mean()),
                    "mean_prediction_scale": float(prediction_scales.mean()),
                    **metrics,
                }
            )
            week_progress.update(1)

    if not combined_predictions:
        raise RuntimeError("No local adaptive CQR windows produced any prediction rows.")

    all_predictions = (
        pd.concat(combined_predictions, ignore_index=True)
        .sort_values(by=[DATE_COLUMN], kind="mergesort")
        .reset_index(drop=True)
    )
    weekly_metrics_df = (
        pd.DataFrame(weekly_metrics)
        .sort_values(by=["window_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    overall_metrics = rfe._compute_interval_metrics_for_frame(all_predictions)
    return all_predictions, weekly_metrics_df, overall_metrics


def _run_single_method(
    *,
    artifacts: rfe.ModelArtifacts,
    target_spec: rfe.TargetSpec,
    prediction_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    scale_series: pd.Series,
    scale_name: str,
    scale_csv_path: Path,
    scale_column_name: str,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
    context_results_root: Path,
    selections: list[MethodSelection],
) -> list[dict[str, object]]:
    result_rows: list[dict[str, object]] = []

    if artifacts.estimation_kind == "point":
        residual_path = artifacts.residual_dir / f"{target_spec.point_residual_name}.csv"
        residual_series = rfe._load_residual_series(residual_path, target_spec.point_residual_name)
        point_variants = [(True, "cp_symmetric"), (False, "cp_asymmetric")]

        for symmetric, method_name in point_variants:
            if not _method_is_selected(
                method_name=method_name,
                artifacts=artifacts,
                selections=selections,
            ):
                continue

            output_dir = output_dir_for(
                context_results_root / target_spec.slug,
                rfe._point_method_output_name(artifacts.model_name, symmetric=symmetric),
                artifacts.objective_name,
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            predictions_df, weekly_metrics_df, overall_metrics = _run_scaled_point_method(
                artifacts=artifacts,
                target_spec=target_spec,
                prediction_frame=prediction_frame,
                y_frame=y_frame,
                residual_series=residual_series,
                scale_series=scale_series,
                scale_name=scale_name,
                symmetric=symmetric,
                calibration_range_weeks=calibration_range_weeks,
                calibration_frequency_weeks=calibration_frequency_weeks,
            )

            target_stem = target_artifact_stem(target_spec.y_column)
            prediction_csv_path = output_dir / f"{target_stem}_predictions.csv"
            weekly_metrics_path = output_dir / "weekly_metrics.csv"
            overall_metrics_path = output_dir / "overall_metrics.csv"

            predictions_df.to_csv(prediction_csv_path, index=False)
            weekly_metrics_df.to_csv(weekly_metrics_path, index=False)
            pd.DataFrame(
                [
                    {
                        "calibration_range_weeks": calibration_range_weeks,
                        "calibration_frequency_weeks": calibration_frequency_weeks,
                        "prediction_rows": int(len(predictions_df)),
                        "window_count": int(len(weekly_metrics_df)),
                        "scale_name": scale_name,
                        **overall_metrics,
                    }
                ]
            ).to_csv(overall_metrics_path, index=False)
            plot_html_path = rfe._save_prediction_plot_html(
                output_dir=output_dir,
                target_spec=target_spec,
                display_name=rfe._method_display_name(artifacts.display_name, method_name),
                prediction_frame=predictions_df,
                metrics=overall_metrics,
                estimation_kind=artifacts.estimation_kind,
            )
            plot_png_path = rfe._save_prediction_plot_png(
                output_dir=output_dir,
                target_spec=target_spec,
                display_name=rfe._method_display_name(artifacts.display_name, method_name),
                prediction_frame=predictions_df,
                metrics=overall_metrics,
                estimation_kind=artifacts.estimation_kind,
            )
            _write_pipeline_metadata(
                output_dir=output_dir,
                target_spec=target_spec,
                artifacts=artifacts,
                method_name=method_name,
                calibration_range_weeks=calibration_range_weeks,
                calibration_frequency_weeks=calibration_frequency_weeks,
                window_count=len(weekly_metrics_df),
                residual_paths=[residual_path],
                extra_metadata={
                    "scale_name": scale_name,
                    "scale_path": str(scale_csv_path),
                    "scale_column_name": scale_column_name,
                },
            )

            result_rows.append(
                {
                    "pipeline": "lacp",
                    "scale_name": scale_name,
                    "target_code": target_spec.code,
                    "target_slug": target_spec.slug,
                    "target_name": target_spec.y_column,
                    "method_name": method_name,
                    "estimation_kind": artifacts.estimation_kind,
                    "model_name": artifacts.model_name,
                    "objective_name": artifacts.objective_name,
                    "display_name": rfe._method_display_name(artifacts.display_name, method_name),
                    "base_display_name": artifacts.display_name,
                    "calibration_range_weeks": calibration_range_weeks,
                    "calibration_frequency_weeks": calibration_frequency_weeks,
                    "alpha": ALPHA,
                    "lower_quantile": LOWER_QUANTILE,
                    "upper_quantile": UPPER_QUANTILE,
                    "window_count": int(len(weekly_metrics_df)),
                    "prediction_rows": int(len(predictions_df)),
                    **overall_metrics,
                    "output_dir": str(output_dir),
                    "prediction_csv": str(prediction_csv_path),
                    "prediction_plot_html": str(plot_html_path),
                    "prediction_plot_png": str(plot_png_path),
                    "weekly_metrics_csv": str(weekly_metrics_path),
                    "overall_metrics_csv": str(overall_metrics_path),
                    "source_prediction_path": str(artifacts.prediction_path),
                    "source_residual_path": str(residual_path),
                    "source_scale_path": str(scale_csv_path),
                    "scale_column_name": scale_column_name,
                }
            )
        return result_rows

    if not _method_is_selected(
        method_name="cqr",
        artifacts=artifacts,
        selections=selections,
    ):
        return result_rows

    residual_low_path = artifacts.residual_dir / f"{target_spec.interval_residual_low_name}.csv"
    residual_up_path = artifacts.residual_dir / f"{target_spec.interval_residual_up_name}.csv"
    residual_low = rfe._load_residual_series(residual_low_path, target_spec.interval_residual_low_name)
    residual_up = rfe._load_residual_series(residual_up_path, target_spec.interval_residual_up_name)

    output_dir = output_dir_for(
        context_results_root / target_spec.slug,
        rfe._interval_method_output_name(artifacts.model_name),
        artifacts.objective_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df, weekly_metrics_df, overall_metrics = _run_scaled_interval_method(
        artifacts=artifacts,
        target_spec=target_spec,
        prediction_frame=prediction_frame,
        y_frame=y_frame,
        residual_low=residual_low,
        residual_up=residual_up,
        scale_series=scale_series,
        scale_name=scale_name,
        calibration_range_weeks=calibration_range_weeks,
        calibration_frequency_weeks=calibration_frequency_weeks,
    )

    target_stem = target_artifact_stem(target_spec.y_column)
    prediction_csv_path = output_dir / f"{target_stem}_predictions.csv"
    weekly_metrics_path = output_dir / "weekly_metrics.csv"
    overall_metrics_path = output_dir / "overall_metrics.csv"

    predictions_df.to_csv(prediction_csv_path, index=False)
    weekly_metrics_df.to_csv(weekly_metrics_path, index=False)
    pd.DataFrame(
        [
            {
                "calibration_range_weeks": calibration_range_weeks,
                "calibration_frequency_weeks": calibration_frequency_weeks,
                "prediction_rows": int(len(predictions_df)),
                "window_count": int(len(weekly_metrics_df)),
                "scale_name": scale_name,
                **overall_metrics,
            }
        ]
    ).to_csv(overall_metrics_path, index=False)
    plot_html_path = rfe._save_prediction_plot_html(
        output_dir=output_dir,
        target_spec=target_spec,
        display_name=rfe._method_display_name(artifacts.display_name, "cqr"),
        prediction_frame=predictions_df,
        metrics=overall_metrics,
        estimation_kind=artifacts.estimation_kind,
    )
    plot_png_path = rfe._save_prediction_plot_png(
        output_dir=output_dir,
        target_spec=target_spec,
        display_name=rfe._method_display_name(artifacts.display_name, "cqr"),
        prediction_frame=predictions_df,
        metrics=overall_metrics,
        estimation_kind=artifacts.estimation_kind,
    )
    _write_pipeline_metadata(
        output_dir=output_dir,
        target_spec=target_spec,
        artifacts=artifacts,
        method_name="cqr",
        calibration_range_weeks=calibration_range_weeks,
        calibration_frequency_weeks=calibration_frequency_weeks,
        window_count=len(weekly_metrics_df),
        residual_paths=[residual_low_path, residual_up_path],
        extra_metadata={
            "scale_name": scale_name,
            "scale_path": str(scale_csv_path),
            "scale_column_name": scale_column_name,
        },
    )

    result_rows.append(
        {
            "pipeline": "lacp",
            "scale_name": scale_name,
            "target_code": target_spec.code,
            "target_slug": target_spec.slug,
            "target_name": target_spec.y_column,
            "method_name": "cqr",
            "estimation_kind": artifacts.estimation_kind,
            "model_name": artifacts.model_name,
            "objective_name": artifacts.objective_name,
            "display_name": rfe._method_display_name(artifacts.display_name, "cqr"),
            "base_display_name": artifacts.display_name,
            "calibration_range_weeks": calibration_range_weeks,
            "calibration_frequency_weeks": calibration_frequency_weeks,
            "alpha": ALPHA,
            "lower_quantile": LOWER_QUANTILE,
            "upper_quantile": UPPER_QUANTILE,
            "window_count": int(len(weekly_metrics_df)),
            "prediction_rows": int(len(predictions_df)),
            **overall_metrics,
            "output_dir": str(output_dir),
            "prediction_csv": str(prediction_csv_path),
            "prediction_plot_html": str(plot_html_path),
            "prediction_plot_png": str(plot_png_path),
            "weekly_metrics_csv": str(weekly_metrics_path),
            "overall_metrics_csv": str(overall_metrics_path),
            "source_prediction_path": str(artifacts.prediction_path),
            "source_residual_low_path": str(residual_low_path),
            "source_residual_up_path": str(residual_up_path),
            "source_scale_path": str(scale_csv_path),
            "scale_column_name": scale_column_name,
        }
    )
    return result_rows


def run_local_adaptive_conformal_prediction(
    *,
    calibration_range_weeks: int = CALIBRATION_RANGE,
    calibration_frequency_weeks: int = CALIBRATION_FREQUENCY,
    target_list: list[str] | tuple[str, ...] = TARGET_LIST,
    predictions_root: Path = PREDICTIONS_ROOT,
    residuals_root: Path = RESIDUALS_ROOT,
    scales_root: Path = SCALES_ROOT,
    y_path: Path = Y_PATH,
    results_root: Path = RESULTS_ROOT,
    method_model_loss_list: Iterable[object] | None = METHOD_MODEL_LOSS_LIST,
    scale_names: Iterable[str] | None = SCALE_NAME_LIST,
    clean_output: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions_root = predictions_root.resolve()
    residuals_root = residuals_root.resolve()
    scales_root = scales_root.resolve()
    y_path = y_path.resolve()
    results_root = results_root.resolve()

    if calibration_range_weeks <= 0:
        raise ValueError(f"calibration_range_weeks must be positive, got {calibration_range_weeks}.")
    if calibration_frequency_weeks <= 0:
        raise ValueError(
            f"calibration_frequency_weeks must be positive, got {calibration_frequency_weeks}."
        )

    if clean_output and results_root.exists():
        shutil.rmtree(results_root)

    target_specs = rfe._resolve_targets(list(target_list))
    y_frame = rfe._load_y_frame(y_path, target_specs)
    scale_specs = _discover_scale_csvs(scales_root, scale_names)
    method_selections = resolve_method_selections(method_model_loss_list)
    artifacts = _filter_artifacts(
        rfe._discover_model_artifacts(
            predictions_root=predictions_root,
            residuals_root=residuals_root,
        ),
        method_selections,
    )
    if not artifacts:
        raise FileNotFoundError("No prediction artifacts matched the requested method selections.")

    all_ranked_results: list[pd.DataFrame] = []
    all_failures: list[dict[str, object]] = []

    with tqdm(total=len(scale_specs), desc="Scales", unit="scale", dynamic_ncols=True) as scale_progress:
        for scale_name, scale_csv_path in scale_specs:
            scale_progress.set_postfix_str(scale_name)
            scale_series, scale_column_name = _load_scale_series(scale_csv_path)
            context_results_root = results_root / scale_name
            prediction_cache: dict[Path, pd.DataFrame] = {}
            result_rows: list[dict[str, object]] = []
            failures: list[dict[str, object]] = []

            for target_spec in target_specs:
                for artifacts_item in artifacts:
                    try:
                        prediction_frame = prediction_cache.get(artifacts_item.prediction_path)
                        if prediction_frame is None:
                            prediction_frame = rfe._load_prediction_frame(artifacts_item.prediction_path)
                            prediction_cache[artifacts_item.prediction_path] = prediction_frame

                        result_rows.extend(
                            _run_single_method(
                                artifacts=artifacts_item,
                                target_spec=target_spec,
                                prediction_frame=prediction_frame,
                                y_frame=y_frame,
                                scale_series=scale_series,
                                scale_name=scale_name,
                                scale_csv_path=scale_csv_path,
                                scale_column_name=scale_column_name,
                                calibration_range_weeks=calibration_range_weeks,
                                calibration_frequency_weeks=calibration_frequency_weeks,
                                context_results_root=context_results_root,
                                selections=method_selections,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        failures.append(
                            {
                                "pipeline": "lacp",
                                "scale_name": scale_name,
                                "target_code": target_spec.code,
                                "estimation_kind": artifacts_item.estimation_kind,
                                "model_name": artifacts_item.model_name,
                                "objective_name": artifacts_item.objective_name,
                                "display_name": artifacts_item.display_name,
                                "error": repr(exc),
                            }
                        )
                        rfe._log(
                            f"FAILED {scale_name} | {target_spec.code} | "
                            f"{artifacts_item.display_name} ({artifacts_item.estimation_kind}): {exc}"
                        )

            if result_rows:
                ranked_results = rfe._write_summary_results(context_results_root, result_rows)
                all_ranked_results.append(ranked_results)
            else:
                context_results_root.mkdir(parents=True, exist_ok=True)
                rfe._log(
                    f"No successful runs for scale {scale_name}; writing failures only."
                )
            rfe._write_failures(context_results_root, failures)
            all_failures.extend(failures)
            scale_progress.update(1)

    ranked_results_df = (
        pd.concat(all_ranked_results, ignore_index=True)
        if all_ranked_results
        else pd.DataFrame()
    )
    failures_df = pd.DataFrame(all_failures)
    return ranked_results_df, failures_df


def main() -> int:
    args = _parse_args()
    ranked_results, failures_df = run_local_adaptive_conformal_prediction(
        calibration_range_weeks=args.calibration_range,
        calibration_frequency_weeks=args.calibration_frequency,
        target_list=args.targets,
        method_model_loss_list=args.method_spec or METHOD_MODEL_LOSS_LIST,
        scale_names=args.scale_names or SCALE_NAME_LIST,
        clean_output=bool(args.clean_output),
    )
    rfe._log(f"Completed runs : {len(ranked_results):,}")
    rfe._log(f"Failures       : {len(failures_df):,}")
    return 0 if failures_df.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
