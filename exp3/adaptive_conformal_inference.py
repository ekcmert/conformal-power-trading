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
    _filter_artifacts,
    _method_is_selected,
    resolve_method_selections,
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
Y_PATH = REPO_ROOT / "data" / "final" / "y.csv"
RESULTS_ROOT = REPO_ROOT / "results" / "regime_aware" / "aci"

# LEARNING_RATES: tuple[float, ...] = (0.001,0.005,0.01,0.05)
LEARNING_RATES: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.1)
METHOD_MODEL_LOSS_LIST: tuple[object, ...] = BASE_METHOD_MODEL_LOSS_LIST
_ALPHA_EPSILON = 1e-6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptive conformal inference on saved predictions and residuals.",
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
        "--learning-rates",
        nargs="+",
        type=float,
        required=not LEARNING_RATES,
        default=list(LEARNING_RATES),
        help="Learning rates used for adaptive alpha updates.",
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
        "--clean-output",
        action="store_true",
        help="Remove existing results/regime_aware/aci outputs before running.",
    )
    return parser.parse_args()


def _clip_alpha(value: float) -> float:
    return float(np.clip(value, _ALPHA_EPSILON, 1.0 - _ALPHA_EPSILON))


def _learning_rate_folder_name(learning_rate: float) -> str:
    return format(float(learning_rate), "g")


def _normalize_learning_rates(learning_rates: Iterable[float] | None) -> list[float]:
    if learning_rates is None:
        return []

    normalized: list[float] = []
    for value in learning_rates:
        learning_rate = float(value)
        if not np.isfinite(learning_rate):
            raise ValueError(f"Learning rates must be finite, got {value!r}.")
        if learning_rate <= 0.0:
            raise ValueError(f"Learning rates must be positive, got {value!r}.")
        normalized.append(learning_rate)
    if not normalized:
        raise ValueError("At least one learning rate must be provided for ACI.")

    deduplicated: list[float] = []
    seen: set[float] = set()
    for value in normalized:
        if value in seen:
            continue
        deduplicated.append(value)
        seen.add(value)
    return deduplicated


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
        "pipeline": "adaptive_conformal_inference",
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


def _build_aci_result_frame(
    *,
    y_true: pd.Series,
    lower_pred: pd.Series,
    upper_pred: pd.Series,
    base_lower_pred: pd.Series,
    base_upper_pred: pd.Series,
    margin_lower: pd.Series,
    margin_upper: pd.Series,
    window_index: int,
    window: rfe.RollingWindow,
    calibration_rows: int,
    learning_rate: float,
    alpha_used: float,
    alpha_next: float,
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
    prediction_frame["learning_rate"] = learning_rate
    prediction_frame["alpha_used"] = alpha_used
    prediction_frame["alpha_next"] = alpha_next
    return prediction_frame


def _update_alpha(current_alpha: float, empirical_coverage: float, learning_rate: float) -> float:
    observed_miscoverage = 1.0 - empirical_coverage
    return _clip_alpha(current_alpha + learning_rate * (ALPHA - observed_miscoverage))


def _run_aci_point_method(
    *,
    artifacts: rfe.ModelArtifacts,
    target_spec: rfe.TargetSpec,
    prediction_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    residual_series: pd.Series,
    symmetric: bool,
    learning_rate: float,
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
        raise RuntimeError("No rolling windows were created for ACI point recalibration.")

    method_name = "cp_symmetric" if symmetric else "cp_asymmetric"
    current_alpha = ALPHA
    combined_predictions: list[pd.DataFrame] = []
    weekly_metrics: list[dict[str, object]] = []

    with tqdm(
        total=len(windows),
        desc=f"Weeks | {target_spec.code} | {method_name} | {artifacts.model_name} | lr={learning_rate:g}",
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

            y_true = y_series.reindex(prediction_slice.index).astype(float)
            base_prediction = prediction_slice["y_pred"].astype(float)

            if symmetric:
                conformal = ConformalPrediction(
                    residuals_upper=calibration_residuals.to_numpy(),
                    residuals_lower=calibration_residuals.to_numpy(),
                    predictions_upper=base_prediction.to_numpy(),
                    predictions_lower=base_prediction.to_numpy(),
                    alpha=current_alpha,
                    symmetric=True,
                )
            else:
                conformal = ConformalPrediction(
                    residuals_upper=calibration_residuals[calibration_residuals >= 0.0].to_numpy(),
                    residuals_lower=calibration_residuals[calibration_residuals < 0.0].to_numpy(),
                    predictions_upper=base_prediction.to_numpy(),
                    predictions_lower=base_prediction.to_numpy(),
                    alpha=current_alpha,
                    symmetric=False,
                )

            result = conformal.get_result()
            lower_pred = pd.Series(result.calibrated_lower, index=prediction_slice.index)
            upper_pred = pd.Series(result.calibrated_upper, index=prediction_slice.index)
            margins_lower = pd.Series(result.margin_lower, index=prediction_slice.index)
            margins_upper = pd.Series(result.margin_upper, index=prediction_slice.index)

            provisional_frame = rfe._build_result_prediction_frame(
                y_true=y_true,
                lower_pred=lower_pred,
                upper_pred=upper_pred,
                base_lower_pred=base_prediction,
                base_upper_pred=base_prediction,
                margin_lower=margins_lower,
                margin_upper=margins_upper,
                window_index=window_index,
                window=window,
                calibration_rows=int(len(calibration_residuals)),
            )
            metrics = rfe._compute_interval_metrics_for_frame(provisional_frame)
            next_alpha = _update_alpha(current_alpha, metrics["empirical_coverage"], learning_rate)

            calibrated_frame = _build_aci_result_frame(
                y_true=y_true,
                lower_pred=lower_pred,
                upper_pred=upper_pred,
                base_lower_pred=base_prediction,
                base_upper_pred=base_prediction,
                margin_lower=margins_lower,
                margin_upper=margins_upper,
                window_index=window_index,
                window=window,
                calibration_rows=int(len(calibration_residuals)),
                learning_rate=learning_rate,
                alpha_used=current_alpha,
                alpha_next=next_alpha,
            )
            combined_predictions.append(calibrated_frame)
            weekly_metrics.append(
                {
                    "window_index": window_index,
                    "calibration_start": window.calibration_start,
                    "calibration_end_exclusive": window.calibration_end,
                    "prediction_start": window.prediction_start,
                    "prediction_end_exclusive": window.prediction_end,
                    "calibration_rows": int(len(calibration_residuals)),
                    "prediction_rows": int(len(prediction_slice)),
                    "learning_rate": learning_rate,
                    "alpha_used": current_alpha,
                    "alpha_next": next_alpha,
                    "observed_miscoverage": float(1.0 - metrics["empirical_coverage"]),
                    **metrics,
                }
            )
            current_alpha = next_alpha
            week_progress.update(1)

    if not combined_predictions:
        raise RuntimeError("No ACI point windows produced any prediction rows.")

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


def _run_aci_interval_method(
    *,
    artifacts: rfe.ModelArtifacts,
    target_spec: rfe.TargetSpec,
    prediction_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    residual_low: pd.Series,
    residual_up: pd.Series,
    learning_rate: float,
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
        raise RuntimeError("No rolling windows were created for ACI CQR recalibration.")

    current_alpha = ALPHA
    combined_predictions: list[pd.DataFrame] = []
    weekly_metrics: list[dict[str, object]] = []

    with tqdm(
        total=len(windows),
        desc=f"Weeks | {target_spec.code} | cqr | {artifacts.model_name} | lr={learning_rate:g}",
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

            y_true = y_series.reindex(prediction_slice.index).astype(float)
            base_lower = prediction_slice["y_pred_lower"].astype(float)
            base_upper = prediction_slice["y_pred_upper"].astype(float)

            conformal = ConformalizedQuantileRegression(
                residuals_lower=calibration_low.to_numpy(),
                residuals_upper=calibration_up.to_numpy(),
                predictions_lower=base_lower.to_numpy(),
                predictions_upper=base_upper.to_numpy(),
                alpha=current_alpha,
            )
            result = conformal.get_result()

            lower_pred = pd.Series(result.calibrated_lower, index=prediction_slice.index)
            upper_pred = pd.Series(result.calibrated_upper, index=prediction_slice.index)
            margins_lower = pd.Series(result.margin_lower, index=prediction_slice.index)
            margins_upper = pd.Series(result.margin_upper, index=prediction_slice.index)

            provisional_frame = rfe._build_result_prediction_frame(
                y_true=y_true,
                lower_pred=lower_pred,
                upper_pred=upper_pred,
                base_lower_pred=base_lower,
                base_upper_pred=base_upper,
                margin_lower=margins_lower,
                margin_upper=margins_upper,
                window_index=window_index,
                window=window,
                calibration_rows=int(len(calibration_low)),
            )
            metrics = rfe._compute_interval_metrics_for_frame(provisional_frame)
            next_alpha = _update_alpha(current_alpha, metrics["empirical_coverage"], learning_rate)

            calibrated_frame = _build_aci_result_frame(
                y_true=y_true,
                lower_pred=lower_pred,
                upper_pred=upper_pred,
                base_lower_pred=base_lower,
                base_upper_pred=base_upper,
                margin_lower=margins_lower,
                margin_upper=margins_upper,
                window_index=window_index,
                window=window,
                calibration_rows=int(len(calibration_low)),
                learning_rate=learning_rate,
                alpha_used=current_alpha,
                alpha_next=next_alpha,
            )
            combined_predictions.append(calibrated_frame)
            weekly_metrics.append(
                {
                    "window_index": window_index,
                    "calibration_start": window.calibration_start,
                    "calibration_end_exclusive": window.calibration_end,
                    "prediction_start": window.prediction_start,
                    "prediction_end_exclusive": window.prediction_end,
                    "calibration_rows": int(len(calibration_low)),
                    "prediction_rows": int(len(prediction_slice)),
                    "learning_rate": learning_rate,
                    "alpha_used": current_alpha,
                    "alpha_next": next_alpha,
                    "observed_miscoverage": float(1.0 - metrics["empirical_coverage"]),
                    **metrics,
                }
            )
            current_alpha = next_alpha
            week_progress.update(1)

    if not combined_predictions:
        raise RuntimeError("No ACI CQR windows produced any prediction rows.")

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
    learning_rate: float,
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

            predictions_df, weekly_metrics_df, overall_metrics = _run_aci_point_method(
                artifacts=artifacts,
                target_spec=target_spec,
                prediction_frame=prediction_frame,
                y_frame=y_frame,
                residual_series=residual_series,
                symmetric=symmetric,
                learning_rate=learning_rate,
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
                        "learning_rate": learning_rate,
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
                    "learning_rate": learning_rate,
                    "nominal_alpha": ALPHA,
                },
            )

            result_rows.append(
                {
                    "pipeline": "aci",
                    "learning_rate": learning_rate,
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

    predictions_df, weekly_metrics_df, overall_metrics = _run_aci_interval_method(
        artifacts=artifacts,
        target_spec=target_spec,
        prediction_frame=prediction_frame,
        y_frame=y_frame,
        residual_low=residual_low,
        residual_up=residual_up,
        learning_rate=learning_rate,
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
                "learning_rate": learning_rate,
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
            "learning_rate": learning_rate,
            "nominal_alpha": ALPHA,
        },
    )

    result_rows.append(
        {
            "pipeline": "aci",
            "learning_rate": learning_rate,
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
        }
    )
    return result_rows


def run_adaptive_conformal_inference(
    *,
    calibration_range_weeks: int = CALIBRATION_RANGE,
    calibration_frequency_weeks: int = CALIBRATION_FREQUENCY,
    target_list: list[str] | tuple[str, ...] = TARGET_LIST,
    predictions_root: Path = PREDICTIONS_ROOT,
    residuals_root: Path = RESIDUALS_ROOT,
    y_path: Path = Y_PATH,
    results_root: Path = RESULTS_ROOT,
    learning_rates: Iterable[float] | None = LEARNING_RATES,
    method_model_loss_list: Iterable[object] | None = METHOD_MODEL_LOSS_LIST,
    clean_output: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions_root = predictions_root.resolve()
    residuals_root = residuals_root.resolve()
    y_path = y_path.resolve()
    results_root = results_root.resolve()

    if calibration_range_weeks <= 0:
        raise ValueError(f"calibration_range_weeks must be positive, got {calibration_range_weeks}.")
    if calibration_frequency_weeks <= 0:
        raise ValueError(
            f"calibration_frequency_weeks must be positive, got {calibration_frequency_weeks}."
        )

    normalized_learning_rates = _normalize_learning_rates(learning_rates)

    if clean_output and results_root.exists():
        shutil.rmtree(results_root)

    target_specs = rfe._resolve_targets(list(target_list))
    y_frame = rfe._load_y_frame(y_path, target_specs)
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

    with tqdm(total=len(normalized_learning_rates), desc="Learning rates", unit="lr", dynamic_ncols=True) as lr_progress:
        for learning_rate in normalized_learning_rates:
            folder_name = _learning_rate_folder_name(learning_rate)
            lr_progress.set_postfix_str(folder_name)
            context_results_root = results_root / folder_name
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
                                learning_rate=learning_rate,
                                calibration_range_weeks=calibration_range_weeks,
                                calibration_frequency_weeks=calibration_frequency_weeks,
                                context_results_root=context_results_root,
                                selections=method_selections,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        failures.append(
                            {
                                "pipeline": "aci",
                                "learning_rate": learning_rate,
                                "target_code": target_spec.code,
                                "estimation_kind": artifacts_item.estimation_kind,
                                "model_name": artifacts_item.model_name,
                                "objective_name": artifacts_item.objective_name,
                                "display_name": artifacts_item.display_name,
                                "error": repr(exc),
                            }
                        )
                        rfe._log(
                            f"FAILED lr={learning_rate:g} | {target_spec.code} | "
                            f"{artifacts_item.display_name} ({artifacts_item.estimation_kind}): {exc}"
                        )

            ranked_results = rfe._write_summary_results(context_results_root, result_rows)
            all_ranked_results.append(ranked_results)
            rfe._write_failures(context_results_root, failures)
            all_failures.extend(failures)
            lr_progress.update(1)

    ranked_results_df = (
        pd.concat(all_ranked_results, ignore_index=True)
        if all_ranked_results
        else pd.DataFrame()
    )
    failures_df = pd.DataFrame(all_failures)
    return ranked_results_df, failures_df


def main() -> int:
    args = _parse_args()
    ranked_results, failures_df = run_adaptive_conformal_inference(
        calibration_range_weeks=args.calibration_range,
        calibration_frequency_weeks=args.calibration_frequency,
        target_list=args.targets,
        learning_rates=args.learning_rates,
        method_model_loss_list=args.method_spec or METHOD_MODEL_LOSS_LIST,
        clean_output=bool(args.clean_output),
    )
    rfe._log(f"Completed runs : {len(ranked_results):,}")
    rfe._log(f"Failures       : {len(failures_df):,}")
    return 0 if failures_df.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
