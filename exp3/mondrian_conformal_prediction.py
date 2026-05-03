from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm.auto import tqdm

from base_models.interval_estimation.common import (
    DEFAULT_LOWER_QUANTILE,
    DEFAULT_UPPER_QUANTILE,
)
from base_models.point_estimation.common import REPO_ROOT, output_dir_for, target_artifact_stem
from conformal_prediction import ConformalPrediction, ConformalizedQuantileRegression
from exp2 import regime_free_experiment as rfe


CALIBRATION_RANGE = 24
CALIBRATION_FREQUENCY = 1
TARGET_LIST = ["DA"]
# TARGET_LIST = ["DA", "ID", "ID3", "ID1", "IMB"]

# TARGET_LIST = ["DA"]

DATE_COLUMN = "date"
LOWER_QUANTILE = DEFAULT_LOWER_QUANTILE
UPPER_QUANTILE = DEFAULT_UPPER_QUANTILE
ALPHA = 1.0 - (UPPER_QUANTILE - LOWER_QUANTILE)

PREDICTIONS_ROOT = REPO_ROOT / "data" / "predictions"
RESIDUALS_ROOT = REPO_ROOT / "data" / "residuals"
REGIMES_ROOT = REPO_ROOT / "data" / "regimes"
Y_PATH = REPO_ROOT / "data" / "final" / "y.csv"
RESULTS_ROOT = REPO_ROOT / "results" / "regime_aware" / "mcp"

REGIME_FILENAME = "regimes.csv"
REGIME_LABEL_COLUMN = "label"
METHOD_MODEL_LOSS_LIST: tuple[object, ...] = (
    ("cqr", "quantile_extra_trees", "squared_error"),
    # ("cqr", "lightgbm", "quantile"),
    # ("cp_asymmetric", "lightgbm", "regression_l1"),
    # ("cp_asymmetric", "hist_gradient_boosting", "absolute_error"),
    # ("cp_symmetric", "lightgbm", "regression_l1"),
    # ("cp_symmetric", "hist_gradient_boosting", "absolute_error"),
)
# METHOD_MODEL_LOSS_LIST: tuple[object, ...] = (
#     ("cqr", "quantile_extra_trees", "squared_error"),
#     ("cqr", "lightgbm", "quantile"),
#     ("cp_asymmetric", "lightgbm", "regression_l1"),
#     ("cp_asymmetric", "hist_gradient_boosting", "absolute_error"),
#     ("cp_symmetric", "lightgbm", "regression_l1"),
#     ("cp_symmetric", "hist_gradient_boosting", "absolute_error"),
#     ("cp_asymmetric", "lightgbm", "quantile_p50"),
#     ("cp_symmetric", "lightgbm", "quantile_p50"),
#     ("cp_asymmetric", "lightgbm", "fair"),
#     ("cp_symmetric", "lightgbm", "fair"),
#     ("cqr", "quantile_regressor", "pinball_loss"),
#     ("cp_asymmetric", "catboost", "rmse"),
#     ("cp_symmetric", "catboost", "rmse"),
# )
REGIME_GROUP_LIST: tuple[str, ...] = ()


@dataclass(frozen=True)
class MethodSelection:
    method_name: str
    model_name: str
    objective_name: str


@dataclass(frozen=True)
class RegimeGroup:
    name: str
    csv_path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run regime-aware Mondrian conformal recalibration on saved predictions and residuals.",
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
        "--regime-groups",
        nargs="+",
        default=list(REGIME_GROUP_LIST),
        help="Optional subset of regime subfolders under data/regimes.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing results/regime_aware/mcp outputs before running.",
    )
    return parser.parse_args()


def _normalize_method_name(method: object, symmetry: object | None = None) -> str:
    normalized_method = str(method).strip().lower().replace("-", "_")
    if normalized_method in {"cp_symmetric", "cp_asymmetric", "cqr"}:
        return normalized_method
    if normalized_method in {"cp", "conformal_prediction", "conformal"}:
        if symmetry is None:
            raise ValueError("Point conformal selections must include a symmetry value.")
        normalized_symmetry = str(symmetry).strip().lower().replace("-", "_")
        if normalized_symmetry in {"sym", "symmetric"}:
            return "cp_symmetric"
        if normalized_symmetry in {"asym", "asymmetric"}:
            return "cp_asymmetric"
        raise ValueError(f"Unsupported symmetry value {symmetry!r}.")
    if normalized_method in {"cqr", "conformalized_quantile_regression"}:
        return "cqr"
    raise ValueError(f"Unsupported method value {method!r}.")


def _parse_folder_method_spec(value: str) -> MethodSelection:
    normalized = value.strip().replace("\\", "/")
    if "/" not in normalized:
        raise ValueError(
            "Method selection strings must follow the folder naming convention "
            "like 'cp_symmetric_lightgbm/regression' or 'cqr_lightgbm/quantile'."
        )

    method_folder, objective_name = normalized.split("/", 1)
    method_folder = method_folder.strip()
    objective_name = objective_name.strip()
    if not method_folder or not objective_name:
        raise ValueError(f"Invalid method selection string {value!r}.")

    prefixes = (
        ("cp_symmetric_", "cp_symmetric"),
        ("cp_asymmetric_", "cp_asymmetric"),
        ("cqr_", "cqr"),
    )
    for prefix, method_name in prefixes:
        if method_folder.startswith(prefix):
            model_name = method_folder[len(prefix) :].strip()
            if not model_name:
                raise ValueError(f"Missing model name in method selection {value!r}.")
            return MethodSelection(
                method_name=method_name,
                model_name=model_name,
                objective_name=objective_name,
            )

    raise ValueError(
        f"Unsupported method folder {method_folder!r}. "
        "Expected prefixes cp_symmetric_, cp_asymmetric_, or cqr_."
    )


def parse_method_selection(value: object) -> MethodSelection:
    if isinstance(value, MethodSelection):
        return value

    if isinstance(value, str):
        return _parse_folder_method_spec(value)

    if isinstance(value, dict):
        if "method_name" in value:
            method_name = _normalize_method_name(value["method_name"])
        else:
            method_name = _normalize_method_name(value.get("method"), value.get("symmetry"))
        model_name = str(value.get("model_name", value.get("model", ""))).strip()
        objective_name = str(value.get("objective_name", value.get("loss", ""))).strip()
        if not model_name or not objective_name:
            raise ValueError(f"Method selection dictionary is missing model or loss: {value!r}")
        return MethodSelection(
            method_name=method_name,
            model_name=model_name,
            objective_name=objective_name,
        )

    if isinstance(value, (tuple, list)):
        if len(value) == 3:
            method_name = _normalize_method_name(value[0])
            model_name = str(value[1]).strip()
            objective_name = str(value[2]).strip()
        elif len(value) == 4:
            method_name = _normalize_method_name(value[0], value[1])
            model_name = str(value[2]).strip()
            objective_name = str(value[3]).strip()
        else:
            raise ValueError(
                "Method selections must be 3-tuples "
                "(method_name, model_name, loss) or 4-tuples "
                "(method, symmetry, model_name, loss)."
            )
        if not model_name or not objective_name:
            raise ValueError(f"Method selection is missing model or loss: {value!r}")
        return MethodSelection(
            method_name=method_name,
            model_name=model_name,
            objective_name=objective_name,
        )

    raise TypeError(f"Unsupported method selection type: {type(value)!r}")


def resolve_method_selections(
    method_model_loss_list: Iterable[object] | None,
) -> list[MethodSelection]:
    if method_model_loss_list is None:
        return []

    resolved = [parse_method_selection(item) for item in method_model_loss_list]
    deduplicated: list[MethodSelection] = []
    seen: set[tuple[str, str, str]] = set()
    for selection in resolved:
        key = (selection.method_name, selection.model_name, selection.objective_name)
        if key in seen:
            continue
        deduplicated.append(selection)
        seen.add(key)
    return deduplicated


def discover_method_selections_from_regime_free(
    results_root: Path = REPO_ROOT / "results" / "regime_free",
) -> list[MethodSelection]:
    results_root = results_root.resolve()
    summary_path = results_root / "all_method_results.csv"
    if summary_path.exists():
        summary_frame = pd.read_csv(summary_path)
        required_columns = {"method_name", "model_name", "objective_name"}
        if not required_columns.issubset(summary_frame.columns):
            raise KeyError(
                f"Expected columns {sorted(required_columns)} in {summary_path}, "
                f"found {summary_frame.columns.tolist()}."
            )
        return resolve_method_selections(
            [
                (
                    str(row["method_name"]),
                    str(row["model_name"]),
                    str(row["objective_name"]),
                )
                for _, row in (
                    summary_frame.loc[:, ["method_name", "model_name", "objective_name"]]
                    .drop_duplicates()
                    .iterrows()
                )
            ]
        )

    discovered: list[MethodSelection] = []
    if not results_root.exists():
        return discovered

    for target_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        for method_dir in sorted(path for path in target_dir.iterdir() if path.is_dir()):
            for objective_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
                discovered.append(
                    parse_method_selection(f"{method_dir.name}/{objective_dir.name}")
                )
    return resolve_method_selections(discovered)


def _method_is_selected(
    *,
    method_name: str,
    artifacts: rfe.ModelArtifacts,
    selections: list[MethodSelection],
) -> bool:
    if not selections:
        return True
    return any(
        selection.method_name == method_name
        and selection.model_name == artifacts.model_name
        and selection.objective_name == artifacts.objective_name
        for selection in selections
    )


def _filter_artifacts(
    artifacts: list[rfe.ModelArtifacts],
    selections: list[MethodSelection],
) -> list[rfe.ModelArtifacts]:
    if not selections:
        return artifacts

    filtered: list[rfe.ModelArtifacts] = []
    for artifacts_item in artifacts:
        if artifacts_item.estimation_kind == "point":
            if _method_is_selected(
                method_name="cp_symmetric",
                artifacts=artifacts_item,
                selections=selections,
            ) or _method_is_selected(
                method_name="cp_asymmetric",
                artifacts=artifacts_item,
                selections=selections,
            ):
                filtered.append(artifacts_item)
            continue

        if _method_is_selected(
            method_name="cqr",
            artifacts=artifacts_item,
            selections=selections,
        ):
            filtered.append(artifacts_item)

    return filtered


def _load_regime_series(csv_path: Path) -> pd.Series:
    regime_frame = pd.read_csv(csv_path, parse_dates=[DATE_COLUMN])
    if DATE_COLUMN not in regime_frame.columns:
        raise ValueError(f"Expected a {DATE_COLUMN!r} column in {csv_path}.")
    if REGIME_LABEL_COLUMN not in regime_frame.columns:
        raise KeyError(f"Missing {REGIME_LABEL_COLUMN!r} column in {csv_path}.")
    return regime_frame.set_index(DATE_COLUMN)[REGIME_LABEL_COLUMN].astype(str).sort_index()


def _discover_regime_groups(
    regimes_root: Path,
    requested_group_names: Iterable[str] | None = None,
) -> list[RegimeGroup]:
    regimes_root = regimes_root.resolve()
    requested = {str(value).strip() for value in (requested_group_names or []) if str(value).strip()}
    groups: list[RegimeGroup] = []

    if not regimes_root.exists():
        raise FileNotFoundError(f"Regimes root does not exist: {regimes_root}")

    for group_dir in sorted(path for path in regimes_root.iterdir() if path.is_dir()):
        if requested and group_dir.name not in requested:
            continue
        csv_path = group_dir / REGIME_FILENAME
        if csv_path.exists():
            groups.append(RegimeGroup(name=group_dir.name, csv_path=csv_path))

    if requested:
        discovered_names = {group.name for group in groups}
        missing = sorted(requested - discovered_names)
        if missing:
            raise FileNotFoundError(
                f"Requested regime groups were not found under {regimes_root}: {missing}"
            )

    if not groups:
        raise FileNotFoundError(
            f"No {REGIME_FILENAME!r} files were found directly under subfolders of {regimes_root}."
        )

    return groups


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
        "pipeline": "mondrian_conformal_prediction",
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


def _build_mondrian_result_frame(
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
    regime_group_name: str,
    regime_label: str,
    regime_calibration_rows: int,
    window_calibration_rows: int,
    calibration_mode: str,
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
    prediction_frame["regime_group_name"] = regime_group_name
    prediction_frame["regime_label"] = regime_label
    prediction_frame["regime_calibration_rows"] = regime_calibration_rows
    prediction_frame["window_calibration_rows"] = window_calibration_rows
    prediction_frame["calibration_mode"] = calibration_mode
    return prediction_frame


def _validate_regime_alignment(
    regime_series: pd.Series,
    index: pd.DatetimeIndex,
    *,
    label: str,
) -> pd.Series:
    aligned = regime_series.reindex(index)
    if aligned.isna().any():
        missing_dates = aligned[aligned.isna()].index.astype(str).tolist()[:5]
        raise KeyError(
            f"Missing regime labels for {label} on dates {missing_dates}. "
            "Ensure regimes.csv spans the full prediction history."
        )
    return aligned.astype(str)


def _run_mondrian_point_method(
    *,
    artifacts: rfe.ModelArtifacts,
    target_spec: rfe.TargetSpec,
    prediction_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    residual_series: pd.Series,
    regime_series: pd.Series,
    regime_group_name: str,
    symmetric: bool,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if "y_pred" not in prediction_frame.columns:
        raise KeyError(f"Missing y_pred column in {artifacts.prediction_path}.")

    y_series = y_frame[target_spec.y_column].reindex(prediction_frame.index)
    if y_series.isna().any():
        raise KeyError(f"Missing y values for target {target_spec.y_column}.")

    full_regimes = _validate_regime_alignment(
        regime_series,
        prediction_frame.index,
        label=f"{target_spec.code} point predictions",
    )
    windows = rfe._build_rolling_windows(
        prediction_frame.index,
        calibration_range_weeks=calibration_range_weeks,
        calibration_frequency_weeks=calibration_frequency_weeks,
    )
    if not windows:
        raise RuntimeError("No rolling windows were created for Mondrian point recalibration.")

    method_name = "cp_symmetric" if symmetric else "cp_asymmetric"
    combined_predictions: list[pd.DataFrame] = []
    weekly_metrics: list[dict[str, object]] = []

    with tqdm(
        total=len(windows),
        desc=f"Weeks | {target_spec.code} | {method_name} | {artifacts.model_name} | {regime_group_name}",
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

            calibration_regimes = _validate_regime_alignment(
                regime_series,
                calibration_residuals.index,
                label=f"{target_spec.code} calibration window {window_index}",
            )
            prediction_regimes = _validate_regime_alignment(
                full_regimes,
                prediction_slice.index,
                label=f"{target_spec.code} prediction window {window_index}",
            )
            y_true = y_series.reindex(prediction_slice.index).astype(float)
            base_prediction = prediction_slice["y_pred"].astype(float)

            window_frames: list[pd.DataFrame] = []
            active_regime_labels: list[str] = []
            fallback_prediction_rows = 0

            for regime_label in pd.unique(prediction_regimes):
                group_index = prediction_regimes.index[prediction_regimes == regime_label]
                if len(group_index) == 0:
                    continue

                regime_calibration = calibration_residuals[calibration_regimes == regime_label]
                if regime_calibration.empty:
                    effective_calibration = calibration_residuals
                    calibration_mode = "window_fallback"
                    fallback_prediction_rows += int(len(group_index))
                else:
                    effective_calibration = regime_calibration
                    calibration_mode = "regime"

                group_prediction = base_prediction.reindex(group_index)
                group_y_true = y_true.reindex(group_index)

                if symmetric:
                    conformal = ConformalPrediction(
                        residuals_upper=effective_calibration.to_numpy(),
                        residuals_lower=effective_calibration.to_numpy(),
                        predictions_upper=group_prediction.to_numpy(),
                        predictions_lower=group_prediction.to_numpy(),
                        alpha=ALPHA,
                        symmetric=True,
                    )
                else:
                    conformal = ConformalPrediction(
                        residuals_upper=effective_calibration[effective_calibration >= 0.0].to_numpy(),
                        residuals_lower=effective_calibration[effective_calibration < 0.0].to_numpy(),
                        predictions_upper=group_prediction.to_numpy(),
                        predictions_lower=group_prediction.to_numpy(),
                        alpha=ALPHA,
                        symmetric=False,
                    )

                result = conformal.get_result()
                lower_pred = pd.Series(result.calibrated_lower, index=group_index)
                upper_pred = pd.Series(result.calibrated_upper, index=group_index)
                margins_lower = pd.Series(result.margin_lower, index=group_index)
                margins_upper = pd.Series(result.margin_upper, index=group_index)

                window_frames.append(
                    _build_mondrian_result_frame(
                        y_true=group_y_true,
                        lower_pred=lower_pred,
                        upper_pred=upper_pred,
                        base_lower_pred=group_prediction,
                        base_upper_pred=group_prediction,
                        margin_lower=margins_lower,
                        margin_upper=margins_upper,
                        window_index=window_index,
                        window=window,
                        calibration_rows=int(len(effective_calibration)),
                        regime_group_name=regime_group_name,
                        regime_label=str(regime_label),
                        regime_calibration_rows=int(len(regime_calibration)),
                        window_calibration_rows=int(len(calibration_residuals)),
                        calibration_mode=calibration_mode,
                    )
                )
                active_regime_labels.append(str(regime_label))

            if not window_frames:
                week_progress.update(1)
                continue

            window_predictions = (
                pd.concat(window_frames, ignore_index=True)
                .sort_values(by=[DATE_COLUMN], kind="mergesort")
                .reset_index(drop=True)
            )
            combined_predictions.append(window_predictions)

            metrics = rfe._compute_interval_metrics_for_frame(window_predictions)
            weekly_metrics.append(
                {
                    "window_index": window_index,
                    "calibration_start": window.calibration_start,
                    "calibration_end_exclusive": window.calibration_end,
                    "prediction_start": window.prediction_start,
                    "prediction_end_exclusive": window.prediction_end,
                    "calibration_rows": int(len(calibration_residuals)),
                    "prediction_rows": int(len(prediction_slice)),
                    "regime_group_name": regime_group_name,
                    "active_regime_count": int(len(active_regime_labels)),
                    "active_regime_labels": "|".join(active_regime_labels),
                    "fallback_prediction_rows": int(fallback_prediction_rows),
                    **metrics,
                }
            )
            week_progress.update(1)

    if not combined_predictions:
        raise RuntimeError("No Mondrian point windows produced any prediction rows.")

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


def _run_mondrian_interval_method(
    *,
    artifacts: rfe.ModelArtifacts,
    target_spec: rfe.TargetSpec,
    prediction_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    residual_low: pd.Series,
    residual_up: pd.Series,
    regime_series: pd.Series,
    regime_group_name: str,
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

    full_regimes = _validate_regime_alignment(
        regime_series,
        prediction_frame.index,
        label=f"{target_spec.code} interval predictions",
    )
    windows = rfe._build_rolling_windows(
        prediction_frame.index,
        calibration_range_weeks=calibration_range_weeks,
        calibration_frequency_weeks=calibration_frequency_weeks,
    )
    if not windows:
        raise RuntimeError("No rolling windows were created for Mondrian CQR recalibration.")

    combined_predictions: list[pd.DataFrame] = []
    weekly_metrics: list[dict[str, object]] = []

    with tqdm(
        total=len(windows),
        desc=f"Weeks | {target_spec.code} | cqr | {artifacts.model_name} | {regime_group_name}",
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

            calibration_regimes = _validate_regime_alignment(
                regime_series,
                calibration_low.index,
                label=f"{target_spec.code} interval calibration window {window_index}",
            )
            prediction_regimes = _validate_regime_alignment(
                full_regimes,
                prediction_slice.index,
                label=f"{target_spec.code} interval prediction window {window_index}",
            )
            y_true = y_series.reindex(prediction_slice.index).astype(float)
            base_lower = prediction_slice["y_pred_lower"].astype(float)
            base_upper = prediction_slice["y_pred_upper"].astype(float)

            window_frames: list[pd.DataFrame] = []
            active_regime_labels: list[str] = []
            fallback_prediction_rows = 0

            for regime_label in pd.unique(prediction_regimes):
                group_index = prediction_regimes.index[prediction_regimes == regime_label]
                if len(group_index) == 0:
                    continue

                regime_calibration_low = calibration_low[calibration_regimes == regime_label]
                regime_calibration_up = calibration_up[calibration_regimes == regime_label]
                if regime_calibration_low.empty or regime_calibration_up.empty:
                    effective_low = calibration_low
                    effective_up = calibration_up
                    calibration_mode = "window_fallback"
                    fallback_prediction_rows += int(len(group_index))
                else:
                    effective_low = regime_calibration_low
                    effective_up = regime_calibration_up
                    calibration_mode = "regime"

                group_base_lower = base_lower.reindex(group_index)
                group_base_upper = base_upper.reindex(group_index)
                group_y_true = y_true.reindex(group_index)

                conformal = ConformalizedQuantileRegression(
                    residuals_lower=effective_low.to_numpy(),
                    residuals_upper=effective_up.to_numpy(),
                    predictions_lower=group_base_lower.to_numpy(),
                    predictions_upper=group_base_upper.to_numpy(),
                    alpha=ALPHA,
                )
                result = conformal.get_result()

                lower_pred = pd.Series(result.calibrated_lower, index=group_index)
                upper_pred = pd.Series(result.calibrated_upper, index=group_index)
                margins_lower = pd.Series(result.margin_lower, index=group_index)
                margins_upper = pd.Series(result.margin_upper, index=group_index)

                window_frames.append(
                    _build_mondrian_result_frame(
                        y_true=group_y_true,
                        lower_pred=lower_pred,
                        upper_pred=upper_pred,
                        base_lower_pred=group_base_lower,
                        base_upper_pred=group_base_upper,
                        margin_lower=margins_lower,
                        margin_upper=margins_upper,
                        window_index=window_index,
                        window=window,
                        calibration_rows=int(len(effective_low)),
                        regime_group_name=regime_group_name,
                        regime_label=str(regime_label),
                        regime_calibration_rows=int(len(regime_calibration_low)),
                        window_calibration_rows=int(len(calibration_low)),
                        calibration_mode=calibration_mode,
                    )
                )
                active_regime_labels.append(str(regime_label))

            if not window_frames:
                week_progress.update(1)
                continue

            window_predictions = (
                pd.concat(window_frames, ignore_index=True)
                .sort_values(by=[DATE_COLUMN], kind="mergesort")
                .reset_index(drop=True)
            )
            combined_predictions.append(window_predictions)

            metrics = rfe._compute_interval_metrics_for_frame(window_predictions)
            weekly_metrics.append(
                {
                    "window_index": window_index,
                    "calibration_start": window.calibration_start,
                    "calibration_end_exclusive": window.calibration_end,
                    "prediction_start": window.prediction_start,
                    "prediction_end_exclusive": window.prediction_end,
                    "calibration_rows": int(len(calibration_low)),
                    "prediction_rows": int(len(prediction_slice)),
                    "regime_group_name": regime_group_name,
                    "active_regime_count": int(len(active_regime_labels)),
                    "active_regime_labels": "|".join(active_regime_labels),
                    "fallback_prediction_rows": int(fallback_prediction_rows),
                    **metrics,
                }
            )
            week_progress.update(1)

    if not combined_predictions:
        raise RuntimeError("No Mondrian CQR windows produced any prediction rows.")

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
    regime_series: pd.Series,
    regime_group: RegimeGroup,
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

            predictions_df, weekly_metrics_df, overall_metrics = _run_mondrian_point_method(
                artifacts=artifacts,
                target_spec=target_spec,
                prediction_frame=prediction_frame,
                y_frame=y_frame,
                residual_series=residual_series,
                regime_series=regime_series,
                regime_group_name=regime_group.name,
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
                        "regime_group_name": regime_group.name,
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
                    "regime_group_name": regime_group.name,
                    "regime_path": str(regime_group.csv_path),
                    "nominal_alpha": ALPHA,
                },
            )

            result_rows.append(
                {
                    "pipeline": "mcp",
                    "regime_group_name": regime_group.name,
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
                    "source_regime_path": str(regime_group.csv_path),
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

    predictions_df, weekly_metrics_df, overall_metrics = _run_mondrian_interval_method(
        artifacts=artifacts,
        target_spec=target_spec,
        prediction_frame=prediction_frame,
        y_frame=y_frame,
        residual_low=residual_low,
        residual_up=residual_up,
        regime_series=regime_series,
        regime_group_name=regime_group.name,
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
                "regime_group_name": regime_group.name,
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
            "regime_group_name": regime_group.name,
            "regime_path": str(regime_group.csv_path),
            "nominal_alpha": ALPHA,
        },
    )

    result_rows.append(
        {
            "pipeline": "mcp",
            "regime_group_name": regime_group.name,
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
            "source_regime_path": str(regime_group.csv_path),
        }
    )
    return result_rows


def run_mondrian_conformal_prediction(
    *,
    calibration_range_weeks: int = CALIBRATION_RANGE,
    calibration_frequency_weeks: int = CALIBRATION_FREQUENCY,
    target_list: list[str] | tuple[str, ...] = TARGET_LIST,
    predictions_root: Path = PREDICTIONS_ROOT,
    residuals_root: Path = RESIDUALS_ROOT,
    regimes_root: Path = REGIMES_ROOT,
    y_path: Path = Y_PATH,
    results_root: Path = RESULTS_ROOT,
    method_model_loss_list: Iterable[object] | None = METHOD_MODEL_LOSS_LIST,
    regime_group_names: Iterable[str] | None = REGIME_GROUP_LIST,
    clean_output: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions_root = predictions_root.resolve()
    residuals_root = residuals_root.resolve()
    regimes_root = regimes_root.resolve()
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
    regime_groups = _discover_regime_groups(regimes_root, regime_group_names)
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

    rfe._log("")
    rfe._log("=" * 80)
    rfe._log("Mondrian Conformal Recalibration")
    rfe._log("=" * 80)
    rfe._log(f"Targets                 : {[target_spec.code for target_spec in target_specs]}")
    rfe._log(f"Calibration range       : {calibration_range_weeks} weeks")
    rfe._log(f"Calibration frequency   : {calibration_frequency_weeks} week(s)")
    rfe._log(f"Nominal coverage        : {UPPER_QUANTILE - LOWER_QUANTILE:.2%}")
    rfe._log(f"Prediction roots        : {predictions_root}")
    rfe._log(f"Residual roots          : {residuals_root}")
    rfe._log(f"Regimes root            : {regimes_root}")
    rfe._log(f"Results root            : {results_root}")
    rfe._log(f"Model-loss directories  : {len(artifacts)}")
    rfe._log(f"Regime groups           : {[group.name for group in regime_groups]}")
    rfe._log("=" * 80)

    with tqdm(
        total=len(regime_groups),
        desc="Regime groups",
        unit="group",
        dynamic_ncols=True,
    ) as group_progress:
        for regime_group in regime_groups:
            group_progress.set_postfix_str(regime_group.name)
            regime_series = _load_regime_series(regime_group.csv_path)
            context_results_root = results_root / regime_group.name
            prediction_cache: dict[Path, pd.DataFrame] = {}
            result_rows: list[dict[str, object]] = []
            failures: list[dict[str, object]] = []

            with tqdm(
                total=len(target_specs),
                desc=f"Targets | {regime_group.name}",
                unit="target",
                dynamic_ncols=True,
                leave=False,
            ) as target_progress:
                for target_spec in target_specs:
                    target_progress.set_postfix_str(target_spec.code)
                    with tqdm(
                        total=len(artifacts),
                        desc=f"Methods | {regime_group.name} | {target_spec.code}",
                        unit="model",
                        dynamic_ncols=True,
                        leave=False,
                    ) as model_progress:
                        for artifacts_item in artifacts:
                            model_progress.set_postfix_str(
                                f"{artifacts_item.estimation_kind}:{artifacts_item.model_name}"
                            )
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
                                        regime_series=regime_series,
                                        regime_group=regime_group,
                                        calibration_range_weeks=calibration_range_weeks,
                                        calibration_frequency_weeks=calibration_frequency_weeks,
                                        context_results_root=context_results_root,
                                        selections=method_selections,
                                    )
                                )
                            except Exception as exc:  # noqa: BLE001
                                failures.append(
                                    {
                                        "pipeline": "mcp",
                                        "regime_group_name": regime_group.name,
                                        "target_code": target_spec.code,
                                        "estimation_kind": artifacts_item.estimation_kind,
                                        "model_name": artifacts_item.model_name,
                                        "objective_name": artifacts_item.objective_name,
                                        "display_name": artifacts_item.display_name,
                                        "error": repr(exc),
                                    }
                                )
                                rfe._log(
                                    f"FAILED {regime_group.name} | {target_spec.code} | "
                                    f"{artifacts_item.display_name} ({artifacts_item.estimation_kind}): {exc}"
                                )
                            finally:
                                model_progress.update(1)
                    target_progress.update(1)

            ranked_results = rfe._write_summary_results(context_results_root, result_rows)
            all_ranked_results.append(ranked_results)
            rfe._write_failures(context_results_root, failures)
            all_failures.extend(failures)
            group_progress.update(1)

    ranked_results_df = (
        pd.concat(all_ranked_results, ignore_index=True)
        if all_ranked_results
        else pd.DataFrame()
    )
    failures_df = pd.DataFrame(all_failures)
    return ranked_results_df, failures_df


def main() -> int:
    args = _parse_args()
    ranked_results, failures_df = run_mondrian_conformal_prediction(
        calibration_range_weeks=args.calibration_range,
        calibration_frequency_weeks=args.calibration_frequency,
        target_list=args.targets,
        method_model_loss_list=args.method_spec or METHOD_MODEL_LOSS_LIST,
        regime_group_names=args.regime_groups or REGIME_GROUP_LIST,
        clean_output=bool(args.clean_output),
    )

    rfe._log("")
    rfe._log("=" * 80)
    rfe._log("Mondrian Summary")
    rfe._log("=" * 80)
    rfe._log(f"Completed runs : {len(ranked_results):,}")
    rfe._log(f"Failures       : {len(failures_df):,}")
    if not ranked_results.empty:
        printable_columns = [
            "regime_group_name",
            "target_code",
            "display_name",
            "mean_pinball_loss",
            "empirical_coverage",
            "mean_interval_width",
            "average_rank",
            "selection_order",
        ]
        available_columns = [column for column in printable_columns if column in ranked_results.columns]
        rfe._log(ranked_results[available_columns].to_string(index=False))
    rfe._log("=" * 80)
    return 0 if failures_df.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
