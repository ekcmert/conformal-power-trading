from __future__ import annotations

import argparse
import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from base_models.interval_estimation.common import (
    DEFAULT_LOWER_QUANTILE,
    DEFAULT_UPPER_QUANTILE,
)
from base_models.point_estimation.common import (
    MCP_TARGETS,
    REPO_ROOT,
    X_PATH,
    output_dir_for,
    standardize_features,
    target_artifact_stem,
)
from base_models.point_estimation.model_registry import _build_experiment_definitions
from conformal_prediction import ConformalPrediction
from exp2 import regime_free_experiment as rfe
from exp3.mondrian_conformal_prediction import (
    MethodSelection,
    parse_method_selection,
)


CALIBRATION_RANGE = 24
CALIBRATION_FREQUENCY = 1
NUM_BATCHES = 20
TARGET_LIST = ["DA", "ID", "ID3", "ID1", "IMB"]

DATE_COLUMN = "date"
LOWER_QUANTILE = DEFAULT_LOWER_QUANTILE
UPPER_QUANTILE = DEFAULT_UPPER_QUANTILE
ALPHA = 1.0 - (UPPER_QUANTILE - LOWER_QUANTILE)

Y_PATH = REPO_ROOT / "data" / "final" / "y.csv"
PREDICTIONS_ROOT = REPO_ROOT / "data" / "predictions"
BEST_PARAMS_ROOT = REPO_ROOT / "results" / "tuning" / "point_estimation" / "mcp_models"
RESULTS_ROOT = REPO_ROOT / "results" / "regime_aware" / "enbpi"

ENBPI_MODEL_NAME = "lightgbm"
ENBPI_OBJECTIVE_NAME = "regression_l1"
RANDOM_STATE = 42
AGGREGATION = "median"


@dataclass(frozen=True)
class EnbPISelection:
    model_name: str
    objective_name: str
    symmetries: tuple[bool, ...]


@dataclass(frozen=True)
class EnbPIModelSpec:
    model_name: str
    objective_name: str
    display_name: str
    params: dict[str, object]
    params_summary: str
    best_params_path: Path | None


@dataclass(frozen=True)
class EnbPIBootstrapResult:
    train_prediction: pd.Series
    prediction: pd.Series
    oob_counts: pd.Series
    fallback_oob_rows: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ensemble Batch Prediction Intervals with internal bootstrap training.",
    )
    parser.add_argument("--calibration-range", type=int, default=CALIBRATION_RANGE)
    parser.add_argument("--calibration-frequency", type=int, default=CALIBRATION_FREQUENCY)
    parser.add_argument("--num-batches", type=int, default=NUM_BATCHES)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=TARGET_LIST,
        help="Subset of targets to run. Available: DA ID ID3 ID1 IMB",
    )
    parser.add_argument(
        "--method-spec",
        default=None,
        help=(
            "Optional single point model selection such as "
            "'lightgbm/regression_l1' or 'enbpi_asymmetric_lightgbm/regression_l1'."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=ENBPI_MODEL_NAME,
        help="Point base learner model name used by EnbPI.",
    )
    parser.add_argument(
        "--objective-name",
        default=ENBPI_OBJECTIVE_NAME,
        help="Point base learner objective/loss name used by EnbPI.",
    )
    parser.add_argument(
        "--symmetry",
        choices=("symmetric", "asymmetric", "both"),
        default="both",
        help="Which EnbPI residual calibration variant to run.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing results/regime_aware/enbpi outputs before running.",
    )
    return parser.parse_args()


def _normalize_objective_name(value: object) -> str:
    normalized = rfe._normalize_objective_name(value)
    if normalized.strip().lower() == "default":
        return ""
    return normalized


def _normalize_requested_symmetries(symmetric: bool | None) -> tuple[bool, ...]:
    if symmetric is None:
        return (True, False)
    return (bool(symmetric),)


def _method_to_symmetry(method_name: str) -> bool | None:
    normalized = str(method_name).strip().lower().replace("-", "_")
    if normalized in {"cp_symmetric", "enbpi_symmetric", "symmetric"}:
        return True
    if normalized in {"cp_asymmetric", "enbpi_asymmetric", "asymmetric"}:
        return False
    return None


def _is_interval_only_selection(value: object) -> bool:
    if isinstance(value, MethodSelection):
        normalized_method = value.method_name
    elif isinstance(value, dict):
        normalized_method = str(value.get("method_name", value.get("method", "")))
    elif isinstance(value, (tuple, list)) and len(value) >= 3:
        normalized_method = str(value[0])
    elif isinstance(value, str):
        normalized_method = value.strip().replace("\\", "/").split("/", 1)[0]
    else:
        return False

    normalized_method = normalized_method.strip().lower().replace("-", "_")
    return normalized_method in {
        "cqr",
        "conformalized_quantile_regression",
    } or normalized_method.startswith("cqr_")


def _parse_enbpi_folder_spec(value: str) -> tuple[str, str, bool | None]:
    normalized = value.strip().replace("\\", "/")
    if "/" not in normalized:
        raise ValueError(
            "Selection strings must follow 'model/objective', "
            "'cp_symmetric_model/objective', or 'enbpi_symmetric_model/objective'."
        )

    method_folder, objective_name = normalized.split("/", 1)
    method_folder = method_folder.strip()
    objective_name = objective_name.strip()
    if not method_folder or not objective_name:
        raise ValueError(f"Invalid EnbPI selection string {value!r}.")

    prefixes = (
        ("enbpi_symmetric_", True),
        ("enbpi_asymmetric_", False),
        ("cp_symmetric_", True),
        ("cp_asymmetric_", False),
    )
    for prefix, symmetry in prefixes:
        if method_folder.startswith(prefix):
            model_name = method_folder[len(prefix) :].strip()
            if not model_name:
                raise ValueError(f"Missing model name in selection {value!r}.")
            return model_name, objective_name, symmetry

    if method_folder.startswith("cqr_"):
        model_name = method_folder[len("cqr_") :].strip()
        return model_name, objective_name, None

    return method_folder, objective_name, None


def _selection_parts(value: object) -> tuple[str, str, bool | None]:
    if isinstance(value, MethodSelection):
        return (
            value.model_name,
            _normalize_objective_name(value.objective_name),
            _method_to_symmetry(value.method_name),
        )

    if isinstance(value, str):
        model_name, objective_name, folder_symmetry = _parse_enbpi_folder_spec(value)
        return model_name, _normalize_objective_name(objective_name), folder_symmetry

    if isinstance(value, dict):
        model_name = str(value.get("model_name", value.get("model", ""))).strip()
        objective_name = _normalize_objective_name(
            value.get("objective_name", value.get("loss", ""))
        )
        method_name = value.get("method_name", value.get("method"))
        if not model_name:
            parsed = parse_method_selection(value)
            return (
                parsed.model_name,
                _normalize_objective_name(parsed.objective_name),
                _method_to_symmetry(parsed.method_name),
            )
        symmetry = _method_to_symmetry(method_name) if method_name is not None else None
        if "symmetric" in value:
            symmetry = bool(value["symmetric"])
        return model_name, objective_name, symmetry

    if isinstance(value, (tuple, list)):
        if len(value) == 2:
            return (
                str(value[0]).strip(),
                _normalize_objective_name(value[1]),
                None,
            )
        parsed = parse_method_selection(value)
        return (
            parsed.model_name,
            _normalize_objective_name(parsed.objective_name),
            _method_to_symmetry(parsed.method_name),
        )

    raise TypeError(f"Unsupported EnbPI method selection type: {type(value)!r}")


def resolve_enbpi_selection(
    model_loss_pair: object | None,
    *,
    model_name: str = ENBPI_MODEL_NAME,
    objective_name: str = ENBPI_OBJECTIVE_NAME,
    symmetric: bool | None,
) -> EnbPISelection:
    requested_symmetries = _normalize_requested_symmetries(symmetric)
    item_symmetry: bool | None = None

    if model_loss_pair is None:
        resolved_model_name = str(model_name).strip()
        resolved_objective_name = _normalize_objective_name(objective_name)
    else:
        if isinstance(model_loss_pair, (list, tuple)) and not isinstance(
            model_loss_pair,
            (str, bytes, dict, MethodSelection),
        ):
            pair_items = list(model_loss_pair)
            if pair_items and all(
                not isinstance(item, (str, bytes)) or "/" in str(item)
                for item in pair_items
            ):
                if len(pair_items) != 1:
                    raise ValueError(
                        "EnbPI accepts exactly one point model/loss pair, "
                        f"got {len(pair_items)} selections."
                    )
                model_loss_pair = pair_items[0]

        if _is_interval_only_selection(model_loss_pair):
            raise ValueError(
                f"EnbPI requires a point model/loss pair, got {model_loss_pair!r}."
            )
        resolved_model_name, resolved_objective_name, item_symmetry = _selection_parts(
            model_loss_pair
        )

    if not resolved_model_name:
        raise ValueError("EnbPI selection is missing model_name.")

    symmetries = requested_symmetries
    if item_symmetry is not None:
        symmetries = tuple(value for value in requested_symmetries if value == item_symmetry)
    if not symmetries:
        raise ValueError(
            "The selected EnbPI method symmetry is incompatible with the requested "
            f"symmetric={symmetric!r} setting."
        )

    return EnbPISelection(
        model_name=resolved_model_name,
        objective_name=resolved_objective_name,
        symmetries=tuple(sorted(symmetries, reverse=True)),
    )


def _load_best_params_table(best_params_root: Path) -> pd.DataFrame:
    best_params_path = best_params_root / "best_params.csv"
    if not best_params_path.exists():
        return pd.DataFrame()
    table = pd.read_csv(best_params_path)
    table["objective_name"] = table["objective_name"].map(_normalize_objective_name)
    return table


def _best_params_json_path(best_params_root: Path, model_name: str, objective_name: str) -> Path:
    objective_part = objective_name or "default"
    return (
        REPO_ROOT
        / "base_models"
        / "point_estimation"
        / "best_params"
        / f"{model_name}__{objective_part}.json"
    )


def _load_model_spec(
    selection: EnbPISelection,
    *,
    best_params_root: Path,
    best_params_table: pd.DataFrame,
) -> EnbPIModelSpec:
    model_name = selection.model_name
    objective_name = _normalize_objective_name(selection.objective_name)
    matching_rows = pd.DataFrame()
    if not best_params_table.empty:
        matching_rows = best_params_table.loc[
            (best_params_table["model_name"].astype(str) == model_name)
            & (best_params_table["objective_name"].map(_normalize_objective_name) == objective_name)
        ]

    best_params_path: Path | None = None
    if not matching_rows.empty:
        row = matching_rows.iloc[0]
        params = json.loads(str(row["best_params_json"]))
        display_name = str(row.get("display_name") or f"{model_name} | {objective_name}")
        params_summary = str(row.get("params_summary") or repr(params))
        raw_path = row.get("best_params_json_path")
        if isinstance(raw_path, str) and raw_path.strip():
            best_params_path = Path(raw_path)
    else:
        candidate_path = _best_params_json_path(best_params_root, model_name, objective_name)
        if not candidate_path.exists():
            raise FileNotFoundError(
                f"No best-params entry found for {model_name}/{objective_name} in "
                f"{best_params_root} or {candidate_path}."
            )
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        params = dict(payload["best_params"])
        display_name = str(payload.get("display_name") or f"{model_name} | {objective_name}")
        params_summary = str(payload.get("params_summary") or repr(params))
        best_params_path = candidate_path

    return EnbPIModelSpec(
        model_name=model_name,
        objective_name=objective_name,
        display_name=display_name,
        params=dict(params),
        params_summary=params_summary,
        best_params_path=best_params_path,
    )


def _definition_for(model_name: str, objective_name: str):
    for definition in _build_experiment_definitions():
        definition_objective = _normalize_objective_name(definition.objective_name)
        if definition.model_name == model_name and definition_objective == objective_name:
            return definition
    raise KeyError(f"No point-estimation model definition for {model_name}/{objective_name}.")


def _batch_params(params: dict[str, object], batch_index: int, random_state: int) -> dict[str, object]:
    batch_params = dict(params)
    seed = int(random_state) + int(batch_index)
    for seed_key in ("random_state", "random_seed"):
        if seed_key in batch_params:
            batch_params[seed_key] = seed
    return batch_params


def _build_model(spec: EnbPIModelSpec, *, batch_index: int, random_state: int) -> object:
    definition = _definition_for(spec.model_name, spec.objective_name)
    return definition.builder_factory(_batch_params(spec.params, batch_index, random_state))


def _month_start(timestamp: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=timestamp.year, month=timestamp.month, day=1, tz=timestamp.tz)


def _default_prediction_index(
    full_index: pd.DatetimeIndex,
    *,
    training_range_months: int,
) -> pd.DatetimeIndex:
    first_available = full_index.min()
    first_month = _month_start(first_available)
    has_full_first_month = first_available == first_month
    prediction_start = first_month + pd.DateOffset(
        months=training_range_months + (0 if has_full_first_month else 1)
    )
    return full_index[full_index >= prediction_start]


def _reference_prediction_index(
    *,
    predictions_root: Path,
    model_name: str,
    objective_name: str,
) -> pd.DatetimeIndex | None:
    model_dir = output_dir_for(
        predictions_root / "point_estimation",
        model_name,
        objective_name,
    )
    if not model_dir.exists():
        return None
    prediction_files = sorted(model_dir.glob("*_predictions.csv"))
    if not prediction_files:
        return None
    prediction_frame = pd.read_csv(prediction_files[0], parse_dates=[DATE_COLUMN])
    if DATE_COLUMN not in prediction_frame.columns:
        return None
    return pd.DatetimeIndex(prediction_frame[DATE_COLUMN]).sort_values()


def _prediction_index_for_model(
    full_index: pd.DatetimeIndex,
    *,
    predictions_root: Path,
    model_name: str,
    objective_name: str,
    training_range_months: int,
) -> pd.DatetimeIndex:
    reference_index = _reference_prediction_index(
        predictions_root=predictions_root,
        model_name=model_name,
        objective_name=objective_name,
    )
    if reference_index is None or reference_index.empty:
        reference_index = _default_prediction_index(
            full_index,
            training_range_months=training_range_months,
        )
    prediction_index = pd.DatetimeIndex(reference_index.intersection(full_index)).sort_values()
    if prediction_index.empty:
        raise RuntimeError(f"No prediction rows found for {model_name}/{objective_name}.")
    return prediction_index


def _load_final_frames(
    *,
    x_path: Path,
    y_path: Path,
    target_specs: list[rfe.TargetSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_targets = sorted({MCP_TARGETS[0], *[target_spec.y_column for target_spec in target_specs]})
    X = pd.read_csv(x_path, index_col=0, parse_dates=True)
    y = pd.read_csv(y_path, index_col=0, parse_dates=True)
    X, y = X.align(y, join="inner", axis=0)

    missing_targets = [target for target in required_targets if target not in y.columns]
    if missing_targets:
        raise KeyError(f"Missing target columns in {y_path}: {missing_targets}")

    return X.sort_index(), y.loc[:, required_targets].sort_index()


def _aggregate_predictions(prediction_matrix: np.ndarray, aggregation: str) -> np.ndarray:
    normalized = aggregation.strip().lower().replace("-", "_")
    if normalized == "mean":
        valid_counts = np.isfinite(prediction_matrix).sum(axis=0)
        result = np.full(prediction_matrix.shape[1], np.nan, dtype=float)
        valid_mask = valid_counts > 0
        result[valid_mask] = (
            np.nansum(prediction_matrix[:, valid_mask], axis=0)
            / valid_counts[valid_mask]
        )
        return result
    if normalized == "median":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmedian(prediction_matrix, axis=0)
    if normalized in {"trimmed_mean", "trimmed"}:
        if prediction_matrix.shape[0] <= 2:
            return _aggregate_predictions(prediction_matrix, "mean")
        ordered = np.sort(prediction_matrix, axis=0)
        return _aggregate_predictions(ordered[1:-1], "mean")
    raise ValueError(f"Unsupported EnbPI aggregation {aggregation!r}.")


def _fit_bootstrap_ensemble(
    *,
    spec: EnbPIModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_prediction: pd.DataFrame,
    num_batches: int,
    aggregation: str,
    random_state: int,
) -> EnbPIBootstrapResult:
    if num_batches <= 0:
        raise ValueError(f"num_batches must be positive, got {num_batches}.")

    X_train_scaled, X_prediction_scaled, _ = standardize_features(X_train, X_prediction)
    y_train = y_train.astype(float)
    train_rows = len(X_train_scaled)
    prediction_rows = len(X_prediction_scaled)
    rng = np.random.default_rng(random_state)

    train_predictions = np.full((num_batches, train_rows), np.nan, dtype=float)
    prediction_matrix = np.empty((num_batches, prediction_rows), dtype=float)
    oob_counts = np.zeros(train_rows, dtype=int)
    fitted_models: list[object] = []

    with tqdm(
        total=num_batches,
        desc=f"Bootstrap | {spec.display_name}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    ) as batch_progress:
        for batch_index in range(num_batches):
            sample_positions = rng.integers(0, train_rows, size=train_rows)
            in_bag = np.zeros(train_rows, dtype=bool)
            in_bag[sample_positions] = True
            oob_positions = np.flatnonzero(~in_bag)

            model = _build_model(spec, batch_index=batch_index, random_state=random_state)
            model.fit(
                X_train_scaled.iloc[sample_positions],
                y_train.iloc[sample_positions],
            )
            fitted_models.append(model)

            if oob_positions.size:
                train_predictions[batch_index, oob_positions] = model.predict(
                    X_train_scaled.iloc[oob_positions]
                )
                oob_counts[oob_positions] += 1
            prediction_matrix[batch_index, :] = model.predict(X_prediction_scaled)
            batch_progress.update(1)

    oob_prediction_values = _aggregate_predictions(train_predictions, aggregation)
    fallback_mask = ~np.isfinite(oob_prediction_values)
    fallback_oob_rows = int(fallback_mask.sum())
    if fallback_oob_rows:
        fallback_positions = np.flatnonzero(fallback_mask)
        fallback_predictions = np.vstack(
            [
                model.predict(X_train_scaled.iloc[fallback_positions])
                for model in fitted_models
            ]
        )
        oob_prediction_values[fallback_mask] = _aggregate_predictions(
            fallback_predictions,
            aggregation,
        )

    return EnbPIBootstrapResult(
        train_prediction=pd.Series(oob_prediction_values, index=X_train.index, name="oob_prediction"),
        prediction=pd.Series(
            _aggregate_predictions(prediction_matrix, aggregation),
            index=X_prediction.index,
            name="y_pred",
        ),
        oob_counts=pd.Series(oob_counts, index=X_train.index, name="oob_model_count"),
        fallback_oob_rows=fallback_oob_rows,
    )


def _prediction_windows(
    prediction_index: pd.DatetimeIndex,
    *,
    calibration_frequency_weeks: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if prediction_index.empty:
        return []
    frequency_delta = pd.Timedelta(weeks=calibration_frequency_weeks)
    prediction_start = prediction_index.min()
    prediction_last_timestamp = prediction_index.max()
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    while prediction_start <= prediction_last_timestamp:
        prediction_end = prediction_start + frequency_delta
        windows.append((prediction_start, prediction_end))
        prediction_start = prediction_end
    return windows


def _enbpi_method_name(*, symmetric: bool) -> str:
    return "enbpi_symmetric" if symmetric else "enbpi_asymmetric"


def _enbpi_output_name(model_name: str, *, symmetric: bool) -> str:
    return f"{_enbpi_method_name(symmetric=symmetric)}_{model_name}"


def _enbpi_display_name(base_display_name: str, *, symmetric: bool) -> str:
    label = "symmetric" if symmetric else "asymmetric"
    return f"EnbPI {label} | {base_display_name}"


def _write_pipeline_metadata(
    *,
    output_dir: Path,
    target_spec: rfe.TargetSpec,
    spec: EnbPIModelSpec,
    method_name: str,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
    num_batches: int,
    window_count: int,
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
    train_rows: int,
    prediction_rows: int,
    oob_count_min: int,
    oob_count_mean: float,
    fallback_oob_rows: int,
    aggregation: str,
) -> None:
    metadata = {
        "pipeline": "ensemble_batch_prediction_intervals",
        "target_code": target_spec.code,
        "target_slug": target_spec.slug,
        "target_name": target_spec.y_column,
        "method_name": method_name,
        "estimation_kind": "point",
        "model_name": spec.model_name,
        "objective_name": spec.objective_name,
        "base_display_name": spec.display_name,
        "calibration_range_weeks": calibration_range_weeks,
        "calibration_frequency_weeks": calibration_frequency_weeks,
        "alpha": ALPHA,
        "lower_quantile": LOWER_QUANTILE,
        "upper_quantile": UPPER_QUANTILE,
        "num_batches": num_batches,
        "aggregation": aggregation,
        "window_count": window_count,
        "train_rows": train_rows,
        "prediction_rows": prediction_rows,
        "prediction_start": str(prediction_start),
        "prediction_end": str(prediction_end),
        "oob_count_min": oob_count_min,
        "oob_count_mean": oob_count_mean,
        "fallback_oob_rows": fallback_oob_rows,
        "best_params_path": str(spec.best_params_path) if spec.best_params_path else "",
        "params": spec.params,
        "params_summary": spec.params_summary,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_enbpi_target_variant(
    *,
    target_spec: rfe.TargetSpec,
    spec: EnbPIModelSpec,
    bootstrap_result: EnbPIBootstrapResult,
    y_frame: pd.DataFrame,
    symmetric: bool,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
    num_batches: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    y_train_target = y_frame.loc[bootstrap_result.train_prediction.index, target_spec.y_column]
    initial_buffer = (y_train_target.astype(float) - bootstrap_result.train_prediction).dropna()
    if initial_buffer.empty:
        raise RuntimeError(f"OOB residual buffer is empty for {target_spec.code}.")

    y_prediction_target = y_frame.loc[bootstrap_result.prediction.index, target_spec.y_column].astype(float)
    prediction_windows = _prediction_windows(
        bootstrap_result.prediction.index,
        calibration_frequency_weeks=calibration_frequency_weeks,
    )
    if not prediction_windows:
        raise RuntimeError("No EnbPI prediction windows were created.")

    method_name = _enbpi_method_name(symmetric=symmetric)
    residual_buffer = initial_buffer.copy()
    combined_predictions: list[pd.DataFrame] = []
    weekly_metrics: list[dict[str, object]] = []

    with tqdm(
        total=len(prediction_windows),
        desc=f"Weeks | {target_spec.code} | {method_name} | {spec.model_name}",
        unit="week",
        dynamic_ncols=True,
        leave=False,
    ) as week_progress:
        for window_index, (prediction_start, prediction_end) in enumerate(prediction_windows):
            prediction_slice = bootstrap_result.prediction.loc[
                (bootstrap_result.prediction.index >= prediction_start)
                & (bootstrap_result.prediction.index < prediction_end)
            ]
            if prediction_slice.empty:
                week_progress.update(1)
                continue

            y_true = y_prediction_target.reindex(prediction_slice.index).astype(float)
            buffer_before_rows = int(len(residual_buffer))
            if symmetric:
                conformal = ConformalPrediction(
                    residuals_upper=residual_buffer.to_numpy(),
                    residuals_lower=residual_buffer.to_numpy(),
                    predictions_upper=prediction_slice.to_numpy(),
                    predictions_lower=prediction_slice.to_numpy(),
                    alpha=ALPHA,
                    symmetric=True,
                )
            else:
                conformal = ConformalPrediction(
                    residuals_upper=residual_buffer[residual_buffer >= 0.0].to_numpy(),
                    residuals_lower=residual_buffer[residual_buffer < 0.0].to_numpy(),
                    predictions_upper=prediction_slice.to_numpy(),
                    predictions_lower=prediction_slice.to_numpy(),
                    alpha=ALPHA,
                    symmetric=False,
                )

            result = conformal.get_result()
            lower_pred = pd.Series(result.calibrated_lower, index=prediction_slice.index)
            upper_pred = pd.Series(result.calibrated_upper, index=prediction_slice.index)
            margins_lower = pd.Series(result.margin_lower, index=prediction_slice.index)
            margins_upper = pd.Series(result.margin_upper, index=prediction_slice.index)

            buffer_start = residual_buffer.index.min()
            window = rfe.RollingWindow(
                calibration_start=buffer_start,
                calibration_end=prediction_start,
                prediction_start=prediction_start,
                prediction_end=prediction_end,
            )
            calibrated_frame = rfe._build_result_prediction_frame(
                y_true=y_true,
                lower_pred=lower_pred,
                upper_pred=upper_pred,
                base_lower_pred=prediction_slice,
                base_upper_pred=prediction_slice,
                margin_lower=margins_lower,
                margin_upper=margins_upper,
                window_index=window_index,
                window=window,
                calibration_rows=buffer_before_rows,
            )

            newest_residuals = (y_true - prediction_slice).dropna()
            if not newest_residuals.empty:
                residual_buffer = pd.concat(
                    [
                        residual_buffer.iloc[len(newest_residuals) :],
                        newest_residuals,
                    ]
                )

            buffer_after_rows = int(len(residual_buffer))
            calibrated_frame["enbpi_symmetric"] = bool(symmetric)
            calibrated_frame["num_batches"] = int(num_batches)
            calibrated_frame["residual_buffer_rows_before"] = buffer_before_rows
            calibrated_frame["residual_buffer_rows_after"] = buffer_after_rows
            calibrated_frame["residual_buffer_update_rows"] = int(len(newest_residuals))
            combined_predictions.append(calibrated_frame)

            metrics = rfe._compute_interval_metrics_for_frame(calibrated_frame)
            weekly_metrics.append(
                {
                    "window_index": window_index,
                    "calibration_start": buffer_start,
                    "calibration_end_exclusive": prediction_start,
                    "prediction_start": prediction_start,
                    "prediction_end_exclusive": prediction_end,
                    "calibration_rows": buffer_before_rows,
                    "prediction_rows": int(len(prediction_slice)),
                    "residual_buffer_rows_before": buffer_before_rows,
                    "residual_buffer_rows_after": buffer_after_rows,
                    "residual_buffer_update_rows": int(len(newest_residuals)),
                    "margin_lower": float(margins_lower.iloc[0]),
                    "margin_upper": float(margins_upper.iloc[0]),
                    "enbpi_symmetric": bool(symmetric),
                    **metrics,
                }
            )
            week_progress.update(1)

    if not combined_predictions:
        raise RuntimeError("No EnbPI windows produced any prediction rows.")

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


def _write_variant_outputs(
    *,
    target_spec: rfe.TargetSpec,
    spec: EnbPIModelSpec,
    bootstrap_result: EnbPIBootstrapResult,
    predictions_df: pd.DataFrame,
    weekly_metrics_df: pd.DataFrame,
    overall_metrics: dict[str, float],
    output_dir: Path,
    symmetric: bool,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
    num_batches: int,
    aggregation: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
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
                "num_batches": num_batches,
                "aggregation": aggregation,
                **overall_metrics,
            }
        ]
    ).to_csv(overall_metrics_path, index=False)

    display_name = _enbpi_display_name(spec.display_name, symmetric=symmetric)
    plot_html_path = rfe._save_prediction_plot_html(
        output_dir=output_dir,
        target_spec=target_spec,
        display_name=display_name,
        prediction_frame=predictions_df,
        metrics=overall_metrics,
        estimation_kind="point",
    )
    plot_png_path = rfe._save_prediction_plot_png(
        output_dir=output_dir,
        target_spec=target_spec,
        display_name=display_name,
        prediction_frame=predictions_df,
        metrics=overall_metrics,
        estimation_kind="point",
    )

    _write_pipeline_metadata(
        output_dir=output_dir,
        target_spec=target_spec,
        spec=spec,
        method_name=_enbpi_method_name(symmetric=symmetric),
        calibration_range_weeks=calibration_range_weeks,
        calibration_frequency_weeks=calibration_frequency_weeks,
        num_batches=num_batches,
        window_count=len(weekly_metrics_df),
        prediction_start=pd.Timestamp(predictions_df[DATE_COLUMN].min()),
        prediction_end=pd.Timestamp(predictions_df[DATE_COLUMN].max()),
        train_rows=int(len(bootstrap_result.train_prediction)),
        prediction_rows=int(len(bootstrap_result.prediction)),
        oob_count_min=int(bootstrap_result.oob_counts.min()),
        oob_count_mean=float(bootstrap_result.oob_counts.mean()),
        fallback_oob_rows=bootstrap_result.fallback_oob_rows,
        aggregation=aggregation,
    )

    return {
        "pipeline": "enbpi",
        "target_code": target_spec.code,
        "target_slug": target_spec.slug,
        "target_name": target_spec.y_column,
        "method_name": _enbpi_method_name(symmetric=symmetric),
        "estimation_kind": "point",
        "model_name": spec.model_name,
        "objective_name": spec.objective_name,
        "display_name": display_name,
        "base_display_name": spec.display_name,
        "calibration_range_weeks": calibration_range_weeks,
        "calibration_frequency_weeks": calibration_frequency_weeks,
        "alpha": ALPHA,
        "lower_quantile": LOWER_QUANTILE,
        "upper_quantile": UPPER_QUANTILE,
        "num_batches": num_batches,
        "aggregation": aggregation,
        "window_count": int(len(weekly_metrics_df)),
        "prediction_rows": int(len(predictions_df)),
        **overall_metrics,
        "output_dir": str(output_dir),
        "prediction_csv": str(prediction_csv_path),
        "prediction_plot_html": str(plot_html_path),
        "prediction_plot_png": str(plot_png_path),
        "weekly_metrics_csv": str(weekly_metrics_path),
        "overall_metrics_csv": str(overall_metrics_path),
        "source_best_params_path": str(spec.best_params_path) if spec.best_params_path else "",
    }


def run_ensemble_batch_prediction_intervals(
    *,
    calibration_range_weeks: int = CALIBRATION_RANGE,
    calibration_frequency_weeks: int = CALIBRATION_FREQUENCY,
    target_list: list[str] | tuple[str, ...] = TARGET_LIST,
    x_path: Path = X_PATH,
    y_path: Path = Y_PATH,
    predictions_root: Path = PREDICTIONS_ROOT,
    best_params_root: Path = BEST_PARAMS_ROOT,
    results_root: Path = RESULTS_ROOT,
    model_loss_pair: object | None = None,
    model_name: str = ENBPI_MODEL_NAME,
    objective_name: str = ENBPI_OBJECTIVE_NAME,
    num_batches: int = NUM_BATCHES,
    symmetric: bool | None = True,
    aggregation: str = AGGREGATION,
    random_state: int = RANDOM_STATE,
    training_range_months: int = 24,
    clean_output: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_path = x_path.resolve()
    y_path = y_path.resolve()
    predictions_root = predictions_root.resolve()
    best_params_root = best_params_root.resolve()
    results_root = results_root.resolve()

    if calibration_range_weeks <= 0:
        raise ValueError(f"calibration_range_weeks must be positive, got {calibration_range_weeks}.")
    if calibration_frequency_weeks <= 0:
        raise ValueError(
            f"calibration_frequency_weeks must be positive, got {calibration_frequency_weeks}."
        )
    if num_batches <= 0:
        raise ValueError(f"num_batches must be positive, got {num_batches}.")

    if clean_output and results_root.exists():
        shutil.rmtree(results_root)

    target_specs = rfe._resolve_targets(list(target_list))
    X, y_frame = _load_final_frames(x_path=x_path, y_path=y_path, target_specs=target_specs)
    selection = resolve_enbpi_selection(
        model_loss_pair,
        model_name=model_name,
        objective_name=objective_name,
        symmetric=symmetric,
    )

    best_params_table = _load_best_params_table(best_params_root)
    all_result_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    rfe._log("")
    rfe._log("=" * 80)
    rfe._log("Ensemble Batch Prediction Intervals")
    rfe._log("=" * 80)
    rfe._log(f"Targets                 : {[target_spec.code for target_spec in target_specs]}")
    rfe._log(f"Calibration frequency   : {calibration_frequency_weeks} week(s)")
    rfe._log(f"Nominal coverage        : {UPPER_QUANTILE - LOWER_QUANTILE:.2%}")
    rfe._log(f"Bootstrap batches       : {num_batches}")
    rfe._log(f"Aggregation             : {aggregation}")
    rfe._log(f"Best params root        : {best_params_root}")
    rfe._log(f"Results root            : {results_root}")
    rfe._log(
        f"Model-loss pair         : "
        f"{selection.model_name}/{selection.objective_name or 'default'}"
    )
    rfe._log("=" * 80)

    with tqdm(total=1, desc="EnbPI model", unit="model", dynamic_ncols=True) as model_progress:
        for selection_index, selection_item in enumerate([selection]):
            model_progress.set_postfix_str(
                f"{selection_item.model_name}/{selection_item.objective_name or 'default'}"
            )
            try:
                spec = _load_model_spec(
                    selection_item,
                    best_params_root=best_params_root,
                    best_params_table=best_params_table,
                )
                prediction_index = _prediction_index_for_model(
                    X.index,
                    predictions_root=predictions_root,
                    model_name=spec.model_name,
                    objective_name=spec.objective_name,
                    training_range_months=training_range_months,
                )
                prediction_start = prediction_index.min()
                train_index = X.index[X.index < prediction_start]
                if train_index.empty:
                    raise RuntimeError(
                        f"Training split is empty before prediction start {prediction_start}."
                    )

                X_train = X.loc[train_index].copy()
                X_prediction = X.loc[prediction_index].copy()
                y_train = y_frame.loc[train_index, MCP_TARGETS[0]].astype(float)
                bootstrap_result = _fit_bootstrap_ensemble(
                    spec=spec,
                    X_train=X_train,
                    y_train=y_train,
                    X_prediction=X_prediction,
                    num_batches=num_batches,
                    aggregation=aggregation,
                    random_state=random_state + selection_index * max(num_batches, 1),
                )

                for target_spec in target_specs:
                    for variant_symmetric in selection_item.symmetries:
                        try:
                            output_dir = output_dir_for(
                                results_root / target_spec.slug,
                                _enbpi_output_name(spec.model_name, symmetric=variant_symmetric),
                                spec.objective_name,
                            )
                            predictions_df, weekly_metrics_df, overall_metrics = _run_enbpi_target_variant(
                                target_spec=target_spec,
                                spec=spec,
                                bootstrap_result=bootstrap_result,
                                y_frame=y_frame,
                                symmetric=variant_symmetric,
                                calibration_range_weeks=calibration_range_weeks,
                                calibration_frequency_weeks=calibration_frequency_weeks,
                                num_batches=num_batches,
                            )
                            all_result_rows.append(
                                _write_variant_outputs(
                                    target_spec=target_spec,
                                    spec=spec,
                                    bootstrap_result=bootstrap_result,
                                    predictions_df=predictions_df,
                                    weekly_metrics_df=weekly_metrics_df,
                                    overall_metrics=overall_metrics,
                                    output_dir=output_dir,
                                    symmetric=variant_symmetric,
                                    calibration_range_weeks=calibration_range_weeks,
                                    calibration_frequency_weeks=calibration_frequency_weeks,
                                    num_batches=num_batches,
                                    aggregation=aggregation,
                                )
                            )
                        except Exception as exc:  # noqa: BLE001
                            failures.append(
                                {
                                    "pipeline": "enbpi",
                                    "target_code": target_spec.code,
                                    "method_name": _enbpi_method_name(symmetric=variant_symmetric),
                                    "estimation_kind": "point",
                                    "model_name": spec.model_name,
                                    "objective_name": spec.objective_name,
                                    "display_name": spec.display_name,
                                    "error": repr(exc),
                                }
                            )
                            rfe._log(
                                f"FAILED EnbPI {target_spec.code} | "
                                f"{_enbpi_method_name(symmetric=variant_symmetric)} | "
                                f"{spec.display_name}: {exc}"
                            )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "pipeline": "enbpi",
                        "target_code": "",
                        "method_name": "",
                        "estimation_kind": "point",
                        "model_name": selection_item.model_name,
                        "objective_name": selection_item.objective_name,
                        "display_name": (
                            f"{selection_item.model_name} | "
                            f"{selection_item.objective_name or 'default'}"
                        ),
                        "error": repr(exc),
                    }
                )
                rfe._log(
                    f"FAILED EnbPI model "
                    f"{selection_item.model_name}/{selection_item.objective_name or 'default'}: {exc}"
                )
            finally:
                model_progress.update(1)

    ranked_results = rfe._write_summary_results(results_root, all_result_rows)
    rfe._write_failures(results_root, failures)
    return ranked_results, pd.DataFrame(failures)


def main() -> int:
    args = _parse_args()
    if args.symmetry == "both":
        symmetric: bool | None = None
    else:
        symmetric = args.symmetry == "symmetric"

    ranked_results, failures_df = run_ensemble_batch_prediction_intervals(
        calibration_range_weeks=args.calibration_range,
        calibration_frequency_weeks=args.calibration_frequency,
        target_list=args.targets,
        model_loss_pair=args.method_spec,
        model_name=args.model_name,
        objective_name=args.objective_name,
        num_batches=args.num_batches,
        symmetric=symmetric,
        clean_output=bool(args.clean_output),
    )
    rfe._log(f"Completed runs : {len(ranked_results):,}")
    rfe._log(f"Failures       : {len(failures_df):,}")
    return 0 if failures_df.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
