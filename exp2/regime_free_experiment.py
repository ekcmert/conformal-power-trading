from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd
from plotly import graph_objects as go
from tqdm.auto import tqdm

from base_models.interval_estimation.common import (
    DEFAULT_LOWER_QUANTILE,
    DEFAULT_UPPER_QUANTILE,
    build_prediction_frame,
    compute_interval_metrics,
)
from base_models.point_estimation.common import REPO_ROOT, output_dir_for, target_artifact_stem
from conformal_prediction import ConformalPrediction, ConformalizedQuantileRegression


matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
RESULTS_ROOT = REPO_ROOT / "results" / "regime_free"


@dataclass(frozen=True)
class TargetSpec:
    code: str
    slug: str
    y_column: str
    point_residual_name: str
    interval_residual_low_name: str
    interval_residual_up_name: str


@dataclass(frozen=True)
class ModelArtifacts:
    estimation_kind: str
    model_name: str
    objective_name: str
    display_name: str
    prediction_dir: Path
    residual_dir: Path
    prediction_path: Path
    metadata: dict[str, object]


@dataclass(frozen=True)
class RollingWindow:
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    prediction_start: pd.Timestamp
    prediction_end: pd.Timestamp


TARGET_SPECS = {
    "DA": TargetSpec(
        code="DA",
        slug="da",
        y_column="DE Price Spot EUR/MWh EPEX H Actual",
        point_residual_name="da_res",
        interval_residual_low_name="da_res_low",
        interval_residual_up_name="da_res_up",
    ),
    "ID": TargetSpec(
        code="ID",
        slug="id",
        y_column="DE Price Intraday VWAP EUR/MWh EPEX H Actual",
        point_residual_name="id_res",
        interval_residual_low_name="id_res_low",
        interval_residual_up_name="id_res_up",
    ),
    "ID3": TargetSpec(
        code="ID3",
        slug="id3",
        y_column="DE Price Intraday VWAP ID3 EUR/MWh EPEX H Actual",
        point_residual_name="id3_res",
        interval_residual_low_name="id3_res_low",
        interval_residual_up_name="id3_res_up",
    ),
    "ID1": TargetSpec(
        code="ID1",
        slug="id1",
        y_column="DE Price Intraday VWAP ID1 EUR/MWh EPEX H Actual",
        point_residual_name="id1_res",
        interval_residual_low_name="id1_res_low",
        interval_residual_up_name="id1_res_up",
    ),
    "IMB": TargetSpec(
        code="IMB",
        slug="imb",
        y_column="DE Volume Imbalance Net MWh 15min Actual",
        point_residual_name="imb_res",
        interval_residual_low_name="imb_res_low",
        interval_residual_up_name="imb_res_up",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run weekly regime-free conformal recalibration on saved predictions and residuals.",
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
        "--clean-output",
        action="store_true",
        help="Remove existing results/regime_free outputs before running.",
    )
    return parser.parse_args()


def _log(message: str) -> None:
    tqdm.write(message)


def _normalize_objective_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value or "")


def _resolve_targets(target_codes: list[str] | tuple[str, ...]) -> list[TargetSpec]:
    resolved: list[TargetSpec] = []
    for target_code in target_codes:
        normalized = str(target_code).strip().upper()
        if normalized not in TARGET_SPECS:
            raise KeyError(
                f"Unsupported target code {target_code!r}. "
                f"Expected one of {list(TARGET_SPECS)}."
            )
        resolved.append(TARGET_SPECS[normalized])
    return resolved


def _load_y_frame(y_path: Path, target_specs: list[TargetSpec]) -> pd.DataFrame:
    y_frame = pd.read_csv(y_path, parse_dates=[DATE_COLUMN])
    if DATE_COLUMN not in y_frame.columns:
        raise ValueError(f"Expected a {DATE_COLUMN!r} column in {y_path}.")

    missing_columns = [
        target_spec.y_column
        for target_spec in target_specs
        if target_spec.y_column not in y_frame.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing target columns in {y_path}: {missing_columns}")

    return y_frame.set_index(DATE_COLUMN).sort_index()


def _load_metadata(metadata_path: Path) -> dict[str, object]:
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _combined_prediction_file(model_dir: Path) -> Path:
    prediction_files = sorted(model_dir.glob("*_predictions.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No combined prediction file found in {model_dir}.")
    if len(prediction_files) > 1:
        raise ValueError(
            f"Expected a single combined prediction file in {model_dir}, "
            f"found {len(prediction_files)}: {[path.name for path in prediction_files]}"
        )
    return prediction_files[0]


def _discover_model_artifacts(
    *,
    predictions_root: Path,
    residuals_root: Path,
) -> list[ModelArtifacts]:
    artifacts: list[ModelArtifacts] = []
    for estimation_folder, estimation_kind in (
        ("point_estimation", "point"),
        ("interval_estimation", "interval"),
    ):
        base_dir = predictions_root / estimation_folder
        if not base_dir.exists():
            continue

        model_dirs = sorted({path.parent for path in base_dir.rglob("*_predictions.csv")})
        for model_dir in model_dirs:
            relative_dir = model_dir.relative_to(predictions_root)
            metadata = _load_metadata(model_dir / "run_metadata.json")
            model_name = str(metadata.get("model_name") or relative_dir.parts[1])
            objective_name = _normalize_objective_name(
                metadata.get("objective_name") or (relative_dir.parts[2] if len(relative_dir.parts) > 2 else "")
            )
            display_name = str(
                metadata.get("display_name")
                or f"{model_name} | {objective_name or 'default'}"
            )
            artifacts.append(
                ModelArtifacts(
                    estimation_kind=estimation_kind,
                    model_name=model_name,
                    objective_name=objective_name,
                    display_name=display_name,
                    prediction_dir=model_dir,
                    residual_dir=residuals_root / relative_dir,
                    prediction_path=_combined_prediction_file(model_dir),
                    metadata=metadata,
                )
            )

    return sorted(
        artifacts,
        key=lambda artifact: (
            artifact.estimation_kind,
            artifact.model_name,
            artifact.objective_name,
        ),
    )


def _load_prediction_frame(prediction_path: Path) -> pd.DataFrame:
    prediction_frame = pd.read_csv(prediction_path, parse_dates=[DATE_COLUMN])
    if DATE_COLUMN not in prediction_frame.columns:
        raise ValueError(f"Expected a {DATE_COLUMN!r} column in {prediction_path}.")
    return prediction_frame.set_index(DATE_COLUMN).sort_index()


def _load_residual_series(residual_path: Path, column_name: str) -> pd.Series:
    residual_frame = pd.read_csv(residual_path, parse_dates=[DATE_COLUMN])
    if DATE_COLUMN not in residual_frame.columns:
        raise ValueError(f"Expected a {DATE_COLUMN!r} column in {residual_path}.")
    if column_name not in residual_frame.columns:
        raise KeyError(f"Missing residual column {column_name!r} in {residual_path}.")
    return residual_frame.set_index(DATE_COLUMN)[column_name].astype(float).sort_index()


def _build_rolling_windows(
    index: pd.DatetimeIndex,
    *,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
) -> list[RollingWindow]:
    if index.empty:
        return []

    calibration_delta = pd.Timedelta(weeks=calibration_range_weeks)
    frequency_delta = pd.Timedelta(weeks=calibration_frequency_weeks)
    prediction_start = index.min() + calibration_delta
    prediction_last_timestamp = index.max()

    windows: list[RollingWindow] = []
    while prediction_start <= prediction_last_timestamp:
        prediction_end = prediction_start + frequency_delta
        windows.append(
            RollingWindow(
                calibration_start=prediction_start - calibration_delta,
                calibration_end=prediction_start,
                prediction_start=prediction_start,
                prediction_end=prediction_end,
            )
        )
        prediction_start = prediction_end

    return windows


def _window_slice(
    series_or_frame: pd.Series | pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series | pd.DataFrame:
    return series_or_frame.loc[(series_or_frame.index >= start) & (series_or_frame.index < end)].copy()


def _compute_interval_metrics_for_frame(prediction_frame: pd.DataFrame) -> dict[str, float]:
    return compute_interval_metrics(
        prediction_frame["y_true"],
        prediction_frame["y_pred_lower"],
        prediction_frame["y_pred_upper"],
        lower_quantile=LOWER_QUANTILE,
        upper_quantile=UPPER_QUANTILE,
    )


def _build_result_prediction_frame(
    *,
    y_true: pd.Series,
    lower_pred: pd.Series,
    upper_pred: pd.Series,
    base_lower_pred: pd.Series,
    base_upper_pred: pd.Series,
    margin_lower: pd.Series,
    margin_upper: pd.Series,
    window_index: int,
    window: RollingWindow,
    calibration_rows: int,
) -> pd.DataFrame:
    prediction_frame = build_prediction_frame(y_true, lower_pred, upper_pred).reset_index()
    prediction_frame.columns = [DATE_COLUMN, *prediction_frame.columns.tolist()[1:]]
    prediction_frame["base_y_pred_lower"] = base_lower_pred.to_numpy()
    prediction_frame["base_y_pred_upper"] = base_upper_pred.to_numpy()
    prediction_frame["base_y_pred_center"] = (
        prediction_frame["base_y_pred_lower"] + prediction_frame["base_y_pred_upper"]
    ) / 2.0
    prediction_frame["initial_y_pred_lower"] = prediction_frame["base_y_pred_lower"]
    prediction_frame["initial_y_pred_upper"] = prediction_frame["base_y_pred_upper"]
    prediction_frame["initial_y_pred_center"] = prediction_frame["base_y_pred_center"]
    prediction_frame["initial_prediction"] = prediction_frame["base_y_pred_center"]
    prediction_frame["margin_lower"] = margin_lower.to_numpy()
    prediction_frame["margin_upper"] = margin_upper.to_numpy()
    prediction_frame["window_index"] = window_index
    prediction_frame["calibration_start"] = window.calibration_start
    prediction_frame["calibration_end_exclusive"] = window.calibration_end
    prediction_frame["prediction_start"] = window.prediction_start
    prediction_frame["prediction_end_exclusive"] = window.prediction_end
    prediction_frame["calibration_rows"] = calibration_rows
    return prediction_frame


def _point_method_output_name(model_name: str, *, symmetric: bool) -> str:
    if symmetric:
        return f"cp_symmetric_{model_name}"
    return f"cp_asymmetric_{model_name}"


def _interval_method_output_name(model_name: str) -> str:
    return f"cqr_{model_name}"


def _method_display_name(base_display_name: str, method_name: str) -> str:
    if method_name == "cp_symmetric":
        return f"CP symmetric | {base_display_name}"
    if method_name == "cp_asymmetric":
        return f"CP asymmetric | {base_display_name}"
    return f"CQR | {base_display_name}"


def _save_prediction_plot_html(
    *,
    output_dir: Path,
    target_spec: TargetSpec,
    display_name: str,
    prediction_frame: pd.DataFrame,
    metrics: dict[str, float],
    estimation_kind: str,
) -> Path:
    target_stem = target_artifact_stem(target_spec.y_column)
    html_path = output_dir / f"{target_stem}_predictions.html"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prediction_frame[DATE_COLUMN],
            y=prediction_frame["y_true"],
            mode="lines",
            name="True",
            line={"color": "#1d3557", "width": 2},
        )
    )
    if estimation_kind == "point":
        fig.add_trace(
            go.Scatter(
                x=prediction_frame[DATE_COLUMN],
                y=prediction_frame["initial_prediction"],
                mode="lines",
                name="Initial prediction",
                line={"color": "#6c757d", "width": 1.1, "dash": "dot"},
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=prediction_frame[DATE_COLUMN],
                y=prediction_frame["initial_y_pred_upper"],
                mode="lines",
                name="Initial upper",
                line={"color": "#6c757d", "width": 1.0, "dash": "dot"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=prediction_frame[DATE_COLUMN],
                y=prediction_frame["initial_y_pred_lower"],
                mode="lines",
                name="Initial lower",
                line={"color": "#6c757d", "width": 1.0, "dash": "dot"},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=prediction_frame[DATE_COLUMN],
            y=prediction_frame["y_pred_upper"],
            mode="lines",
            name="Upper",
            line={"color": "#2a9d8f", "width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prediction_frame[DATE_COLUMN],
            y=prediction_frame["y_pred_lower"],
            mode="lines",
            name="Lower",
            line={"color": "#2a9d8f", "width": 1},
            fill="tonexty",
            fillcolor="rgba(42, 157, 143, 0.18)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prediction_frame[DATE_COLUMN],
            y=prediction_frame["y_pred_center"],
            mode="lines",
            name="Interval midpoint",
            line={"color": "#e76f51", "width": 1.2, "dash": "dash"},
        )
    )
    fig.update_layout(
        title=(
            f"{target_spec.code} | {display_name}<br>"
            f"Pinball={metrics['mean_pinball_loss']:.3f} "
            f"Coverage={metrics['empirical_coverage']:.3f} "
            f"Width={metrics['mean_interval_width']:.3f}"
        ),
        xaxis_title="Date",
        yaxis_title=target_spec.y_column,
        hovermode="x unified",
        template="plotly_white",
        legend_title="Series",
    )
    fig.write_html(html_path, include_plotlyjs="cdn")
    return html_path


def _save_prediction_plot_png(
    *,
    output_dir: Path,
    target_spec: TargetSpec,
    display_name: str,
    prediction_frame: pd.DataFrame,
    metrics: dict[str, float],
    estimation_kind: str,
) -> Path:
    target_stem = target_artifact_stem(target_spec.y_column)
    png_path = output_dir / f"{target_stem}_predictions.png"

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(
        prediction_frame[DATE_COLUMN],
        prediction_frame["y_true"],
        label="True",
        linewidth=1.6,
        color="#1d3557",
    )
    if estimation_kind == "point":
        ax.plot(
            prediction_frame[DATE_COLUMN],
            prediction_frame["initial_prediction"],
            label="Initial prediction",
            linewidth=1.0,
            linestyle=":",
            color="#6c757d",
        )
    else:
        ax.plot(
            prediction_frame[DATE_COLUMN],
            prediction_frame["initial_y_pred_upper"],
            label="Initial upper",
            linewidth=0.9,
            linestyle=":",
            color="#6c757d",
        )
        ax.plot(
            prediction_frame[DATE_COLUMN],
            prediction_frame["initial_y_pred_lower"],
            label="Initial lower",
            linewidth=0.9,
            linestyle=":",
            color="#6c757d",
        )
    ax.plot(
        prediction_frame[DATE_COLUMN],
        prediction_frame["y_pred_center"],
        label="Interval midpoint",
        linewidth=1.1,
        linestyle="--",
        color="#e76f51",
    )
    ax.fill_between(
        prediction_frame[DATE_COLUMN],
        prediction_frame["y_pred_lower"],
        prediction_frame["y_pred_upper"],
        color="#2a9d8f",
        alpha=0.25,
        label=f"{int(round((UPPER_QUANTILE - LOWER_QUANTILE) * 100))}% interval",
    )
    ax.set_title(
        f"{target_spec.code} | {display_name}\n"
        f"Pinball={metrics['mean_pinball_loss']:.3f} "
        f"Coverage={metrics['empirical_coverage']:.3f} "
        f"Width={metrics['mean_interval_width']:.3f}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(target_spec.y_column)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return png_path


def _rank_results(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df.copy()

    ranked_groups: list[pd.DataFrame] = []
    nominal_coverage = UPPER_QUANTILE - LOWER_QUANTILE
    for _, target_group in results_df.groupby("target_code", sort=False):
        ranked = target_group.copy()
        ranked["pinball_rank"] = ranked["mean_pinball_loss"].rank(method="dense", ascending=True)
        ranked["mean_interval_width_rank"] = ranked["mean_interval_width"].rank(method="dense", ascending=True)
        ranked["empirical_coverage_rank"] = ranked["empirical_coverage"].rank(method="dense", ascending=False)
        ranked["coverage_gap_to_nominal"] = (ranked["empirical_coverage"] - nominal_coverage).abs()
        ranked["average_rank"] = ranked[
            ["pinball_rank", "mean_interval_width_rank", "empirical_coverage_rank"]
        ].mean(axis=1)
        ranked = ranked.sort_values(
            by=["average_rank", "mean_pinball_loss", "mean_interval_width", "empirical_coverage"],
            ascending=[True, True, True, False],
            kind="mergesort",
        ).reset_index(drop=True)
        ranked["selection_order"] = range(1, len(ranked) + 1)
        ranked_groups.append(ranked)

    return pd.concat(ranked_groups, ignore_index=True)


def _write_run_metadata(
    *,
    output_dir: Path,
    target_spec: TargetSpec,
    artifacts: ModelArtifacts,
    method_name: str,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
    window_count: int,
    residual_paths: list[Path],
) -> None:
    metadata = {
        "pipeline": "regime_free",
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
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_point_method(
    *,
    artifacts: ModelArtifacts,
    target_spec: TargetSpec,
    prediction_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    residual_series: pd.Series,
    symmetric: bool,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if "y_pred" not in prediction_frame.columns:
        raise KeyError(f"Missing y_pred column in {artifacts.prediction_path}.")

    y_series = y_frame[target_spec.y_column].reindex(prediction_frame.index)
    if y_series.isna().any():
        raise KeyError(f"Missing y values for target {target_spec.y_column}.")

    windows = _build_rolling_windows(
        prediction_frame.index,
        calibration_range_weeks=calibration_range_weeks,
        calibration_frequency_weeks=calibration_frequency_weeks,
    )
    if not windows:
        raise RuntimeError("No rolling windows were created for point conformal recalibration.")

    method_name = "cp_symmetric" if symmetric else "cp_asymmetric"
    combined_predictions: list[pd.DataFrame] = []
    weekly_metrics: list[dict[str, object]] = []

    with tqdm(
        total=len(windows),
        desc=f"Weeks | {target_spec.code} | {method_name} | {artifacts.model_name}",
        unit="week",
        dynamic_ncols=True,
        leave=False,
    ) as week_progress:
        for window_index, window in enumerate(windows):
            calibration_residuals = _window_slice(
                residual_series,
                start=window.calibration_start,
                end=window.calibration_end,
            )
            prediction_slice = _window_slice(
                prediction_frame,
                start=window.prediction_start,
                end=window.prediction_end,
            )
            if calibration_residuals.empty or prediction_slice.empty:
                week_progress.update(1)
                continue

            y_true = y_series.reindex(prediction_slice.index)
            base_prediction = prediction_slice["y_pred"].astype(float)

            if symmetric:
                conformal = ConformalPrediction(
                    residuals_upper=calibration_residuals.to_numpy(),
                    residuals_lower=calibration_residuals.to_numpy(),
                    predictions_upper=base_prediction.to_numpy(),
                    predictions_lower=base_prediction.to_numpy(),
                    alpha=ALPHA,
                    symmetric=True,
                )
            else:
                conformal = ConformalPrediction(
                    residuals_upper=calibration_residuals[calibration_residuals >= 0.0].to_numpy(),
                    residuals_lower=calibration_residuals[calibration_residuals < 0.0].to_numpy(),
                    predictions_upper=base_prediction.to_numpy(),
                    predictions_lower=base_prediction.to_numpy(),
                    alpha=ALPHA,
                    symmetric=False,
                )

            result = conformal.get_result()
            lower_pred = pd.Series(result.calibrated_lower, index=prediction_slice.index)
            upper_pred = pd.Series(result.calibrated_upper, index=prediction_slice.index)
            margins_lower = pd.Series(result.margin_lower, index=prediction_slice.index)
            margins_upper = pd.Series(result.margin_upper, index=prediction_slice.index)

            calibrated_frame = _build_result_prediction_frame(
                y_true=y_true.astype(float),
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
            combined_predictions.append(calibrated_frame)

            metrics = _compute_interval_metrics_for_frame(calibrated_frame)
            weekly_metrics.append(
                {
                    "window_index": window_index,
                    "calibration_start": window.calibration_start,
                    "calibration_end_exclusive": window.calibration_end,
                    "prediction_start": window.prediction_start,
                    "prediction_end_exclusive": window.prediction_end,
                    "calibration_rows": int(len(calibration_residuals)),
                    "prediction_rows": int(len(prediction_slice)),
                    **metrics,
                }
            )
            week_progress.update(1)

    if not combined_predictions:
        raise RuntimeError("No point conformal windows produced any prediction rows.")

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
    overall_metrics = _compute_interval_metrics_for_frame(all_predictions)
    return all_predictions, weekly_metrics_df, overall_metrics


def _run_interval_method(
    *,
    artifacts: ModelArtifacts,
    target_spec: TargetSpec,
    prediction_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    residual_low: pd.Series,
    residual_up: pd.Series,
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

    windows = _build_rolling_windows(
        prediction_frame.index,
        calibration_range_weeks=calibration_range_weeks,
        calibration_frequency_weeks=calibration_frequency_weeks,
    )
    if not windows:
        raise RuntimeError("No rolling windows were created for CQR recalibration.")

    combined_predictions: list[pd.DataFrame] = []
    weekly_metrics: list[dict[str, object]] = []

    with tqdm(
        total=len(windows),
        desc=f"Weeks | {target_spec.code} | cqr | {artifacts.model_name}",
        unit="week",
        dynamic_ncols=True,
        leave=False,
    ) as week_progress:
        for window_index, window in enumerate(windows):
            calibration_low = _window_slice(
                residual_low,
                start=window.calibration_start,
                end=window.calibration_end,
            )
            calibration_up = _window_slice(
                residual_up,
                start=window.calibration_start,
                end=window.calibration_end,
            )
            prediction_slice = _window_slice(
                prediction_frame,
                start=window.prediction_start,
                end=window.prediction_end,
            )
            if calibration_low.empty or calibration_up.empty or prediction_slice.empty:
                week_progress.update(1)
                continue

            y_true = y_series.reindex(prediction_slice.index)
            base_lower = prediction_slice["y_pred_lower"].astype(float)
            base_upper = prediction_slice["y_pred_upper"].astype(float)

            conformal = ConformalizedQuantileRegression(
                residuals_lower=calibration_low.to_numpy(),
                residuals_upper=calibration_up.to_numpy(),
                predictions_lower=base_lower.to_numpy(),
                predictions_upper=base_upper.to_numpy(),
                alpha=ALPHA,
            )
            result = conformal.get_result()

            lower_pred = pd.Series(result.calibrated_lower, index=prediction_slice.index)
            upper_pred = pd.Series(result.calibrated_upper, index=prediction_slice.index)
            margins_lower = pd.Series(result.margin_lower, index=prediction_slice.index)
            margins_upper = pd.Series(result.margin_upper, index=prediction_slice.index)

            calibrated_frame = _build_result_prediction_frame(
                y_true=y_true.astype(float),
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
            combined_predictions.append(calibrated_frame)

            metrics = _compute_interval_metrics_for_frame(calibrated_frame)
            weekly_metrics.append(
                {
                    "window_index": window_index,
                    "calibration_start": window.calibration_start,
                    "calibration_end_exclusive": window.calibration_end,
                    "prediction_start": window.prediction_start,
                    "prediction_end_exclusive": window.prediction_end,
                    "calibration_rows": int(len(calibration_low)),
                    "prediction_rows": int(len(prediction_slice)),
                    **metrics,
                }
            )
            week_progress.update(1)

    if not combined_predictions:
        raise RuntimeError("No CQR windows produced any prediction rows.")

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
    overall_metrics = _compute_interval_metrics_for_frame(all_predictions)
    return all_predictions, weekly_metrics_df, overall_metrics


def _run_single_method(
    *,
    artifacts: ModelArtifacts,
    target_spec: TargetSpec,
    prediction_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    calibration_range_weeks: int,
    calibration_frequency_weeks: int,
    results_root: Path,
) -> list[dict[str, object]]:
    result_rows: list[dict[str, object]] = []

    if artifacts.estimation_kind == "point":
        residual_path = artifacts.residual_dir / f"{target_spec.point_residual_name}.csv"
        residual_series = _load_residual_series(residual_path, target_spec.point_residual_name)
        point_variants = [(True, "cp_symmetric"), (False, "cp_asymmetric")]

        for symmetric, method_name in point_variants:
            output_dir = output_dir_for(
                results_root / target_spec.slug,
                _point_method_output_name(artifacts.model_name, symmetric=symmetric),
                artifacts.objective_name,
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            predictions_df, weekly_metrics_df, overall_metrics = _run_point_method(
                artifacts=artifacts,
                target_spec=target_spec,
                prediction_frame=prediction_frame,
                y_frame=y_frame,
                residual_series=residual_series,
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
                        **overall_metrics,
                    }
                ]
            ).to_csv(overall_metrics_path, index=False)
            plot_html_path = _save_prediction_plot_html(
                output_dir=output_dir,
                target_spec=target_spec,
                display_name=_method_display_name(artifacts.display_name, method_name),
                prediction_frame=predictions_df,
                metrics=overall_metrics,
                estimation_kind=artifacts.estimation_kind,
            )
            plot_png_path = _save_prediction_plot_png(
                output_dir=output_dir,
                target_spec=target_spec,
                display_name=_method_display_name(artifacts.display_name, method_name),
                prediction_frame=predictions_df,
                metrics=overall_metrics,
                estimation_kind=artifacts.estimation_kind,
            )
            _write_run_metadata(
                output_dir=output_dir,
                target_spec=target_spec,
                artifacts=artifacts,
                method_name=method_name,
                calibration_range_weeks=calibration_range_weeks,
                calibration_frequency_weeks=calibration_frequency_weeks,
                window_count=len(weekly_metrics_df),
                residual_paths=[residual_path],
            )

            result_rows.append(
                {
                    "target_code": target_spec.code,
                    "target_slug": target_spec.slug,
                    "target_name": target_spec.y_column,
                    "method_name": method_name,
                    "estimation_kind": artifacts.estimation_kind,
                    "model_name": artifacts.model_name,
                    "objective_name": artifacts.objective_name,
                    "display_name": _method_display_name(artifacts.display_name, method_name),
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

    residual_low_path = artifacts.residual_dir / f"{target_spec.interval_residual_low_name}.csv"
    residual_up_path = artifacts.residual_dir / f"{target_spec.interval_residual_up_name}.csv"
    residual_low = _load_residual_series(residual_low_path, target_spec.interval_residual_low_name)
    residual_up = _load_residual_series(residual_up_path, target_spec.interval_residual_up_name)

    output_dir = output_dir_for(
        results_root / target_spec.slug,
        _interval_method_output_name(artifacts.model_name),
        artifacts.objective_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df, weekly_metrics_df, overall_metrics = _run_interval_method(
        artifacts=artifacts,
        target_spec=target_spec,
        prediction_frame=prediction_frame,
        y_frame=y_frame,
        residual_low=residual_low,
        residual_up=residual_up,
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
                **overall_metrics,
            }
        ]
    ).to_csv(overall_metrics_path, index=False)
    plot_html_path = _save_prediction_plot_html(
        output_dir=output_dir,
        target_spec=target_spec,
        display_name=_method_display_name(artifacts.display_name, "cqr"),
        prediction_frame=predictions_df,
        metrics=overall_metrics,
        estimation_kind=artifacts.estimation_kind,
    )
    plot_png_path = _save_prediction_plot_png(
        output_dir=output_dir,
        target_spec=target_spec,
        display_name=_method_display_name(artifacts.display_name, "cqr"),
        prediction_frame=predictions_df,
        metrics=overall_metrics,
        estimation_kind=artifacts.estimation_kind,
    )
    _write_run_metadata(
        output_dir=output_dir,
        target_spec=target_spec,
        artifacts=artifacts,
        method_name="cqr",
        calibration_range_weeks=calibration_range_weeks,
        calibration_frequency_weeks=calibration_frequency_weeks,
        window_count=len(weekly_metrics_df),
        residual_paths=[residual_low_path, residual_up_path],
    )

    result_rows.append(
        {
            "target_code": target_spec.code,
            "target_slug": target_spec.slug,
            "target_name": target_spec.y_column,
            "method_name": "cqr",
            "estimation_kind": artifacts.estimation_kind,
            "model_name": artifacts.model_name,
            "objective_name": artifacts.objective_name,
            "display_name": _method_display_name(artifacts.display_name, "cqr"),
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


def _write_summary_results(results_root: Path, result_rows: list[dict[str, object]]) -> pd.DataFrame:
    ranked_results = _rank_results(pd.DataFrame(result_rows))
    results_root.mkdir(parents=True, exist_ok=True)
    ranked_results.to_csv(results_root / "all_method_results.csv", index=False)

    for target_slug, target_df in ranked_results.groupby("target_slug", sort=False):
        target_dir = results_root / str(target_slug)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_df.to_csv(target_dir / "all_method_results.csv", index=False)

    return ranked_results


def _write_failures(results_root: Path, failures: list[dict[str, object]]) -> None:
    failures_path = results_root / "failures.csv"
    if failures:
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        return
    failures_path.unlink(missing_ok=True)


def run_regime_free_experiment(
    *,
    calibration_range_weeks: int = CALIBRATION_RANGE,
    calibration_frequency_weeks: int = CALIBRATION_FREQUENCY,
    target_list: list[str] | tuple[str, ...] = TARGET_LIST,
    predictions_root: Path = PREDICTIONS_ROOT,
    residuals_root: Path = RESIDUALS_ROOT,
    y_path: Path = Y_PATH,
    results_root: Path = RESULTS_ROOT,
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

    if clean_output and results_root.exists():
        shutil.rmtree(results_root)

    target_specs = _resolve_targets(list(target_list))
    y_frame = _load_y_frame(y_path, target_specs)
    artifacts = _discover_model_artifacts(
        predictions_root=predictions_root,
        residuals_root=residuals_root,
    )
    if not artifacts:
        raise FileNotFoundError("No combined prediction artifacts were found to calibrate.")

    result_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    _log("")
    _log("=" * 80)
    _log("Regime-Free Conformal Recalibration")
    _log("=" * 80)
    _log(f"Targets                 : {[target_spec.code for target_spec in target_specs]}")
    _log(f"Calibration range       : {calibration_range_weeks} weeks")
    _log(f"Calibration frequency   : {calibration_frequency_weeks} week(s)")
    _log(f"Nominal coverage        : {UPPER_QUANTILE - LOWER_QUANTILE:.2%}")
    _log(f"Prediction roots        : {predictions_root}")
    _log(f"Residual roots          : {residuals_root}")
    _log(f"Results root            : {results_root}")
    _log(f"Model-loss directories  : {len(artifacts)}")
    _log("=" * 80)

    with tqdm(
        total=len(target_specs),
        desc="Targets",
        unit="target",
        dynamic_ncols=True,
    ) as target_progress:
        for target_spec in target_specs:
            target_progress.set_postfix_str(target_spec.code)
            prediction_cache: dict[Path, pd.DataFrame] = {}
            with tqdm(
                total=len(artifacts),
                desc=f"Methods | {target_spec.code}",
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
                            prediction_frame = _load_prediction_frame(artifacts_item.prediction_path)
                            prediction_cache[artifacts_item.prediction_path] = prediction_frame

                        result_rows.extend(
                            _run_single_method(
                                artifacts=artifacts_item,
                                target_spec=target_spec,
                                prediction_frame=prediction_frame,
                                y_frame=y_frame,
                                calibration_range_weeks=calibration_range_weeks,
                                calibration_frequency_weeks=calibration_frequency_weeks,
                                results_root=results_root,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        failures.append(
                            {
                                "target_code": target_spec.code,
                                "estimation_kind": artifacts_item.estimation_kind,
                                "model_name": artifacts_item.model_name,
                                "objective_name": artifacts_item.objective_name,
                                "display_name": artifacts_item.display_name,
                                "error": repr(exc),
                            }
                        )
                        _log(
                            f"FAILED {target_spec.code} | {artifacts_item.display_name} "
                            f"({artifacts_item.estimation_kind}): {exc}"
                        )
                    finally:
                        model_progress.update(1)
            target_progress.update(1)

    ranked_results = _write_summary_results(results_root, result_rows)
    failures_df = pd.DataFrame(failures)
    _write_failures(results_root, failures)
    return ranked_results, failures_df


def main() -> int:
    args = _parse_args()
    ranked_results, failures_df = run_regime_free_experiment(
        calibration_range_weeks=args.calibration_range,
        calibration_frequency_weeks=args.calibration_frequency,
        target_list=args.targets,
        clean_output=bool(args.clean_output),
    )

    _log("")
    _log("=" * 80)
    _log("Regime-Free Summary")
    _log("=" * 80)
    _log(f"Completed runs : {len(ranked_results):,}")
    _log(f"Failures       : {len(failures_df):,}")
    if not ranked_results.empty:
        printable_columns = [
            "target_code",
            "display_name",
            "mean_pinball_loss",
            "empirical_coverage",
            "mean_interval_width",
            "average_rank",
            "selection_order",
        ]
        _log(ranked_results[printable_columns].to_string(index=False))
    _log("=" * 80)
    return 0 if failures_df.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
