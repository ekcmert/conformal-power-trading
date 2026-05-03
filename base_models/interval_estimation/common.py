from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
from plotly import graph_objects as go
from sklearn.metrics import mean_pinball_loss

from base_models.point_estimation.common import (
    MCP_TARGETS,
    REPO_ROOT,
    SPREAD_TARGETS,
    TEST_YEAR,
    load_final_datasets,
    output_dir_for,
    plot_feature_importances,
    reset_results_directory,
    split_train_and_tuning_validation,
    split_train_test_by_year,
    standardize_features,
    target_artifact_stem,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LOWER_QUANTILE = 0.05
DEFAULT_UPPER_QUANTILE = 0.95


def validate_quantile_pair(lower_quantile: float, upper_quantile: float) -> None:
    if not 0.0 < lower_quantile < 1.0:
        raise ValueError(f"lower_quantile must be between 0 and 1, got {lower_quantile}.")
    if not 0.0 < upper_quantile < 1.0:
        raise ValueError(f"upper_quantile must be between 0 and 1, got {upper_quantile}.")
    if lower_quantile >= upper_quantile:
        raise ValueError(
            f"lower_quantile must be smaller than upper_quantile, got {lower_quantile} >= {upper_quantile}."
        )


def interval_label(lower_quantile: float, upper_quantile: float) -> str:
    interval_pct = int(round((upper_quantile - lower_quantile) * 100))
    return f"{interval_pct}% interval"


def normalize_interval_bounds(
    lower_pred: pd.Series,
    upper_pred: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    lower_aligned, upper_aligned = lower_pred.align(upper_pred, join="inner")
    return lower_aligned.combine(upper_aligned, min), lower_aligned.combine(upper_aligned, max)


def build_prediction_frame(
    y_true: pd.Series,
    lower_pred: pd.Series,
    upper_pred: pd.Series,
) -> pd.DataFrame:
    y_true = y_true.copy()
    lower_pred = lower_pred.reindex(y_true.index)
    upper_pred = upper_pred.reindex(y_true.index)
    lower_pred, upper_pred = normalize_interval_bounds(lower_pred, upper_pred)

    prediction_frame = pd.DataFrame(index=y_true.index)
    prediction_frame["y_true"] = y_true.astype(float)
    prediction_frame["y_pred_lower"] = lower_pred.astype(float)
    prediction_frame["y_pred_upper"] = upper_pred.astype(float)
    prediction_frame["y_pred_center"] = (
        prediction_frame["y_pred_lower"] + prediction_frame["y_pred_upper"]
    ) / 2.0
    prediction_frame["interval_width"] = (
        prediction_frame["y_pred_upper"] - prediction_frame["y_pred_lower"]
    )
    prediction_frame["covered"] = (
        (prediction_frame["y_true"] >= prediction_frame["y_pred_lower"])
        & (prediction_frame["y_true"] <= prediction_frame["y_pred_upper"])
    ).astype(int)
    return prediction_frame


def compute_interval_metrics(
    y_true: pd.Series,
    lower_pred: pd.Series,
    upper_pred: pd.Series,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, float]:
    prediction_frame = build_prediction_frame(y_true, lower_pred, upper_pred)
    lower_loss = mean_pinball_loss(
        prediction_frame["y_true"],
        prediction_frame["y_pred_lower"],
        alpha=lower_quantile,
    )
    upper_loss = mean_pinball_loss(
        prediction_frame["y_true"],
        prediction_frame["y_pred_upper"],
        alpha=upper_quantile,
    )
    return {
        "mean_pinball_loss": float((lower_loss + upper_loss) / 2.0),
        "empirical_coverage": float(prediction_frame["covered"].mean()),
        "mean_interval_width": float(prediction_frame["interval_width"].mean()),
    }


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {hex_color}.")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def plot_interval_predictions(
    model_label: str,
    target_name: str,
    prediction_frame: pd.DataFrame,
    output_dir: Path,
    metrics: dict[str, float],
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> Path:
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(
        prediction_frame.index,
        prediction_frame["y_true"].values,
        label="True",
        linewidth=1.6,
        color="#1d3557",
    )
    ax.plot(
        prediction_frame.index,
        prediction_frame["y_pred_center"].values,
        label="Interval midpoint",
        linewidth=1.1,
        linestyle="--",
        color="#e76f51",
    )
    ax.fill_between(
        prediction_frame.index,
        prediction_frame["y_pred_lower"].values,
        prediction_frame["y_pred_upper"].values,
        color="#2a9d8f",
        alpha=0.25,
        label=interval_label(lower_quantile, upper_quantile),
    )
    ax.set_title(
        f"{target_name} | {model_label} | {TEST_YEAR}\n"
        f"Pinball={metrics['mean_pinball_loss']:.3f} "
        f"Coverage={metrics['empirical_coverage']:.3f} "
        f"Width={metrics['mean_interval_width']:.3f}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(target_name)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_path = output_dir / f"{target_artifact_stem(target_name)}_predictions.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_combined_plotly_predictions(
    targets: list[str] | tuple[str, ...],
    model_label: str,
    output_dir: Path,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> Path:
    color_map = {
        "DAID": "#1d3557",
        "DAID3": "#2a9d8f",
        "DAID1": "#e76f51",
        "DAIMB": "#8d5fd3",
        "DE Price Spot EUR/MWh EPEX H Actual": "#1d3557",
    }

    fig = go.Figure()
    for target in targets:
        prediction_frame = pd.read_csv(
            output_dir / f"{target_artifact_stem(target)}_predictions.csv",
            parse_dates=["timestamp"],
        )
        base_color = color_map.get(target, "#264653")
        band_color = _hex_to_rgba(base_color, 0.15)

        fig.add_trace(
            go.Scatter(
                x=prediction_frame["timestamp"],
                y=prediction_frame["y_true"],
                mode="lines",
                name=f"{target} true",
                line={"color": base_color, "width": 2},
                legendgroup=target,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=prediction_frame["timestamp"],
                y=prediction_frame["y_pred_upper"],
                mode="lines",
                name=f"{target} upper",
                line={"color": base_color, "width": 0.7},
                legendgroup=target,
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=prediction_frame["timestamp"],
                y=prediction_frame["y_pred_lower"],
                mode="lines",
                name=f"{target} {interval_label(lower_quantile, upper_quantile)}",
                line={"color": base_color, "width": 0.7},
                fill="tonexty",
                fillcolor=band_color,
                legendgroup=target,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=prediction_frame["timestamp"],
                y=prediction_frame["y_pred_center"],
                mode="lines",
                name=f"{target} midpoint",
                line={"color": base_color, "width": 1.2, "dash": "dash"},
                legendgroup=target,
            )
        )

    fig.update_layout(
        title=f"{model_label} | {TEST_YEAR} Interval Predictions vs True Values",
        xaxis_title="Date",
        yaxis_title="Target value",
        hovermode="x unified",
        template="plotly_white",
        legend_title="Series",
    )

    output_path = output_dir / "all_targets_predictions.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path
