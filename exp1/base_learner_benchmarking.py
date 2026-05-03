from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Patch


RANK_INTERVAL = 8
RANK_POINT = 8

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "base_learner_benchmarking"

POINT_RESULTS_DIR = RESULTS_ROOT / "point_estimation" / "mcp_models"
POINT_TUNING_RESULTS_DIR = RESULTS_ROOT / "tuning" / "point_estimation" / "mcp_models"
INTERVAL_RESULTS_DIR = RESULTS_ROOT / "interval_estimation" / "mcp_models"
INTERVAL_TUNING_RESULTS_DIR = RESULTS_ROOT / "tuning" / "interval_estimation" / "mcp_models"

SOURCE_COLORS = {
    "standard": "#1d3557",
    "tuning": "#e76f51",
}


@dataclass(frozen=True)
class ResultSource:
    label: str
    results_dir: Path
    tuning_active: bool


def _normalize_objective_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value or "")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _result_key(row: pd.Series) -> tuple[str, str, str]:
    return (
        str(row["model_name"]),
        _normalize_objective_name(row["objective_name"]),
        str(row["target"]),
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
    results_df["source_tuning_active"] = source.tuning_active
    results_df["source_preference"] = int(source.tuning_active)
    return results_df


def _rank_metric(series: pd.Series, *, ascending: bool) -> pd.Series:
    return series.rank(method="dense", ascending=ascending)


def _select_best_results(estimation_kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [_load_results_for_source(source) for source in _results_sources(estimation_kind)]
    available_frames = [frame for frame in frames if not frame.empty]
    if not available_frames:
        raise FileNotFoundError(f"No result CSVs were found for {estimation_kind} estimation.")

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


def _print_selected_rows(estimation_kind: str, selected_df: pd.DataFrame) -> None:
    print("")
    print("=" * 80)
    if estimation_kind == "point":
        print("Selected Point Models")
        printable_columns = [
            "selection_order",
            "display_name",
            "source_label",
            "mae",
            "rmse",
            "r2",
            "mae_rank",
            "rmse_rank",
            "r2_rank",
            "average_rank",
        ]
    else:
        print("Selected Interval Models")
        printable_columns = [
            "selection_order",
            "display_name",
            "source_label",
            "mean_pinball_loss",
            "empirical_coverage",
            "mean_interval_width",
            "pinball_rank",
            "empirical_coverage_rank",
            "mean_interval_width_rank",
            "average_rank",
        ]
    print("=" * 80)
    print(selected_df[printable_columns].to_string(index=False))
    print("=" * 80)


def _write_selection_artifacts(
    estimation_kind: str,
    ranking_df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> None:
    if estimation_kind == "point":
        preferred_selected_columns = [
            "selection_order",
            "model_name",
            "objective_name",
            "display_name",
            "target",
            "source_label",
            "source_tuning_active",
            "mae",
            "rmse",
            "r2",
            "mae_rank",
            "rmse_rank",
            "r2_rank",
        ]
    else:
        preferred_selected_columns = [
            "selection_order",
            "model_name",
            "objective_name",
            "display_name",
            "target",
            "source_label",
            "source_tuning_active",
            "mean_pinball_loss",
            "empirical_coverage",
            "mean_interval_width",
            "pinball_rank",
            "empirical_coverage_rank",
            "mean_interval_width_rank",
        ]

    ordered_selected_columns = [
        column_name for column_name in preferred_selected_columns if column_name in selected_df.columns
    ]
    ordered_selected_columns.extend(
        column_name
        for column_name in selected_df.columns
        if column_name not in ordered_selected_columns and column_name != "average_rank"
    )
    if "average_rank" in selected_df.columns and "average_rank" not in ordered_selected_columns:
        ordered_selected_columns.append("average_rank")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking_df.to_csv(OUTPUT_DIR / f"{estimation_kind}_model_ranking_summary.csv", index=False)
    selected_df[ordered_selected_columns].to_csv(
        OUTPUT_DIR / f"{estimation_kind}_selected_models.csv",
        index=False,
    )


def _point_metric_label(row: pd.Series) -> str:
    return (
        f"MAE={row['mae']:.3f}  "
        f"RMSE={row['rmse']:.3f}  "
        f"R2={row['r2']:.3f}"
    )


def _interval_metric_label(row: pd.Series) -> str:
    return (
        f"Pinball={row['mean_pinball_loss']:.3f}  "
        f"Coverage={row['empirical_coverage']:.3f}  "
        f"Width={row['mean_interval_width']:.3f}"
    )


def _plot_selection_panel(
    ax: plt.Axes,
    selected_df: pd.DataFrame,
    *,
    title: str,
    metric_label_builder,
) -> None:
    plot_df = selected_df.copy().sort_values(by=["selection_order"], kind="mergesort")
    labels = [
        f"{int(row['selection_order'])}. {row['display_name']}"
        for _, row in plot_df.iterrows()
    ]
    colors = [SOURCE_COLORS.get(str(label), "#6c757d") for label in plot_df["source_label"]]

    bars = ax.barh(labels, plot_df["average_rank"], color=colors, edgecolor="none")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Average rank (lower is better)")
    ax.grid(axis="x", alpha=0.2, linewidth=0.8)
    ax.set_axisbelow(True)

    max_rank = float(plot_df["average_rank"].max()) if not plot_df.empty else 1.0
    ax.set_xlim(0.0, max_rank + 0.75)

    text_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for bar, (_, row) in zip(bars, plot_df.iterrows(), strict=False):
        y_position = bar.get_y() + (bar.get_height() / 2)
        ax.text(
            1.01,
            y_position,
            metric_label_builder(row),
            transform=text_transform,
            va="center",
            ha="left",
            fontsize=9,
            color="#264653",
        )


def plot_selected_model_benchmarks(
    point_selected_df: pd.DataFrame,
    interval_selected_df: pd.DataFrame,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(20, 12),
        gridspec_kw={"height_ratios": [max(len(point_selected_df), 1), max(len(interval_selected_df), 1)]},
    )

    _plot_selection_panel(
        axes[0],
        point_selected_df,
        title=f"Point Estimation | Top {len(point_selected_df)} models by average rank",
        metric_label_builder=_point_metric_label,
    )
    _plot_selection_panel(
        axes[1],
        interval_selected_df,
        title=f"Interval Estimation | Top {len(interval_selected_df)} models by average rank",
        metric_label_builder=_interval_metric_label,
    )

    legend_handles = [
        Patch(facecolor=SOURCE_COLORS["standard"], label="Standard results"),
        Patch(facecolor=SOURCE_COLORS["tuning"], label="Tuning results"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        f"Base Learner Benchmarking\nPoint top {RANK_POINT} | Interval top {RANK_INTERVAL}",
        fontsize=16,
        y=0.98,
    )
    fig.subplots_adjust(left=0.26, right=0.76, top=0.90, bottom=0.06, hspace=0.42)

    output_path = OUTPUT_DIR / "best_model_benchmarks.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    point_ranking_df, point_selected_df = _select_best_results("point")
    interval_ranking_df, interval_selected_df = _select_best_results("interval")

    _write_selection_artifacts("point", point_ranking_df, point_selected_df)
    _write_selection_artifacts("interval", interval_ranking_df, interval_selected_df)

    _print_selected_rows("point", point_selected_df)
    _print_selected_rows("interval", interval_selected_df)

    output_path = plot_selected_model_benchmarks(point_selected_df, interval_selected_df)
    print("")
    print(f"Saved benchmark plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
