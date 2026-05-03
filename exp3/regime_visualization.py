from __future__ import annotations

import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATE_COLUMN = "date"
LABEL_COLUMN = "label"

TIMESERIES_PATH = REPO_ROOT / "data" / "processed" / "timeseries_data.csv"
REGIMES_ROOT = REPO_ROOT / "data" / "regimes"
RESULTS_ROOT = REPO_ROOT / "results" / "regime_visualization"

REGIME_FILENAME = "regimes.csv"
DEFAULT_DPI = 300
DEFAULT_FIGSIZE: tuple[float, float] | None = None
DEFAULT_CELL_WIDTH = 7.0
DEFAULT_CELL_HEIGHT = 5.0


@dataclass(frozen=True)
class PlotPair:
    x_column: str
    y_column: str
    x_label: str
    y_label: str
    title: str


REGIME_FOLDERS = (
    "kmeans_pca_6",
    "kmedoids_6",
    "hmm_8",
)
REGIME_TITLES = (
    "K-Means PCA 6",
    "K-Medoids 6",
    "HMM 8",
)
PLOT_COLUMN_PAIRS = (
    PlotPair(
        x_column="DE Residual Load MWh/h 15min Actual",
        y_column="DE Price Spot EUR/MWh EPEX H Actual",
        x_label="Residual load (MWh/h)",
        y_label="Day-ahead spot price (EUR/MWh)",
        title="Residual load vs spot price",
    ),
    PlotPair(
        x_column="DE Wind Power Production MWh/h 15min Actual",
        y_column="DE Solar Photovoltaic Production MWh/h 15min Actual",
        x_label="Wind production (MWh/h)",
        y_label="Solar production (MWh/h)",
        title="Wind vs solar production",
    ),
    PlotPair(
        x_column="DE Consumption MWh/h 15min Actual",
        y_column="DE Residual Load MWh/h 15min Actual",
        x_label="Consumption (MWh/h)",
        y_label="Residual load (MWh/h)",
        title="Consumption vs residual load",
    ),
)
OUTPUT_NAME: str | None = None
DPI = DEFAULT_DPI
FIGSIZE = DEFAULT_FIGSIZE


@dataclass(frozen=True)
class RegimeVisualizationResult:
    output_path: Path
    regime_folders: tuple[str, ...]
    plot_column_pairs: tuple[PlotPair, ...]


def _resolve_existing_path(raw_path: str | Path, *, description: str) -> Path:
    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, REPO_ROOT / path]

    deduplicated: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(candidate)

    for candidate in deduplicated:
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join(str(candidate.resolve(strict=False)) for candidate in deduplicated)
    raise FileNotFoundError(f"Could not find {description} {raw_path!r}. Searched: {searched}")


def _natural_sort_key(value: object) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(\d+)", str(value))
    key_parts: list[tuple[int, object]] = []
    for part in parts:
        if part.isdigit():
            key_parts.append((0, int(part)))
        else:
            key_parts.append((1, part.lower()))
    return tuple(key_parts)


def _validate_nonempty(values: Sequence[object], *, description: str) -> None:
    if not values:
        raise ValueError(f"Expected at least one {description}.")


def _validate_plot_pairs(
    plot_column_pairs: Sequence[PlotPair],
    data_columns: Sequence[str],
    *,
    date_column: str,
) -> tuple[PlotPair, ...]:
    _validate_nonempty(plot_column_pairs, description="plot column pair")

    pairs = tuple(plot_column_pairs)
    available_columns = set(data_columns)
    missing_columns = sorted(
        {
            column
            for pair in pairs
            for column in (pair.x_column, pair.y_column)
            if column not in available_columns
        }
    )
    if date_column not in available_columns:
        missing_columns.insert(0, date_column)
    if missing_columns:
        raise KeyError(f"Missing plot columns in timeseries_data.csv: {missing_columns}")

    return pairs


def _plot_columns(plot_column_pairs: Sequence[PlotPair]) -> list[str]:
    return list(
        dict.fromkeys(
            column
            for pair in plot_column_pairs
            for column in (pair.x_column, pair.y_column)
        )
    )


def _load_csv_with_date(
    path: Path,
    *,
    date_column: str,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, usecols=usecols)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{path} is empty.") from exc

    if date_column not in frame.columns:
        raise ValueError(f"Expected a {date_column!r} column in {path}.")

    frame = frame.copy()
    # Normalize every source to UTC before joining regime labels to plot data.
    frame[date_column] = pd.to_datetime(frame[date_column], utc=True, errors="coerce")
    invalid_dates = int(frame[date_column].isna().sum())
    if invalid_dates:
        raise ValueError(f"{path} contains {invalid_dates} invalid {date_column!r} values.")

    return (
        frame.sort_values(date_column)
        .drop_duplicates(subset=[date_column], keep="last")
        .reset_index(drop=True)
    )


def _load_plot_data_frame(
    plot_data_path: Path,
    *,
    plot_column_pairs: Sequence[PlotPair],
    date_column: str,
) -> tuple[pd.DataFrame, tuple[PlotPair, ...]]:
    try:
        data_columns = pd.read_csv(plot_data_path, nrows=0).columns
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{plot_data_path} is empty.") from exc

    pairs = _validate_plot_pairs(plot_column_pairs, data_columns, date_column=date_column)
    usecols = [date_column, *_plot_columns(pairs)]
    frame = _load_csv_with_date(plot_data_path, date_column=date_column, usecols=usecols)
    for column in _plot_columns(pairs):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, pairs


def _resolve_regime_file(regime_folder: str | Path, *, regimes_root: Path) -> Path:
    raw_path = Path(regime_folder).expanduser()
    candidates = (
        [raw_path]
        if raw_path.is_absolute()
        else [regimes_root / raw_path, REPO_ROOT / raw_path, Path.cwd() / raw_path]
    )

    searched: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)

        regime_file = resolved / REGIME_FILENAME if resolved.is_dir() else resolved
        searched.append(str(regime_file))
        if regime_file.is_file():
            return regime_file.resolve()

    raise FileNotFoundError(
        f"Could not find {REGIME_FILENAME!r} for regime folder {str(regime_folder)!r}. "
        f"Searched: {', '.join(searched)}"
    )


def _load_regime_frame(
    regime_path: Path,
    *,
    date_column: str,
    label_column: str,
) -> pd.DataFrame:
    frame = _load_csv_with_date(regime_path, date_column=date_column)
    if label_column not in frame.columns:
        raise ValueError(f"Expected a {label_column!r} column in {regime_path}.")

    frame = frame.loc[:, [date_column, label_column]].copy()
    frame[label_column] = frame[label_column].astype("string").fillna("<missing>").astype(str)
    if frame.empty:
        raise ValueError(f"{regime_path} has no regime rows.")
    return frame


def _regime_color_map(labels: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    palettes = ["tab20", "tab20b", "tab20c", "Set3"]
    colors: list[tuple[float, float, float, float]] = []
    for palette in palettes:
        cmap = plt.get_cmap(palette)
        palette_colors = getattr(cmap, "colors", None)
        if palette_colors is not None:
            colors.extend(tuple(color) for color in palette_colors)
        else:
            colors.extend(cmap(index / 20) for index in range(20))

    sorted_labels = sorted(labels, key=_natural_sort_key)
    return {
        label: colors[index % len(colors)]
        for index, label in enumerate(sorted_labels)
    }


def _regime_legend_label(label: str) -> str:
    display_label = re.sub(r"^(cluster|regime)[_-]?", "", str(label), flags=re.IGNORECASE)
    return f"Regime {display_label}"


def _format_regime_title(regime_name: str) -> str:
    name = str(regime_name).strip().replace("\\", "/").split("/")[-1]
    title = name.replace("_", " ").replace("-", " ").title()
    return (
        title.replace("Pca", "PCA")
        .replace("Hmm", "HMM")
        .replace("Gmm", "GMM")
    )


def _resolve_regime_titles(
    regime_names: Sequence[str],
    regime_titles: Sequence[str] | None,
) -> tuple[str, ...]:
    if regime_titles is None or len(regime_titles) != len(regime_names):
        return tuple(_format_regime_title(name) for name in regime_names)
    return tuple(str(title) for title in regime_titles)


def _resolve_figsize(
    figsize: tuple[float, float] | None,
    *,
    row_count: int,
    column_count: int,
) -> tuple[float, float]:
    if figsize is not None:
        if len(figsize) != 2 or figsize[0] <= 0 or figsize[1] <= 0:
            raise ValueError("figsize must contain two positive numbers.")
        return float(figsize[0]), float(figsize[1])

    return (
        DEFAULT_CELL_WIDTH * column_count,
        DEFAULT_CELL_HEIGHT * row_count,
    )


def _default_output_name(regime_folders: Sequence[str]) -> str:
    joined = "__".join(
        str(folder).strip().replace("\\", "/").split("/")[-1]
        for folder in regime_folders
    )
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", joined).strip("._")
    return f"{safe_name or 'regimes'}_cluster_grid.png"


def _resolve_output_path(
    *,
    output_name: str | Path | None,
    regime_folders: Sequence[str],
    results_root: Path,
) -> Path:
    name = Path(output_name) if output_name else Path(_default_output_name(regime_folders))
    if not name.suffix:
        name = name.with_suffix(".png")
    if name.is_absolute():
        return name.resolve()
    return (results_root / name).resolve()


def _merge_for_plot(
    *,
    regime_frame: pd.DataFrame,
    plot_data_frame: pd.DataFrame,
    plot_columns: Sequence[str],
    date_column: str,
) -> pd.DataFrame:
    plot_frame = regime_frame.merge(
        plot_data_frame.loc[:, [date_column, *plot_columns]],
        on=date_column,
        how="left",
        indicator=True,
    )
    matched_rows = int((plot_frame["_merge"] == "both").sum())
    if matched_rows == 0:
        raise ValueError(
            "No overlapping UTC timestamps between the regime file and timeseries_data.csv."
        )
    return plot_frame.drop(columns=["_merge"])


def _draw_regime_grid(
    *,
    regime_frames: Sequence[pd.DataFrame],
    regime_titles: Sequence[str],
    plot_data_frame: pd.DataFrame,
    plot_column_pairs: Sequence[PlotPair],
    date_column: str,
    label_column: str,
    output_path: Path,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    row_count = len(regime_frames)
    column_count = len(plot_column_pairs)
    if row_count != len(regime_titles):
        raise ValueError(
            "regime_titles must have the same length as regime_frames "
            f"({len(regime_titles)} != {row_count})."
        )
    if column_count == 0:
        raise ValueError("Expected at least one plot column pair.")

    plt.rcParams.update(
        {
            "axes.edgecolor": "#4b5563",
            "axes.linewidth": 0.8,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "font.size": 12,
            "grid.color": "#d1d5db",
            "grid.linewidth": 0.55,
        }
    )

    fig, axes = plt.subplots(row_count, column_count, figsize=figsize, squeeze=False)
    axes = np.asarray(axes)
    columns = _plot_columns(plot_column_pairs)

    for row_index, (regime_frame, regime_title) in enumerate(
        zip(regime_frames, regime_titles)
    ):
        plot_frame = _merge_for_plot(
            regime_frame=regime_frame,
            plot_data_frame=plot_data_frame,
            plot_columns=columns,
            date_column=date_column,
        )
        labels = sorted(plot_frame[label_column].unique().tolist(), key=_natural_sort_key)
        color_map = _regime_color_map(labels)
        legend_handles = []
        legend_labels = []

        for column_index, plot_pair in enumerate(plot_column_pairs):
            ax = axes[row_index, column_index]
            valid_total = 0

            for label in labels:
                label_frame = plot_frame.loc[plot_frame[label_column] == label]
                label_frame = label_frame.loc[
                    label_frame[[plot_pair.x_column, plot_pair.y_column]]
                    .notna()
                    .all(axis=1),
                    [plot_pair.x_column, plot_pair.y_column],
                ]
                valid_total += len(label_frame)
                if label_frame.empty:
                    continue

                scatter = ax.scatter(
                    label_frame[plot_pair.x_column],
                    label_frame[plot_pair.y_column],
                    s=7,
                    alpha=0.68,
                    color=color_map[label],
                    edgecolors="none",
                    rasterized=True,
                    label=_regime_legend_label(label),
                )
                if column_index == 0:
                    legend_handles.append(scatter)
                    legend_labels.append(_regime_legend_label(label))

            if row_index == 0:
                ax.set_title(
                    textwrap.fill(plot_pair.title, width=44),
                    pad=8,
                    fontweight="bold",
                )
            ax.set_xlabel(textwrap.fill(plot_pair.x_label, width=34), labelpad=6)
            ax.set_ylabel(textwrap.fill(plot_pair.y_label, width=34), labelpad=6)
            ax.tick_params(axis="both", labelsize=11)
            ax.grid(True, alpha=0.7)
            ax.margins(x=0.04, y=0.04)
            ax.text(
                0.02,
                0.98,
                f"n={valid_total:,}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "#d1d5db",
                    "alpha": 0.9,
                },
            )

        axes[row_index, 0].annotate(
            regime_title,
            xy=(-0.20, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            rotation=90,
            fontsize=15,
            fontweight="bold",
        )
        axes[row_index, -1].legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.005, 0.5),
            frameon=True,
            fontsize=11,
            title="Clusters",
            title_fontsize=12,
            borderaxespad=0.0,
            markerscale=1.5,
        )

    fig.subplots_adjust(
        left=0.065,
        right=0.875,
        top=0.955,
        bottom=0.075,
        hspace=0.18,
        wspace=0.18,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_regime_visualization(
    *,
    regime_folders: Sequence[str],
    plot_column_pairs: Sequence[PlotPair],
    plot_data_path: str | Path = TIMESERIES_PATH,
    regimes_root: str | Path = REGIMES_ROOT,
    results_root: str | Path = RESULTS_ROOT,
    regime_titles: Sequence[str] | None = None,
    output_name: str | Path | None = None,
    date_column: str = DATE_COLUMN,
    label_column: str = LABEL_COLUMN,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] | None = DEFAULT_FIGSIZE,
) -> RegimeVisualizationResult:
    _validate_nonempty(regime_folders, description="regime folder")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")

    plot_data_path = _resolve_existing_path(
        plot_data_path,
        description="timeseries_data.csv",
    )
    regimes_root = _resolve_existing_path(regimes_root, description="regimes root")
    results_root = Path(results_root).expanduser()
    if not results_root.is_absolute():
        results_root = REPO_ROOT / results_root
    results_root = results_root.resolve()

    plot_data_frame, pairs = _load_plot_data_frame(
        plot_data_path,
        plot_column_pairs=plot_column_pairs,
        date_column=date_column,
    )
    regime_names = tuple(str(folder) for folder in regime_folders)
    row_titles = _resolve_regime_titles(regime_names, regime_titles)
    resolved_figsize = _resolve_figsize(
        figsize,
        row_count=len(regime_names),
        column_count=len(pairs),
    )
    regime_paths = [
        _resolve_regime_file(folder, regimes_root=regimes_root)
        for folder in regime_names
    ]
    regime_frames = [
        _load_regime_frame(path, date_column=date_column, label_column=label_column)
        for path in regime_paths
    ]

    output_path = _resolve_output_path(
        output_name=output_name,
        regime_folders=regime_names,
        results_root=results_root,
    )
    _draw_regime_grid(
        regime_frames=regime_frames,
        regime_titles=row_titles,
        plot_data_frame=plot_data_frame,
        plot_column_pairs=pairs,
        date_column=date_column,
        label_column=label_column,
        output_path=output_path,
        dpi=dpi,
        figsize=resolved_figsize,
    )

    return RegimeVisualizationResult(
        output_path=output_path,
        regime_folders=regime_names,
        plot_column_pairs=pairs,
    )


def main() -> int:
    result = create_regime_visualization(
        regime_folders=REGIME_FOLDERS,
        plot_column_pairs=PLOT_COLUMN_PAIRS,
        plot_data_path=TIMESERIES_PATH,
        regimes_root=REGIMES_ROOT,
        results_root=RESULTS_ROOT,
        regime_titles=REGIME_TITLES,
        output_name=OUTPUT_NAME,
        date_column=DATE_COLUMN,
        label_column=LABEL_COLUMN,
        dpi=DPI,
        figsize=FIGSIZE,
    )
    print(f"Saved regime visualization: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
