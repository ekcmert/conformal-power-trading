from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from regime_discovery.heuristic_regime import (
    HEURISTIC_REQUIRED_COLUMNS,
    assign_german_da_regimes,
)
from regime_discovery.regime_config import REGIME_COLUMNS, REGIME_COLUMNS_PCA
from regime_discovery.regime_clustering import (
    RegimeClustering,
    default_output_folder_for_method,
    resolve_clustering_method_name,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
X_PATH = REPO_ROOT / "data" / "final" / "X.csv"
REGIMES_ROOT = REPO_ROOT / "data" / "regimes"
HEURISTIC_FOLDER_NAME = "heuristic"
OUTPUT_FILENAME = "regimes.csv"
LABEL_COLUMN_NAME = "label"
REGIME_RANGE = 24
REGIME_FREQUENCY = 1
NUM_CLUSTERS_GRID = [6,8,10]
HEURISTIC_METHOD_NAME = "heuristic"


@dataclass(frozen=True)
class RegimeGenerationSummary:
    rows_written: int
    output_path: Path
    label_column: str
    skipped: bool = False


@dataclass(frozen=True)
class RollingRegimeWindow:
    sample_start: pd.Timestamp
    prediction_start: pd.Timestamp
    prediction_end: pd.Timestamp


def _load_x_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, parse_dates=["date"])
    if "date" not in frame.columns:
        raise ValueError(f"Expected a date column in {csv_path}.")
    return frame.sort_values("date")


def _resolve_output_dir(regimes_root: Path, folder_name: str) -> Path:
    base_dir = regimes_root.resolve()
    output_dir = (base_dir / folder_name).resolve()

    if output_dir != base_dir and base_dir not in output_dir.parents:
        raise ValueError(
            f"Resolved output directory escaped regimes_root: {output_dir} not under {base_dir}"
        )

    return output_dir


def _has_non_empty_regimes_csv(output_path: Path) -> bool:
    if not output_path.is_file():
        return False

    try:
        existing_frame = pd.read_csv(output_path, nrows=1)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return False

    return not existing_frame.empty


def _month_start(timestamp: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(
        year=timestamp.year,
        month=timestamp.month,
        day=1,
        tz=timestamp.tz,
    )


def _build_rolling_windows(
    dates: pd.Series,
    *,
    regime_range: int,
    regime_frequency: int,
) -> list[RollingRegimeWindow]:
    if regime_range < 1:
        raise ValueError("regime_range must be at least 1 month.")
    if regime_frequency < 1:
        raise ValueError("regime_frequency must be at least 1 month.")
    if dates.empty:
        return []

    first_available = dates.min()
    last_available = dates.max()
    first_month = _month_start(first_available)
    last_month = _month_start(last_available)
    has_full_first_month = first_available == first_month

    prediction_start = first_month + pd.DateOffset(
        months=regime_range + (0 if has_full_first_month else 1)
    )
    windows: list[RollingRegimeWindow] = []

    while prediction_start <= last_month:
        prediction_end = prediction_start + pd.DateOffset(months=regime_frequency)
        sample_start = prediction_start - pd.DateOffset(months=regime_range)
        windows.append(
            RollingRegimeWindow(
                sample_start=sample_start,
                prediction_start=prediction_start,
                prediction_end=prediction_end,
            )
        )
        prediction_start = prediction_end

    return windows


def _resolve_clustering_columns(
    *,
    clustering_columns: list[str] | None,
    is_pca: bool,
) -> list[str]:
    if clustering_columns is not None:
        if not clustering_columns:
            raise ValueError("clustering_columns must contain at least one column.")
        return list(dict.fromkeys(clustering_columns))

    if is_pca:
        return list(REGIME_COLUMNS_PCA)
    return list(REGIME_COLUMNS)


def _resolve_method_name(method_name: str) -> str:
    normalized = method_name.strip().lower()
    if normalized == HEURISTIC_METHOD_NAME:
        return HEURISTIC_METHOD_NAME
    return resolve_clustering_method_name(normalized)


def _default_output_folder_name(
    method_name: str,
    *,
    is_pca: bool,
    num_clusters: int,
) -> str:
    if method_name == HEURISTIC_METHOD_NAME:
        return HEURISTIC_FOLDER_NAME
    return default_output_folder_for_method(
        method_name,
        is_pca=is_pca,
        num_clusters=num_clusters,
    )


def generate_regimes(
    *,
    x_path: Path = X_PATH,
    regimes_root: Path = REGIMES_ROOT,
    method_name: str = HEURISTIC_METHOD_NAME,
    output_folder_name: str | None = None,
    output_filename: str = OUTPUT_FILENAME,
    label_column: str = LABEL_COLUMN_NAME,
    num_clusters: int = 8,
    clustering_columns: list[str] | None = None,
    is_pca: bool = False,
    pca_dim: int = 16,
    regime_range: int = REGIME_RANGE,
    regime_frequency: int = REGIME_FREQUENCY,
    skip_existing: bool = True,
) -> RegimeGenerationSummary:
    resolved_method_name = _resolve_method_name(method_name)
    resolved_output_folder_name = output_folder_name or _default_output_folder_name(
        resolved_method_name,
        is_pca=is_pca,
        num_clusters=num_clusters,
    )
    output_dir = _resolve_output_dir(regimes_root.resolve(), resolved_output_folder_name)
    output_path = output_dir / output_filename

    if skip_existing and _has_non_empty_regimes_csv(output_path):
        tqdm.write(
            "Skipping regime generation for "
            f"'{resolved_output_folder_name}': found existing non-empty {output_path}."
        )
        return RegimeGenerationSummary(
            rows_written=0,
            output_path=output_path,
            label_column=label_column,
            skipped=True,
        )

    x_frame = _load_x_frame(x_path.resolve())

    if resolved_method_name == HEURISTIC_METHOD_NAME:
        required_columns = HEURISTIC_REQUIRED_COLUMNS
        resolved_clustering_columns: list[str] | None = None
    else:
        resolved_clustering_columns = _resolve_clustering_columns(
            clustering_columns=clustering_columns,
            is_pca=is_pca,
        )
        required_columns = resolved_clustering_columns

    missing_features = [column for column in required_columns if column not in x_frame.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns in X.csv: {missing_features}")

    rolling_windows = _build_rolling_windows(
        x_frame["date"],
        regime_range=regime_range,
        regime_frequency=regime_frequency,
    )
    if not rolling_windows:
        raise RuntimeError("No rolling monthly regime windows could be built from X.csv.")

    regime_frames: list[pd.DataFrame] = []
    for window in rolling_windows:
        sample_mask = (
            (x_frame["date"] >= window.sample_start) &
            (x_frame["date"] < window.prediction_start)
        )
        prediction_mask = (
            (x_frame["date"] >= window.prediction_start) &
            (x_frame["date"] < window.prediction_end)
        )

        sample_frame = x_frame.loc[sample_mask]
        prediction_frame = x_frame.loc[prediction_mask]

        if prediction_frame.empty:
            continue
        if sample_frame.empty:
            raise RuntimeError(
                "A rolling regime window had prediction rows but no historical sample rows: "
                f"sample_start={window.sample_start}, prediction_start={window.prediction_start}."
            )

        if resolved_method_name == HEURISTIC_METHOD_NAME:
            regime_frames.append(
                assign_german_da_regimes(
                    df_sample=sample_frame,
                    df_pred=prediction_frame,
                    output_col=label_column,
                )
            )
            continue

        regime_clustering = RegimeClustering(
            sample_frame,
            prediction_frame,
            num_clusters=num_clusters,
            column_names=resolved_clustering_columns or [],
            is_pca=is_pca,
            pca_dim=pca_dim,
        )
        regime_frames.append(
            regime_clustering.assign(
                resolved_method_name,
                output_col=label_column,
            )
        )

    if not regime_frames:
        raise RuntimeError("No regime rows were generated from the rolling monthly windows.")

    regime_frame = pd.concat(regime_frames, ignore_index=False).sort_values("date")

    export_frame = regime_frame.loc[:, ["date", label_column]].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    export_frame.to_csv(output_path, index=False)

    return RegimeGenerationSummary(
        rows_written=len(export_frame),
        output_path=output_path,
        label_column=label_column,
    )


generate_heuristic_regimes = generate_regimes


def generate_regime_grid(
    *,
    method_name: str,
    num_clusters_grid: list[int] = NUM_CLUSTERS_GRID,
    **kwargs: object,
) -> list[RegimeGenerationSummary]:
    resolved_method_name = _resolve_method_name(method_name)
    if resolved_method_name == HEURISTIC_METHOD_NAME:
        return [generate_regimes(method_name=resolved_method_name, **kwargs)]

    if not num_clusters_grid:
        raise ValueError("num_clusters_grid must contain at least one cluster count.")

    summaries: list[RegimeGenerationSummary] = []
    for num_clusters in num_clusters_grid:
        summaries.append(
            generate_regimes(
                method_name=resolved_method_name,
                num_clusters=int(num_clusters),
                **kwargs,
            )
        )
    return summaries


def generate_all_regimes(
    *,
    x_path: Path = X_PATH,
    regimes_root: Path = REGIMES_ROOT,
    output_filename: str = OUTPUT_FILENAME,
    label_column: str = LABEL_COLUMN_NAME,
    num_clusters_grid: list[int] = NUM_CLUSTERS_GRID,
    clustering_columns: list[str] | None = None,
    pca_dim: int = 16,
    regime_range: int = REGIME_RANGE,
    regime_frequency: int = REGIME_FREQUENCY,
    skip_existing: bool = True,
    show_progress: bool = False,
) -> list[RegimeGenerationSummary]:
    method_names = RegimeClustering.available_methods()
    total_runs = 1 + (len(method_names) * 2 * len(num_clusters_grid))
    summaries: list[RegimeGenerationSummary] = []

    with tqdm(
        total=total_runs,
        desc="Generating regimes",
        unit="file",
        disable=not show_progress,
    ) as progress_bar:
        heuristic_summary = generate_regimes(
            x_path=x_path,
            regimes_root=regimes_root,
            method_name=HEURISTIC_METHOD_NAME,
            output_filename=output_filename,
            label_column=label_column,
            regime_range=regime_range,
            regime_frequency=regime_frequency,
            skip_existing=skip_existing,
        )
        summaries.append(heuristic_summary)
        if show_progress:
            progress_bar.set_postfix_str("heuristic", refresh=False)
            progress_bar.update(1)

        for method_name in method_names:
            for is_pca in (False, True):
                for num_clusters in num_clusters_grid:
                    regime_summary = generate_regimes(
                        x_path=x_path,
                        regimes_root=regimes_root,
                        method_name=method_name,
                        output_filename=output_filename,
                        label_column=label_column,
                        num_clusters=int(num_clusters),
                        clustering_columns=clustering_columns,
                        is_pca=is_pca,
                        pca_dim=pca_dim,
                        regime_range=regime_range,
                        regime_frequency=regime_frequency,
                        skip_existing=skip_existing,
                    )
                    summaries.append(regime_summary)
                    if show_progress:
                        progress_bar.set_postfix_str(
                            f"{method_name} | {'pca' if is_pca else 'raw'} | k={num_clusters}",
                            refresh=False,
                        )
                        progress_bar.update(1)

        if show_progress:
            progress_bar.set_postfix_str("done", refresh=False)

    return summaries


def generate_kmeans_regimes(**kwargs: object) -> RegimeGenerationSummary:
    return generate_regimes(method_name="kmeans", **kwargs)


def generate_kmedoids_regimes(**kwargs: object) -> RegimeGenerationSummary:
    return generate_regimes(method_name="kmedoids", **kwargs)


def generate_agglomerative_regimes(**kwargs: object) -> RegimeGenerationSummary:
    return generate_regimes(method_name="agglomerative", **kwargs)


def generate_divisive_regimes(**kwargs: object) -> RegimeGenerationSummary:
    return generate_regimes(method_name="divisive", **kwargs)


def generate_spectral_regimes(**kwargs: object) -> RegimeGenerationSummary:
    return generate_regimes(method_name="spectral", **kwargs)


def generate_gmm_regimes(**kwargs: object) -> RegimeGenerationSummary:
    return generate_regimes(method_name="gmm", **kwargs)


def generate_hmm_regimes(**kwargs: object) -> RegimeGenerationSummary:
    return generate_regimes(method_name="hmm", **kwargs)


def main() -> None:
    summaries = generate_all_regimes(show_progress=True)
    generated_count = sum(not summary.skipped for summary in summaries)
    skipped_count = sum(summary.skipped for summary in summaries)
    print(f"Generated files: {generated_count}")
    print(f"Skipped files  : {skipped_count}")
    for summary in summaries:
        print(f"Status       : {'skipped' if summary.skipped else 'generated'}")
        print(f"Rows written : {summary.rows_written}")
        print(f"Output path  : {summary.output_path}")
        print(f"Label column : {summary.label_column}")


if __name__ == "__main__":
    main()
