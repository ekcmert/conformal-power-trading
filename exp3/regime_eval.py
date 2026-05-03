from __future__ import annotations

import argparse
import fnmatch
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly import colors as plotly_colors
from plotly.subplots import make_subplots
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DATE_COLUMN = "date"
LABEL_COLUMN = "label"

X_PATH = REPO_ROOT / "data" / "final" / "X.csv"
Y_PATH = REPO_ROOT / "data" / "final" / "y.csv"
REGIMES_ROOT = REPO_ROOT / "data" / "regimes"
RESIDUALS_ROOT = REPO_ROOT / "data" / "residuals"
RESULTS_ROOT = REPO_ROOT / "results" / "regimes"

REGIME_FILENAME = "regimes.csv"
PLOT_FILENAME = "regime_plots.html"
SUMMARY_FILENAME = "summary.csv"
REQUESTED_SUMMARY_ALIAS = "summar.csv"
FINAL_SUMMARY_FILENAME = "final_summary.csv"


@dataclass(frozen=True)
class ResidualSeries:
    column_name: str
    frame: pd.DataFrame
    source_path: Path


@dataclass(frozen=True)
class RegimeEvaluationResult:
    regime_groups_evaluated: int
    residual_series_evaluated: int
    results_root: Path
    summary_path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated regime assignments against bin-size balance, "
            "reference residuals, target prices, and diagnostic plots."
        ),
    )
    parser.add_argument(
        "--plot-column-pair",
        "--plot-pair",
        dest="plot_column_pairs",
        nargs=2,
        action="append",
        metavar=("LEFT_COLUMN", "RIGHT_COLUMN"),
        required=True,
        help=(
            "Two X.csv columns to draw as one X/Y scatter subplot. "
            "Repeat exactly three times."
        ),
    )
    parser.add_argument(
        "--reference-residual-model-path",
        required=True,
        help=(
            "Single residual CSV file to evaluate. Accepts an absolute path, a path "
            "relative to the repo, or a path relative to data/residuals, for example "
            "'point_estimation/lightgbm/regression/da_res.csv'."
        ),
    )
    parser.add_argument(
        "--target-price-column-name",
        required=True,
        help="Target price column from data/final/y.csv to summarize by regime.",
    )
    parser.add_argument("--x-path", type=Path, default=X_PATH)
    parser.add_argument("--y-path", type=Path, default=Y_PATH)
    parser.add_argument("--regimes-root", type=Path, default=REGIMES_ROOT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--label-column", default=LABEL_COLUMN)
    parser.add_argument(
        "--regime-glob",
        default="*",
        help=(
            "Optional glob matched against regime folder names relative to "
            "data/regimes, for example 'kmeans_*' or 'heuristic'."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV metrics without generating Plotly HTML files.",
    )
    parser.add_argument(
        "--include-plotlyjs",
        default="cdn",
        choices=("cdn", "directory", "true", "false"),
        help="Plotly JavaScript embedding mode used by plotly.write_html.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def _plotlyjs_mode(value: str) -> str | bool:
    if value == "true":
        return True
    if value == "false":
        return False
    return value


def _resolve_existing_path(
    raw_path: str | Path,
    *,
    fallback_root: Path | None = None,
    description: str = "path",
) -> Path:
    path = Path(raw_path).expanduser()
    candidates: list[Path]
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [Path.cwd() / path, REPO_ROOT / path]
        if fallback_root is not None:
            candidates.append(fallback_root / path)

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


def _path_id(path: Path, *, root: Path | None = None) -> str:
    resolved = path.resolve()
    if root is not None:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _natural_sort_key(value: object) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(\d+)", str(value))
    key_parts: list[tuple[int, object]] = []
    for part in parts:
        if part.isdigit():
            key_parts.append((0, int(part)))
        else:
            key_parts.append((1, part.lower()))
    return tuple(key_parts)


def _load_csv_with_date(path: Path, *, date_column: str = DATE_COLUMN) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{path} is empty.") from exc

    if date_column not in frame.columns:
        raise ValueError(f"Expected a {date_column!r} column in {path}.")

    frame = frame.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], utc=True, errors="coerce")
    invalid_dates = int(frame[date_column].isna().sum())
    if invalid_dates:
        raise ValueError(f"{path} contains {invalid_dates} invalid {date_column!r} values.")

    return (
        frame.sort_values(date_column)
        .drop_duplicates(subset=[date_column], keep="last")
        .reset_index(drop=True)
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


def _discover_regime_files(regimes_root: Path, regime_glob: str) -> list[Path]:
    paths = sorted(
        (path for path in regimes_root.rglob(REGIME_FILENAME) if path.is_file()),
        key=lambda path: path.parent.relative_to(regimes_root).as_posix(),
    )
    if regime_glob in {"", "*"}:
        return paths

    return [
        path
        for path in paths
        if fnmatch.fnmatch(path.parent.relative_to(regimes_root).as_posix(), regime_glob)
    ]


def _validate_plot_pairs(
    plot_column_pairs: Sequence[tuple[str, str]],
    x_columns: Sequence[str],
) -> list[tuple[str, str]]:
    pairs = [(str(left), str(right)) for left, right in plot_column_pairs]
    if len(pairs) != 3:
        raise ValueError(
            f"Expected exactly three --plot-column-pair values, received {len(pairs)}."
        )

    missing_columns = sorted(
        {
            column
            for pair in pairs
            for column in pair
            if column not in set(x_columns)
        }
    )
    if missing_columns:
        raise KeyError(f"Missing plot columns in X.csv: {missing_columns}")
    return pairs


def _load_residual_series(
    reference_residual_path: Path,
    *,
    date_column: str,
) -> list[ResidualSeries]:
    if not reference_residual_path.is_file():
        raise ValueError(
            "reference_residual_model_path must point to one residual CSV file, "
            f"not a directory: {reference_residual_path}"
        )

    series: list[ResidualSeries] = []
    skipped: list[str] = []
    frame = _load_csv_with_date(reference_residual_path, date_column=date_column)

    value_columns = [column for column in frame.columns if column != date_column]
    for column in value_columns:
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        if numeric_values.notna().sum() == 0:
            skipped.append(f"{reference_residual_path}::{column}")
            continue

        residual_frame = frame.loc[:, [date_column]].copy()
        residual_frame[column] = numeric_values
        series.append(
            ResidualSeries(
                column_name=column,
                frame=residual_frame,
                source_path=reference_residual_path,
            )
        )

    if not series:
        skipped_display = ", ".join(skipped[:5])
        raise ValueError(
            "No numeric residual series were found in "
            f"{reference_residual_path}. Skipped examples: {skipped_display}"
        )
    if len(series) > 1:
        columns = [item.column_name for item in series]
        raise ValueError(
            "Expected exactly one numeric residual column in "
            f"{reference_residual_path}, found {columns}."
        )
    return series


def _gini(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or float(array.sum()) == 0.0:
        return float("nan")

    sorted_array = np.sort(array)
    n = sorted_array.size
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(index * sorted_array)) / (n * np.sum(sorted_array)) - ((n + 1) / n))


def _compute_balance_metrics(
    regime_frame: pd.DataFrame,
    *,
    regime_name: str,
    regime_source_path: Path,
    date_column: str,
    label_column: str,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    counts = regime_frame[label_column].value_counts(dropna=False)
    labels = sorted(counts.index.tolist(), key=_natural_sort_key)
    counts = counts.reindex(labels).astype(int)

    total = int(counts.sum())
    n_regimes = int(len(counts))
    shares = counts.astype(float) / total if total else counts.astype(float)
    expected_count = total / n_regimes if n_regimes else float("nan")
    expected_share = 1.0 / n_regimes if n_regimes else float("nan")

    if n_regimes <= 1:
        normalized_entropy = 1.0 if n_regimes == 1 else float("nan")
        bin_size_cv = 0.0 if n_regimes == 1 else float("nan")
        imbalance_ratio = 1.0 if n_regimes == 1 else float("nan")
        chi_square = 0.0 if n_regimes == 1 else float("nan")
    else:
        positive_shares = shares[shares > 0.0].to_numpy(dtype=float)
        entropy = -float(np.sum(positive_shares * np.log(positive_shares)))
        normalized_entropy = entropy / math.log(n_regimes)
        bin_size_cv = float(counts.std(ddof=0) / counts.mean()) if counts.mean() else float("nan")
        min_count = int(counts.min())
        imbalance_ratio = float(counts.max() / min_count) if min_count else float("inf")
        chi_square = float((((counts - expected_count) ** 2) / expected_count).sum())

    summary = {
        "total_observations": total,
        "n_regimes": n_regimes,
        "min_bin_count": int(counts.min()) if n_regimes else 0,
        "max_bin_count": int(counts.max()) if n_regimes else 0,
        "mean_bin_count": float(counts.mean()) if n_regimes else float("nan"),
        "std_bin_count": float(counts.std(ddof=0)) if n_regimes else float("nan"),
        "min_bin_share": float(shares.min()) if n_regimes else float("nan"),
        "max_bin_share": float(shares.max()) if n_regimes else float("nan"),
        "bin_size_imbalance_ratio": imbalance_ratio,
        "bin_size_cv": bin_size_cv,
        "bin_size_normalized_entropy": float(normalized_entropy),
        "bin_size_gini": _gini(counts.to_numpy(dtype=float)),
        "bin_size_chi_square": chi_square,
        "bin_size_chi_square_per_regime": chi_square / n_regimes if n_regimes else float("nan"),
        "max_abs_share_deviation": (
            float((shares - expected_share).abs().max()) if n_regimes else float("nan")
        ),
    }

    rows: list[dict[str, object]] = []
    for label in labels:
        count = int(counts.loc[label])
        share = float(shares.loc[label])
        rows.append(
            {
                "regime_method": regime_name,
                "regime_source_path": _path_id(regime_source_path),
                "regime_label": label,
                "bin_count": count,
                "bin_share": share,
                "expected_bin_count": expected_count,
                "expected_bin_share": expected_share,
                "bin_count_delta": count - expected_count,
                "abs_share_deviation": abs(share - expected_share),
                **summary,
            }
        )

    return pd.DataFrame(rows), summary


def _value_stats_by_regime(
    regime_frame: pd.DataFrame,
    value_frame: pd.DataFrame,
    *,
    value_column: str,
    metric_family: str,
    source_path: Path,
    regime_name: str,
    regime_source_path: Path,
    reference_residual_model_id: str,
    target_price_column_name: str,
    date_column: str,
    label_column: str,
) -> pd.DataFrame:
    joined = regime_frame.loc[:, [date_column, label_column]].merge(
        value_frame.loc[:, [date_column, value_column]],
        on=date_column,
        how="left",
    )
    joined[value_column] = pd.to_numeric(joined[value_column], errors="coerce")

    rows: list[dict[str, object]] = []
    labels = sorted(regime_frame[label_column].unique().tolist(), key=_natural_sort_key)
    for label in labels:
        group = joined.loc[joined[label_column] == label, value_column]
        values = group.dropna()
        n = int(values.shape[0])
        regime_count = int(group.shape[0])
        rows.append(
            {
                "regime_method": regime_name,
                "regime_source_path": _path_id(regime_source_path),
                "regime_label": label,
                "metric_family": metric_family,
                "value_column": value_column,
                "source_file": _path_id(source_path),
                "reference_residual_model_path": reference_residual_model_id,
                "target_price_column_name": target_price_column_name,
                "regime_count": regime_count,
                "n": n,
                "missing_count": regime_count - n,
                "coverage": n / regime_count if regime_count else float("nan"),
                "min": float(values.min()) if n else float("nan"),
                "max": float(values.max()) if n else float("nan"),
                "mean": float(values.mean()) if n else float("nan"),
                "std": float(values.std(ddof=1)) if n > 1 else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def _summarize_value_metrics(
    metrics: pd.DataFrame,
    *,
    balance_summary: dict[str, float | int],
) -> dict[str, object]:
    valid = metrics.loc[metrics["n"] > 0].copy()
    n_observations = int(valid["n"].sum()) if not valid.empty else 0

    if n_observations:
        weights = valid["n"].astype(float)
        means = valid["mean"].astype(float)
        stds = valid["std"].fillna(0.0).astype(float)
        weighted_mean = float(np.average(means, weights=weights))
        weighted_within_regime_std = float(np.average(stds, weights=weights))
    else:
        weighted_mean = float("nan")
        weighted_within_regime_std = float("nan")

    regime_means = valid["mean"].dropna().astype(float)
    regime_stds = valid["std"].dropna().astype(float)

    first = metrics.iloc[0].to_dict()
    return {
        "regime_method": first["regime_method"],
        "metric_family": first["metric_family"],
        "value_column": first["value_column"],
        "source_file": first["source_file"],
        "reference_residual_model_path": first["reference_residual_model_path"],
        "target_price_column_name": first["target_price_column_name"],
        "n_observations": n_observations,
        "missing_count": int(metrics["missing_count"].sum()),
        "mean_coverage": float(metrics["coverage"].mean()),
        "weighted_mean": weighted_mean,
        "global_min": float(valid["min"].min()) if not valid.empty else float("nan"),
        "global_max": float(valid["max"].max()) if not valid.empty else float("nan"),
        "weighted_within_regime_std": weighted_within_regime_std,
        "mean_within_regime_std": (
            float(regime_stds.mean()) if not regime_stds.empty else float("nan")
        ),
        "min_regime_mean": (
            float(regime_means.min()) if not regime_means.empty else float("nan")
        ),
        "max_regime_mean": (
            float(regime_means.max()) if not regime_means.empty else float("nan")
        ),
        "range_of_regime_means": (
            float(regime_means.max() - regime_means.min())
            if not regime_means.empty
            else float("nan")
        ),
        "mean_abs_regime_mean": (
            float(regime_means.abs().mean()) if not regime_means.empty else float("nan")
        ),
        "std_of_regime_means": (
            float(regime_means.std(ddof=1)) if len(regime_means) > 1 else float("nan")
        ),
        **balance_summary,
    }


def _variance_decomposition(metrics: pd.DataFrame) -> dict[str, float]:
    valid = metrics.loc[metrics["n"] > 0].copy()
    if valid.empty:
        return {
            "global_mean": float("nan"),
            "global_std": float("nan"),
            "within_ss": float("nan"),
            "between_ss": float("nan"),
            "total_ss": float("nan"),
            "eta_squared": float("nan"),
        }

    weights = valid["n"].astype(float)
    means = valid["mean"].astype(float)
    global_mean = float(np.average(means, weights=weights))

    within_ss = float(
        (
            (valid["n"].astype(float) - 1.0).clip(lower=0.0)
            * valid["std"].fillna(0.0).astype(float).pow(2.0)
        ).sum()
    )
    between_ss = float((weights * (means - global_mean).pow(2.0)).sum())
    total_ss = within_ss + between_ss
    eta_squared = between_ss / total_ss if total_ss > 0.0 else float("nan")
    global_std = math.sqrt(total_ss / (float(weights.sum()) - 1.0)) if weights.sum() > 1 else float("nan")

    return {
        "global_mean": global_mean,
        "global_std": global_std,
        "within_ss": within_ss,
        "between_ss": between_ss,
        "total_ss": total_ss,
        "eta_squared": float(eta_squared),
    }


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not bool(valid.any()):
        return float("nan")
    return float(np.average(values.loc[valid].astype(float), weights=weights.loc[valid].astype(float)))


def _cv(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    if clean.empty:
        return float("nan")
    mean_value = float(clean.mean())
    if math.isclose(mean_value, 0.0):
        return float("nan")
    return float(clean.std(ddof=0) / abs(mean_value))


def _std_ratio(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    clean = clean.loc[clean > 0.0]
    if clean.empty:
        return float("nan")
    min_value = float(clean.min())
    return float(clean.max() / min_value) if min_value > 0.0 else float("inf")


def _bounded_inverse(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(1.0 / (1.0 + max(value, 0.0)))


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _build_method_evaluation_summary(
    *,
    balance_frame: pd.DataFrame,
    target_metrics: pd.DataFrame,
    residual_metrics: pd.DataFrame,
    rank: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for regime_method in sorted(balance_frame["regime_method"].unique(), key=_natural_sort_key):
        balance_rows = balance_frame.loc[balance_frame["regime_method"] == regime_method]
        target_rows = target_metrics.loc[target_metrics["regime_method"] == regime_method]
        residual_rows = residual_metrics.loc[residual_metrics["regime_method"] == regime_method]
        if balance_rows.empty or target_rows.empty or residual_rows.empty:
            continue

        balance = balance_rows.iloc[0]
        target_variance = _variance_decomposition(target_rows)
        residual_variance = _variance_decomposition(residual_rows)

        target_weights = target_rows["n"].astype(float)
        residual_weights = residual_rows["n"].astype(float)
        target_stds = target_rows["std"].astype(float)
        residual_stds = residual_rows["std"].astype(float)
        residual_means = residual_rows["mean"].astype(float)

        target_weighted_within_std = _weighted_average(target_stds, target_weights)
        residual_weighted_within_std = _weighted_average(residual_stds, residual_weights)
        target_std_cv = _cv(target_stds)
        residual_std_cv = _cv(residual_stds)
        target_std_ratio = _std_ratio(target_stds)
        residual_std_ratio = _std_ratio(residual_stds)

        residual_global_std = residual_variance["global_std"]
        residual_bias_abs_mean = float(abs(residual_variance["global_mean"]))
        residual_bias_to_std = (
            residual_bias_abs_mean / residual_global_std
            if np.isfinite(residual_global_std) and residual_global_std > 0.0
            else float("nan")
        )

        size_balance_score = _clip01(
            float(balance["bin_size_normalized_entropy"])
            * _bounded_inverse(float(balance["bin_size_cv"]))
            * _bounded_inverse(math.log(max(float(balance["bin_size_imbalance_ratio"]), 1.0)))
            * (1.0 - min(float(balance["max_abs_share_deviation"]), 1.0))
        )
        residual_variance_consistency_score = _clip01(
            0.70 * _bounded_inverse(residual_std_cv)
            + 0.20 * _bounded_inverse(math.log(max(residual_std_ratio, 1.0)))
            + 0.10 * _bounded_inverse(residual_bias_to_std)
        )
        price_profile_matching_score = _clip01(
            0.75 * target_variance["eta_squared"]
            + 0.25 * _bounded_inverse(target_weighted_within_std / target_variance["global_std"])
            if np.isfinite(target_variance["global_std"]) and target_variance["global_std"] > 0.0
            else target_variance["eta_squared"]
        )
        residual_regime_signal_score = _clip01(residual_variance["eta_squared"])

        overall_score = _clip01(
            0.35 * size_balance_score
            + 0.35 * residual_variance_consistency_score
            + 0.25 * price_profile_matching_score
            + 0.05 * residual_regime_signal_score
        )

        rows.append(
            {
                "regime_method": regime_method,
                "recommendation_score": overall_score,
                "size_balance_score": size_balance_score,
                "residual_variance_consistency_score": residual_variance_consistency_score,
                "price_profile_matching_score": price_profile_matching_score,
                "residual_regime_signal_score": residual_regime_signal_score,
                "n_regimes": int(balance["n_regimes"]),
                "total_observations": int(balance["total_observations"]),
                "min_bin_count": int(balance["min_bin_count"]),
                "max_bin_count": int(balance["max_bin_count"]),
                "min_bin_share": float(balance["min_bin_share"]),
                "max_bin_share": float(balance["max_bin_share"]),
                "bin_size_cv": float(balance["bin_size_cv"]),
                "bin_size_imbalance_ratio": float(balance["bin_size_imbalance_ratio"]),
                "bin_size_normalized_entropy": float(balance["bin_size_normalized_entropy"]),
                "bin_size_gini": float(balance["bin_size_gini"]),
                "max_abs_share_deviation": float(balance["max_abs_share_deviation"]),
                "residual_column": str(residual_rows["value_column"].iloc[0]),
                "residual_weighted_within_regime_std": residual_weighted_within_std,
                "residual_std_cv_across_regimes": residual_std_cv,
                "residual_std_ratio_across_regimes": residual_std_ratio,
                "residual_eta_squared": residual_variance["eta_squared"],
                "residual_global_mean": residual_variance["global_mean"],
                "residual_abs_global_mean": residual_bias_abs_mean,
                "residual_bias_to_global_std": residual_bias_to_std,
                "residual_min_regime_mean": float(residual_means.min()),
                "residual_max_regime_mean": float(residual_means.max()),
                "residual_range_of_regime_means": float(residual_means.max() - residual_means.min()),
                "target_price_column": str(target_rows["value_column"].iloc[0]),
                "target_price_weighted_within_regime_std": target_weighted_within_std,
                "target_price_std_cv_across_regimes": target_std_cv,
                "target_price_std_ratio_across_regimes": target_std_ratio,
                "target_price_eta_squared": target_variance["eta_squared"],
                "target_price_global_mean": target_variance["global_mean"],
                "target_price_global_std": target_variance["global_std"],
                "target_price_min_regime_mean": float(target_rows["mean"].astype(float).min()),
                "target_price_max_regime_mean": float(target_rows["mean"].astype(float).max()),
                "target_price_range_of_regime_means": float(
                    target_rows["mean"].astype(float).max() - target_rows["mean"].astype(float).min()
                ),
                "reference_residual_model_path": str(residual_rows["reference_residual_model_path"].iloc[0]),
                "regime_source_path": str(balance["regime_source_path"]),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary = summary.sort_values(
        [
            "recommendation_score",
            "size_balance_score",
            "residual_variance_consistency_score",
            "price_profile_matching_score",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    if rank:
        summary.insert(0, "rank", np.arange(1, len(summary) + 1))
        summary["recommendation"] = np.select(
            [
                summary["rank"] <= 5,
                summary["rank"] <= 15,
            ],
            [
                "strong_candidate",
                "candidate",
            ],
            default="review",
        )

    return summary


def _build_final_summary(method_evaluation_summary: pd.DataFrame) -> pd.DataFrame:
    if method_evaluation_summary.empty:
        return method_evaluation_summary.copy()

    compact = method_evaluation_summary.loc[
        :,
        [
            "regime_method",
            "size_balance_score",
            "residual_variance_consistency_score",
            "price_profile_matching_score",
            "n_regimes",
        ],
    ].copy()
    compact = compact.rename(
        columns={
            "size_balance_score": "cluster_size_balance_score",
        }
    )

    category_weights = {
        "cluster_size_balance_score": 0.35 / 0.95,
        "residual_variance_consistency_score": 0.35 / 0.95,
        "price_profile_matching_score": 0.25 / 0.95,
    }
    compact["final_score"] = (
        compact["cluster_size_balance_score"] * category_weights["cluster_size_balance_score"]
        + compact["residual_variance_consistency_score"]
        * category_weights["residual_variance_consistency_score"]
        + compact["price_profile_matching_score"] * category_weights["price_profile_matching_score"]
    )

    compact = compact.sort_values(
        [
            "final_score",
            "cluster_size_balance_score",
            "residual_variance_consistency_score",
            "price_profile_matching_score",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    compact.insert(0, "final_rank", np.arange(1, len(compact) + 1))

    compact["cluster_size_balance_rank"] = (
        compact["cluster_size_balance_score"].rank(method="min", ascending=False).astype(int)
    )
    compact["residual_variance_consistency_rank"] = (
        compact["residual_variance_consistency_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    compact["price_profile_matching_rank"] = (
        compact["price_profile_matching_score"].rank(method="min", ascending=False).astype(int)
    )
    compact["recommendation_label"] = np.select(
        [
            compact["final_rank"] <= 5,
            compact["final_rank"] <= 15,
        ],
        [
            "strong_candidate",
            "candidate",
        ],
        default="review",
    )

    return compact.loc[
        :,
        [
            "final_rank",
            "regime_method",
            "final_score",
            "cluster_size_balance_score",
            "residual_variance_consistency_score",
            "price_profile_matching_score",
            "cluster_size_balance_rank",
            "residual_variance_consistency_rank",
            "price_profile_matching_rank",
            "recommendation_label",
            "n_regimes",
        ],
    ]


def _regime_color_map(labels: Sequence[str]) -> dict[str, str]:
    palette = (
        plotly_colors.qualitative.Safe
        + plotly_colors.qualitative.Dark24
        + plotly_colors.qualitative.Set3
        + plotly_colors.qualitative.Pastel
    )
    sorted_labels = sorted(labels, key=_natural_sort_key)
    return {
        label: palette[index % len(palette)]
        for index, label in enumerate(sorted_labels)
    }


def _write_regime_plot(
    *,
    regime_frame: pd.DataFrame,
    x_frame: pd.DataFrame,
    plot_column_pairs: Sequence[tuple[str, str]],
    output_path: Path,
    title: str,
    date_column: str,
    label_column: str,
    include_plotlyjs: str | bool,
) -> None:
    plot_columns = list(dict.fromkeys(column for pair in plot_column_pairs for column in pair))
    plot_frame = regime_frame.loc[:, [date_column, label_column]].merge(
        x_frame.loc[:, [date_column, *plot_columns]],
        on=date_column,
        how="left",
    )

    labels = sorted(regime_frame[label_column].unique().tolist(), key=_natural_sort_key)
    color_map = _regime_color_map(labels)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.09,
        subplot_titles=[f"{x_column} vs {y_column}" for x_column, y_column in plot_column_pairs],
    )

    for row, (x_column, y_column) in enumerate(plot_column_pairs, start=1):
        for label in labels:
            label_frame = plot_frame.loc[plot_frame[label_column] == label]
            fig.add_trace(
                go.Scattergl(
                    x=label_frame[x_column],
                    y=label_frame[y_column],
                    mode="markers",
                    name=f"Regime {label}",
                    legendgroup=f"regime_{label}",
                    showlegend=row == 1,
                    marker={
                        "color": color_map[label],
                        "size": 5,
                        "opacity": 0.72,
                        "line": {"width": 0},
                    },
                    customdata=np.stack(
                        [
                            label_frame[date_column].astype(str).to_numpy(),
                            label_frame[label_column].astype(str).to_numpy(),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "date=%{customdata[0]}<br>"
                        "regime=%{customdata[1]}<br>"
                        f"{x_column}=%{{x}}<br>"
                        f"{y_column}=%{{y}}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )

        fig.update_xaxes(title_text=x_column, row=row, col=1)
        fig.update_yaxes(title_text=y_column, row=row, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=1050,
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 70, "r": 70, "t": 115, "b": 45},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=include_plotlyjs)


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_regime_evaluation(
    *,
    plot_column_pairs: Sequence[tuple[str, str]],
    reference_residual_model_path: str | Path,
    target_price_column_name: str,
    x_path: Path = X_PATH,
    y_path: Path = Y_PATH,
    regimes_root: Path = REGIMES_ROOT,
    results_root: Path = RESULTS_ROOT,
    label_column: str = LABEL_COLUMN,
    date_column: str = DATE_COLUMN,
    regime_glob: str = "*",
    skip_plots: bool = False,
    include_plotlyjs: str | bool = "cdn",
    show_progress: bool = True,
) -> RegimeEvaluationResult:
    x_path = _resolve_existing_path(x_path, description="X.csv")
    y_path = _resolve_existing_path(y_path, description="y.csv")
    regimes_root = _resolve_existing_path(regimes_root, description="regimes root")
    residual_model_path = _resolve_existing_path(
        reference_residual_model_path,
        fallback_root=RESIDUALS_ROOT,
        description="reference residual model path",
    )
    results_root = results_root.resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    x_frame = _load_csv_with_date(x_path, date_column=date_column)
    y_frame = _load_csv_with_date(y_path, date_column=date_column)
    pairs = _validate_plot_pairs(plot_column_pairs, x_frame.columns)

    if target_price_column_name not in y_frame.columns:
        raise KeyError(f"Missing target price column in y.csv: {target_price_column_name!r}")
    target_frame = y_frame.loc[:, [date_column, target_price_column_name]].copy()
    target_frame[target_price_column_name] = pd.to_numeric(
        target_frame[target_price_column_name],
        errors="coerce",
    )

    residual_series = _load_residual_series(
        residual_model_path,
        date_column=date_column,
    )
    regime_paths = _discover_regime_files(regimes_root, regime_glob)
    if not regime_paths:
        raise FileNotFoundError(
            f"No {REGIME_FILENAME} files found under {regimes_root} matching {regime_glob!r}."
        )

    residual_model_id = _path_id(
        residual_model_path,
        root=RESIDUALS_ROOT if residual_model_path.is_dir() else RESIDUALS_ROOT,
    )

    all_balance_frames: list[pd.DataFrame] = []
    all_target_frames: list[pd.DataFrame] = []
    all_residual_frames: list[pd.DataFrame] = []
    all_summary_rows: list[dict[str, object]] = []

    iterator = tqdm(
        regime_paths,
        desc="Evaluating regimes",
        unit="group",
        disable=not show_progress,
    )
    for regime_path in iterator:
        relative_parent = regime_path.parent.relative_to(regimes_root)
        regime_name = relative_parent.as_posix()
        output_dir = results_root / relative_parent
        output_dir.mkdir(parents=True, exist_ok=True)
        iterator.set_postfix_str(regime_name, refresh=False)

        regime_frame = _load_regime_frame(
            regime_path,
            date_column=date_column,
            label_column=label_column,
        )
        balance_frame, balance_summary = _compute_balance_metrics(
            regime_frame,
            regime_name=regime_name,
            regime_source_path=regime_path,
            date_column=date_column,
            label_column=label_column,
        )

        target_metrics = _value_stats_by_regime(
            regime_frame,
            target_frame,
            value_column=target_price_column_name,
            metric_family="target_price",
            source_path=y_path,
            regime_name=regime_name,
            regime_source_path=regime_path,
            reference_residual_model_id=residual_model_id,
            target_price_column_name=target_price_column_name,
            date_column=date_column,
            label_column=label_column,
        )
        summary_rows = [
            _summarize_value_metrics(
                target_metrics,
                balance_summary=balance_summary,
            )
        ]

        residual_metric_frames: list[pd.DataFrame] = []
        for residual in residual_series:
            metrics = _value_stats_by_regime(
                regime_frame,
                residual.frame,
                value_column=residual.column_name,
                metric_family="residual",
                source_path=residual.source_path,
                regime_name=regime_name,
                regime_source_path=regime_path,
                reference_residual_model_id=residual_model_id,
                target_price_column_name=target_price_column_name,
                date_column=date_column,
                label_column=label_column,
            )
            residual_metric_frames.append(metrics)
            summary_rows.append(
                _summarize_value_metrics(
                    metrics,
                    balance_summary=balance_summary,
                )
            )

        residual_metrics = pd.concat(residual_metric_frames, ignore_index=True)
        metric_family_summary = pd.DataFrame(summary_rows)
        method_summary = _build_method_evaluation_summary(
            balance_frame=balance_frame,
            target_metrics=target_metrics,
            residual_metrics=residual_metrics,
            rank=False,
        )

        _write_frame(output_dir / "bin_size_balance.csv", balance_frame)
        _write_frame(output_dir / "target_price_metrics.csv", target_metrics)
        _write_frame(output_dir / "model_specific_residual_metrics.csv", residual_metrics)
        _write_frame(output_dir / "metric_family_summary.csv", metric_family_summary)
        _write_frame(output_dir / SUMMARY_FILENAME, method_summary)

        if not skip_plots:
            _write_regime_plot(
                regime_frame=regime_frame,
                x_frame=x_frame,
                plot_column_pairs=pairs,
                output_path=output_dir / PLOT_FILENAME,
                title=f"Regime diagnostics: {regime_name}",
                date_column=date_column,
                label_column=label_column,
                include_plotlyjs=include_plotlyjs,
            )

        all_balance_frames.append(balance_frame)
        all_target_frames.append(target_metrics)
        all_residual_frames.append(residual_metrics)
        all_summary_rows.extend(summary_rows)

    all_summary = pd.DataFrame(all_summary_rows)
    all_balance = pd.concat(all_balance_frames, ignore_index=True)
    all_target = pd.concat(all_target_frames, ignore_index=True)
    all_residual = pd.concat(all_residual_frames, ignore_index=True)
    metric_family_summary = all_summary
    model_specific_summary = metric_family_summary.loc[
        all_summary["metric_family"] == "residual"
    ].reset_index(drop=True)
    method_evaluation_summary = _build_method_evaluation_summary(
        balance_frame=all_balance,
        target_metrics=all_target,
        residual_metrics=all_residual,
        rank=True,
    )
    final_summary = _build_final_summary(method_evaluation_summary)

    _write_frame(results_root / SUMMARY_FILENAME, method_evaluation_summary)
    _write_frame(results_root / REQUESTED_SUMMARY_ALIAS, method_evaluation_summary)
    _write_frame(results_root / FINAL_SUMMARY_FILENAME, final_summary)
    _write_frame(results_root / "method_evaluation_summary.csv", method_evaluation_summary)
    _write_frame(results_root / "metric_family_summary.csv", metric_family_summary)
    _write_frame(results_root / "model_specific_summary.csv", model_specific_summary)
    _write_frame(results_root / "all_bin_size_balance.csv", all_balance)
    _write_frame(results_root / "all_target_price_metrics.csv", all_target)
    _write_frame(results_root / "all_model_specific_residual_metrics.csv", all_residual)

    metadata = {
        "plot_column_pairs": list(map(list, pairs)),
        "reference_residual_model_path": residual_model_id,
        "target_price_column_name": target_price_column_name,
        "x_path": _path_id(x_path),
        "y_path": _path_id(y_path),
        "regimes_root": _path_id(regimes_root),
        "results_root": _path_id(results_root),
        "regime_glob": regime_glob,
        "regime_groups_evaluated": len(regime_paths),
        "residual_series_evaluated": len(residual_series),
        "skip_plots": skip_plots,
        "summary_methodology": {
            "unit": "one row per regime method",
            "recommendation_score": (
                "0.35 * size_balance_score + "
                "0.35 * residual_variance_consistency_score + "
                "0.25 * price_profile_matching_score + "
                "0.05 * residual_regime_signal_score"
            ),
            "size_balance_score": (
                "Rewards high normalized entropy and penalizes bin-size CV, "
                "imbalance ratio, and max absolute share deviation."
            ),
            "residual_variance_consistency_score": (
                "Rewards similar residual standard deviations across regimes, "
                "low residual std ratio, and low global residual bias."
            ),
            "price_profile_matching_score": (
                "Rewards target-price separation across regimes using eta-squared "
                "and penalizes high within-regime target-price dispersion."
            ),
            "residual_regime_signal_score": (
                "Residual eta-squared; small weight because the primary residual goal "
                "is stable calibration groups, not over-separating residual means."
            ),
        },
        "final_summary_methodology": {
            "unit": "one row per regime method",
            "final_score": (
                "0.368421 * cluster_size_balance_score + "
                "0.368421 * residual_variance_consistency_score + "
                "0.263158 * price_profile_matching_score"
            ),
            "note": (
                "final_summary.csv intentionally uses only the three benchmark categories "
                "and omits the residual_regime_signal_score from the wider summary.csv."
            ),
        },
    }
    (results_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return RegimeEvaluationResult(
        regime_groups_evaluated=len(regime_paths),
        residual_series_evaluated=len(residual_series),
        results_root=results_root,
        summary_path=results_root / SUMMARY_FILENAME,
    )


def main() -> int:
    args = _parse_args()
    result = run_regime_evaluation(
        plot_column_pairs=args.plot_column_pairs,
        reference_residual_model_path=args.reference_residual_model_path,
        target_price_column_name=args.target_price_column_name,
        x_path=args.x_path,
        y_path=args.y_path,
        regimes_root=args.regimes_root,
        results_root=args.results_root,
        label_column=args.label_column,
        regime_glob=args.regime_glob,
        skip_plots=args.skip_plots,
        include_plotlyjs=_plotlyjs_mode(args.include_plotlyjs),
        show_progress=not args.no_progress,
    )
    print(f"Regime groups evaluated  : {result.regime_groups_evaluated}")
    print(f"Residual series evaluated: {result.residual_series_evaluated}")
    print(f"Results root             : {result.results_root}")
    print(f"Summary CSV              : {result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
