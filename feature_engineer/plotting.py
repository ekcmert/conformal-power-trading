from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
import numpy as np
import pandas as pd

from .common import coerce_numeric_frame, ensure_directory


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize_feature_correlations(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    X = coerce_numeric_frame(features)
    y = coerce_numeric_frame(targets)

    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    if X.empty or y.empty:
        return pd.DataFrame()

    # Constant columns produce invalid-correlation runtime warnings and do not add signal to the EDA.
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if X.empty:
        return pd.DataFrame()

    corr_matrix = pd.DataFrame(index=X.columns, dtype=float)
    n_matrix = pd.DataFrame(index=X.columns, dtype=float)

    for target_column in y.columns:
        target_series = y[target_column]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr_matrix[target_column] = X.corrwith(target_series)
        n_matrix[target_column] = X.notna().mul(target_series.notna(), axis=0).sum()

    corr_matrix = corr_matrix.dropna(how="all")
    if corr_matrix.empty:
        return pd.DataFrame()

    abs_corr = corr_matrix.abs()
    best_target = abs_corr.idxmax(axis=1)
    best_position = abs_corr.to_numpy().argmax(axis=1)

    row_positions = np.arange(len(corr_matrix.index))
    signed_best = corr_matrix.to_numpy()[row_positions, best_position]
    best_abs = abs_corr.to_numpy()[row_positions, best_position]
    best_n = n_matrix.reindex(corr_matrix.index).to_numpy()[row_positions, best_position]

    summary = pd.DataFrame(
        {
            "best_corr": signed_best,
            "best_abs_corr": best_abs,
            "best_target": best_target.to_numpy(),
            "n_observations": best_n,
        },
        index=corr_matrix.index,
    )

    return summary.sort_values("best_abs_corr", ascending=False)


def save_target_correlation_plot(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    output_path: str | Path,
    title: str,
    top_n: int = 40,
) -> pd.DataFrame:
    output_file = Path(output_path)
    ensure_directory(output_file.parent)

    summary = summarize_feature_correlations(features, targets)
    summary.to_csv(output_file.with_suffix(".csv"), index_label="feature")

    if summary.empty:
        return summary

    plot_df = summary.head(top_n).sort_values("best_corr")

    fig, ax = plt.subplots(figsize=(14, max(8, int(len(plot_df) * 0.35))))
    ax.barh(plot_df.index, plot_df["best_corr"], color="#0b7285")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Best Pearson correlation across target columns")
    ax.set_ylabel("Feature")

    x_offset = plot_df["best_corr"].abs().max() * 0.02 if not plot_df.empty else 0.01
    for row_index, (feature_name, row) in enumerate(plot_df.iterrows()):
        label = f"{row['best_target']} | n={int(row['n_observations'])}"
        x_position = row["best_corr"] + (x_offset if row["best_corr"] >= 0 else -x_offset)
        ha = "left" if row["best_corr"] >= 0 else "right"
        ax.text(x_position, row_index, label, va="center", ha=ha, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summary


__all__ = [
    "save_target_correlation_plot",
    "summarize_feature_correlations",
]
