from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from feature_engineer.common import (
    EDA_DIR,
    FINAL_DIR,
    align_and_drop_nulls,
    ensure_directory,
    load_merged_datasets,
    merge_feature_frames,
    save_dataframe,
)
from feature_engineer.instance import engineer_instance_features
from feature_engineer.ohlc import engineer_ohlc_features
from feature_engineer.plotting import save_target_correlation_plot
from feature_engineer.scenario import engineer_scenario_features
from feature_engineer.time_features import add_time_features_de
from feature_engineer.timeseries import build_target_frame, engineer_timeseries_features


@dataclass
class FinalizationResult:
    X: pd.DataFrame
    y: pd.DataFrame
    X_path: Path
    y_path: Path
    data_source: str
    dropped_feature_columns: list[str]
    dropped_target_columns: list[str]


def run_finalization_pipeline(
    cache_rebuilt_processed: bool = True,
) -> FinalizationResult:
    datasets, data_source = load_merged_datasets(cache_rebuilt=cache_rebuilt_processed)

    timeseries_frame = datasets["timeseries"]
    scenario_frame = datasets["scenario"]
    ohlc_frame = datasets["ohlc"]
    instance_frame = datasets["instance"]

    y = build_target_frame(timeseries_frame)

    feature_blocks = {
        "timeseries": engineer_timeseries_features(timeseries_frame),
        "scenario": engineer_scenario_features(scenario_frame),
        "instance": engineer_instance_features(instance_frame),
        "ohlc": engineer_ohlc_features(ohlc_frame, target_index=y.index),
    }

    eda_dir = ensure_directory(EDA_DIR)
    for block_name, block in feature_blocks.items():
        save_target_correlation_plot(
            features=block,
            targets=y,
            output_path=eda_dir / f"{block_name}_correlations.png",
            title=f"{block_name.title()} Features vs Targets",
            top_n=40,
        )

    X = merge_feature_frames(feature_blocks.values(), how="inner")
    X = add_time_features_de(X, drop_original_time_cols=False)
    X, y, dropped_feature_columns, dropped_target_columns = align_and_drop_nulls(X, y)

    if X.empty or y.empty:
        raise RuntimeError(
            "The final feature or target dataframe is empty after aligning indices and dropping nulls."
        )

    save_target_correlation_plot(
        features=X,
        targets=y,
        output_path=eda_dir / "final_correlations.png",
        title="Final Feature Set vs Targets",
        top_n=60,
    )

    final_dir = ensure_directory(FINAL_DIR)
    X_path = save_dataframe(X, final_dir / "X.csv", index_label="date")
    y_path = save_dataframe(y, final_dir / "y.csv", index_label="date")

    return FinalizationResult(
        X=X,
        y=y,
        X_path=X_path,
        y_path=y_path,
        data_source=data_source,
        dropped_feature_columns=dropped_feature_columns,
        dropped_target_columns=dropped_target_columns,
    )


def main() -> None:
    result = run_finalization_pipeline()

    print(f"Data source: {result.data_source}")
    print(f"X saved to: {result.X_path}")
    print(f"y saved to: {result.y_path}")
    print(f"X shape: {result.X.shape}")
    print(f"y shape: {result.y.shape}")

    if result.dropped_feature_columns:
        print(f"Dropped all-null feature columns: {len(result.dropped_feature_columns)}")
    if result.dropped_target_columns:
        print(f"Dropped all-null target columns: {len(result.dropped_target_columns)}")


if __name__ == "__main__":
    main()
