from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from data_preprocessor.data_prep import (
    DEFAULT_CUTOFF,
    DEFAULT_CURVES_DIR,
    preprocess_all_from_curves_folder,
)


INPUT_DIR = DEFAULT_CURVES_DIR
OUTPUT_DIR = INPUT_DIR.parent / "processed"
FILE_EXT = ".csv"
READ_CSV_KWARGS = {"sep": ","}
CUTOFF_UTC = DEFAULT_CUTOFF

MERGED_OUTPUT_FILENAMES = {
    "timeseries": "timeseries_data.csv",
    "scenario": "scenario_data.csv",
    "ohlc": "ohlc_data.csv",
    "instance": "instance_data.csv",
}


def save_merged_outputs(
    merged: dict[str, pd.DataFrame],
    output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    for name, dataframe in tqdm(
        merged.items(),
        desc="Saving processed data",
        unit="dataset",
        total=len(merged),
    ):
        if dataframe is None or dataframe.empty:
            print(f"[skip] merged {name}: no data")
            continue

        path = directory / MERGED_OUTPUT_FILENAMES[name]
        dataframe.to_csv(path, index_label=dataframe.index.name or "date")
        saved_paths[name] = path
        print(f"[saved] merged {name}: {path}")

    return saved_paths


def run_preprocess_pipeline(
    input_dir: str | Path = INPUT_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    cutoff_utc: pd.Timestamp | None = CUTOFF_UTC,
    ext: str = FILE_EXT,
    read_csv_kwargs: dict | None = None,
) -> dict[str, object]:
    results = preprocess_all_from_curves_folder(
        data_folder=input_dir,
        cutoff_utc=cutoff_utc,
        read_csv_kwargs=READ_CSV_KWARGS if read_csv_kwargs is None else read_csv_kwargs,
        ext=ext,
    )

    saved_paths = save_merged_outputs(results["merged"], output_dir=output_dir)
    results["saved_paths"] = saved_paths
    return results


def main() -> None:
    results = run_preprocess_pipeline()

    print("\nRemoved")
    for name, removed in results["removed"].items():
        print(f"{name}: {len(removed)}")

    print("\nMerged Shapes")
    for name, dataframe in results["merged"].items():
        shape = dataframe.shape if isinstance(dataframe, pd.DataFrame) else None
        print(f"{name}: {shape}")


if __name__ == "__main__":
    main()
