from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
X_PATH = REPO_ROOT / "data" / "final" / "X.csv"
SCALES_ROOT = REPO_ROOT / "data" / "scales"
OUTPUT_FILENAME = "scales.csv"
TARGET_COLUMN_NAME = "target_column"

# Populate this mapping or call generate_scales(...) directly from another module.
COLUMN_FOLDER_MAP: dict[str, str] = {
"DE Residual Load MWh/h 15min Forecast__std" : "res_load_fc",
"DE Residual Load MWh/h 15min Climate__std" : "res_load_cli",
"DE Wind Power Production MWh/h 15min Forecast__std" : "wind_fc",
"DE Wind Power Production MWh/h 15min Climate__std" : "wind_cli",
"DE Solar Photovoltaic Production MWh/h 15min Forecast__std" : "solar_fc",
"DE Solar Photovoltaic Production MWh/h 15min Climate__std" : "solar_cli",
"DE Consumption MWh/h 15min Forecast__std" : "cons_fc",
"DE Consumption MWh/h 15min Climate__std" : "cons_cli",
"DE Consumption Temperature °C 15min Forecast__std" : "cons_temp_fc"
}


@dataclass(frozen=True)
class ScaleGenerationSummary:
    folders_written: int
    files_written: int
    columns_extracted: int
    row_count: int


def _load_x_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, parse_dates=["date"])
    if "date" not in frame.columns:
        raise ValueError(f"Expected a date column in {csv_path}.")
    return frame.sort_values("date")


def _normalize_mapping(column_folder_map: Mapping[str, str]) -> dict[str, str]:
    if not column_folder_map:
        raise ValueError(
            "column_folder_map is empty. Populate COLUMN_FOLDER_MAP or pass a mapping to generate_scales(...)."
        )

    normalized_mapping: dict[str, str] = {}
    seen_folders: dict[str, str] = {}

    for column_name, folder_name in column_folder_map.items():
        normalized_folder_name = str(folder_name).strip()
        if not normalized_folder_name:
            raise ValueError(f"Folder name for column {column_name!r} cannot be empty.")

        existing_column = seen_folders.get(normalized_folder_name)
        if existing_column is not None and existing_column != column_name:
            raise ValueError(
                "Each folder_name must map to exactly one X.csv column. "
                f"Folder {normalized_folder_name!r} was assigned to both "
                f"{existing_column!r} and {column_name!r}."
            )

        normalized_mapping[str(column_name)] = normalized_folder_name
        seen_folders[normalized_folder_name] = str(column_name)

    return normalized_mapping


def _resolve_output_dir(scales_root: Path, folder_name: str) -> Path:
    base_dir = scales_root.resolve()
    output_dir = (base_dir / folder_name).resolve()

    if output_dir != base_dir and base_dir not in output_dir.parents:
        raise ValueError(
            f"Resolved output directory escaped scales_root: {output_dir} not under {base_dir}"
        )

    return output_dir


def generate_scales(
    column_folder_map: Mapping[str, str] | None = None,
    *,
    x_path: Path = X_PATH,
    scales_root: Path = SCALES_ROOT,
    output_filename: str = OUTPUT_FILENAME,
) -> ScaleGenerationSummary:
    if column_folder_map is None:
        column_folder_map = COLUMN_FOLDER_MAP

    normalized_mapping = _normalize_mapping(column_folder_map)
    x_frame = _load_x_frame(x_path.resolve())

    missing_features = [column for column in normalized_mapping if column not in x_frame.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns in X.csv: {missing_features}")

    scales_root = scales_root.resolve()
    files_written = 0

    for column_name, folder_name in normalized_mapping.items():
        output_dir = _resolve_output_dir(scales_root, folder_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        export_frame = (
            x_frame.loc[:, ["date", column_name]]
            .rename(columns={column_name: TARGET_COLUMN_NAME})
            .copy()
        )
        export_frame.to_csv(output_dir / output_filename, index=False)
        files_written += 1

    return ScaleGenerationSummary(
        folders_written=len(normalized_mapping),
        files_written=files_written,
        columns_extracted=len(normalized_mapping),
        row_count=len(x_frame),
    )


def main() -> None:
    try:
        summary = generate_scales()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Folders written  : {summary.folders_written}")
    print(f"Files written    : {summary.files_written}")
    print(f"Columns extracted: {summary.columns_extracted}")
    print(f"Rows per export  : {summary.row_count}")


if __name__ == "__main__":
    main()
