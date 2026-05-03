from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd


IssuePolicy = Literal["latest", "earliest"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BEGIN = "2020-01-01"
DEFAULT_END = "2026-04-20"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "curves"


@dataclass(frozen=True)
class RelativeInstanceSettings:
    days_ahead: int = 1
    before_time_of_day: time = time(12, 0)
    issued: IssuePolicy = "latest"
    frequency: Any = None


def curve_name_to_filename(
    name: str,
    ext: str = ".csv",
    max_len: int = 180,
) -> str:
    value = name.strip().lower()
    value = value.replace(">", "_to_")
    value = value.replace("&", "_and_")
    value = value.replace("/", "_per_")
    value = re.sub(r"[^a-z0-9._ -]+", "", value)
    value = re.sub(r"[ .-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")

    suffix = ext if ext.startswith(".") else f".{ext}"
    if max_len and len(value) + len(suffix) > max_len:
        value = value[: max_len - len(suffix)].rstrip("_")

    return f"{value}{suffix}"


def curve_name_to_path(
    name: str,
    folder: str | Path = DEFAULT_OUTPUT_DIR,
    ext: str = ".csv",
    prefix: str | None = None,
) -> Path:
    directory = Path(folder)
    directory.mkdir(parents=True, exist_ok=True)

    filename = curve_name_to_filename(name, ext=ext)
    if prefix:
        safe_prefix = re.sub(r"[^a-zA-Z0-9._-]+", "_", prefix.strip()).strip("_")
        if safe_prefix:
            filename = f"{safe_prefix}_{filename}"

    return directory / filename


def save_dataframe(
    dataframe: pd.DataFrame,
    path: str | Path,
    **write_kwargs: Any,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        dataframe.to_csv(output_path, index=True, **write_kwargs)
        return output_path
    if output_path.suffix.lower() == ".parquet":
        dataframe.to_parquet(output_path, index=True, **write_kwargs)
        return output_path

    raise ValueError("Unsupported export format. Use '.csv' or '.parquet'.")


def save_curve_dataframe(
    curve_name: str,
    dataframe: pd.DataFrame | None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ext: str = ".csv",
    prefix: str | None = None,
    overwrite: bool = True,
    **write_kwargs: Any,
) -> Path | None:
    if dataframe is None or dataframe.empty:
        return None

    output_path = curve_name_to_path(
        name=curve_name,
        folder=output_dir,
        ext=ext,
        prefix=prefix,
    )

    if output_path.exists() and not overwrite:
        return output_path

    return save_dataframe(dataframe=dataframe, path=output_path, **write_kwargs)


__all__ = [
    "DEFAULT_BEGIN",
    "DEFAULT_END",
    "DEFAULT_OUTPUT_DIR",
    "REPO_ROOT",
    "RelativeInstanceSettings",
    "curve_name_to_filename",
    "curve_name_to_path",
    "save_curve_dataframe",
    "save_dataframe",
]
