from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_ROOT = REPO_ROOT / "data" / "predictions"
RESIDUALS_ROOT = REPO_ROOT / "data" / "residuals"
Y_PATH = REPO_ROOT / "data" / "final" / "y.csv"

TARGET_RESIDUAL_STEMS = {
    "DE Price Spot EUR/MWh EPEX H Actual": "da_res",
    "DE Price Intraday VWAP EUR/MWh EPEX H Actual": "id_res",
    "DE Price Intraday VWAP ID1 EUR/MWh EPEX H Actual": "id1_res",
    "DE Price Intraday VWAP ID3 EUR/MWh EPEX H Actual": "id3_res",
    "DE Volume Imbalance Net MWh 15min Actual": "imb_res",
}
PREDICTION_SUFFIX = "_predictions.csv"


@dataclass(frozen=True)
class ResidualGenerationSummary:
    model_loss_directories: int
    point_files_written: int
    interval_files_written: int
    skipped_prediction_files: int


def _target_artifact_stem(target_name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]+', "_", target_name).strip()
    return re.sub(r"\s+", " ", sanitized)


def _load_y_frame(y_path: Path) -> pd.DataFrame:
    y_frame = pd.read_csv(y_path, parse_dates=["date"])
    if "date" not in y_frame.columns:
        raise ValueError(f"Expected a date column in {y_path}.")

    missing_targets = [target for target in TARGET_RESIDUAL_STEMS if target not in y_frame.columns]
    if missing_targets:
        raise KeyError(f"Missing target columns in {y_path}: {missing_targets}")

    return y_frame.set_index("date").sort_index()


def _prediction_model_directories(predictions_root: Path) -> list[Path]:
    return sorted({path.parent for path in predictions_root.rglob(f"*{PREDICTION_SUFFIX}")})


def _load_estimation_kind(model_dir: Path) -> str:
    metadata_path = model_dir / "run_metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text())
        estimation_kind = str(payload.get("estimation_kind", "")).strip().lower()
        if estimation_kind in {"point", "interval"}:
            return estimation_kind

    prediction_files = sorted(model_dir.glob(f"*{PREDICTION_SUFFIX}"))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found in {model_dir}.")

    sample_columns = pd.read_csv(prediction_files[0], nrows=0).columns
    if "y_pred" in sample_columns:
        return "point"
    if {"y_pred_lower", "y_pred_upper"}.issubset(sample_columns):
        return "interval"

    raise ValueError(
        f"Unable to infer estimation_kind for {model_dir}. "
        "Expected point columns (y_pred) or interval columns (y_pred_lower, y_pred_upper)."
    )


def _combined_prediction_files(model_dir: Path) -> list[Path]:
    prediction_files = sorted(model_dir.glob(f"*{PREDICTION_SUFFIX}"))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found in {model_dir}.")
    if len(prediction_files) > 1:
        raise ValueError(
            f"Expected a single combined prediction file in {model_dir}, "
            f"found {len(prediction_files)}: {[path.name for path in prediction_files]}"
        )
    return prediction_files


def _load_prediction_frame(prediction_path: Path) -> pd.DataFrame:
    prediction_frame = pd.read_csv(prediction_path, parse_dates=["date"])
    if "date" not in prediction_frame.columns:
        raise ValueError(f"Expected a date column in {prediction_path}.")
    return prediction_frame


def _align_target_values(
    prediction_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    *,
    target_name: str,
) -> tuple[pd.Series, pd.Series]:
    indexed_predictions = prediction_frame.set_index("date")
    target_values = y_frame[target_name].reindex(indexed_predictions.index)

    if target_values.isna().any():
        missing_dates = target_values[target_values.isna()].index.astype(str).tolist()[:5]
        raise KeyError(
            f"Missing y.csv values for target {target_name} on dates {missing_dates} "
            f"while processing predictions."
        )

    return indexed_predictions.index.to_series(index=indexed_predictions.index), target_values.astype(float)


def _build_point_residual_frame(
    prediction_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    *,
    target_name: str,
    residual_name: str,
) -> pd.DataFrame:
    if "y_pred" not in prediction_frame.columns:
        raise ValueError("Point prediction file must contain a y_pred column.")

    dates, y_true = _align_target_values(prediction_frame, y_frame, target_name=target_name)
    residuals = y_true - prediction_frame.set_index("date")["y_pred"].astype(float)

    return pd.DataFrame(
        {
            "date": dates.to_numpy(),
            residual_name: residuals.to_numpy(),
        }
    )


def _build_interval_residual_frames(
    prediction_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    *,
    target_name: str,
    residual_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {"y_pred_lower", "y_pred_upper"}
    if not required_columns.issubset(prediction_frame.columns):
        raise ValueError(
            "Interval prediction file must contain y_pred_lower and y_pred_upper columns."
        )

    indexed_predictions = prediction_frame.set_index("date")
    dates, y_true = _align_target_values(prediction_frame, y_frame, target_name=target_name)

    lower_residuals = y_true - indexed_predictions["y_pred_lower"].astype(float)
    upper_residuals = y_true - indexed_predictions["y_pred_upper"].astype(float)

    residual_low = pd.DataFrame(
        {
            "date": dates.to_numpy(),
            f"{residual_name}_low": lower_residuals.to_numpy(),
        }
    )
    residual_up = pd.DataFrame(
        {
            "date": dates.to_numpy(),
            f"{residual_name}_up": upper_residuals.to_numpy(),
        }
    )
    return residual_low, residual_up


def generate_residuals(
    *,
    predictions_root: Path = PREDICTIONS_ROOT,
    residuals_root: Path = RESIDUALS_ROOT,
    y_path: Path = Y_PATH,
) -> ResidualGenerationSummary:
    predictions_root = predictions_root.resolve()
    residuals_root = residuals_root.resolve()
    y_path = y_path.resolve()

    y_frame = _load_y_frame(y_path)
    model_directories = _prediction_model_directories(predictions_root)
    point_files_written = 0
    interval_files_written = 0
    skipped_prediction_files = 0

    for model_dir in model_directories:
        estimation_kind = _load_estimation_kind(model_dir)
        output_dir = residuals_root / model_dir.relative_to(predictions_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        for prediction_path in _combined_prediction_files(model_dir):
            prediction_frame = _load_prediction_frame(prediction_path)

            if estimation_kind == "point":
                for target_name, residual_name in TARGET_RESIDUAL_STEMS.items():
                    residual_frame = _build_point_residual_frame(
                        prediction_frame,
                        y_frame,
                        target_name=target_name,
                        residual_name=residual_name,
                    )
                    residual_frame.to_csv(output_dir / f"{residual_name}.csv", index=False)
                    point_files_written += 1
                continue

            for target_name, residual_name in TARGET_RESIDUAL_STEMS.items():
                residual_low, residual_up = _build_interval_residual_frames(
                    prediction_frame,
                    y_frame,
                    target_name=target_name,
                    residual_name=residual_name,
                )
                residual_low.to_csv(output_dir / f"{residual_name}_low.csv", index=False)
                residual_up.to_csv(output_dir / f"{residual_name}_up.csv", index=False)
                interval_files_written += 2

    return ResidualGenerationSummary(
        model_loss_directories=len(model_directories),
        point_files_written=point_files_written,
        interval_files_written=interval_files_written,
        skipped_prediction_files=skipped_prediction_files,
    )


def main() -> None:
    summary = generate_residuals()
    print(f"Processed model-loss directories : {summary.model_loss_directories}")
    print(f"Point residual files written     : {summary.point_files_written}")
    print(f"Interval residual files written  : {summary.interval_files_written}")
    print(f"Skipped prediction files         : {summary.skipped_prediction_files}")


if __name__ == "__main__":
    main()
