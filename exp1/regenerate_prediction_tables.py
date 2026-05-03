from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_ROOT = REPO_ROOT / "data" / "predictions"
OUTPUT_DIR = REPO_ROOT / "results" / "exp1_full"
PREDICTIONS_FILENAME = "DE Price Spot EUR_MWh EPEX H Actual_predictions.csv"

YEARS = {2025, 2026}
LOWER_QUANTILE = 0.05
UPPER_QUANTILE = 0.95


@dataclass(frozen=True)
class ModelSpec:
    model: str
    loss: str
    model_slug: str
    loss_slug: str


POINT_MODELS = [
    ModelSpec("LightGBM", "regression-l1", "lightgbm", "regression_l1"),
    ModelSpec("XGBoost", "squared-error", "xgboost", "squared_error"),
    ModelSpec("LightGBM", "fair", "lightgbm", "fair"),
    ModelSpec("HistGradientBoosting", "squared-error", "hist_gradient_boosting", "squared_error"),
    ModelSpec("LightGBM", "quantile-p50", "lightgbm", "quantile_p50"),
    ModelSpec("CatBoost", "rmse", "catboost", "rmse"),
    ModelSpec("LightGBM", "regression", "lightgbm", "regression"),
    ModelSpec("HistGradientBoosting", "absolute-error", "hist_gradient_boosting", "absolute_error"),
]

INTERVAL_MODELS = [
    ModelSpec("Quantile ET", "squared-error", "quantile_extra_trees", "squared_error"),
    ModelSpec("Quantile RF", "squared-error", "quantile_random_forest", "squared_error"),
    ModelSpec("QuantileRegressor", "pinball loss", "quantile_regressor", "pinball_loss"),
    ModelSpec("LightGBM", "quantile", "lightgbm", "quantile"),
]


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def prediction_path(kind: str, spec: ModelSpec) -> Path:
    return PREDICTIONS_ROOT / kind / spec.model_slug / spec.loss_slug / PREDICTIONS_FILENAME


def filtered_rows(path: Path) -> Iterable[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            date = parse_datetime(row["date"])
            if date.year in YEARS:
                yield row


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    mean_y = sum(y_true) / len(y_true)
    sse = sum((pred - actual) ** 2 for actual, pred in zip(y_true, y_pred))
    sst = sum((actual - mean_y) ** 2 for actual in y_true)
    return float("nan") if sst == 0 else 1.0 - sse / sst


def pinball_loss(actual: float, prediction: float, quantile: float) -> float:
    error = actual - prediction
    return max(quantile * error, (quantile - 1.0) * error)


def competition_ranks(rows: list[dict[str, object]], metric: str, higher_is_better: bool = False) -> list[int]:
    order = sorted(
        range(len(rows)),
        key=lambda index: rows[index][metric],
        reverse=higher_is_better,
    )
    ranks = [0] * len(rows)
    previous_value: object | None = None
    current_rank = 0

    for position, index in enumerate(order, start=1):
        value = rows[index][metric]
        if previous_value is None or value != previous_value:
            current_rank = position
            previous_value = value
        ranks[index] = current_rank

    return ranks


def add_ranks(rows: list[dict[str, object]], rank_specs: list[tuple[str, str, bool]]) -> None:
    for metric, rank_name, higher_is_better in rank_specs:
        ranks = competition_ranks(rows, metric, higher_is_better=higher_is_better)
        for row, rank in zip(rows, ranks):
            row[rank_name] = rank

    rank_names = [rank_name for _, rank_name, _ in rank_specs]
    for row in rows:
        row["avg_rank"] = sum(int(row[rank_name]) for rank_name in rank_names) / len(rank_names)


def point_metrics(spec: ModelSpec) -> dict[str, object]:
    path = prediction_path("point_estimation", spec)
    dates: list[datetime] = []
    y_true: list[float] = []
    y_pred: list[float] = []

    for row in filtered_rows(path):
        actual = parse_float(row.get("y_true"))
        prediction = parse_float(row.get("y_pred"))
        if actual is None or prediction is None:
            continue
        dates.append(parse_datetime(row["date"]))
        y_true.append(actual)
        y_pred.append(prediction)

    if not y_true:
        raise ValueError(f"No usable rows after filtering {path}")

    errors = [prediction - actual for actual, prediction in zip(y_true, y_pred)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error**2 for error in errors) / len(errors))

    return {
        "model": spec.model,
        "loss": spec.loss,
        "rmse": rmse,
        "mae": mae,
        "r2": r2_score(y_true, y_pred),
        "rows": len(y_true),
        "date_start": min(dates).isoformat(),
        "date_end": max(dates).isoformat(),
    }


def interval_metrics(spec: ModelSpec) -> dict[str, object]:
    path = prediction_path("interval_estimation", spec)
    dates: list[datetime] = []
    pinball_losses: list[float] = []
    widths: list[float] = []
    coverages: list[float] = []

    for row in filtered_rows(path):
        actual = parse_float(row.get("y_true"))
        lower = parse_float(row.get("y_pred_lower"))
        upper = parse_float(row.get("y_pred_upper"))
        if actual is None or lower is None or upper is None:
            continue

        width = parse_float(row.get("interval_width"))
        covered = parse_float(row.get("covered"))
        if width is None:
            width = upper - lower
        if covered is None:
            covered = 1.0 if lower <= actual <= upper else 0.0

        dates.append(parse_datetime(row["date"]))
        pinball_losses.append(
            (
                pinball_loss(actual, lower, LOWER_QUANTILE)
                + pinball_loss(actual, upper, UPPER_QUANTILE)
            )
            / 2.0
        )
        widths.append(width)
        coverages.append(covered)

    if not pinball_losses:
        raise ValueError(f"No usable rows after filtering {path}")

    return {
        "model": spec.model,
        "loss": spec.loss,
        "mean_pinball_loss": sum(pinball_losses) / len(pinball_losses),
        "mean_interval_width": sum(widths) / len(widths),
        "empirical_coverage": sum(coverages) / len(coverages),
        "rows": len(pinball_losses),
        "date_start": min(dates).isoformat(),
        "date_end": max(dates).isoformat(),
    }


def format_row(row: dict[str, object], field_map: list[tuple[str, str, str]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for source, destination, format_spec in field_map:
        value = row[source]
        if format_spec:
            output[destination] = format(value, format_spec)
        else:
            output[destination] = value
    return output


def write_point_table(rows: list[dict[str, object]], path: Path) -> None:
    rows.sort(key=lambda row: (float(row["avg_rank"]), str(row["model"]), str(row["loss"])))
    field_map = [
        ("model", "Model", ""),
        ("loss", "Loss", ""),
        ("rmse_rank", "RMSE Rank", ""),
        ("rmse", "RMSE Result", ".3f"),
        ("mae_rank", "MAE Rank", ""),
        ("mae", "MAE Result", ".3f"),
        ("r2_rank", "R2 Rank", ""),
        ("r2", "R2 Result", ".3f"),
        ("avg_rank", "Avg. Rank", ".2f"),
    ]
    write_table(rows, path, field_map)


def write_interval_table(rows: list[dict[str, object]], path: Path) -> None:
    rows.sort(key=lambda row: (float(row["avg_rank"]), str(row["model"]), str(row["loss"])))
    field_map = [
        ("model", "Model", ""),
        ("loss", "Loss", ""),
        ("pinball_rank", "Mean Pinball Loss Rank", ""),
        ("mean_pinball_loss", "Mean Pinball Loss Result", ".3f"),
        ("width_rank", "Mean Interval Width Rank", ""),
        ("mean_interval_width", "Mean Interval Width Result", ".3f"),
        ("coverage_rank", "Empirical Coverage Rank", ""),
        ("empirical_coverage", "Empirical Coverage Result", ".3f"),
        ("avg_rank", "Avg. Rank", ".2f"),
    ]
    write_table(rows, path, field_map)


def write_table(rows: list[dict[str, object]], path: Path, field_map: list[tuple[str, str, str]]) -> None:
    fieldnames = [destination for _, destination, _ in field_map]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_row(row, field_map))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    point_rows = [point_metrics(spec) for spec in POINT_MODELS]
    add_ranks(
        point_rows,
        [
            ("rmse", "rmse_rank", False),
            ("mae", "mae_rank", False),
            ("r2", "r2_rank", True),
        ],
    )

    interval_rows = [interval_metrics(spec) for spec in INTERVAL_MODELS]
    add_ranks(
        interval_rows,
        [
            ("mean_pinball_loss", "pinball_rank", False),
            ("mean_interval_width", "width_rank", False),
            ("empirical_coverage", "coverage_rank", True),
        ],
    )

    point_path = OUTPUT_DIR / "point_estimation_table.csv"
    interval_path = OUTPUT_DIR / "interval_estimation_table.csv"
    write_point_table(point_rows, point_path)
    write_interval_table(interval_rows, interval_path)

    print(f"Wrote {point_path}")
    print(f"Wrote {interval_path}")
    print(f"Filtered years: {', '.join(str(year) for year in sorted(YEARS))}")
    print(f"Point rows per model: {point_rows[0]['rows']}")
    print(f"Interval rows per model: {interval_rows[0]['rows']}")


if __name__ == "__main__":
    main()
