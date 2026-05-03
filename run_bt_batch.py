from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.config import REGIME_AWARE_RESULTS_ROOT, REGIME_FREE_RESULTS_ROOT, Y_PATH
from backtest.io import infer_output_dir, load_prediction_frame, load_price_frame, resolve_prediction_csv
from bt import optimize_strategy


REPO_ROOT = Path(__file__).resolve().parent
BACKTEST_BATCH_RESULTS_ROOT = REPO_ROOT / "results" / "backtest_batch"

RUN_BACKTEST_BATCH = True
STOP_ON_ERROR = False
MAX_TRIALS_PER_STAGE: int | None = None
RANDOM_SEED = optimize_strategy.DEFAULT_RANDOM_SEED

BACKTEST_PARAMS = {
    "position_method": "interval_band",
    "entry_band_fraction": 0.3,
    "exit_band_fraction": 0.75,
    "signal_family": "softsign",
    "signal_scale": 1.5,
    "signal_power": 0.5,
    "position_cap": 60.0,
    "min_interval_width": 1e-6,
}

OBJECTIVE_WEIGHTS = {
    "unit_pnl": 0.35,
    "hit_ratio": 0.35,
    "mean_over_std": 0.10,
    "drawdown": 0.10,
    "cumulative_pnl": 0.10,
}

GRID_STAGES = copy.deepcopy(optimize_strategy.GRID_STAGES)

# MODEL_METHOD_PATH_SPECS = [
#     ("enbpi", "", "enbpi_asymmetric_lightgbm", "regression_l1"),
#     ("enbpi", "", "enbpi_symmetric_lightgbm", "regression_l1"),
#     ("mcp", "kmeans_pca_6", "cqr_quantile_extra_trees", "squared_error"),
#     ("mcp", "kmedoids_6", "cqr_quantile_extra_trees", "squared_error"),
#     ("mcp", "hmm_8", "cqr_quantile_extra_trees", "squared_error"),
#     ("mcp", "kmeans_6", "cqr_quantile_extra_trees", "squared_error"),
#     ("mcp", "kmedoids_pca_6", "cqr_quantile_extra_trees", "squared_error"),
#     ("mcp", "agglomerative_pca_6", "cqr_quantile_extra_trees", "squared_error"),
#     ("mcp", "divisive_6", "cqr_quantile_extra_trees", "squared_error"),
#     ("aci", "0.1", "cqr_quantile_extra_trees", "squared_error"),
#     ("aci", "0.05", "cqr_lightgbm", "quantile"),
#     ("lacp", "cons_cli", "cqr_quantile_extra_trees", "squared_error"),
#     ("lacp", "res_load_cli", "cqr_quantile_extra_trees", "squared_error"),
#     ("lacp", "wind_cli", "cqr_quantile_extra_trees", "squared_error"),
#     ("lacp", "cons_temp_fc", "cqr_quantile_extra_trees", "squared_error"),
#     ("aci", "0.1", "cqr_lightgbm", "quantile"),
# ]

MODEL_METHOD_PATH_SPECS = [
    ("mcp", "kmeans_6", "cp_symmetric_lightgbm", "regression_l1"),
    ("aci", "0.05", "cp_asymmetric_lightgbm", "regression_l1"),
    ("lacp", "res_load_cli", "cp_symmetric_hist_gradient_boosting", "absolute_error"),
]

SUMMARY_SCENARIOS = [
    "strategy_close_id",
    "strategy_close_id3",
    "strategy_close_id1",
    "strategy_close_imb",
]

SUMMARY_COLUMN_RENAMES = {
    "display_name": "Scenario",
    "avg_hourly_pnl": "Avg Hourly PnL",
    "avg_unit_pnl": "Unit PnL (EUR/MWh)",
    "cumulative_pnl": "Cumulative PnL",
    "hit_ratio": "Hit Ratio",
    "pnl_volatility": "PnL Volatility",
    "pnl_mean_over_std": "Mean / Std",
    "max_drawdown": "Max Drawdown",
    "avg_abs_position_size": "Avg |Position|",
}


@dataclass(frozen=True)
class BatchRunResult:
    source_path: Path
    prediction_csv: Path
    optimization_dir: Path
    best_run_dir: Path
    best_params: dict[str, object]
    best_row: dict[str, object]
    best_summary: pd.DataFrame


def regime_aware_path(
    method: str,
    method_setting: str,
    model_name: str,
    objective_name: str,
    *,
    market: str = "da",
) -> Path:
    return REGIME_AWARE_RESULTS_ROOT / method / method_setting / market / model_name / objective_name


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(resolved)
    return unique_paths


PRED_PATHS = dedupe_paths(
    [
        *[
            regime_aware_path(method, method_setting, model_name, objective_name)
            for method, method_setting, model_name, objective_name in MODEL_METHOD_PATH_SPECS
        ],
    ]
)


def safe_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [safe_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): safe_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_json_value(item) for item in value]
    return value


def describe_prediction_path(path: Path) -> dict[str, str]:
    source_dir = path.parent if path.is_file() else path
    resolved_source_dir = source_dir.expanduser().resolve()

    try:
        parts = resolved_source_dir.relative_to(REGIME_AWARE_RESULTS_ROOT.resolve()).parts
    except ValueError:
        parts = ()
    if len(parts) >= 5:
        return {
            "source_family": "regime_aware",
            "method": parts[0],
            "method_setting": parts[1],
            "market": parts[2],
            "model": parts[3],
            "objective": parts[4],
        }
    if len(parts) >= 4:
        return {
            "source_family": "regime_aware",
            "method": parts[0],
            "method_setting": "",
            "market": parts[1],
            "model": parts[2],
            "objective": parts[3],
        }

    try:
        parts = resolved_source_dir.relative_to(REGIME_FREE_RESULTS_ROOT.resolve()).parts
    except ValueError:
        parts = ()
    if len(parts) >= 3:
        return {
            "source_family": "regime_free",
            "method": "regime_free",
            "method_setting": "",
            "market": parts[0],
            "model": parts[1],
            "objective": parts[2],
        }

    return {
        "source_family": "",
        "method": "",
        "method_setting": "",
        "market": "",
        "model": "",
        "objective": "",
    }


def run_optimizer_for_path(
    *,
    pred_path: Path,
    y_path: Path,
    output_root: Path,
    base_params: dict[str, object],
    objective_weights: dict[str, float],
    grid_stages: list[dict[str, object]],
    max_trials: int | None,
    random_seed: int,
) -> BatchRunResult:
    run_start = time.perf_counter()
    resolved_base_params = dict(optimize_strategy.BASE_PARAMS)
    resolved_base_params.update(base_params)
    resolved_objective_weights = optimize_strategy.normalize_objective_weights(objective_weights)
    rng = np.random.default_rng(random_seed)

    prediction_csv = resolve_prediction_csv(pred_path)
    prediction_frame = load_prediction_frame(prediction_csv)
    price_frame = load_price_frame(y_path)

    optimization_dir = infer_output_dir(prediction_csv, results_root=output_root) / "optimization"
    optimization_dir.mkdir(parents=True, exist_ok=True)

    print(f"Prediction CSV: {prediction_csv}")
    print(f"Optimization dir: {optimization_dir}")
    print(f"Max trials per stage: {max_trials if max_trials is not None else 'all'}")
    print(f"Random seed: {random_seed}")
    print(f"Objective weights: {resolved_objective_weights}")

    seen_keys: set[tuple[object, ...]] = set()
    stage_frames: list[pd.DataFrame] = []
    current_best_params = dict(resolved_base_params)

    for stage in grid_stages:
        stage_name = str(stage["stage_name"])
        stage_results, current_best_params = optimize_strategy.run_stage(
            stage_name=stage_name,
            grid=dict(stage["grid"]),
            base_params=current_best_params,
            max_trials=max_trials,
            pred_path=pred_path,
            y_path=y_path,
            prediction_frame=prediction_frame,
            price_frame=price_frame,
            seen_keys=seen_keys,
            rng=rng,
            objective_weights=resolved_objective_weights,
        )
        stage_results.to_csv(optimization_dir / f"{stage_name}_results.csv", index=False)
        stage_frames.append(stage_results)

        best_row = stage_results.iloc[0]
        print(
            f"[{stage_name}] best score={best_row['stage_objective_score']:.4f}, "
            f"weighted_unit_pnl={best_row['weighted_avg_unit_pnl']:.4f}, "
            f"weighted_hit_ratio={best_row['weighted_avg_hit_ratio']:.4f}, "
            f"weighted_mean_std={best_row['weighted_avg_pnl_mean_over_std']:.4f}, "
            f"weighted_dd_per_mwh={best_row['weighted_avg_max_drawdown_per_active_mwh']:.4f}, "
            f"weighted_cum_pnl={best_row['weighted_avg_cumulative_pnl']:.4f}, "
            f"priority_min_cum_pnl={best_row['priority_min_cumulative_pnl']:.4f}"
        )

    refine_stage = optimize_strategy.build_refine_stage(current_best_params)
    refine_results, current_best_params = optimize_strategy.run_stage(
        stage_name=str(refine_stage["stage_name"]),
        grid=dict(refine_stage["grid"]),
        base_params=current_best_params,
        max_trials=max_trials,
        pred_path=pred_path,
        y_path=y_path,
        prediction_frame=prediction_frame,
        price_frame=price_frame,
        seen_keys=seen_keys,
        rng=rng,
        objective_weights=resolved_objective_weights,
        allow_empty=True,
    )
    if not refine_results.empty:
        refine_results.to_csv(optimization_dir / f"{refine_stage['stage_name']}_results.csv", index=False)
        stage_frames.append(refine_results)

    all_trials = optimize_strategy.add_objective_scores(
        pd.concat(stage_frames, ignore_index=True),
        resolved_objective_weights,
        score_prefix="global",
        objective_column="global_objective_score",
    )
    all_trials = all_trials.sort_values(
        [
            "global_objective_score",
            "weighted_avg_unit_pnl",
            "weighted_avg_hit_ratio",
            "weighted_avg_pnl_mean_over_std",
            "weighted_avg_max_drawdown_per_active_mwh",
            "weighted_avg_cumulative_pnl",
            "priority_min_cumulative_pnl",
        ],
        ascending=[False, False, False, False, True, False, False],
    ).reset_index(drop=True)
    all_trials.insert(0, "global_rank", np.arange(1, len(all_trials) + 1))
    all_trials.to_csv(optimization_dir / "all_trials.csv", index=False)

    best_row = all_trials.iloc[0].to_dict()
    best_params = {
        "position_method": "interval_band",
        "entry_band_fraction": float(best_row["entry_band_fraction"]),
        "exit_band_fraction": float(best_row["exit_band_fraction"]),
        "signal_family": str(best_row["signal_family"]),
        "signal_scale": float(best_row["signal_scale"]),
        "signal_power": float(best_row["signal_power"]),
        "position_cap": float(best_row["position_cap"]),
        "min_interval_width": float(resolved_base_params["min_interval_width"]),
    }

    best_artifacts = optimize_strategy.run_best_backtest(
        pred_path=pred_path,
        y_path=y_path,
        output_dir=optimization_dir,
        best_params=best_params,
    )

    elapsed_seconds = time.perf_counter() - run_start
    metadata = {
        "prediction_csv": str(prediction_csv),
        "optimization_dir": str(optimization_dir),
        "batch_results_root": str(output_root),
        "base_params": resolved_base_params,
        "max_trials_per_stage": max_trials,
        "random_seed": random_seed,
        "grid_stages": grid_stages + [refine_stage],
        "real_strategy_scenarios": optimize_strategy.REAL_STRATEGY_SCENARIOS,
        "summary_scenarios": SUMMARY_SCENARIOS,
        "optimization_scenario_weights": optimize_strategy.OPTIMIZATION_SCENARIO_WEIGHTS,
        "priority_scenarios": optimize_strategy.PRIORITY_SCENARIOS,
        "global_scoring": {
            "weighted_avg_unit_pnl": "maximize",
            "weighted_avg_hit_ratio": "maximize",
            "weighted_avg_pnl_mean_over_std": "maximize",
            "weighted_avg_max_drawdown_per_active_mwh": "minimize",
            "weighted_avg_cumulative_pnl": "maximize",
            "objective_weights": resolved_objective_weights,
            "aggregation": "weighted percentile-rank average across enabled objectives",
        },
        "best_params": best_params,
        "best_result": {
            "global_objective_score": float(best_row["global_objective_score"]),
            "weighted_avg_unit_pnl": float(best_row["weighted_avg_unit_pnl"]),
            "weighted_avg_hit_ratio": float(best_row["weighted_avg_hit_ratio"]),
            "weighted_avg_pnl_mean_over_std": float(best_row["weighted_avg_pnl_mean_over_std"]),
            "weighted_avg_max_drawdown_per_active_mwh": float(
                best_row["weighted_avg_max_drawdown_per_active_mwh"]
            ),
            "weighted_avg_cumulative_pnl": float(best_row["weighted_avg_cumulative_pnl"]),
            "priority_min_cumulative_pnl": float(best_row["priority_min_cumulative_pnl"]),
        },
        "best_backtest_artifacts": best_artifacts,
        "elapsed_seconds": elapsed_seconds,
    }
    (optimization_dir / "optimization_metadata.json").write_text(
        json.dumps(safe_json_value(metadata), indent=2),
        encoding="utf-8",
    )

    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"All trials CSV : {optimization_dir / 'all_trials.csv'}")
    print(f"Best run dir   : {best_artifacts['output_dir']}")

    return BatchRunResult(
        source_path=Path(pred_path).expanduser().resolve(),
        prediction_csv=prediction_csv,
        optimization_dir=optimization_dir,
        best_run_dir=Path(best_artifacts["output_dir"]),
        best_params=best_params,
        best_row=best_row,
        best_summary=pd.read_csv(best_artifacts["summary_path"]),
    )


def build_batch_summary_rows(result: BatchRunResult) -> list[dict[str, object]]:
    descriptor = describe_prediction_path(result.source_path)
    summary = result.best_summary.copy()
    if SUMMARY_SCENARIOS:
        summary = summary[summary["scenario_name"].isin(SUMMARY_SCENARIOS)].copy()

    rows: list[dict[str, object]] = []
    for _, summary_row in summary.iterrows():
        row = {
            "source_family": descriptor["source_family"],
            "method": descriptor["method"],
            "method_setting": descriptor["method_setting"],
            "market": descriptor["market"],
            "model": descriptor["model"],
            "objective": descriptor["objective"],
            "source_path": str(result.source_path),
            "prediction_csv": str(result.prediction_csv),
            "optimization_dir": str(result.optimization_dir),
            "best_run_dir": str(result.best_run_dir),
            "global_objective_score": result.best_row.get("global_objective_score"),
            "weighted_avg_unit_pnl": result.best_row.get("weighted_avg_unit_pnl"),
            "weighted_avg_hit_ratio": result.best_row.get("weighted_avg_hit_ratio"),
            "weighted_avg_pnl_mean_over_std": result.best_row.get("weighted_avg_pnl_mean_over_std"),
            "weighted_avg_max_drawdown_per_active_mwh": result.best_row.get(
                "weighted_avg_max_drawdown_per_active_mwh"
            ),
            "weighted_avg_cumulative_pnl": result.best_row.get("weighted_avg_cumulative_pnl"),
            "priority_min_cumulative_pnl": result.best_row.get("priority_min_cumulative_pnl"),
            "position_method": result.best_params["position_method"],
            "entry_band_fraction": result.best_params["entry_band_fraction"],
            "exit_band_fraction": result.best_params["exit_band_fraction"],
            "signal_family": result.best_params["signal_family"],
            "signal_scale": result.best_params["signal_scale"],
            "signal_power": result.best_params["signal_power"],
            "position_cap": result.best_params["position_cap"],
            "min_interval_width": result.best_params["min_interval_width"],
        }
        for source_column, summary_column in SUMMARY_COLUMN_RENAMES.items():
            row[summary_column] = summary_row[source_column]
        rows.append(row)
    return rows


def write_batch_summary(results: list[BatchRunResult], output_root: Path) -> Path:
    rows = [row for result in results for row in build_batch_summary_rows(result)]
    summary_path = output_root / "batch_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path


def write_batch_failures(failures: list[dict[str, object]], output_root: Path) -> Path | None:
    if not failures:
        return None
    failures_path = output_root / "batch_failures.csv"
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures).to_csv(failures_path, index=False)
    return failures_path


def grid_stage_settings(grid_stages: list[dict[str, object]]) -> list[dict[str, object]]:
    stage_settings: list[dict[str, object]] = []
    for stage in grid_stages:
        grid = dict(stage["grid"])
        candidate_count_before_constraints = 1
        for values in grid.values():
            candidate_count_before_constraints *= len(values)
        stage_settings.append(
            {
                "stage_name": stage["stage_name"],
                "candidate_count_before_constraints": candidate_count_before_constraints,
                "grid": grid,
            }
        )
    return stage_settings


def print_run_settings(
    *,
    output_root: Path,
    y_path: Path,
    grid_stages: list[dict[str, object]],
) -> None:
    resolved_base_params = dict(optimize_strategy.BASE_PARAMS)
    resolved_base_params.update(BACKTEST_PARAMS)
    resolved_objective_weights = optimize_strategy.normalize_objective_weights(OBJECTIVE_WEIGHTS)
    path_settings = [
        {
            "run_index": index,
            "path": path,
            "random_seed": RANDOM_SEED + index - 1,
            **describe_prediction_path(path),
        }
        for index, path in enumerate(PRED_PATHS, start=1)
    ]
    settings = {
        "run_controls": {
            "run_backtest_batch": RUN_BACKTEST_BATCH,
            "stop_on_error": STOP_ON_ERROR,
            "max_trials_per_stage": MAX_TRIALS_PER_STAGE,
            "base_random_seed": RANDOM_SEED,
            "per_path_random_seed_rule": "base_random_seed + run_index - 1",
        },
        "paths": {
            "repo_root": REPO_ROOT,
            "y_path": y_path,
            "batch_output_root": output_root,
            "model_method_path_specs": MODEL_METHOD_PATH_SPECS,
            "deduped_prediction_path_count": len(PRED_PATHS),
            "deduped_prediction_paths": path_settings,
        },
        "backtest_params": BACKTEST_PARAMS,
        "effective_base_params": resolved_base_params,
        "objective_weights": {
            "configured": OBJECTIVE_WEIGHTS,
            "normalized": resolved_objective_weights,
            "score_specs": optimize_strategy.OBJECTIVE_SCORE_SPECS,
        },
        "grid_stages": grid_stage_settings(grid_stages),
        "strategy_selection": {
            "optimizer_real_strategy_scenarios": optimize_strategy.REAL_STRATEGY_SCENARIOS,
            "optimizer_scenario_weights": optimize_strategy.OPTIMIZATION_SCENARIO_WEIGHTS,
            "priority_scenarios": optimize_strategy.PRIORITY_SCENARIOS,
            "batch_summary_scenarios": SUMMARY_SCENARIOS,
            "batch_summary_columns": SUMMARY_COLUMN_RENAMES,
        },
    }

    print("")
    print("=== Batch run settings ===")
    print(json.dumps(safe_json_value(settings), indent=2))
    print("=== End batch run settings ===")
    print("")


def main() -> int:
    if not RUN_BACKTEST_BATCH:
        print("RUN_BACKTEST_BATCH is False; nothing to run.")
        return 0

    output_root = BACKTEST_BATCH_RESULTS_ROOT.resolve()
    y_path = Path(Y_PATH).resolve()
    grid_stages = copy.deepcopy(GRID_STAGES)
    successes: list[BatchRunResult] = []
    failures: list[dict[str, object]] = []

    print(f"Batch output root: {output_root}")
    print(f"Prediction paths : {len(PRED_PATHS)}")
    print_run_settings(output_root=output_root, y_path=y_path, grid_stages=grid_stages)

    for run_index, pred_path in enumerate(PRED_PATHS, start=1):
        print("")
        print(f"=== [{run_index}/{len(PRED_PATHS)}] {pred_path} ===")
        try:
            result = run_optimizer_for_path(
                pred_path=Path(pred_path),
                y_path=y_path,
                output_root=output_root,
                base_params=BACKTEST_PARAMS,
                objective_weights=OBJECTIVE_WEIGHTS,
                grid_stages=grid_stages,
                max_trials=MAX_TRIALS_PER_STAGE,
                random_seed=RANDOM_SEED + run_index - 1,
            )
        except Exception as exc:
            failure = {
                "run_index": run_index,
                "source_path": str(pred_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"FAILED: {type(exc).__name__}: {exc}")
            write_batch_failures(failures, output_root)
            if STOP_ON_ERROR:
                raise
            continue

        successes.append(result)
        summary_path = write_batch_summary(successes, output_root)
        print(f"Batch summary CSV: {summary_path}")

    failures_path = write_batch_failures(failures, output_root)
    if failures_path is not None:
        print(f"Batch failures CSV: {failures_path}")

    print("")
    print(f"Completed batch: {len(successes)} succeeded, {len(failures)} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
