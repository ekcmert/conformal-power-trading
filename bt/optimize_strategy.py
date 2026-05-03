from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from backtest import BacktestConfig, IntervalBacktester, infer_output_dir, is_openable_interval_band
from backtest.config import Y_PATH
from backtest.io import load_prediction_frame, load_price_frame, resolve_prediction_csv


PRED_PATH = Path(
    r"C:\Users\mert.ekici\Desktop\CPT\results\regime_aware\mcp\heuristic\da\cp_asymmetric_catboost\rmse"
)

BASE_PARAMS = {
    "position_method": "interval_band",
    "entry_band_fraction": 0.15,
    "exit_band_fraction": 0.75,
    "signal_family": "arctan",
    "signal_scale": 2.0,
    "signal_power": 2.0,
    "position_cap": 60.0,
    "min_interval_width": 1e-6,
}

REAL_STRATEGY_SCENARIOS = [
    "strategy_close_id",
    "strategy_close_id3",
    "strategy_close_id1",
    "strategy_close_imb",
]

OPTIMIZATION_SCENARIO_WEIGHTS = {
    "strategy_close_id": 0.2,
    "strategy_close_id3": 0.3,
    "strategy_close_id1": 0.4,
    "strategy_close_imb": 0.1,
}

PRIORITY_SCENARIOS = [
    "strategy_close_id3",
    "strategy_close_id1",
]

GRID_STAGES = [
    {
        "stage_name": "band_and_family",
        "grid": {
            "entry_band_fraction": [0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
            "exit_band_fraction": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50, 1.65, 1.80, 2.00],
            "signal_family": ["linear", "arctan", "tanh", "softsign"],
        },
    },
    {
        "stage_name": "signal_shape",
        "grid": {
            "signal_scale": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0],
            "signal_power": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0],
            # "position_cap": [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0, 125.0, 150.0],
            "position_cap": [40.0, 60.0]
        },
    },
]

DEFAULT_RANDOM_SEED = 42

DEFAULT_OBJECTIVE_WEIGHTS = {
    "unit_pnl": 0.5,
    "hit_ratio": 0.5,
    "mean_over_std": 0.0,
    "drawdown": 0.0,
    "cumulative_pnl": 0.0,
}

OBJECTIVE_SCORE_SPECS = {
    "unit_pnl": {
        "metric_column": "weighted_avg_unit_pnl",
        "higher_is_better": True,
        "description": "weighted average unit PnL",
    },
    "hit_ratio": {
        "metric_column": "weighted_avg_hit_ratio",
        "higher_is_better": True,
        "description": "weighted average active hit ratio",
    },
    "mean_over_std": {
        "metric_column": "weighted_avg_pnl_mean_over_std",
        "higher_is_better": True,
        "description": "weighted average PnL mean over standard deviation",
    },
    "drawdown": {
        "metric_column": "weighted_avg_max_drawdown_per_active_mwh",
        "higher_is_better": False,
        "description": "weighted average max drawdown per active MWh",
    },
    "cumulative_pnl": {
        "metric_column": "weighted_avg_cumulative_pnl",
        "higher_is_better": True,
        "description": "weighted average cumulative PnL",
    },
}


def unique_sorted_float_grid(values: list[float], *, lower: float | None = None, upper: float | None = None) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        rounded = round(float(value), 6)
        if lower is not None and rounded < lower:
            continue
        if upper is not None and rounded > upper:
            continue
        cleaned.append(rounded)
    return sorted(set(cleaned))


def normalize_objective_weights(objective_weights: dict[str, float] | None = None) -> dict[str, float]:
    if objective_weights is None:
        resolved_weights = dict(DEFAULT_OBJECTIVE_WEIGHTS)
    else:
        unknown_weights = sorted(set(objective_weights) - set(OBJECTIVE_SCORE_SPECS))
        if unknown_weights:
            raise ValueError(f"Unsupported objective weight(s): {unknown_weights}")
        resolved_weights = {key: 0.0 for key in OBJECTIVE_SCORE_SPECS}
        resolved_weights.update(objective_weights)

    for key, value in resolved_weights.items():
        resolved_weights[key] = float(value)
        if not np.isfinite(resolved_weights[key]) or resolved_weights[key] < 0:
            raise ValueError(f"Objective weight {key!r} must be a finite non-negative value.")

    total_weight = sum(resolved_weights.values())
    if total_weight <= 0:
        raise ValueError("At least one objective weight must be greater than zero.")

    return {key: value / total_weight for key, value in resolved_weights.items()}


def build_refine_stage(best_params: dict[str, object]) -> dict[str, object]:
    best_entry = float(best_params["entry_band_fraction"])
    best_exit = float(best_params["exit_band_fraction"])
    best_scale = float(best_params["signal_scale"])
    best_power = float(best_params["signal_power"])
    best_cap = float(best_params["position_cap"])

    return {
        "stage_name": "local_refine",
        "grid": {
            "entry_band_fraction": unique_sorted_float_grid(
                [best_entry - 0.05, best_entry, best_entry + 0.05],
                lower=0.0,
                upper=0.70,
            ),
            "exit_band_fraction": unique_sorted_float_grid(
                [best_exit - 0.10, best_exit, best_exit + 0.10],
                lower=0.05,
                upper=2.00,
            ),
            "signal_scale": unique_sorted_float_grid(
                [best_scale - 0.5, best_scale, best_scale + 0.5],
                lower=0.25,
            ),
            "signal_power": unique_sorted_float_grid(
                [best_power - 0.5, best_power, best_power + 0.5],
                lower=0.5,
            ),
            "position_cap": unique_sorted_float_grid(
                [best_cap * 0.75, best_cap, best_cap * 1.25],
                lower=5.0,
            ),
        },
    }


def iter_grid_candidates(grid: dict[str, list[object]]) -> list[dict[str, object]]:
    keys = list(grid)
    value_lists = [grid[key] for key in keys]
    return [dict(zip(keys, values, strict=True)) for values in product(*value_lists)]


def select_stage_candidates(
    *,
    grid: dict[str, list[object]],
    base_params: dict[str, object],
    max_trials: int | None,
    rng: np.random.Generator,
    seen_keys: set[tuple[object, ...]],
) -> tuple[list[dict[str, object]], int, int]:
    valid_candidates: list[dict[str, object]] = []
    skipped_impossible_interval_bands = 0
    for overrides in iter_grid_candidates(grid):
        candidate_params = dict(base_params)
        candidate_params.update(overrides)
        if (
            str(candidate_params.get("position_method", "interval_band")).lower() == "interval_band"
            and not is_openable_interval_band(
                float(candidate_params["entry_band_fraction"]),
                float(candidate_params["exit_band_fraction"]),
            )
        ):
            skipped_impossible_interval_bands += 1
            continue
        candidate_key = (
            float(candidate_params["entry_band_fraction"]),
            float(candidate_params["exit_band_fraction"]),
            str(candidate_params["signal_family"]),
            float(candidate_params["signal_scale"]),
            float(candidate_params["signal_power"]),
            float(candidate_params["position_cap"]),
        )
        if candidate_key in seen_keys:
            continue
        valid_candidates.append(overrides)

    total_possible = len(valid_candidates)
    if max_trials is None or max_trials <= 0 or max_trials >= total_possible:
        return valid_candidates, total_possible, skipped_impossible_interval_bands

    sampled_indices = rng.choice(total_possible, size=max_trials, replace=False)
    sampled_candidates = [valid_candidates[int(index)] for index in sampled_indices.tolist()]
    return sampled_candidates, total_possible, skipped_impossible_interval_bands


def normalize_score(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    filled = series.astype("float64").copy()
    if higher_is_better:
        worst_value = filled.min(skipna=True)
        filled = filled.fillna(worst_value - 1.0 if pd.notna(worst_value) else -1.0)
        return filled.rank(method="average", pct=True, ascending=True)

    worst_value = filled.max(skipna=True)
    filled = filled.fillna(worst_value + 1.0 if pd.notna(worst_value) else 1.0)
    return filled.rank(method="average", pct=True, ascending=False)


def add_objective_scores(
    results: pd.DataFrame,
    objective_weights: dict[str, float],
    *,
    score_prefix: str = "",
    objective_column: str,
) -> pd.DataFrame:
    scored = results.copy()
    weighted_terms: list[pd.Series] = []

    for objective_name, weight in objective_weights.items():
        if weight <= 0:
            continue
        score_spec = OBJECTIVE_SCORE_SPECS[objective_name]
        metric_column = str(score_spec["metric_column"])
        if metric_column not in scored.columns:
            raise KeyError(f"Missing objective metric column {metric_column!r}.")

        score_column = f"{score_prefix}_{objective_name}_score" if score_prefix else f"{objective_name}_score"
        scored[score_column] = normalize_score(
            scored[metric_column],
            higher_is_better=bool(score_spec["higher_is_better"]),
        )
        weighted_terms.append(float(weight) * scored[score_column])

    scored[objective_column] = sum(weighted_terms)
    return scored


def weighted_metric(strategy_summary: pd.DataFrame, column: str) -> float:
    weighted_values: list[float] = []
    weights: list[float] = []
    for scenario_name, weight in OPTIMIZATION_SCENARIO_WEIGHTS.items():
        if weight <= 0:
            continue
        value = float(strategy_summary.loc[scenario_name, column])
        if not np.isfinite(value):
            continue
        weighted_values.append(value)
        weights.append(float(weight))

    if not weighted_values or not weights:
        return float("nan")
    return float(np.average(weighted_values, weights=weights))


def evaluate_candidate(
    *,
    pred_path: Path,
    y_path: Path,
    prediction_frame: pd.DataFrame,
    price_frame: pd.DataFrame,
    params: dict[str, object],
) -> dict[str, object]:
    config = BacktestConfig(
        pred_path=pred_path,
        y_path=y_path,
        output_dir=None,
        **params,
    )
    backtester = IntervalBacktester(config)
    hourly_results, scenario_specs, _ = backtester._build_hourly_results(prediction_frame, price_frame)
    summary = backtester._build_summary_frame(hourly_results, scenario_specs).set_index("scenario_name")
    strategy_summary = summary.loc[REAL_STRATEGY_SCENARIOS].copy()

    active_position_mask = hourly_results["strategy_abs_position"] > 0
    active_avg_abs_position = float(hourly_results.loc[active_position_mask, "strategy_abs_position"].mean())
    if not np.isfinite(active_avg_abs_position) or active_avg_abs_position <= 0:
        active_avg_abs_position = float("nan")

    strategy_summary["max_drawdown_per_active_mwh"] = (
        strategy_summary["max_drawdown"] / active_avg_abs_position
        if np.isfinite(active_avg_abs_position)
        else np.nan
    )
    priority_summary = strategy_summary.loc[PRIORITY_SCENARIOS]

    result = {
        "entry_band_fraction": float(config.entry_band_fraction),
        "exit_band_fraction": float(config.exit_band_fraction),
        "signal_family": config.signal_family,
        "signal_scale": float(config.signal_scale),
        "signal_power": float(config.signal_power),
        "position_cap": float(config.position_cap),
        "active_avg_abs_position": active_avg_abs_position,
        "weighted_avg_unit_pnl": weighted_metric(strategy_summary, "avg_unit_pnl"),
        "weighted_avg_hit_ratio": weighted_metric(strategy_summary, "hit_ratio"),
        "weighted_avg_pnl_mean_over_std": weighted_metric(strategy_summary, "pnl_mean_over_std"),
        "weighted_avg_cumulative_pnl": weighted_metric(strategy_summary, "cumulative_pnl"),
        "weighted_avg_max_drawdown": weighted_metric(strategy_summary, "max_drawdown"),
        "weighted_avg_max_drawdown_per_active_mwh": weighted_metric(
            strategy_summary,
            "max_drawdown_per_active_mwh",
        ),
        "priority_min_unit_pnl": float(priority_summary["avg_unit_pnl"].min()),
        "priority_min_hit_ratio": float(priority_summary["hit_ratio"].min()),
        "priority_min_cumulative_pnl": float(priority_summary["cumulative_pnl"].min()),
        "priority_avg_max_drawdown_per_active_mwh": float(priority_summary["max_drawdown_per_active_mwh"].mean()),
        "priority_worst_max_drawdown_per_active_mwh": float(priority_summary["max_drawdown_per_active_mwh"].max()),
        "avg_unit_pnl": float(strategy_summary["avg_unit_pnl"].mean()),
        "min_unit_pnl": float(strategy_summary["avg_unit_pnl"].min()),
        "avg_hit_ratio": float(strategy_summary["hit_ratio"].mean()),
        "min_hit_ratio": float(strategy_summary["hit_ratio"].min()),
        "avg_cumulative_pnl": float(strategy_summary["cumulative_pnl"].mean()),
        "min_cumulative_pnl": float(strategy_summary["cumulative_pnl"].min()),
        "avg_max_drawdown": float(strategy_summary["max_drawdown"].mean()),
        "worst_max_drawdown": float(strategy_summary["max_drawdown"].max()),
        "avg_max_drawdown_per_active_mwh": float(strategy_summary["max_drawdown_per_active_mwh"].mean()),
        "worst_max_drawdown_per_active_mwh": float(strategy_summary["max_drawdown_per_active_mwh"].max()),
        "avg_abs_position_size": float(strategy_summary["avg_abs_position_size"].mean()),
    }

    for scenario_name in REAL_STRATEGY_SCENARIOS:
        scenario_row = strategy_summary.loc[scenario_name]
        market_slug = scenario_name.removeprefix("strategy_close_")
        result[f"{market_slug}_avg_unit_pnl"] = float(scenario_row["avg_unit_pnl"])
        result[f"{market_slug}_hit_ratio"] = float(scenario_row["hit_ratio"])
        result[f"{market_slug}_cumulative_pnl"] = float(scenario_row["cumulative_pnl"])
        result[f"{market_slug}_max_drawdown"] = float(scenario_row["max_drawdown"])
        result[f"{market_slug}_max_drawdown_per_active_mwh"] = float(
            scenario_row["max_drawdown_per_active_mwh"]
        )

    return result


def score_stage_results(stage_results: pd.DataFrame, objective_weights: dict[str, float]) -> pd.DataFrame:
    scored = add_objective_scores(
        stage_results,
        objective_weights,
        objective_column="stage_objective_score",
    )
    return scored.sort_values(
        [
            "stage_objective_score",
            "weighted_avg_unit_pnl",
            "weighted_avg_hit_ratio",
            "weighted_avg_pnl_mean_over_std",
            "weighted_avg_max_drawdown_per_active_mwh",
            "weighted_avg_cumulative_pnl",
            "priority_min_cumulative_pnl",
        ],
        ascending=[False, False, False, False, True, False, False],
    ).reset_index(drop=True)


def run_stage(
    *,
    stage_name: str,
    grid: dict[str, list[object]],
    base_params: dict[str, object],
    max_trials: int | None,
    pred_path: Path,
    y_path: Path,
    prediction_frame: pd.DataFrame,
    price_frame: pd.DataFrame,
    seen_keys: set[tuple[object, ...]],
    rng: np.random.Generator,
    objective_weights: dict[str, float],
    allow_empty: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    candidates, total_possible, skipped_impossible_interval_bands = select_stage_candidates(
        grid=grid,
        base_params=base_params,
        max_trials=max_trials,
        rng=rng,
        seen_keys=seen_keys,
    )
    if skipped_impossible_interval_bands:
        print(
            f"[{stage_name}] skipped {skipped_impossible_interval_bands} impossible interval-band "
            "candidate(s) with exit_band_fraction <= entry_band_fraction."
        )
    if not candidates:
        if allow_empty:
            print(f"[{stage_name}] skipped: no new candidate combinations available.")
            return pd.DataFrame(), base_params
        raise ValueError(
            f"No candidate combinations available for stage {stage_name!r}. "
            "Try increasing max_trials or widening the grid."
        )
    rows: list[dict[str, object]] = []
    stage_start = time.perf_counter()

    progress_label = f"{stage_name} [{len(candidates)}/{total_possible}]"
    for idx, overrides in enumerate(tqdm(candidates, desc=progress_label, unit="trial"), start=1):
        params = dict(base_params)
        params.update(overrides)

        param_key = (
            float(params["entry_band_fraction"]),
            float(params["exit_band_fraction"]),
            str(params["signal_family"]),
            float(params["signal_scale"]),
            float(params["signal_power"]),
            float(params["position_cap"]),
        )
        if param_key in seen_keys:
            continue
        seen_keys.add(param_key)

        evaluation = evaluate_candidate(
            pred_path=pred_path,
            y_path=y_path,
            prediction_frame=prediction_frame,
            price_frame=price_frame,
            params=params,
        )
        evaluation["stage_name"] = stage_name
        evaluation["stage_candidate_index"] = idx
        rows.append(evaluation)

    if not rows:
        if allow_empty:
            print(f"[{stage_name}] skipped: all sampled candidates were already evaluated.")
            return pd.DataFrame(), base_params
        raise ValueError(
            f"No candidate rows were evaluated for stage {stage_name!r}. "
            "Try increasing max_trials or widening the grid."
        )

    stage_results = score_stage_results(pd.DataFrame(rows), objective_weights)
    elapsed = time.perf_counter() - stage_start
    print(f"[{stage_name}] evaluated {len(rows)}/{len(candidates)} sampled trials in {elapsed:.1f}s")
    best_row = stage_results.iloc[0].to_dict()
    best_params = {
        "position_method": "interval_band",
        "entry_band_fraction": float(best_row["entry_band_fraction"]),
        "exit_band_fraction": float(best_row["exit_band_fraction"]),
        "signal_family": str(best_row["signal_family"]),
        "signal_scale": float(best_row["signal_scale"]),
        "signal_power": float(best_row["signal_power"]),
        "position_cap": float(best_row["position_cap"]),
        "min_interval_width": float(base_params["min_interval_width"]),
    }
    return stage_results, best_params


def run_best_backtest(
    *,
    pred_path: Path,
    y_path: Path,
    output_dir: Path,
    best_params: dict[str, object],
) -> dict[str, str]:
    best_run_dir = output_dir / "best_run"
    artifacts = IntervalBacktester(
        BacktestConfig(
            pred_path=pred_path,
            y_path=y_path,
            output_dir=best_run_dir,
            **best_params,
        )
    ).run()
    return {
        "output_dir": str(artifacts.output_dir),
        "hourly_results_path": str(artifacts.hourly_results_path),
        "summary_path": str(artifacts.summary_path),
        "dashboard_path": str(artifacts.dashboard_path),
        "metadata_path": str(artifacts.metadata_path),
    }


def main(
    *,
    pred_path: Path = PRED_PATH,
    y_path: Path | None = None,
    base_params: dict[str, object] | None = None,
    objective_weights: dict[str, float] | None = None,
    max_trials: int | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> int:
    y_path = Path(Y_PATH) if y_path is None else y_path
    resolved_base_params = dict(BASE_PARAMS)
    if base_params is not None:
        resolved_base_params.update(base_params)
    resolved_objective_weights = normalize_objective_weights(objective_weights)
    rng = np.random.default_rng(random_seed)

    prediction_csv = resolve_prediction_csv(pred_path)
    prediction_frame = load_prediction_frame(prediction_csv)
    price_frame = load_price_frame(y_path)

    base_output_dir = infer_output_dir(prediction_csv) / "optimization"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Prediction CSV: {prediction_csv}")
    print(f"Optimization dir: {base_output_dir}")
    print(f"Max trials per stage: {max_trials if max_trials is not None else 'all'}")
    print(f"Random seed: {random_seed}")
    print(f"Objective weights: {resolved_objective_weights}")

    seen_keys: set[tuple[object, ...]] = set()
    stage_frames: list[pd.DataFrame] = []
    current_best_params = dict(resolved_base_params)

    for stage in GRID_STAGES:
        stage_results, current_best_params = run_stage(
            stage_name=str(stage["stage_name"]),
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
        stage_results.to_csv(base_output_dir / f"{stage['stage_name']}_results.csv", index=False)
        stage_frames.append(stage_results)
        best_row = stage_results.iloc[0]
        print(
            f"[{stage['stage_name']}] best score={best_row['stage_objective_score']:.4f}, "
            f"weighted_unit_pnl={best_row['weighted_avg_unit_pnl']:.4f}, "
            f"weighted_hit_ratio={best_row['weighted_avg_hit_ratio']:.4f}, "
            f"weighted_mean_std={best_row['weighted_avg_pnl_mean_over_std']:.4f}, "
            f"weighted_dd_per_mwh={best_row['weighted_avg_max_drawdown_per_active_mwh']:.4f}, "
            f"weighted_cum_pnl={best_row['weighted_avg_cumulative_pnl']:.4f}, "
            f"priority_min_cum_pnl={best_row['priority_min_cumulative_pnl']:.4f}"
        )

    refine_stage = build_refine_stage(current_best_params)
    refine_results, current_best_params = run_stage(
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
        refine_results.to_csv(base_output_dir / f"{refine_stage['stage_name']}_results.csv", index=False)
        stage_frames.append(refine_results)

    all_trials = add_objective_scores(
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
    all_trials.to_csv(base_output_dir / "all_trials.csv", index=False)

    best_row = all_trials.iloc[0]
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

    best_artifacts = run_best_backtest(
        pred_path=pred_path,
        y_path=y_path,
        output_dir=base_output_dir,
        best_params=best_params,
    )

    optimization_metadata = {
        "prediction_csv": str(prediction_csv),
        "optimization_dir": str(base_output_dir),
        "base_params": resolved_base_params,
        "max_trials_per_stage": max_trials,
        "random_seed": random_seed,
        "grid_stages": GRID_STAGES + [refine_stage],
        "real_strategy_scenarios": REAL_STRATEGY_SCENARIOS,
        "drawdown_normalization": (
            "max_drawdown is divided by active_avg_abs_position, "
            "where active_avg_abs_position is the average absolute strategy position across hours with non-zero exposure."
        ),
        "metric_definitions": {
            "avg_unit_pnl": "sum(pnl) / sum(abs(position)); EUR/MWh over all opened-position volume",
            "hit_ratio": "share of opened-position rows with pnl > 0",
        },
        "optimization_scenario_weights": OPTIMIZATION_SCENARIO_WEIGHTS,
        "priority_scenarios": PRIORITY_SCENARIOS,
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
    }
    (base_output_dir / "optimization_metadata.json").write_text(
        json.dumps(optimization_metadata, indent=2),
        encoding="utf-8",
    )

    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"All trials CSV : {base_output_dir / 'all_trials.csv'}")
    print(f"Best run dir   : {best_artifacts['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
