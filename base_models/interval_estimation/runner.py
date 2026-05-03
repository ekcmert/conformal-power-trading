from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import optuna
import pandas as pd
from tqdm.auto import tqdm

from base_models.tuning_grids import (
    INTERVAL_TUNING_PARAM_GRIDS,
    canonicalize_params,
    count_param_grid_candidates,
    resolve_interval_tuning_budget,
    sample_params_from_trial,
)

from .common import (
    DEFAULT_LOWER_QUANTILE,
    DEFAULT_UPPER_QUANTILE,
    TEST_YEAR,
    build_prediction_frame,
    compute_interval_metrics,
    load_final_datasets,
    output_dir_for,
    plot_feature_importances,
    plot_interval_predictions,
    save_combined_plotly_predictions,
    split_train_and_tuning_validation,
    split_train_test_by_year,
    standardize_features,
    target_artifact_stem,
    validate_quantile_pair,
)
from .importance import compute_importance
from .model_registry import ExperimentGroup, ExperimentSpec, get_experiment_spec_groups


optuna.logging.set_verbosity(optuna.logging.WARNING)
BEST_PARAMS_DIR = Path(__file__).resolve().parent / "best_params"
COMPLETED_PAIR_ARTIFACTS = ("metrics_summary.csv", "all_targets_predictions.html")


def _log(message: str) -> None:
    tqdm.write(message)


def _format_period(data: pd.DataFrame | pd.Series) -> str:
    if data.empty:
        return "empty"
    return f"{data.index.min()} -> {data.index.max()} ({len(data):,} rows)"


def _planned_trial_budget(total_candidates: int, tuning_n_trials: int | None) -> int:
    return min(total_candidates, tuning_n_trials or total_candidates)


def _completed_output_dir_for(
    config: "IntervalEstimationConfig",
    group: ExperimentGroup,
) -> Path:
    return output_dir_for(config.results_dir, group.model_name, group.objective_name)


def _has_completed_pair_outputs(output_dir: Path) -> bool:
    if not output_dir.exists() or not output_dir.is_dir():
        return False
    if not any(output_dir.iterdir()):
        return False
    return all((output_dir / artifact_name).exists() for artifact_name in COMPLETED_PAIR_ARTIFACTS)


def _load_csv_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    if df.empty:
        return []
    return df.to_dict(orient="records")


def _normalize_key_part(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _pair_key(model_name: object, objective_name: object) -> tuple[str, str]:
    return _normalize_key_part(model_name), _normalize_key_part(objective_name)


def _result_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        _normalize_key_part(row.get("model_name", "")),
        _normalize_key_part(row.get("objective_name", "")),
        _normalize_key_part(row.get("target", "")),
    )


def _upsert_result_rows(
    store: dict[tuple[str, str, str], dict[str, object]],
    rows: list[dict[str, object]],
) -> None:
    for row in rows:
        store[_result_key(row)] = row


def _write_run_summaries(
    config: "IntervalEstimationConfig",
    results_store: dict[tuple[str, str, str], dict[str, object]],
    failures_store: dict[tuple[str, str], dict[str, object]],
    tuning_selection_store: dict[tuple[str, str], dict[str, object]],
    best_params_store: dict[tuple[str, str], dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_df = pd.DataFrame(list(results_store.values()))
    failures_df = pd.DataFrame(list(failures_store.values()))

    results_df.to_csv(config.results_dir / "all_model_results.csv", index=False)

    failed_experiments_path = config.results_dir / "failed_experiments.csv"
    if failures_df.empty:
        failed_experiments_path.unlink(missing_ok=True)
    else:
        failures_df.to_csv(failed_experiments_path, index=False)

    tuning_selection_path = config.results_dir / "tuning_selection_summary.csv"
    if config.tuning_active and tuning_selection_store:
        pd.DataFrame(list(tuning_selection_store.values())).to_csv(tuning_selection_path, index=False)
    else:
        tuning_selection_path.unlink(missing_ok=True)

    best_params_path = config.results_dir / "best_params.csv"
    if config.tuning_active and best_params_store:
        pd.DataFrame(list(best_params_store.values())).to_csv(best_params_path, index=False)
    else:
        best_params_path.unlink(missing_ok=True)

    return results_df, failures_df


def _group_tuning_budget(
    config: "IntervalEstimationConfig",
    group: ExperimentGroup,
):
    return resolve_interval_tuning_budget(
        model_name=group.model_name,
        objective_name=group.objective_name,
        global_n_trials=config.tuning_n_trials,
        global_patience=config.tuning_patience,
    )


def _print_run_summary(
    config: "IntervalEstimationConfig",
    experiment_groups: list[ExperimentGroup],
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    X_tuning_train_raw: pd.DataFrame | None = None,
    X_tuning_validation_raw: pd.DataFrame | None = None,
) -> None:
    total_grid_combinations = sum(len(group.candidates) for group in experiment_groups)
    total_planned_trials = sum(
        _planned_trial_budget(len(group.candidates), _group_tuning_budget(config, group).n_trials)
        for group in experiment_groups
    )

    _log("")
    _log("=" * 80)
    _log("Interval Estimation Pipeline")
    _log("=" * 80)
    _log(f"Results directory : {config.results_dir}")
    _log(f"Targets           : {list(config.targets)}")
    _log(f"Quantiles         : ({config.lower_quantile:.2f}, {config.upper_quantile:.2f})")
    _log(f"Model-loss pairs  : {len(experiment_groups):,}")
    _log(f"Final-train data  : {_format_period(X_train_raw)}")
    if config.tuning_active:
        _log(f"Tuning train data : {_format_period(X_tuning_train_raw) if X_tuning_train_raw is not None else 'n/a'}")
        _log(
            "Validation data  : "
            f"{_format_period(X_tuning_validation_raw) if X_tuning_validation_raw is not None else 'n/a'}"
        )
        _log(f"Grid combinations : {total_grid_combinations:,} total across all model-loss pairs")
        _log(
            "Optuna trials     : "
            f"{total_planned_trials:,} planned total across all model-loss pairs"
        )
        _log(
            "Budget policy     : "
            "model-specific budgets with optional global caps "
            f"(trial_cap={config.tuning_n_trials if config.tuning_n_trials is not None else 'none'}, "
            f"patience_cap={config.tuning_patience if config.tuning_patience is not None else 'none'})"
        )
        _log(
            "Resume policy     : "
            "skip completed model-loss pairs when both metrics_summary.csv "
            "and all_targets_predictions.html already exist"
        )
    _log(f"Test data         : {_format_period(X_test_raw)}")
    _log("=" * 80)


def _best_params_filename(spec: ExperimentSpec) -> str:
    objective_label = spec.objective_name or "default"
    return f"{target_artifact_stem(spec.model_name)}__{target_artifact_stem(objective_label)}.json"


def _save_best_params_artifacts(
    config: "IntervalEstimationConfig",
    spec: ExperimentSpec,
    tuning_metadata: dict[str, object],
) -> tuple[Path, dict[str, object]]:
    BEST_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    best_params_path = BEST_PARAMS_DIR / _best_params_filename(spec)

    payload = {
        "model_name": spec.model_name,
        "objective_name": spec.objective_name or "",
        "display_name": spec.display_name,
        "best_params": spec.params,
        "params_summary": spec.params_summary,
        "pipeline": "interval_estimation",
        "targets": list(config.targets),
        "quantiles": [config.lower_quantile, config.upper_quantile],
        "results_dir": str(config.results_dir),
        "test_year": TEST_YEAR,
        "tuning": {
            "active": bool(config.tuning_active),
            "months": int(config.tuning_months),
            "patience": tuning_metadata.get("tuning_patience", config.tuning_patience),
            "n_trials": tuning_metadata.get("tuning_n_trials", config.tuning_n_trials),
            "budget_label": tuning_metadata.get("tuning_budget_label", ""),
            "group_patience": tuning_metadata.get("tuning_group_patience", ""),
            "group_n_trials": tuning_metadata.get("tuning_group_n_trials", ""),
            "global_patience_cap": tuning_metadata.get("configured_tuning_patience", config.tuning_patience),
            "global_n_trials_cap": tuning_metadata.get("configured_tuning_n_trials", config.tuning_n_trials),
            "metric_name": tuning_metadata.get("tuning_metric_name", ""),
            "metric_value": tuning_metadata.get("tuning_metric_value", ""),
            "trial_budget": tuning_metadata.get("tuning_trial_budget", ""),
            "trials_sampled": tuning_metadata.get("tuning_trials_sampled", ""),
            "successful_trials": tuning_metadata.get("tuning_candidates_evaluated", ""),
            "duplicate_suggestions": tuning_metadata.get("tuning_duplicate_suggestions", ""),
            "early_stopped": tuning_metadata.get("tuning_early_stopped", False),
            "validation_start": tuning_metadata.get("tuning_validation_start", ""),
            "validation_end": tuning_metadata.get("tuning_validation_end", ""),
        },
    }
    best_params_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    best_params_row = {
        "model_name": spec.model_name,
        "objective_name": spec.objective_name or "",
        "display_name": spec.display_name,
        "best_params_json_path": str(best_params_path),
        "best_params_json": json.dumps(spec.params, sort_keys=True, ensure_ascii=True),
        "params_summary": spec.params_summary,
        **tuning_metadata,
    }
    return best_params_path, best_params_row


@dataclass(frozen=True)
class IntervalEstimationConfig:
    targets: list[str] | tuple[str, ...]
    results_dir: Path
    feature_columns: list[str] | tuple[str, ...] | None = None
    lower_quantile: float = DEFAULT_LOWER_QUANTILE
    upper_quantile: float = DEFAULT_UPPER_QUANTILE
    tuning_active: bool = False
    tuning_months: int = 6
    tuning_patience: int | None = None
    tuning_n_trials: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "results_dir", self.results_dir.resolve())
        validate_quantile_pair(self.lower_quantile, self.upper_quantile)
        if self.tuning_active and self.tuning_months <= 0:
            raise ValueError(f"tuning_months must be positive when tuning_active=True, got {self.tuning_months}.")
        if self.tuning_patience is not None and self.tuning_patience <= 0:
            raise ValueError(f"tuning_patience must be positive when provided, got {self.tuning_patience}.")
        if self.tuning_n_trials is not None and self.tuning_n_trials <= 0:
            raise ValueError(f"tuning_n_trials must be positive when provided, got {self.tuning_n_trials}.")


def _fit_and_predict_interval(
    spec: ExperimentSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    model = spec.builder()
    model.fit(X_train, y_train)
    if not hasattr(model, "predict_interval"):
        raise TypeError(f"{spec.display_name} does not implement predict_interval().")

    lower_values, upper_values = model.predict_interval(X_test)
    prediction_frame = pd.DataFrame(
        {
            "y_pred_lower": pd.Series(lower_values, index=X_test.index, dtype=float),
            "y_pred_upper": pd.Series(upper_values, index=X_test.index, dtype=float),
        },
        index=X_test.index,
    )
    prediction_frame.attrs["model"] = model
    return prediction_frame


def _evaluate_spec_on_validation(
    config: IntervalEstimationConfig,
    spec: ExperimentSpec,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.DataFrame,
    y_validation: pd.DataFrame,
) -> dict[str, object]:
    validation_rows: list[dict[str, object]] = []

    for target in config.targets:
        interval_predictions = _fit_and_predict_interval(spec, X_train, y_train[target], X_validation)
        metrics = compute_interval_metrics(
            y_validation[target],
            interval_predictions["y_pred_lower"],
            interval_predictions["y_pred_upper"],
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
        )
        validation_rows.append(
            {
                "target": target,
                **metrics,
            }
        )

    validation_df = pd.DataFrame(validation_rows)
    expected_coverage = config.upper_quantile - config.lower_quantile
    empirical_coverage = float(validation_df["empirical_coverage"].mean())
    coverage_gap = abs(empirical_coverage - expected_coverage)

    return {
        "validation_mean_pinball_loss": float(validation_df["mean_pinball_loss"].mean()),
        "validation_empirical_coverage": empirical_coverage,
        "validation_coverage_gap": float(coverage_gap),
        "validation_mean_interval_width": float(validation_df["mean_interval_width"].mean()),
    }


def _select_best_tuned_spec(
    config: IntervalEstimationConfig,
    group: ExperimentGroup,
    X_tuning_train: pd.DataFrame,
    X_tuning_validation: pd.DataFrame,
    y_tuning_train: pd.DataFrame,
    y_tuning_validation: pd.DataFrame,
) -> tuple[ExperimentSpec, dict[str, object]]:
    param_grid = INTERVAL_TUNING_PARAM_GRIDS.get((group.model_name, group.objective_name))
    resolved_budget = _group_tuning_budget(config, group)
    spec_lookup = {
        canonicalize_params(spec.params): spec
        for spec in group.candidates
    }
    total_candidates = int(count_param_grid_candidates(param_grid))
    trial_budget = _planned_trial_budget(total_candidates, resolved_budget.n_trials)
    tuning_patience = resolved_budget.patience
    startup_trials = min(8, max(4, trial_budget // 3), trial_budget)
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=startup_trials)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    candidate_failures = 0
    candidates_evaluated = 0
    unique_trials_sampled = 0
    consecutive_non_improvements = 0
    early_stopped = False
    duplicate_suggestions = 0
    ask_attempts = 0
    best_spec: ExperimentSpec | None = None
    best_summary: dict[str, object] | None = None
    best_rank: tuple[float, float, float] | None = None
    attempted_param_keys: set[tuple[tuple[str, object], ...]] = set()
    evaluated_scores: dict[tuple[tuple[str, object], ...], float] = {}

    _log(
        f"Tuning {group.display_name}: "
        f"grid_combinations={total_candidates}, "
        f"trial_budget={trial_budget}, "
        f"validation_period={_format_period(X_tuning_validation)}, "
        f"budget_policy={resolved_budget.label}, "
        f"patience={tuning_patience if tuning_patience is not None else 'none'}"
    )

    max_ask_attempts = max(trial_budget * 5, total_candidates * 2)
    with tqdm(
        total=trial_budget,
        desc=f"Tuning | {group.display_name}",
        unit="trial",
        leave=False,
        dynamic_ncols=True,
    ) as tuning_progress:
        while (
            unique_trials_sampled < trial_budget
            and unique_trials_sampled < total_candidates
            and ask_attempts < max_ask_attempts
        ):
            trial = study.ask()
            ask_attempts += 1
            params = sample_params_from_trial(trial, group.default_spec.params, param_grid)
            params_key = canonicalize_params(params)
            spec = spec_lookup.get(params_key)

            if spec is None:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                raise KeyError(
                    f"Optuna sampled params not found in candidate lookup for {group.display_name}: {params}"
                )

            if params_key in attempted_param_keys:
                duplicate_suggestions += 1
                if params_key in evaluated_scores:
                    study.tell(trial, evaluated_scores[params_key])
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                tuning_progress.set_postfix_str(
                    f"best={best_summary['validation_mean_pinball_loss']:.4f}"
                    if best_summary is not None
                    else "best=n/a"
                )
                continue

            attempted_param_keys.add(params_key)
            unique_trials_sampled += 1

            try:
                summary = _evaluate_spec_on_validation(
                    config,
                    spec,
                    X_tuning_train,
                    X_tuning_validation,
                    y_tuning_train,
                    y_tuning_validation,
                )
                objective_value = float(summary["validation_mean_pinball_loss"])
                evaluated_scores[params_key] = objective_value
                study.tell(trial, objective_value)
                candidates_evaluated += 1
                current_rank = (
                    objective_value,
                    float(summary["validation_coverage_gap"]),
                    float(summary["validation_mean_interval_width"]),
                )
                improved = best_rank is None or current_rank < best_rank
                if improved:
                    best_spec = spec
                    best_summary = summary
                    best_rank = current_rank
                    consecutive_non_improvements = 0
                else:
                    consecutive_non_improvements += 1

                _log(
                    f"  trial {unique_trials_sampled}/{trial_budget} | {group.display_name}: "
                    f"Pinball={summary['validation_mean_pinball_loss']:.4f}, "
                    f"Coverage={summary['validation_empirical_coverage']:.4f}, "
                    f"Width={summary['validation_mean_interval_width']:.4f}"
                    f"{' | new best' if improved else ''}"
                )
            except Exception as exc:  # noqa: BLE001
                candidate_failures += 1
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                _log(f"  tuning trial failed for {group.display_name}: {exc}")
            finally:
                tuning_progress.update(1)
                tuning_progress.set_postfix_str(
                    f"best={best_summary['validation_mean_pinball_loss']:.4f}"
                    if best_summary is not None
                    else f"failures={candidate_failures}"
                )

            if (
                tuning_patience is not None
                and consecutive_non_improvements >= tuning_patience
            ):
                early_stopped = True
                _log(
                    f"  early stopping tuning for {group.display_name} after "
                    f"{consecutive_non_improvements} non-improving candidate(s)."
                )
                break

    if (
        ask_attempts >= max_ask_attempts
        and unique_trials_sampled < trial_budget
        and unique_trials_sampled < total_candidates
    ):
        _log(
            f"  stopping Optuna asks for {group.display_name} after {ask_attempts} attempts "
            f"because too many duplicate suggestions were produced."
        )

    if best_spec is None or best_summary is None:
        raise RuntimeError(f"All tuning candidates failed for {group.display_name}.")

    tuning_metadata = {
        "tuning_active": True,
        "tuning_months": int(config.tuning_months),
        "tuning_budget_label": resolved_budget.label,
        "tuning_group_patience": resolved_budget.group_patience,
        "tuning_group_n_trials": resolved_budget.group_n_trials,
        "configured_tuning_patience": resolved_budget.global_patience,
        "configured_tuning_n_trials": resolved_budget.global_n_trials,
        "tuning_patience": tuning_patience,
        "tuning_n_trials": resolved_budget.n_trials,
        "tuning_metric_name": "mean_pinball_loss",
        "tuning_metric_value": float(best_summary["validation_mean_pinball_loss"]),
        "tuning_validation_coverage": float(best_summary["validation_empirical_coverage"]),
        "tuning_validation_coverage_gap": float(best_summary["validation_coverage_gap"]),
        "tuning_validation_mean_interval_width": float(best_summary["validation_mean_interval_width"]),
        "tuning_candidate_count": total_candidates,
        "tuning_candidates_evaluated": int(candidates_evaluated),
        "tuning_trials_sampled": int(unique_trials_sampled),
        "tuning_trial_budget": int(trial_budget),
        "tuning_candidate_failures": int(candidate_failures),
        "tuning_duplicate_suggestions": int(duplicate_suggestions),
        "tuning_early_stopped": bool(early_stopped),
        "tuning_train_rows": int(len(X_tuning_train)),
        "tuning_validation_rows": int(len(X_tuning_validation)),
        "tuning_validation_start": str(X_tuning_validation.index.min()),
        "tuning_validation_end": str(X_tuning_validation.index.max()),
        "selected_params_summary": best_spec.params_summary,
    }

    _log(
        f"Selected {group.display_name}: "
        f"Pinball={best_summary['validation_mean_pinball_loss']:.4f}, "
        f"params={best_spec.params_summary}, "
        f"sampled={unique_trials_sampled}/{trial_budget}, "
        f"successful={candidates_evaluated}, duplicates={duplicate_suggestions}, total_grid={total_candidates}"
    )

    return best_spec, tuning_metadata


def run_single_experiment(
    config: IntervalEstimationConfig,
    spec: ExperimentSpec,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    extra_result_fields: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    output_dir = output_dir_for(config.results_dir, spec.model_name, spec.objective_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics_summary.csv"

    if X_train.empty or X_test.empty:
        raise RuntimeError(f"Empty train/test split for {spec.display_name}.")

    shared_result_fields = dict(extra_result_fields or {})
    results: list[dict[str, object]] = []
    for target in config.targets:
        _log(f"Starting fit for {spec.display_name} | {target}")
        target_train = y_train[target]
        target_test = y_test[target]
        interval_predictions = _fit_and_predict_interval(spec, X_train, target_train, X_test)
        model = interval_predictions.attrs["model"]
        prediction_frame = build_prediction_frame(
            target_test,
            interval_predictions["y_pred_lower"],
            interval_predictions["y_pred_upper"],
        )

        metrics = compute_interval_metrics(
            target_test,
            prediction_frame["y_pred_lower"],
            prediction_frame["y_pred_upper"],
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
        )
        importance = compute_importance(model, X_test, target_test)

        prediction_plot_path = plot_interval_predictions(
            model_label=spec.display_name,
            target_name=target,
            prediction_frame=prediction_frame,
            output_dir=output_dir,
            metrics=metrics,
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
        )
        importance_plot_path = None
        if not importance.empty:
            importance_plot_path = plot_feature_importances(
                model_label=spec.display_name,
                target_name=target,
                importance=importance,
                output_dir=output_dir,
            )

        target_stem = target_artifact_stem(target)
        prediction_frame_to_save = prediction_frame.reset_index()
        prediction_frame_to_save.columns = ["timestamp", *prediction_frame.columns.tolist()]
        prediction_frame_to_save.to_csv(output_dir / f"{target_stem}_predictions.csv", index=False)
        if not importance.empty:
            importance.to_csv(output_dir / f"{target_stem}_feature_importances.csv", header=True)

        results.append(
            {
                "model_name": spec.model_name,
                "objective_name": spec.objective_name or "",
                "display_name": spec.display_name,
                "target": target,
                **metrics,
                "train_rows": int(len(target_train)),
                "test_rows": int(len(target_test)),
                "prediction_plot": str(prediction_plot_path),
                "importance_plot": str(importance_plot_path or ""),
                "params_summary": spec.params_summary,
                **shared_result_fields,
            }
        )
        pd.DataFrame(results).to_csv(metrics_path, index=False)
        _log(
            f"{spec.display_name} | {target}: "
            f"Pinball={metrics['mean_pinball_loss']:.4f}, "
            f"Coverage={metrics['empirical_coverage']:.4f}, "
            f"Width={metrics['mean_interval_width']:.4f}"
        )

    pd.DataFrame(results).to_csv(metrics_path, index=False)
    combined_plot_path = save_combined_plotly_predictions(
        config.targets,
        spec.display_name,
        output_dir,
        lower_quantile=config.lower_quantile,
        upper_quantile=config.upper_quantile,
    )
    _log(f"{spec.display_name}: metrics saved to {metrics_path}")
    _log(f"{spec.display_name}: combined Plotly HTML saved to {combined_plot_path}")
    return results


def run_all_experiments(config: IntervalEstimationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_final_datasets(config.targets, config.feature_columns)
    X_train_raw, X_test_raw, y_train, y_test = split_train_test_by_year(X, y, test_year=TEST_YEAR)
    X_train, X_test, _ = standardize_features(X_train_raw, X_test_raw)

    tuning_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    tuning_window_start = None
    tuning_window_end = None
    if config.tuning_active:
        (
            X_tuning_train_raw,
            X_tuning_validation_raw,
            y_tuning_train,
            y_tuning_validation,
            tuning_window_start,
            tuning_window_end,
        ) = split_train_and_tuning_validation(
            X_train_raw,
            y_train,
            tuning_months=config.tuning_months,
            test_year=TEST_YEAR,
        )
        X_tuning_train, X_tuning_validation, _ = standardize_features(
            X_tuning_train_raw,
            X_tuning_validation_raw,
        )
        tuning_inputs = (
            X_tuning_train,
            X_tuning_validation,
            y_tuning_train,
            y_tuning_validation,
        )

    experiment_groups = get_experiment_spec_groups(
        config.lower_quantile,
        config.upper_quantile,
        tuning_active=config.tuning_active,
    )

    results_store = {
        _result_key(row): row
        for row in _load_csv_records(config.results_dir / "all_model_results.csv")
    }
    failures_store = {
        _pair_key(row.get("model_name", ""), row.get("objective_name", "")): row
        for row in _load_csv_records(config.results_dir / "failed_experiments.csv")
    }
    tuning_selection_store = {
        _pair_key(row.get("model_name", ""), row.get("objective_name", "")): row
        for row in _load_csv_records(config.results_dir / "tuning_selection_summary.csv")
    }
    best_params_store = {
        _pair_key(row.get("model_name", ""), row.get("objective_name", "")): row
        for row in _load_csv_records(config.results_dir / "best_params.csv")
    }

    _print_run_summary(
        config,
        experiment_groups,
        X_train_raw,
        X_test_raw,
        X_tuning_train_raw if config.tuning_active else None,
        X_tuning_validation_raw if config.tuning_active else None,
    )
    if config.tuning_active and tuning_window_start is not None and tuning_window_end is not None:
        _log(
            f"Tuning window: [{tuning_window_start}, {tuning_window_end}) "
            f"with {config.tuning_months} month(s) held out before TEST_YEAR={TEST_YEAR}."
        )

    with tqdm(
        total=len(experiment_groups),
        desc="Interval Model-Loss Pairs",
        unit="pair",
        dynamic_ncols=True,
    ) as pair_progress:
        for group_index, group in enumerate(experiment_groups, start=1):
            pair_progress.set_postfix_str(group.display_name)
            _log(f"[{group_index}/{len(experiment_groups)}] {group.display_name}")
            pair_key = _pair_key(group.model_name, group.objective_name or "")
            output_dir = _completed_output_dir_for(config, group)

            if _has_completed_pair_outputs(output_dir):
                existing_results = _load_csv_records(output_dir / "metrics_summary.csv")
                if existing_results:
                    _upsert_result_rows(results_store, existing_results)
                failures_store.pop(pair_key, None)
                _log(
                    f"Skipping {group.display_name}: found completed artifacts in {output_dir}"
                )
                _write_run_summaries(
                    config,
                    results_store,
                    failures_store,
                    tuning_selection_store,
                    best_params_store,
                )
                pair_progress.update(1)
                continue

            best_spec = group.default_spec
            extra_result_fields: dict[str, object] | None = None
            selection_row: dict[str, object] | None = None
            if config.tuning_active:
                assert tuning_inputs is not None
                try:
                    best_spec, extra_result_fields = _select_best_tuned_spec(
                        config,
                        group,
                        tuning_inputs[0],
                        tuning_inputs[1],
                        tuning_inputs[2],
                        tuning_inputs[3],
                    )
                    selection_row = {
                        "model_name": group.model_name,
                        "objective_name": group.objective_name or "",
                        "display_name": group.display_name,
                        **extra_result_fields,
                    }
                    tuning_selection_store[pair_key] = selection_row
                except Exception as exc:  # noqa: BLE001
                    failures_store[pair_key] = {
                        "model_name": group.model_name,
                        "objective_name": group.objective_name or "",
                        "display_name": group.display_name,
                        "error": f"tuning selection failed: {exc!r}",
                    }
                    _log(f"FAILED {group.display_name}: {exc}")
                    _write_run_summaries(
                        config,
                        results_store,
                        failures_store,
                        tuning_selection_store,
                        best_params_store,
                    )
                    pair_progress.update(1)
                    continue

            try:
                run_rows = run_single_experiment(
                    config,
                    best_spec,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    extra_result_fields=extra_result_fields,
                )
                _upsert_result_rows(results_store, run_rows)
                failures_store.pop(pair_key, None)
                if config.tuning_active and extra_result_fields is not None:
                    best_params_path, best_params_row = _save_best_params_artifacts(
                        config,
                        best_spec,
                        extra_result_fields,
                    )
                    best_params_store[pair_key] = best_params_row
                    if selection_row is not None:
                        selection_row["best_params_json_path"] = str(best_params_path)
                        selection_row["best_params_json"] = best_params_row["best_params_json"]
                        tuning_selection_store[pair_key] = selection_row
            except Exception as exc:  # noqa: BLE001
                failures_store[pair_key] = {
                    "model_name": group.model_name,
                    "objective_name": group.objective_name or "",
                    "display_name": group.display_name,
                    "error": repr(exc),
                }
                _log(f"FAILED {group.display_name}: {exc}")

            _write_run_summaries(
                config,
                results_store,
                failures_store,
                tuning_selection_store,
                best_params_store,
            )
            pair_progress.update(1)

    return _write_run_summaries(
        config,
        results_store,
        failures_store,
        tuning_selection_store,
        best_params_store,
    )
