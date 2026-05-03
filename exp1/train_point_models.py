from __future__ import annotations

from pathlib import Path

from base_models.point_estimation import PointEstimationConfig, run_all_experiments
from base_models.point_estimation.common import MCP_TARGETS, REPO_ROOT, SPREAD_TARGETS


TUNING_ACTIVE = True
TUNING_MONTHS = 6
TUNING_PATIENCE = 12
TUNING_N_TRIALS = 32


def _format_global_cap(value: int | None) -> str:
    return str(value) if value is not None else "none"


def _results_dir(experiment_name: str, *, tuning_active: bool) -> Path:
    base_results_dir = REPO_ROOT / "results"
    if tuning_active:
        base_results_dir = base_results_dir / "tuning"
    return base_results_dir / "point_estimation" / experiment_name


def build_config(
    *,
    targets: list[str] | tuple[str, ...],
    experiment_name: str,
    tuning_active: bool,
) -> PointEstimationConfig:
    return PointEstimationConfig(
        targets=targets,
        results_dir=_results_dir(experiment_name, tuning_active=tuning_active),
        feature_columns=None,
        tuning_active=tuning_active,
        tuning_months=TUNING_MONTHS,
        tuning_patience=TUNING_PATIENCE,
        tuning_n_trials=TUNING_N_TRIALS,
    )


SPREAD_CONFIG = build_config(
    targets=SPREAD_TARGETS,
    experiment_name="spread_models",
    tuning_active=TUNING_ACTIVE,
)

MCP_CONFIG = build_config(
    targets=MCP_TARGETS,
    experiment_name="mcp_models",
    tuning_active=TUNING_ACTIVE,
)

TUNING_SPREAD_CONFIG = build_config(
    targets=SPREAD_TARGETS,
    experiment_name="spread_models",
    tuning_active=True,
)

TUNING_MCP_CONFIG = build_config(
    targets=MCP_TARGETS,
    experiment_name="mcp_models",
    tuning_active=True,
)

ACTIVE_CONFIG = MCP_CONFIG


def main() -> None:
    print(
        f"Active point config: results_dir={ACTIVE_CONFIG.results_dir}, "
        f"targets={list(ACTIVE_CONFIG.targets)}, "
        f"tuning_active={ACTIVE_CONFIG.tuning_active}, "
        f"tuning_months={ACTIVE_CONFIG.tuning_months}, "
        f"global_patience_cap={_format_global_cap(ACTIVE_CONFIG.tuning_patience)}, "
        f"global_trial_cap={_format_global_cap(ACTIVE_CONFIG.tuning_n_trials)}, "
        "skip_completed_pairs=True",
        flush=True,
    )
    results_df, failures_df = run_all_experiments(ACTIVE_CONFIG)
    print(f"Saved point-estimation outputs to {ACTIVE_CONFIG.results_dir}", flush=True)
    if not results_df.empty:
        print(
            results_df[["model_name", "objective_name", "target", "mae", "rmse", "r2"]].to_string(index=False),
            flush=True,
        )
    if not failures_df.empty:
        print("Some experiments failed:", flush=True)
        print(failures_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
