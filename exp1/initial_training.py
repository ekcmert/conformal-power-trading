from __future__ import annotations

import traceback
from dataclasses import dataclass

from tqdm.auto import tqdm

from exp1 import train_interval_models, train_point_models


USE_TUNING_CONFIGS = True
RUN_POINT_MCP = False
RUN_POINT_SPREAD = False
RUN_INTERVAL_MCP = True
RUN_INTERVAL_SPREAD = False


@dataclass(frozen=True)
class StageSpec:
    label: str
    module: object
    standard_config_name: str
    tuning_config_name: str
    enabled: bool = True

    def resolve_config(self) -> object:
        config_name = self.tuning_config_name if USE_TUNING_CONFIGS else self.standard_config_name
        return getattr(self.module, config_name)


def _describe_config(config: object) -> str:
    targets = getattr(config, "targets", [])
    results_dir = getattr(config, "results_dir", "n/a")
    tuning_active = getattr(config, "tuning_active", False)
    tuning_months = getattr(config, "tuning_months", None)
    tuning_patience = getattr(config, "tuning_patience", None)
    tuning_n_trials = getattr(config, "tuning_n_trials", None)

    return (
        f"results_dir={results_dir}, "
        f"targets={list(targets)}, "
        f"tuning_active={tuning_active}, "
        f"tuning_months={tuning_months}, "
        f"global_patience_cap={tuning_patience if tuning_patience is not None else 'none'}, "
        f"global_trial_cap={tuning_n_trials if tuning_n_trials is not None else 'none'}, "
        "skip_completed_pairs=True"
    )


def build_stage_specs() -> list[StageSpec]:
    return [
        StageSpec(
            label="Point Models | MCP_CONFIG",
            module=train_point_models,
            standard_config_name="MCP_CONFIG",
            tuning_config_name="TUNING_MCP_CONFIG",
            enabled=RUN_POINT_MCP,
        ),
        StageSpec(
            label="Point Models | SPREAD_CONFIG",
            module=train_point_models,
            standard_config_name="SPREAD_CONFIG",
            tuning_config_name="TUNING_SPREAD_CONFIG",
            enabled=RUN_POINT_SPREAD,
        ),
        StageSpec(
            label="Interval Models | MCP_CONFIG",
            module=train_interval_models,
            standard_config_name="MCP_CONFIG",
            tuning_config_name="TUNING_MCP_CONFIG",
            enabled=RUN_INTERVAL_MCP,
        ),
        StageSpec(
            label="Interval Models | SPREAD_CONFIG",
            module=train_interval_models,
            standard_config_name="SPREAD_CONFIG",
            tuning_config_name="TUNING_SPREAD_CONFIG",
            enabled=RUN_INTERVAL_SPREAD,
        ),
    ]


def run_stage(stage: StageSpec) -> bool:
    config = stage.resolve_config()

    print(f"\n{'=' * 80}")
    print(f"Starting {stage.label}")
    print(f"Configuration: {_describe_config(config)}")
    print(f"{'=' * 80}")

    try:
        stage.module.ACTIVE_CONFIG = config
        stage.module.main()
        print(f"Finished {stage.label}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {stage.label}: {exc}")
        traceback.print_exc()
        return False


def main() -> int:
    stage_specs = [stage for stage in build_stage_specs() if stage.enabled]
    if not stage_specs:
        print("No training stages are enabled.")
        return 0

    print(
        f"Preparing {len(stage_specs)} training stage(s) "
        f"with USE_TUNING_CONFIGS={USE_TUNING_CONFIGS}."
    )

    results: list[bool] = []
    with tqdm(total=len(stage_specs), desc="Training Stages", unit="stage", dynamic_ncols=True) as stage_progress:
        for stage in stage_specs:
            stage_progress.set_postfix_str(stage.label)
            results.append(run_stage(stage))
            stage_progress.update(1)

    failed_runs = sum(not ok for ok in results)
    if failed_runs:
        print(f"\nCompleted with {failed_runs} failed run(s) out of {len(stage_specs)}.")
        return 1

    print(f"\nAll {len(stage_specs)} training run(s) completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
