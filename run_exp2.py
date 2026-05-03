from __future__ import annotations

from exp2.regime_free_experiment import (
    CALIBRATION_FREQUENCY,
    CALIBRATION_RANGE,
    TARGET_LIST,
    run_regime_free_experiment,
)


def main() -> int:
    ranked_results, failures_df = run_regime_free_experiment(
        calibration_range_weeks=CALIBRATION_RANGE,
        calibration_frequency_weeks=CALIBRATION_FREQUENCY,
        target_list=TARGET_LIST,
        clean_output=False,
    )

    print(f"Completed regime-free runs: {len(ranked_results):,}")
    print(f"Failures                  : {len(failures_df):,}")
    return 0 if failures_df.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
