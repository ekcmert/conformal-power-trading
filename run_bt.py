from __future__ import annotations

from pathlib import Path

from backtest import BacktestConfig, IntervalBacktester
from backtest.config import Y_PATH
from bt import optimize_strategy


RUN_BACKTEST_PIPELINE = False
RUN_OPTIMIZE_STRATEGY = True

PRED_PATH = Path(
    r"C:\Users\mert.ekici\Desktop\CPT\results\regime_aware\aci\0.1\da\cqr_quantile_extra_trees\squared_error"
)
OUTPUT_DIR: Path | None = None

BACKTEST_PARAMS = {
    "position_method": "interval_band",
    "entry_band_fraction": 0.3,
    "exit_band_fraction": 0.75,
    "signal_family": "softsign",
    "signal_scale": 1.5,
    "signal_power": 0.5,
    "position_cap": 100.0,
    "min_interval_width": 1e-6,
}

OBJECTIVE_WEIGHTS = {
    "unit_pnl": 1.0,
    "hit_ratio": 0.0,
}


def main() -> int:
    if RUN_BACKTEST_PIPELINE:
        config = BacktestConfig(
            pred_path=PRED_PATH,
            output_dir=OUTPUT_DIR,
            y_path=Path(Y_PATH),
            **BACKTEST_PARAMS,
        )
        artifacts = IntervalBacktester(config).run()

        print(f"Backtest output dir: {artifacts.output_dir}")
        print(f"Hourly results     : {artifacts.hourly_results_path}")
        print(f"Summary CSV        : {artifacts.summary_path}")
        print(f"Dashboard HTML     : {artifacts.dashboard_path}")
        print(f"Metadata JSON      : {artifacts.metadata_path}")

    if RUN_OPTIMIZE_STRATEGY:
        optimize_strategy.main(
            pred_path=PRED_PATH,
            base_params=BACKTEST_PARAMS,
            objective_weights=OBJECTIVE_WEIGHTS,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
