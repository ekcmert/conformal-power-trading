from __future__ import annotations

from pathlib import Path

from backtest import BacktestConfig, IntervalBacktester
from backtest.config import Y_PATH


def main() -> int:
    pred_path = Path(
        r"C:\Users\mert.ekici\Desktop\CPT\results\regime_aware\mcp\heuristic\da\cp_asymmetric_catboost\rmse"
    )
    position_method = "interval_band"
    entry_band_fraction = 0.3
    exit_band_fraction = 0.75
    signal_family = "softsign"
    signal_scale = 1.5
    signal_power = 0.5
    position_cap = 100.0
    min_interval_width = 1e-6
    output_dir = None
    y_path = Y_PATH

    config = BacktestConfig(
        pred_path=pred_path,
        position_method=position_method,
        entry_band_fraction=entry_band_fraction,
        exit_band_fraction=exit_band_fraction,
        signal_family=signal_family,
        signal_scale=signal_scale,
        signal_power=signal_power,
        position_cap=position_cap,
        min_interval_width=min_interval_width,
        output_dir=Path(output_dir) if output_dir else None,
        y_path=Path(y_path),
    )
    artifacts = IntervalBacktester(config).run()

    print(f"Backtest output dir: {artifacts.output_dir}")
    print(f"Hourly results     : {artifacts.hourly_results_path}")
    print(f"Summary CSV        : {artifacts.summary_path}")
    print(f"Dashboard HTML     : {artifacts.dashboard_path}")
    print(f"Metadata JSON      : {artifacts.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
