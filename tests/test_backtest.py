from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest import BacktestConfig, IntervalBacktester, apply_size_mapping, compute_scenario_metrics


def test_apply_size_mapping_step_mode_respects_threshold_bands() -> None:
    margins = pd.Series([-12.0, -7.0, -1.0, 0.0, 3.0, 9.0, 20.0])
    size_map = {-10.0: 5.0, -5.0: 2.0, 0.0: 0.0, 5.0: -2.0, 10.0: -5.0}

    mapped = apply_size_mapping(margins, size_map=size_map, mode="step")

    assert mapped.tolist() == [5.0, 2.0, 0.0, 0.0, 0.0, -2.0, -5.0]


def test_scenario_metrics_use_volume_weighted_unit_pnl_and_active_hit_ratio() -> None:
    pnl = pd.Series([100.0, -20.0, 20.0, 0.0])
    position = pd.Series([10.0, -5.0, 2.0, 0.0])

    metrics = compute_scenario_metrics(pnl, position)

    assert abs(metrics["avg_unit_pnl"] - (100.0 / 17.0)) < 1e-6
    assert abs(metrics["hit_ratio"] - (2.0 / 3.0)) < 1e-6

    all_profitable_metrics = compute_scenario_metrics(
        pd.Series([10.0, 5.0, 0.0]),
        pd.Series([2.0, -1.0, 0.0]),
    )
    assert abs(all_profitable_metrics["hit_ratio"] - 1.0) < 1e-6


def test_interval_backtester_generates_expected_pnl_scenarios(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "results" / "regime_aware" / "mcp" / "heuristic" / "da" / "model" / "objective"
    prediction_dir.mkdir(parents=True)
    prediction_csv = prediction_dir / "DE Price Spot EUR_MWh EPEX H Actual_predictions.csv"
    prediction_frame = pd.DataFrame(
        {
            "date": [
                "2025-01-01 00:00:00+00:00",
                "2025-01-01 01:00:00+00:00",
                "2025-01-01 02:00:00+00:00",
                "2025-01-01 03:00:00+00:00",
            ],
            "y_pred_lower": [80.0, 80.0, 80.0, 80.0],
            "y_pred_upper": [120.0, 120.0, 120.0, 120.0],
            "interval_width": [40.0, 40.0, 40.0, 40.0],
            "initial_y_pred_center": [100.0, 100.0, 100.0, 100.0],
        }
    )
    prediction_frame.to_csv(prediction_csv, index=False)

    y_csv = tmp_path / "data" / "final" / "y.csv"
    y_csv.parent.mkdir(parents=True)
    price_frame = pd.DataFrame(
        {
            "date": [
                "2025-01-01 00:00:00+00:00",
                "2025-01-01 01:00:00+00:00",
                "2025-01-01 02:00:00+00:00",
                "2025-01-01 03:00:00+00:00",
            ],
            "DE Price Spot EUR/MWh EPEX H Actual": [110.0, 118.0, 90.0, 82.0],
            "DE Price Intraday VWAP EUR/MWh EPEX H Actual": [100.0, 100.0, 100.0, 100.0],
            "DE Price Intraday VWAP ID3 EUR/MWh EPEX H Actual": [100.0, 100.0, 100.0, 100.0],
            "DE Price Intraday VWAP ID1 EUR/MWh EPEX H Actual": [100.0, 100.0, 100.0, 100.0],
            "DE Price Imbalance Single EUR/MWh 15min Actual": [100.0, 100.0, 100.0, 100.0],
        }
    )
    price_frame.to_csv(y_csv, index=False)

    backtester = IntervalBacktester(
        BacktestConfig(
            pred_path=prediction_dir,
            y_path=y_csv,
            output_dir=tmp_path / "results" / "backtest" / "test_run",
            position_method="interval_band",
            entry_band_fraction=0.25,
            exit_band_fraction=0.75,
            signal_family="linear",
            signal_scale=1.0,
            signal_power=1.0,
            position_cap=10.0,
        )
    )
    artifacts = backtester.run()

    hourly = artifacts.hourly_results
    assert hourly["strategy_position"].tolist() == [-5.0, 0.0, 5.0, 0.0]
    assert hourly["strategy_close_id_unit_pnl"].dropna().tolist() == [10.0, 10.0]

    summary = artifacts.summary.set_index("scenario_name")
    assert abs(summary.loc["strategy_close_id", "cumulative_pnl"] - 100.0) < 1e-6
    assert abs(summary.loc["strategy_close_id", "avg_unit_pnl"] - 10.0) < 1e-6
    assert abs(summary.loc["strategy_close_id", "hit_ratio"] - 1.0) < 1e-6
    assert abs(summary.loc["strategy_perfect_close", "cumulative_pnl"] - 100.0) < 1e-6
    assert abs(summary.loc["perfect_direction_close_imb", "cumulative_pnl"] - 100.0) < 1e-6
    assert abs(summary.loc["perfect_direction_perfect_close", "cumulative_pnl"] - 100.0) < 1e-6
    assert artifacts.dashboard_path.exists()
    assert artifacts.hourly_results_path.exists()
    assert artifacts.summary_path.exists()
