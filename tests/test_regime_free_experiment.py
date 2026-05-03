from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from exp2.regime_free_experiment import run_regime_free_experiment


class RegimeFreeExperimentTests(unittest.TestCase):
    def test_pipeline_writes_ranked_results_for_point_and_interval_methods(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions_root = root / "predictions"
            residuals_root = root / "residuals"
            results_root = root / "results"
            y_path = root / "y.csv"

            dates = pd.date_range("2024-01-01", periods=35, freq="D", tz="UTC")
            y_frame = pd.DataFrame(
                {
                    "date": dates,
                    "DE Price Spot EUR/MWh EPEX H Actual": range(35),
                    "DE Price Intraday VWAP EUR/MWh EPEX H Actual": range(100, 135),
                }
            )
            y_frame.to_csv(y_path, index=False)

            point_prediction_dir = predictions_root / "point_estimation" / "lightgbm" / "regression"
            point_prediction_dir.mkdir(parents=True, exist_ok=True)
            point_residual_dir = residuals_root / "point_estimation" / "lightgbm" / "regression"
            point_residual_dir.mkdir(parents=True, exist_ok=True)
            (point_prediction_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "display_name": "LightGBM | regression",
                        "model_name": "lightgbm",
                        "objective_name": "regression",
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "date": dates,
                    "y_pred": [value - 1.0 for value in range(35)],
                }
            ).to_csv(
                point_prediction_dir / "DE Price Spot EUR_MWh EPEX H Actual_predictions.csv",
                index=False,
            )
            pd.DataFrame({"date": dates, "da_res": [1.0] * 35}).to_csv(
                point_residual_dir / "da_res.csv",
                index=False,
            )
            pd.DataFrame({"date": dates, "id_res": [2.0] * 35}).to_csv(
                point_residual_dir / "id_res.csv",
                index=False,
            )

            interval_prediction_dir = predictions_root / "interval_estimation" / "lightgbm" / "quantile"
            interval_prediction_dir.mkdir(parents=True, exist_ok=True)
            interval_residual_dir = residuals_root / "interval_estimation" / "lightgbm" / "quantile"
            interval_residual_dir.mkdir(parents=True, exist_ok=True)
            (interval_prediction_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "display_name": "LightGBM | quantile",
                        "model_name": "lightgbm",
                        "objective_name": "quantile",
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "date": dates,
                    "y_pred_lower": [value - 2.0 for value in range(35)],
                    "y_pred_upper": [value + 2.0 for value in range(35)],
                }
            ).to_csv(
                interval_prediction_dir / "DE Price Spot EUR_MWh EPEX H Actual_predictions.csv",
                index=False,
            )
            pd.DataFrame({"date": dates, "da_res_low": [2.0] * 35}).to_csv(
                interval_residual_dir / "da_res_low.csv",
                index=False,
            )
            pd.DataFrame({"date": dates, "da_res_up": [-2.0] * 35}).to_csv(
                interval_residual_dir / "da_res_up.csv",
                index=False,
            )
            pd.DataFrame({"date": dates, "id_res_low": [3.0] * 35}).to_csv(
                interval_residual_dir / "id_res_low.csv",
                index=False,
            )
            pd.DataFrame({"date": dates, "id_res_up": [-1.0] * 35}).to_csv(
                interval_residual_dir / "id_res_up.csv",
                index=False,
            )

            ranked_results, failures_df = run_regime_free_experiment(
                calibration_range_weeks=2,
                calibration_frequency_weeks=1,
                target_list=["DA", "ID"],
                predictions_root=predictions_root,
                residuals_root=residuals_root,
                y_path=y_path,
                results_root=results_root,
                clean_output=True,
            )

            self.assertTrue(failures_df.empty)
            self.assertEqual(set(ranked_results["target_code"]), {"DA", "ID"})
            self.assertEqual(set(ranked_results["method_name"]), {"cp_symmetric", "cp_asymmetric", "cqr"})

            required_rank_columns = {
                "pinball_rank",
                "mean_interval_width_rank",
                "empirical_coverage_rank",
                "average_rank",
                "selection_order",
            }
            self.assertTrue(required_rank_columns.issubset(set(ranked_results.columns)))
            self.assertTrue((results_root / "all_method_results.csv").exists())
            self.assertTrue((results_root / "da" / "all_method_results.csv").exists())
            self.assertTrue((results_root / "id" / "all_method_results.csv").exists())

            cp_prediction_csv = (
                results_root
                / "da"
                / "cp_symmetric_lightgbm"
                / "regression"
                / "DE Price Spot EUR_MWh EPEX H Actual_predictions.csv"
            )
            cqr_prediction_csv = (
                results_root
                / "id"
                / "cqr_lightgbm"
                / "quantile"
                / "DE Price Intraday VWAP EUR_MWh EPEX H Actual_predictions.csv"
            )
            self.assertTrue(cp_prediction_csv.exists())
            self.assertTrue(cqr_prediction_csv.exists())

            cp_predictions = pd.read_csv(cp_prediction_csv)
            cqr_predictions = pd.read_csv(cqr_prediction_csv)
            self.assertIn("margin_lower", cp_predictions.columns)
            self.assertIn("margin_upper", cp_predictions.columns)
            self.assertIn("calibration_start", cp_predictions.columns)
            self.assertIn("prediction_end_exclusive", cp_predictions.columns)
            self.assertIn("initial_prediction", cp_predictions.columns)
            self.assertIn("initial_y_pred_lower", cqr_predictions.columns)
            self.assertIn("initial_y_pred_upper", cqr_predictions.columns)


if __name__ == "__main__":
    unittest.main()
