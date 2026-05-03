from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from generate.generate_residuals import generate_residuals


class GenerateResidualsTests(unittest.TestCase):
    def test_generate_residuals_writes_point_and_interval_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions_root = root / "predictions"
            residuals_root = root / "residuals"
            y_path = root / "y.csv"

            dates = pd.to_datetime(
                [
                    "2024-01-01 00:00:00+00:00",
                    "2024-01-01 01:00:00+00:00",
                ]
            )
            y_frame = pd.DataFrame(
                {
                    "date": dates,
                    "DE Price Spot EUR/MWh EPEX H Actual": [10.0, 20.0],
                    "DE Price Intraday VWAP EUR/MWh EPEX H Actual": [1.0, 2.0],
                    "DE Price Intraday VWAP ID1 EUR/MWh EPEX H Actual": [3.0, 4.0],
                    "DE Price Intraday VWAP ID3 EUR/MWh EPEX H Actual": [5.0, 6.0],
                    "DE Volume Imbalance Net MWh 15min Actual": [7.0, 8.0],
                }
            )
            y_frame.to_csv(y_path, index=False)

            point_dir = predictions_root / "point_estimation" / "lightgbm" / "regression"
            point_dir.mkdir(parents=True, exist_ok=True)
            (point_dir / "run_metadata.json").write_text('{"estimation_kind": "point"}')
            pd.DataFrame(
                {
                    "date": dates,
                    "y_true": [999.0, 999.0],
                    "y_pred": [8.0, 23.0],
                }
            ).to_csv(point_dir / "DE Price Spot EUR_MWh EPEX H Actual_predictions.csv", index=False)

            interval_dir = predictions_root / "interval_estimation" / "lightgbm" / "quantile"
            interval_dir.mkdir(parents=True, exist_ok=True)
            (interval_dir / "run_metadata.json").write_text('{"estimation_kind": "interval"}')
            pd.DataFrame(
                {
                    "date": dates,
                    "y_true": [999.0, 999.0],
                    "y_pred_lower": [7.0, 18.0],
                    "y_pred_upper": [12.0, 21.0],
                }
            ).to_csv(interval_dir / "DE Price Spot EUR_MWh EPEX H Actual_predictions.csv", index=False)

            summary = generate_residuals(
                predictions_root=predictions_root,
                residuals_root=residuals_root,
                y_path=y_path,
            )

            self.assertEqual(summary.model_loss_directories, 2)
            self.assertEqual(summary.point_files_written, 5)
            self.assertEqual(summary.interval_files_written, 10)
            self.assertEqual(summary.skipped_prediction_files, 0)

            point_residual = pd.read_csv(
                residuals_root / "point_estimation" / "lightgbm" / "regression" / "da_res.csv"
            )
            point_id_residual = pd.read_csv(
                residuals_root / "point_estimation" / "lightgbm" / "regression" / "id_res.csv"
            )
            interval_low = pd.read_csv(
                residuals_root / "interval_estimation" / "lightgbm" / "quantile" / "da_res_low.csv"
            )
            interval_up = pd.read_csv(
                residuals_root / "interval_estimation" / "lightgbm" / "quantile" / "da_res_up.csv"
            )
            interval_id_low = pd.read_csv(
                residuals_root / "interval_estimation" / "lightgbm" / "quantile" / "id_res_low.csv"
            )

            self.assertEqual(list(point_residual.columns), ["date", "da_res"])
            self.assertEqual(list(point_id_residual.columns), ["date", "id_res"])
            self.assertEqual(list(interval_low.columns), ["date", "da_res_low"])
            self.assertEqual(list(interval_up.columns), ["date", "da_res_up"])
            self.assertEqual(list(interval_id_low.columns), ["date", "id_res_low"])

            self.assertEqual(point_residual["da_res"].tolist(), [2.0, -3.0])
            self.assertEqual(point_id_residual["id_res"].tolist(), [-7.0, -21.0])
            self.assertEqual(interval_low["da_res_low"].tolist(), [3.0, 2.0])
            self.assertEqual(interval_up["da_res_up"].tolist(), [-2.0, -1.0])
            self.assertEqual(interval_id_low["id_res_low"].tolist(), [-6.0, -16.0])


if __name__ == "__main__":
    unittest.main()
