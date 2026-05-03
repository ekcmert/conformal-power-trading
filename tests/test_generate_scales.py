from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from generate.generate_scales import generate_scales


class GenerateScalesTests(unittest.TestCase):
    def test_generate_scales_writes_one_timeseries_per_mapped_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x_path = root / "X.csv"
            scales_root = root / "scales"

            dates = pd.to_datetime(
                [
                    "2024-01-01 00:00:00+00:00",
                    "2024-01-01 01:00:00+00:00",
                ]
            )
            pd.DataFrame(
                {
                    "date": dates,
                    "feature_a": [1.0, 2.0],
                    "feature_b": [10.0, 20.0],
                }
            ).to_csv(x_path, index=False)

            summary = generate_scales(
                {
                    "feature_a": "alpha",
                    "feature_b": "beta",
                },
                x_path=x_path,
                scales_root=scales_root,
            )

            self.assertEqual(summary.folders_written, 2)
            self.assertEqual(summary.files_written, 2)
            self.assertEqual(summary.columns_extracted, 2)
            self.assertEqual(summary.row_count, 2)

            alpha_frame = pd.read_csv(scales_root / "alpha" / "scales.csv")
            beta_frame = pd.read_csv(scales_root / "beta" / "scales.csv")

            self.assertEqual(list(alpha_frame.columns), ["date", "target_column"])
            self.assertEqual(list(beta_frame.columns), ["date", "target_column"])
            self.assertEqual(alpha_frame["target_column"].tolist(), [1.0, 2.0])
            self.assertEqual(beta_frame["target_column"].tolist(), [10.0, 20.0])

    def test_generate_scales_raises_for_missing_feature_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x_path = root / "X.csv"

            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-01 00:00:00+00:00"]),
                    "feature_a": [1.0],
                }
            ).to_csv(x_path, index=False)

            with self.assertRaisesRegex(KeyError, "Missing feature columns in X.csv"):
                generate_scales(
                    {"missing_feature": "alpha"},
                    x_path=x_path,
                    scales_root=root / "scales",
                )

    def test_generate_scales_raises_for_duplicate_folder_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x_path = root / "X.csv"

            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-01 00:00:00+00:00"]),
                    "feature_a": [1.0],
                    "feature_b": [2.0],
                }
            ).to_csv(x_path, index=False)

            with self.assertRaisesRegex(ValueError, "must map to exactly one X.csv column"):
                generate_scales(
                    {
                        "feature_a": "alpha",
                        "feature_b": "alpha",
                    },
                    x_path=x_path,
                    scales_root=root / "scales",
                )


if __name__ == "__main__":
    unittest.main()
