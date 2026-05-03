from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from generate.generate_regimes import (
    LABEL_COLUMN_NAME,
    NUM_CLUSTERS_GRID,
    RegimeGenerationSummary,
    generate_all_regimes,
    generate_regimes as generate_regimes_generic,
    generate_regime_grid,
    generate_heuristic_regimes,
)
from regime_discovery.heuristic_regime import (
    DEFAULT_HOUR_COLUMN,
    DEFAULT_MONTH_COLUMN,
    DEFAULT_RESIDUAL_LOAD_COLUMN,
    DEFAULT_SOLAR_COLUMN,
    DEFAULT_WEEKEND_COLUMN,
    DEFAULT_WIND_COLUMN,
)
from regime_discovery.regime_config import REGIME_COLUMNS, REGIME_COLUMNS_PCA


class GenerateRegimesTests(unittest.TestCase):
    def test_generate_regimes_skips_existing_non_empty_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            regimes_root = root / "regimes"
            output_path = regimes_root / "heuristic" / "regimes.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-03-01 00:00:00+00:00"]),
                    "label": ["existing"],
                }
            ).to_csv(output_path, index=False)

            with patch("generate.generate_regimes.tqdm.write") as mock_write:
                summary = generate_heuristic_regimes(
                    x_path=root / "missing_X.csv",
                    regimes_root=regimes_root,
                )

            self.assertTrue(summary.skipped)
            self.assertEqual(summary.rows_written, 0)
            self.assertEqual(summary.output_path.resolve(), output_path.resolve())
            self.assertEqual(summary.label_column, LABEL_COLUMN_NAME)
            mock_write.assert_called_once()
            self.assertIn("Skipping regime generation for 'heuristic'", mock_write.call_args.args[0])

    def test_generate_heuristic_regimes_writes_out_of_sample_rolling_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x_path = root / "X.csv"
            regimes_root = root / "regimes"

            input_frame = pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        [
                            "2024-01-01 00:00:00+00:00",
                            "2024-02-01 00:00:00+00:00",
                            "2024-03-01 00:00:00+00:00",
                            "2024-04-01 00:00:00+00:00",
                            "2024-05-01 00:00:00+00:00",
                        ]
                    ),
                    DEFAULT_HOUR_COLUMN: [0, 0, 0, 0, 0],
                    DEFAULT_MONTH_COLUMN: [1, 2, 3, 4, 5],
                    DEFAULT_RESIDUAL_LOAD_COLUMN: [10.0, 20.0, 30.0, 40.0, 50.0],
                    DEFAULT_WIND_COLUMN: [3.0, 4.0, 5.0, 6.0, 7.0],
                    DEFAULT_SOLAR_COLUMN: [0.0, 0.0, 1.0, 2.0, 3.0],
                    DEFAULT_WEEKEND_COLUMN: [0, 0, 0, 0, 0],
                }
            )
            input_frame.to_csv(x_path, index=False)

            expected_windows = [
                (
                    [
                        pd.Timestamp("2024-01-01 00:00:00+0000", tz="UTC"),
                        pd.Timestamp("2024-02-01 00:00:00+0000", tz="UTC"),
                    ],
                    [pd.Timestamp("2024-03-01 00:00:00+0000", tz="UTC")],
                ),
                (
                    [
                        pd.Timestamp("2024-02-01 00:00:00+0000", tz="UTC"),
                        pd.Timestamp("2024-03-01 00:00:00+0000", tz="UTC"),
                    ],
                    [pd.Timestamp("2024-04-01 00:00:00+0000", tz="UTC")],
                ),
                (
                    [
                        pd.Timestamp("2024-03-01 00:00:00+0000", tz="UTC"),
                        pd.Timestamp("2024-04-01 00:00:00+0000", tz="UTC"),
                    ],
                    [pd.Timestamp("2024-05-01 00:00:00+0000", tz="UTC")],
                ),
            ]
            seen_windows: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]] = []

            def fake_assign(
                df_sample: pd.DataFrame,
                df_pred: pd.DataFrame,
                **kwargs: object,
            ) -> pd.DataFrame:
                self.assertEqual(kwargs, {"output_col": LABEL_COLUMN_NAME})
                seen_windows.append((df_sample["date"].tolist(), df_pred["date"].tolist()))

                return df_pred.assign(
                    label=df_pred["date"].dt.strftime("%Y-%m").tolist()
                )

            with patch("generate.generate_regimes.assign_german_da_regimes", side_effect=fake_assign):
                summary = generate_heuristic_regimes(
                    x_path=x_path,
                    regimes_root=regimes_root,
                    regime_range=2,
                    regime_frequency=1,
                )

            output_path = regimes_root / "heuristic" / "regimes.csv"
            exported_frame = pd.read_csv(output_path, parse_dates=["date"])

            self.assertEqual(seen_windows, expected_windows)
            self.assertEqual(summary.rows_written, 3)
            self.assertEqual(summary.output_path.resolve(), output_path.resolve())
            self.assertEqual(summary.label_column, LABEL_COLUMN_NAME)
            self.assertEqual(list(exported_frame.columns), ["date", "label"])
            self.assertEqual(
                exported_frame["date"].tolist(),
                [window[1][0] for window in expected_windows],
            )
            self.assertEqual(exported_frame["label"].tolist(), ["2024-03", "2024-04", "2024-05"])

    def test_generate_heuristic_regimes_skips_partial_first_month_when_building_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x_path = root / "X.csv"
            regimes_root = root / "regimes"

            pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        [
                            "2024-01-15 00:00:00+00:00",
                            "2024-02-01 00:00:00+00:00",
                            "2024-03-01 00:00:00+00:00",
                            "2024-04-01 00:00:00+00:00",
                            "2024-05-01 00:00:00+00:00",
                        ]
                    ),
                    DEFAULT_HOUR_COLUMN: [0, 0, 0, 0, 0],
                    DEFAULT_MONTH_COLUMN: [1, 2, 3, 4, 5],
                    DEFAULT_RESIDUAL_LOAD_COLUMN: [10.0, 20.0, 30.0, 40.0, 50.0],
                    DEFAULT_WIND_COLUMN: [3.0, 4.0, 5.0, 6.0, 7.0],
                    DEFAULT_SOLAR_COLUMN: [0.0, 0.0, 1.0, 2.0, 3.0],
                    DEFAULT_WEEKEND_COLUMN: [0, 0, 0, 0, 0],
                }
            ).to_csv(x_path, index=False)

            seen_prediction_dates: list[list[pd.Timestamp]] = []

            def fake_assign(
                df_sample: pd.DataFrame,
                df_pred: pd.DataFrame,
                **_: object,
            ) -> pd.DataFrame:
                seen_prediction_dates.append(df_pred["date"].tolist())
                return df_pred.assign(label=["shifted"])

            with patch("generate.generate_regimes.assign_german_da_regimes", side_effect=fake_assign):
                summary = generate_heuristic_regimes(
                    x_path=x_path,
                    regimes_root=regimes_root,
                    regime_range=2,
                    regime_frequency=1,
                )

            self.assertEqual(
                seen_prediction_dates,
                [
                    [pd.Timestamp("2024-04-01 00:00:00+0000", tz="UTC")],
                    [pd.Timestamp("2024-05-01 00:00:00+0000", tz="UTC")],
                ],
            )
            self.assertEqual(summary.rows_written, 2)

    def test_generate_heuristic_regimes_raises_for_missing_feature_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x_path = root / "X.csv"

            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-01 00:00:00+00:00"]),
                    DEFAULT_HOUR_COLUMN: [0],
                    DEFAULT_MONTH_COLUMN: [1],
                    DEFAULT_RESIDUAL_LOAD_COLUMN: [10.0],
                    DEFAULT_WIND_COLUMN: [3.0],
                    DEFAULT_WEEKEND_COLUMN: [0],
                }
            ).to_csv(x_path, index=False)

            with self.assertRaisesRegex(KeyError, "Missing feature columns in X.csv"):
                generate_heuristic_regimes(
                    x_path=x_path,
                    regimes_root=root / "regimes",
                )

    def test_generate_heuristic_regimes_raises_when_no_rolling_windows_can_be_built(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x_path = root / "X.csv"

            pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        [
                            "2024-01-01 00:00:00+00:00",
                            "2024-02-01 00:00:00+00:00",
                        ]
                    ),
                    DEFAULT_HOUR_COLUMN: [0, 0],
                    DEFAULT_MONTH_COLUMN: [1, 2],
                    DEFAULT_RESIDUAL_LOAD_COLUMN: [10.0, 20.0],
                    DEFAULT_WIND_COLUMN: [3.0, 4.0],
                    DEFAULT_SOLAR_COLUMN: [1.0, 2.0],
                    DEFAULT_WEEKEND_COLUMN: [0, 0],
                }
            ).to_csv(x_path, index=False)

            with self.assertRaisesRegex(RuntimeError, "No rolling monthly regime windows could be built"):
                generate_heuristic_regimes(
                    x_path=x_path,
                    regimes_root=root / "regimes",
                    regime_range=2,
                    regime_frequency=1,
                )

    def test_generate_regimes_dispatches_clustering_methods_through_regime_clustering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x_path = root / "X.csv"
            regimes_root = root / "regimes"

            pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        [
                            "2024-01-01 00:00:00+00:00",
                            "2024-02-01 00:00:00+00:00",
                            "2024-03-01 00:00:00+00:00",
                            "2024-04-01 00:00:00+00:00",
                            "2024-05-01 00:00:00+00:00",
                        ]
                    ),
                    DEFAULT_HOUR_COLUMN: [0, 1, 2, 3, 4],
                    DEFAULT_MONTH_COLUMN: [1, 2, 3, 4, 5],
                    DEFAULT_RESIDUAL_LOAD_COLUMN: [10.0, 20.0, 30.0, 40.0, 50.0],
                    DEFAULT_WIND_COLUMN: [3.0, 4.0, 5.0, 6.0, 7.0],
                    DEFAULT_SOLAR_COLUMN: [0.0, 1.0, 2.0, 3.0, 4.0],
                    DEFAULT_WEEKEND_COLUMN: [0, 0, 0, 0, 0],
                }
            ).to_csv(x_path, index=False)

            calls: list[dict[str, object]] = []

            class FakeRegimeClustering:
                def __init__(
                    self,
                    df_sample: pd.DataFrame,
                    df_pred: pd.DataFrame,
                    *,
                    num_clusters: int,
                    column_names: list[str],
                    is_pca: bool,
                    pca_dim: int,
                ) -> None:
                    calls.append(
                        {
                            "sample_dates": df_sample["date"].tolist(),
                            "pred_dates": df_pred["date"].tolist(),
                            "num_clusters": num_clusters,
                            "column_names": column_names,
                            "is_pca": is_pca,
                            "pca_dim": pca_dim,
                        }
                    )
                    self.df_pred = df_pred

                def assign(self, method_name: str, *, output_col: str = "regime") -> pd.DataFrame:
                    calls[-1]["method_name"] = method_name
                    return self.df_pred.assign(
                        **{output_col: self.df_pred["date"].dt.strftime("%Y-%m").tolist()}
                    )

            with patch("generate.generate_regimes.RegimeClustering", FakeRegimeClustering):
                summary = generate_regimes_generic(
                    method_name="kmeans",
                    x_path=x_path,
                    regimes_root=regimes_root,
                    clustering_columns=[DEFAULT_HOUR_COLUMN, DEFAULT_MONTH_COLUMN],
                    num_clusters=3,
                    is_pca=True,
                    pca_dim=2,
                    regime_range=2,
                    regime_frequency=1,
                )

            output_path = regimes_root / "kmeans_pca_3" / "regimes.csv"
            exported_frame = pd.read_csv(output_path, parse_dates=["date"])

            self.assertEqual(summary.rows_written, 3)
            self.assertEqual(summary.output_path.resolve(), output_path.resolve())
            self.assertEqual(
                calls,
                [
                    {
                        "sample_dates": [
                            pd.Timestamp("2024-01-01 00:00:00+0000", tz="UTC"),
                            pd.Timestamp("2024-02-01 00:00:00+0000", tz="UTC"),
                        ],
                        "pred_dates": [pd.Timestamp("2024-03-01 00:00:00+0000", tz="UTC")],
                        "num_clusters": 3,
                        "column_names": [DEFAULT_HOUR_COLUMN, DEFAULT_MONTH_COLUMN],
                        "is_pca": True,
                        "pca_dim": 2,
                        "method_name": "kmeans",
                    },
                    {
                        "sample_dates": [
                            pd.Timestamp("2024-02-01 00:00:00+0000", tz="UTC"),
                            pd.Timestamp("2024-03-01 00:00:00+0000", tz="UTC"),
                        ],
                        "pred_dates": [pd.Timestamp("2024-04-01 00:00:00+0000", tz="UTC")],
                        "num_clusters": 3,
                        "column_names": [DEFAULT_HOUR_COLUMN, DEFAULT_MONTH_COLUMN],
                        "is_pca": True,
                        "pca_dim": 2,
                        "method_name": "kmeans",
                    },
                    {
                        "sample_dates": [
                            pd.Timestamp("2024-03-01 00:00:00+0000", tz="UTC"),
                            pd.Timestamp("2024-04-01 00:00:00+0000", tz="UTC"),
                        ],
                        "pred_dates": [pd.Timestamp("2024-05-01 00:00:00+0000", tz="UTC")],
                        "num_clusters": 3,
                        "column_names": [DEFAULT_HOUR_COLUMN, DEFAULT_MONTH_COLUMN],
                        "is_pca": True,
                        "pca_dim": 2,
                        "method_name": "kmeans",
                    },
                ],
            )
            self.assertEqual(list(exported_frame.columns), ["date", "label"])
            self.assertEqual(exported_frame["label"].tolist(), ["2024-03", "2024-04", "2024-05"])

    def test_generate_regime_grid_uses_cluster_grid_in_output_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x_path = root / "X.csv"
            regimes_root = root / "regimes"

            pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        [
                            "2024-01-01 00:00:00+00:00",
                            "2024-02-01 00:00:00+00:00",
                            "2024-03-01 00:00:00+00:00",
                            "2024-04-01 00:00:00+00:00",
                        ]
                    ),
                    DEFAULT_HOUR_COLUMN: [0, 1, 2, 3],
                    DEFAULT_MONTH_COLUMN: [1, 2, 3, 4],
                    DEFAULT_RESIDUAL_LOAD_COLUMN: [10.0, 20.0, 30.0, 40.0],
                    DEFAULT_WIND_COLUMN: [3.0, 4.0, 5.0, 6.0],
                    DEFAULT_SOLAR_COLUMN: [0.0, 1.0, 2.0, 3.0],
                    DEFAULT_WEEKEND_COLUMN: [0, 0, 0, 0],
                }
            ).to_csv(x_path, index=False)

            class FakeRegimeClustering:
                def __init__(
                    self,
                    df_sample: pd.DataFrame,
                    df_pred: pd.DataFrame,
                    *,
                    num_clusters: int,
                    column_names: list[str],
                    is_pca: bool,
                    pca_dim: int,
                ) -> None:
                    self.df_pred = df_pred
                    self.num_clusters = num_clusters

                def assign(self, method_name: str, *, output_col: str = "regime") -> pd.DataFrame:
                    return self.df_pred.assign(**{output_col: [f"{method_name}_{self.num_clusters}"]})

            with patch("generate.generate_regimes.RegimeClustering", FakeRegimeClustering):
                summaries = generate_regime_grid(
                    method_name="kmeans",
                    num_clusters_grid=[6, 10],
                    x_path=x_path,
                    regimes_root=regimes_root,
                    clustering_columns=[DEFAULT_HOUR_COLUMN, DEFAULT_MONTH_COLUMN],
                    regime_range=2,
                    regime_frequency=1,
                )

            self.assertEqual(NUM_CLUSTERS_GRID, [6, 8, 10])
            self.assertEqual(
                [summary.output_path.resolve() for summary in summaries],
                [
                    (regimes_root / "kmeans_6" / "regimes.csv").resolve(),
                    (regimes_root / "kmeans_10" / "regimes.csv").resolve(),
                ],
            )

    def test_generate_regimes_uses_regime_config_defaults_for_pca_switch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x_path = root / "X.csv"
            regimes_root = root / "regimes"

            all_config_columns = list(dict.fromkeys(REGIME_COLUMNS + REGIME_COLUMNS_PCA))
            frame_data: dict[str, object] = {
                "date": pd.to_datetime(
                    [
                        "2024-01-01 00:00:00+00:00",
                        "2024-02-01 00:00:00+00:00",
                        "2024-03-01 00:00:00+00:00",
                        "2024-04-01 00:00:00+00:00",
                    ]
                ),
            }

            for column_name in all_config_columns:
                if column_name == "hour":
                    frame_data[column_name] = [0, 1, 2, 3]
                elif column_name == "month":
                    frame_data[column_name] = [1, 2, 3, 4]
                elif column_name == "is_weekend":
                    frame_data[column_name] = [0, 0, 0, 0]
                else:
                    frame_data[column_name] = [1.0, 2.0, 3.0, 4.0]

            pd.DataFrame(frame_data).to_csv(x_path, index=False)

            seen_column_names: list[list[str]] = []

            class FakeRegimeClustering:
                def __init__(
                    self,
                    df_sample: pd.DataFrame,
                    df_pred: pd.DataFrame,
                    *,
                    num_clusters: int,
                    column_names: list[str],
                    is_pca: bool,
                    pca_dim: int,
                ) -> None:
                    seen_column_names.append(column_names)
                    self.df_pred = df_pred

                def assign(self, method_name: str, *, output_col: str = "regime") -> pd.DataFrame:
                    return self.df_pred.assign(**{output_col: ["cluster_0"]})

            with patch("generate.generate_regimes.RegimeClustering", FakeRegimeClustering):
                generate_regimes_generic(
                    method_name="kmeans",
                    x_path=x_path,
                    regimes_root=regimes_root,
                    regime_range=2,
                    regime_frequency=1,
                    is_pca=False,
                )
                generate_regimes_generic(
                    method_name="kmeans",
                    x_path=x_path,
                    regimes_root=regimes_root,
                    regime_range=2,
                    regime_frequency=1,
                    is_pca=True,
                )

            self.assertEqual(seen_column_names[:2], [REGIME_COLUMNS, REGIME_COLUMNS])
            self.assertEqual(seen_column_names[2:], [REGIME_COLUMNS_PCA, REGIME_COLUMNS_PCA])

    def test_generate_all_regimes_runs_heuristic_and_all_clustering_grids(self) -> None:
        heuristic_summary = RegimeGenerationSummary(
            rows_written=1,
            output_path=Path("heuristic.csv"),
            label_column="label",
        )
        file_summaries = [
            RegimeGenerationSummary(
                rows_written=index,
                output_path=Path(f"grid_{index}.csv"),
                label_column="label",
            )
            for index in range(1, 9)
        ]

        generate_results = [heuristic_summary, *file_summaries]

        with patch(
            "generate.generate_regimes.generate_regimes",
            side_effect=generate_results,
        ) as mock_generate, patch(
            "generate.generate_regimes.RegimeClustering.available_methods",
            return_value=("kmeans", "gmm"),
        ):
            summaries = generate_all_regimes(
                num_clusters_grid=[6, 10],
                pca_dim=4,
            )

        self.assertEqual(summaries, generate_results)
        self.assertEqual(mock_generate.call_count, 9)
        first_call = mock_generate.call_args_list[0].kwargs
        self.assertEqual(
            first_call,
            {
                "x_path": first_call["x_path"],
                "regimes_root": first_call["regimes_root"],
                "method_name": "heuristic",
                "output_filename": "regimes.csv",
                "label_column": "label",
                "regime_range": 24,
                "regime_frequency": 1,
                "skip_existing": True,
            },
        )
        self.assertEqual(
            [call.kwargs["method_name"] for call in mock_generate.call_args_list[1:]],
            ["kmeans", "kmeans", "kmeans", "kmeans", "gmm", "gmm", "gmm", "gmm"],
        )
        self.assertEqual(
            [call.kwargs["is_pca"] for call in mock_generate.call_args_list[1:]],
            [False, False, True, True, False, False, True, True],
        )
        self.assertEqual(
            [call.kwargs["num_clusters"] for call in mock_generate.call_args_list[1:]],
            [6, 10, 6, 10, 6, 10, 6, 10],
        )
        self.assertTrue(all(call.kwargs["pca_dim"] == 4 for call in mock_generate.call_args_list[1:]))
        self.assertTrue(
            all(call.kwargs["x_path"] == first_call["x_path"] for call in mock_generate.call_args_list[1:])
        )
        self.assertTrue(
            all(
                call.kwargs["regimes_root"] == first_call["regimes_root"]
                for call in mock_generate.call_args_list[1:]
            )
        )
        self.assertTrue(
            all(call.kwargs["output_filename"] == "regimes.csv" for call in mock_generate.call_args_list[1:])
        )
        self.assertTrue(
            all(call.kwargs["label_column"] == "label" for call in mock_generate.call_args_list[1:])
        )
        self.assertTrue(
            all(call.kwargs["regime_range"] == 24 for call in mock_generate.call_args_list[1:])
        )
        self.assertTrue(
            all(call.kwargs["regime_frequency"] == 1 for call in mock_generate.call_args_list[1:])
        )
        self.assertTrue(
            all(call.kwargs["skip_existing"] is True for call in mock_generate.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
