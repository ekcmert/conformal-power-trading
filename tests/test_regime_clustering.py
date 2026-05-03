from __future__ import annotations

import unittest

import pandas as pd

from regime_discovery.regime_clustering import RegimeClustering, RegimeFinder


class RegimeClusteringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_frame = pd.DataFrame(
            {
                "feature_a": [
                    -3.2,
                    -3.0,
                    -2.9,
                    -3.1,
                    -2.8,
                    -3.3,
                    -3.1,
                    -2.7,
                    2.8,
                    3.0,
                    3.1,
                    2.9,
                    3.2,
                    2.7,
                    3.3,
                    2.8,
                ],
                "feature_b": [
                    -3.1,
                    -2.9,
                    -3.2,
                    -2.8,
                    -3.0,
                    -3.3,
                    -2.7,
                    -3.1,
                    2.9,
                    3.2,
                    2.8,
                    3.1,
                    2.7,
                    3.0,
                    3.3,
                    2.8,
                ],
                "ignored_feature": list(range(16)),
            }
        )
        self.pred_frame = pd.DataFrame(
            {
                "feature_a": [-3.05, 3.05],
                "feature_b": [-2.95, 3.10],
                "ignored_feature": [100, 200],
            }
        )

    def test_available_methods_and_alias_are_exposed(self) -> None:
        self.assertEqual(
            RegimeClustering.available_methods(),
            ("agglomerative", "divisive", "gmm", "hmm", "kmeans", "kmedoids", "spectral"),
        )
        self.assertIs(RegimeFinder, RegimeClustering)

    def test_all_clustering_methods_assign_two_clear_clusters(self) -> None:
        for method_name in RegimeClustering.available_methods():
            with self.subTest(method_name=method_name):
                clustering = RegimeClustering(
                    self.sample_frame,
                    self.pred_frame,
                    num_clusters=2,
                    column_names=["feature_a", "feature_b"],
                )
                output = clustering.assign(method_name, output_col="label")
                self.assertEqual(output["label"].tolist(), ["cluster_0", "cluster_1"])
                self.assertEqual(output["ignored_feature"].tolist(), [100, 200])

    def test_prepare_features_applies_scaling_and_optional_pca(self) -> None:
        clustering = RegimeClustering(
            self.sample_frame,
            self.pred_frame,
            num_clusters=2,
            column_names=["feature_a", "feature_b", "ignored_feature"],
            is_pca=True,
            pca_dim=2,
        )
        prepared_sample, prepared_pred = clustering.prepare_features()

        self.assertEqual(list(prepared_sample.columns), ["pc_1", "pc_2"])
        self.assertEqual(list(prepared_pred.columns), ["pc_1", "pc_2"])
        self.assertEqual(prepared_sample.shape, (16, 2))
        self.assertEqual(prepared_pred.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
