from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from regime_discovery.agglomerative_regime import assign_agglomerative_regimes
from regime_discovery.divisive_regime import assign_divisive_regimes
from regime_discovery.gmm_regime import assign_gmm_regimes
from regime_discovery.hmm_regime import assign_hmm_regimes
from regime_discovery.kmeans_regime import assign_kmeans_regimes
from regime_discovery.kmedoids_regime import assign_kmedoids_regimes
from regime_discovery.spectral_regime import assign_spectral_regimes


RegimeAssigner = Callable[[pd.DataFrame, pd.DataFrame, int, str], pd.DataFrame]


@dataclass(frozen=True)
class ClusteringMethodSpec:
    method_name: str
    folder_name: str
    assigner: RegimeAssigner


CLUSTERING_METHOD_SPECS: dict[str, ClusteringMethodSpec] = {
    "kmeans": ClusteringMethodSpec(
        method_name="kmeans",
        folder_name="kmeans",
        assigner=assign_kmeans_regimes,
    ),
    "kmedoids": ClusteringMethodSpec(
        method_name="kmedoids",
        folder_name="kmedoids",
        assigner=assign_kmedoids_regimes,
    ),
    "agglomerative": ClusteringMethodSpec(
        method_name="agglomerative",
        folder_name="agglomerative",
        assigner=assign_agglomerative_regimes,
    ),
    "divisive": ClusteringMethodSpec(
        method_name="divisive",
        folder_name="divisive",
        assigner=assign_divisive_regimes,
    ),
    "spectral": ClusteringMethodSpec(
        method_name="spectral",
        folder_name="spectral",
        assigner=assign_spectral_regimes,
    ),
    "gmm": ClusteringMethodSpec(
        method_name="gmm",
        folder_name="gmm",
        assigner=assign_gmm_regimes,
    ),
    "hmm": ClusteringMethodSpec(
        method_name="hmm",
        folder_name="hmm",
        assigner=assign_hmm_regimes,
    ),
}


CLUSTERING_METHOD_ALIASES = {
    "hierarchical_agglomerative": "agglomerative",
    "hierarchical_divisive": "divisive",
    "gaussian_mixture": "gmm",
    "hidden_markov_model": "hmm",
}


def resolve_clustering_method_name(method_name: str) -> str:
    normalized = method_name.strip().lower()
    normalized = CLUSTERING_METHOD_ALIASES.get(normalized, normalized)
    if normalized not in CLUSTERING_METHOD_SPECS:
        available = ", ".join(sorted(CLUSTERING_METHOD_SPECS))
        raise KeyError(f"Unknown clustering regime method {method_name!r}. Available: {available}")
    return normalized


def default_output_folder_for_method(
    method_name: str,
    *,
    is_pca: bool,
    num_clusters: int,
) -> str:
    spec = CLUSTERING_METHOD_SPECS[resolve_clustering_method_name(method_name)]
    if is_pca:
        return f"{spec.folder_name}_pca_{num_clusters}"
    return f"{spec.folder_name}_{num_clusters}"


class RegimeClustering:
    def __init__(
        self,
        df_sample: pd.DataFrame,
        df_pred: pd.DataFrame,
        *,
        num_clusters: int = 8,
        column_names: list[str],
        is_pca: bool = False,
        pca_dim: int = 16,
    ) -> None:
        if not column_names:
            raise ValueError("column_names must contain at least one column name.")
        if pca_dim < 1:
            raise ValueError("pca_dim must be at least 1.")

        self.df_sample = df_sample.copy()
        self.df_pred = df_pred.copy()
        self.num_clusters = num_clusters
        self.column_names = list(dict.fromkeys(column_names))
        self.is_pca = is_pca
        self.pca_dim = pca_dim

        self.scaler = StandardScaler()
        self.pca_model: PCA | None = None

    @classmethod
    def available_methods(cls) -> tuple[str, ...]:
        return tuple(sorted(CLUSTERING_METHOD_SPECS))

    def _filtered_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        missing_sample = [
            column_name
            for column_name in self.column_names
            if column_name not in self.df_sample.columns
        ]
        missing_pred = [
            column_name
            for column_name in self.column_names
            if column_name not in self.df_pred.columns
        ]
        if missing_sample or missing_pred:
            raise KeyError(
                "Selected clustering columns are missing from the provided dataframes: "
                f"sample_missing={missing_sample}, pred_missing={missing_pred}"
            )

        sample_filtered = self.df_sample.loc[:, self.column_names].astype(float)
        pred_filtered = self.df_pred.loc[:, self.column_names].astype(float)

        if sample_filtered.isna().any().any() or pred_filtered.isna().any().any():
            raise ValueError("Selected clustering columns must not contain missing values.")
        if not np.isfinite(sample_filtered.to_numpy()).all() or not np.isfinite(pred_filtered.to_numpy()).all():
            raise ValueError("Selected clustering columns must contain only finite values.")

        return sample_filtered, pred_filtered

    def prepare_features(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        sample_filtered, pred_filtered = self._filtered_frames()
        sample_scaled = pd.DataFrame(
            self.scaler.fit_transform(sample_filtered),
            index=sample_filtered.index,
            columns=self.column_names,
        )
        pred_scaled = pd.DataFrame(
            self.scaler.transform(pred_filtered),
            index=pred_filtered.index,
            columns=self.column_names,
        )

        if not self.is_pca:
            return sample_scaled, pred_scaled

        n_components = min(
            self.pca_dim,
            sample_scaled.shape[0],
            sample_scaled.shape[1],
        )
        self.pca_model = PCA(n_components=n_components)
        component_names = [f"pc_{component_idx + 1}" for component_idx in range(n_components)]
        sample_reduced = pd.DataFrame(
            self.pca_model.fit_transform(sample_scaled),
            index=sample_scaled.index,
            columns=component_names,
        )
        pred_reduced = pd.DataFrame(
            self.pca_model.transform(pred_scaled),
            index=pred_scaled.index,
            columns=component_names,
        )
        return sample_reduced, pred_reduced

    def assign(
        self,
        method_name: str,
        *,
        output_col: str = "regime",
    ) -> pd.DataFrame:
        resolved_method_name = resolve_clustering_method_name(method_name)
        prepared_sample, prepared_pred = self.prepare_features()

        clustered_pred = CLUSTERING_METHOD_SPECS[resolved_method_name].assigner(
            prepared_sample,
            prepared_pred,
            self.num_clusters,
            output_col,
        )
        if output_col not in clustered_pred.columns:
            raise KeyError(f"Expected clustering method to create column {output_col!r}.")

        out = self.df_pred.copy()
        out[output_col] = clustered_pred[output_col].to_numpy()
        return out


RegimeFinder = RegimeClustering
