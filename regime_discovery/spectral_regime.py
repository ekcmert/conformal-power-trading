from __future__ import annotations

import pandas as pd
from sklearn.cluster import SpectralClustering

from regime_discovery.clustering_common import (
    DEFAULT_RANDOM_STATE,
    assign_to_nearest_centers,
    build_regime_frame,
    cluster_centers_from_labels,
    remap_labels_by_reference,
    to_numpy_frame,
    validate_clustering_inputs,
)


def assign_spectral_regimes(
    df_sample: pd.DataFrame,
    df_pred: pd.DataFrame,
    num_clusters: int,
    output_col: str = "regime",
) -> pd.DataFrame:
    validate_clustering_inputs(df_sample, df_pred, num_clusters)

    sample_array = to_numpy_frame(df_sample)
    pred_array = to_numpy_frame(df_pred)

    if len(sample_array) <= 1:
        raise ValueError("Spectral clustering requires at least two sample rows.")

    n_neighbors = min(10, len(sample_array) - 1)
    model = SpectralClustering(
        n_clusters=num_clusters,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels="kmeans",
        random_state=DEFAULT_RANDOM_STATE,
    )
    sample_labels = model.fit_predict(sample_array)

    # Spectral clustering has no out-of-sample predict in sklearn, so we use
    # nearest sample-cluster centroids in the processed feature space.
    label_ids, centers = cluster_centers_from_labels(sample_array, sample_labels)
    raw_pred_labels = assign_to_nearest_centers(pred_array, label_ids, centers)
    pred_labels = remap_labels_by_reference(raw_pred_labels, label_ids, centers)
    return build_regime_frame(df_pred, pred_labels, output_col=output_col)
