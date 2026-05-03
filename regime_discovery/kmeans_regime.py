from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from regime_discovery.clustering_common import (
    DEFAULT_RANDOM_STATE,
    build_regime_frame,
    remap_labels_by_reference,
    to_numpy_frame,
    validate_clustering_inputs,
)


def assign_kmeans_regimes(
    df_sample: pd.DataFrame,
    df_pred: pd.DataFrame,
    num_clusters: int,
    output_col: str = "regime",
) -> pd.DataFrame:
    validate_clustering_inputs(df_sample, df_pred, num_clusters)

    sample_array = to_numpy_frame(df_sample)
    pred_array = to_numpy_frame(df_pred)

    model = KMeans(
        n_clusters=num_clusters,
        n_init=20,
        random_state=DEFAULT_RANDOM_STATE,
    )
    model.fit(sample_array)

    raw_pred_labels = model.predict(pred_array)
    pred_labels = remap_labels_by_reference(
        raw_pred_labels,
        label_ids=np.arange(model.cluster_centers_.shape[0]),
        reference_vectors=model.cluster_centers_,
    )
    return build_regime_frame(df_pred, pred_labels, output_col=output_col)
