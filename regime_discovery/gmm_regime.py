from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from regime_discovery.clustering_common import (
    DEFAULT_RANDOM_STATE,
    build_regime_frame,
    remap_labels_by_reference,
    to_numpy_frame,
    validate_clustering_inputs,
)


def assign_gmm_regimes(
    df_sample: pd.DataFrame,
    df_pred: pd.DataFrame,
    num_clusters: int,
    output_col: str = "regime",
) -> pd.DataFrame:
    validate_clustering_inputs(df_sample, df_pred, num_clusters)

    sample_array = to_numpy_frame(df_sample)
    pred_array = to_numpy_frame(df_pred)

    model = GaussianMixture(
        n_components=num_clusters,
        covariance_type="full",
        reg_covar=1e-6,
        n_init=5,
        random_state=DEFAULT_RANDOM_STATE,
    )
    model.fit(sample_array)

    raw_pred_labels = model.predict(pred_array)
    pred_labels = remap_labels_by_reference(
        raw_pred_labels,
        label_ids=np.arange(model.means_.shape[0]),
        reference_vectors=model.means_,
    )
    return build_regime_frame(df_pred, pred_labels, output_col=output_col)
