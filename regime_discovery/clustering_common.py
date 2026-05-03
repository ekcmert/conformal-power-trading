from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_RANDOM_STATE = 0


def validate_clustering_inputs(
    df_sample: pd.DataFrame,
    df_pred: pd.DataFrame,
    num_clusters: int,
) -> None:
    if num_clusters < 1:
        raise ValueError("num_clusters must be at least 1.")
    if df_sample.empty:
        raise ValueError("df_sample must contain at least one row.")
    if len(df_sample) < num_clusters:
        raise ValueError(
            "df_sample must contain at least as many rows as num_clusters: "
            f"rows={len(df_sample)}, num_clusters={num_clusters}."
        )
    if df_pred.empty:
        raise ValueError("df_pred must contain at least one row.")


def to_numpy_frame(frame: pd.DataFrame) -> np.ndarray:
    array = frame.to_numpy(dtype=float, copy=True)
    if not np.isfinite(array).all():
        raise ValueError("Clustering inputs must contain only finite numeric values.")
    return array


def cluster_centers_from_labels(
    sample_array: np.ndarray,
    sample_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    label_ids = np.unique(sample_labels)
    centers = np.vstack(
        [sample_array[sample_labels == label_id].mean(axis=0) for label_id in label_ids]
    )
    return label_ids, centers


def remap_labels_by_reference(
    labels: np.ndarray,
    label_ids: np.ndarray,
    reference_vectors: np.ndarray,
) -> np.ndarray:
    if len(label_ids) != len(reference_vectors):
        raise ValueError("label_ids and reference_vectors must have the same length.")

    sort_order = np.lexsort(reference_vectors.T[::-1])
    ordered_label_ids = label_ids[sort_order]
    mapping = {
        raw_label: dense_label
        for dense_label, raw_label in enumerate(ordered_label_ids.tolist())
    }
    return np.asarray([mapping[label] for label in labels], dtype=int)


def assign_to_nearest_centers(
    pred_array: np.ndarray,
    label_ids: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    if centers.ndim != 2:
        raise ValueError("centers must be a 2D array.")
    squared_distances = ((pred_array[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return label_ids[np.argmin(squared_distances, axis=1)]


def build_regime_frame(
    df_pred: pd.DataFrame,
    labels: np.ndarray,
    *,
    output_col: str,
) -> pd.DataFrame:
    if len(labels) != len(df_pred):
        raise ValueError(
            "The number of predicted labels must match df_pred: "
            f"labels={len(labels)}, rows={len(df_pred)}."
        )

    out = df_pred.copy()
    out[output_col] = [f"cluster_{int(label)}" for label in labels]
    return out
