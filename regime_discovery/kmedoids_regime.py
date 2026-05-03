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


def _distance_matrix(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)


def _assign_to_medoids(points: np.ndarray, medoid_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = _distance_matrix(points, medoid_points)
    labels = np.argmin(distances, axis=1)
    return labels, distances


def _choose_unique_indices(
    points: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    distances = _distance_matrix(points, centers)
    chosen: list[int] = []
    for cluster_idx in range(centers.shape[0]):
        ranked = np.argsort(distances[:, cluster_idx])
        next_index = next((int(idx) for idx in ranked if int(idx) not in chosen), None)
        if next_index is None:
            raise RuntimeError("Could not choose a unique medoid initialization point.")
        chosen.append(next_index)
    return np.asarray(chosen, dtype=int)


def _initialize_medoids(
    points: np.ndarray,
    num_clusters: int,
    *,
    random_state: int,
) -> np.ndarray:
    kmeans = KMeans(
        n_clusters=num_clusters,
        n_init=10,
        random_state=random_state,
    )
    kmeans.fit(points)
    return _choose_unique_indices(points, kmeans.cluster_centers_)


def _pam_refine(
    points: np.ndarray,
    medoid_indices: np.ndarray,
    *,
    max_iter: int = 20,
) -> np.ndarray:
    medoid_indices = medoid_indices.astype(int, copy=True)

    for _ in range(max_iter):
        medoid_points = points[medoid_indices]
        labels, distances = _assign_to_medoids(points, medoid_points)
        new_medoids: list[int] = []

        for cluster_idx in range(len(medoid_indices)):
            member_indices = np.flatnonzero(labels == cluster_idx)
            if len(member_indices) == 0:
                nearest_distances = distances.min(axis=1)
                fallback_index = int(
                    next(
                        idx
                        for idx in np.argsort(nearest_distances)[::-1]
                        if int(idx) not in new_medoids
                    )
                )
                new_medoids.append(fallback_index)
                continue

            cluster_points = points[member_indices]
            intra_cluster = _distance_matrix(cluster_points, cluster_points)
            best_local_index = int(np.argmin(intra_cluster.sum(axis=1)))
            candidate_index = int(member_indices[best_local_index])

            if candidate_index in new_medoids:
                ranked = np.argsort(intra_cluster.sum(axis=1))
                candidate_index = int(
                    next(
                        member_indices[local_idx]
                        for local_idx in ranked
                        if int(member_indices[local_idx]) not in new_medoids
                    )
                )
            new_medoids.append(candidate_index)

        new_medoid_indices = np.asarray(new_medoids, dtype=int)
        if np.array_equal(new_medoid_indices, medoid_indices):
            return medoid_indices
        medoid_indices = new_medoid_indices

    return medoid_indices


def _total_cost(points: np.ndarray, medoid_points: np.ndarray) -> float:
    _, distances = _assign_to_medoids(points, medoid_points)
    return float(distances.min(axis=1).sum())


def _select_medoid_indices(
    points: np.ndarray,
    num_clusters: int,
) -> np.ndarray:
    n_samples = len(points)
    if n_samples <= 1024:
        initial = _initialize_medoids(
            points,
            num_clusters,
            random_state=DEFAULT_RANDOM_STATE,
        )
        return _pam_refine(points, initial)

    sample_size = min(n_samples, 1024)
    n_subsamples = 5
    rng = np.random.default_rng(DEFAULT_RANDOM_STATE)

    best_indices = _initialize_medoids(
        points,
        num_clusters,
        random_state=DEFAULT_RANDOM_STATE,
    )
    best_cost = _total_cost(points, points[best_indices])

    for sample_id in range(n_subsamples):
        subset_indices = np.sort(rng.choice(n_samples, size=sample_size, replace=False))
        subset_points = points[subset_indices]
        initial = _initialize_medoids(
            subset_points,
            num_clusters,
            random_state=DEFAULT_RANDOM_STATE + sample_id + 1,
        )
        subset_medoids = _pam_refine(subset_points, initial)
        medoid_indices = subset_indices[subset_medoids]
        cost = _total_cost(points, points[medoid_indices])
        if cost < best_cost:
            best_cost = cost
            best_indices = medoid_indices

    return best_indices


def assign_kmedoids_regimes(
    df_sample: pd.DataFrame,
    df_pred: pd.DataFrame,
    num_clusters: int,
    output_col: str = "regime",
) -> pd.DataFrame:
    validate_clustering_inputs(df_sample, df_pred, num_clusters)

    sample_array = to_numpy_frame(df_sample)
    pred_array = to_numpy_frame(df_pred)

    medoid_indices = _select_medoid_indices(sample_array, num_clusters)
    medoid_points = sample_array[medoid_indices]
    raw_pred_labels, _ = _assign_to_medoids(pred_array, medoid_points)
    pred_labels = remap_labels_by_reference(
        raw_pred_labels,
        label_ids=np.arange(len(medoid_points)),
        reference_vectors=medoid_points,
    )
    return build_regime_frame(df_pred, pred_labels, output_col=output_col)
