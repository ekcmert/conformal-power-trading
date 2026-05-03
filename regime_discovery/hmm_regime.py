from __future__ import annotations

import math

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


LOG_2PI = math.log(2.0 * math.pi)


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    stable = values - max_values
    summed = np.exp(stable).sum(axis=axis, keepdims=True)
    result = max_values + np.log(summed)
    if axis is None:
        return result.reshape(())
    return np.squeeze(result, axis=axis)


class _GaussianHMM:
    def __init__(
        self,
        n_states: int,
        *,
        max_iter: int = 50,
        tol: float = 1e-3,
        covariance_floor: float = 1e-6,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.covariance_floor = covariance_floor
        self.random_state = random_state

    def _initialize(self, x: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = x.shape
        overall_var = np.maximum(x.var(axis=0), self.covariance_floor)

        if self.n_states == 1:
            labels = np.zeros(n_samples, dtype=int)
        else:
            labels = KMeans(
                n_clusters=self.n_states,
                n_init=10,
                random_state=self.random_state,
            ).fit_predict(x)

        self.startprob_ = np.full(self.n_states, 1.0 / self.n_states, dtype=float)
        transition_counts = np.full((self.n_states, self.n_states), 1e-2, dtype=float)
        if len(labels) > 1:
            np.add.at(transition_counts, (labels[:-1], labels[1:]), 1.0)
        self.transmat_ = transition_counts / transition_counts.sum(axis=1, keepdims=True)

        self.means_ = np.zeros((self.n_states, n_features), dtype=float)
        self.vars_ = np.zeros((self.n_states, n_features), dtype=float)
        for state_idx in range(self.n_states):
            members = x[labels == state_idx]
            if len(members) == 0:
                self.means_[state_idx] = x[rng.integers(n_samples)]
                self.vars_[state_idx] = overall_var
            else:
                self.means_[state_idx] = members.mean(axis=0)
                self.vars_[state_idx] = np.maximum(
                    members.var(axis=0),
                    self.covariance_floor,
                )

        self.next_startprob_ = self.startprob_.copy()

    def _emission_log_prob(self, x: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - self.means_[None, :, :]
        log_det = np.log(self.vars_).sum(axis=1)
        mahalanobis = ((diff ** 2) / self.vars_[None, :, :]).sum(axis=2)
        return -0.5 * (x.shape[1] * LOG_2PI + log_det[None, :] + mahalanobis)

    def _forward_backward(self, log_emission: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        n_steps, _ = log_emission.shape
        log_start = np.log(np.clip(self.startprob_, 1e-12, None))
        log_trans = np.log(np.clip(self.transmat_, 1e-12, None))

        log_alpha = np.empty_like(log_emission)
        log_alpha[0] = log_start + log_emission[0]
        for step_idx in range(1, n_steps):
            log_alpha[step_idx] = log_emission[step_idx] + _logsumexp(
                log_alpha[step_idx - 1][:, None] + log_trans,
                axis=0,
            )

        log_beta = np.zeros_like(log_emission)
        for step_idx in range(n_steps - 2, -1, -1):
            log_beta[step_idx] = _logsumexp(
                log_trans
                + log_emission[step_idx + 1][None, :]
                + log_beta[step_idx + 1][None, :],
                axis=1,
            )

        log_likelihood = float(_logsumexp(log_alpha[-1], axis=0))
        return log_alpha, log_beta, log_likelihood

    def fit(self, x: np.ndarray) -> "_GaussianHMM":
        self._initialize(x)
        rng = np.random.default_rng(self.random_state)
        previous_log_likelihood: float | None = None
        overall_var = np.maximum(x.var(axis=0), self.covariance_floor)

        for _ in range(self.max_iter):
            log_emission = self._emission_log_prob(x)
            log_alpha, log_beta, log_likelihood = self._forward_backward(log_emission)

            log_gamma = log_alpha + log_beta - log_likelihood
            gamma = np.exp(log_gamma)
            gamma = gamma / np.clip(gamma.sum(axis=1, keepdims=True), 1e-12, None)

            xi_sum = np.zeros((self.n_states, self.n_states), dtype=float)
            for step_idx in range(len(x) - 1):
                log_xi = (
                    log_alpha[step_idx][:, None]
                    + np.log(np.clip(self.transmat_, 1e-12, None))
                    + log_emission[step_idx + 1][None, :]
                    + log_beta[step_idx + 1][None, :]
                    - log_likelihood
                )
                xi = np.exp(log_xi)
                xi = xi / np.clip(xi.sum(), 1e-12, None)
                xi_sum += xi

            state_weights = gamma.sum(axis=0)
            self.startprob_ = gamma[0] + 1e-2
            self.startprob_ = self.startprob_ / self.startprob_.sum()

            transition_counts = xi_sum + 1e-2
            self.transmat_ = transition_counts / transition_counts.sum(axis=1, keepdims=True)

            weighted_x = gamma.T @ x
            means = np.zeros_like(self.means_)
            variances = np.zeros_like(self.vars_)
            for state_idx in range(self.n_states):
                if state_weights[state_idx] <= 1e-8:
                    means[state_idx] = x[rng.integers(len(x))]
                    variances[state_idx] = overall_var
                    continue

                means[state_idx] = weighted_x[state_idx] / state_weights[state_idx]
                centered = x - means[state_idx]
                variances[state_idx] = (
                    gamma[:, state_idx][:, None] * (centered ** 2)
                ).sum(axis=0) / state_weights[state_idx]
                variances[state_idx] = np.maximum(
                    variances[state_idx],
                    self.covariance_floor,
                )

            self.means_ = means
            self.vars_ = variances
            self.next_startprob_ = gamma[-1] @ self.transmat_
            self.next_startprob_ = self.next_startprob_ / np.clip(
                self.next_startprob_.sum(),
                1e-12,
                None,
            )

            if previous_log_likelihood is not None and abs(
                log_likelihood - previous_log_likelihood
            ) < self.tol:
                break
            previous_log_likelihood = log_likelihood

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        log_emission = self._emission_log_prob(x)
        n_steps, _ = log_emission.shape
        log_start = np.log(np.clip(self.next_startprob_, 1e-12, None))
        log_trans = np.log(np.clip(self.transmat_, 1e-12, None))

        delta = np.empty_like(log_emission)
        psi = np.zeros_like(log_emission, dtype=int)
        delta[0] = log_start + log_emission[0]

        for step_idx in range(1, n_steps):
            transition_scores = delta[step_idx - 1][:, None] + log_trans
            psi[step_idx] = np.argmax(transition_scores, axis=0)
            delta[step_idx] = log_emission[step_idx] + transition_scores[
                psi[step_idx],
                np.arange(self.n_states),
            ]

        states = np.zeros(n_steps, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for step_idx in range(n_steps - 2, -1, -1):
            states[step_idx] = psi[step_idx + 1, states[step_idx + 1]]
        return states


def assign_hmm_regimes(
    df_sample: pd.DataFrame,
    df_pred: pd.DataFrame,
    num_clusters: int,
    output_col: str = "regime",
) -> pd.DataFrame:
    validate_clustering_inputs(df_sample, df_pred, num_clusters)

    sample_array = to_numpy_frame(df_sample)
    pred_array = to_numpy_frame(df_pred)

    model = _GaussianHMM(num_clusters)
    model.fit(sample_array)

    raw_pred_labels = model.predict(pred_array)
    pred_labels = remap_labels_by_reference(
        raw_pred_labels,
        label_ids=np.arange(model.means_.shape[0]),
        reference_vectors=model.means_,
    )
    return build_regime_frame(df_pred, pred_labels, output_col=output_col)
