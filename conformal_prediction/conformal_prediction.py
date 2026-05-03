from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .conformal_base import (
    BaseConformalMethod,
    ConformalBase,
    _as_prediction_array,
    _as_residual_array,
    _validate_alpha,
)


class ConformalPrediction(BaseConformalMethod):
    """Split conformal calibration for point predictions."""

    def __init__(
        self,
        *,
        residuals_upper: ArrayLike,
        residuals_lower: ArrayLike,
        predictions_upper: ArrayLike,
        predictions_lower: ArrayLike,
        alpha: float,
        symmetric: bool = True,
    ) -> None:
        self.alpha = _validate_alpha(alpha)
        self.symmetric = bool(symmetric)

        self.residuals_upper = _as_residual_array(
            residuals_upper,
            name="residuals_upper",
        )
        self.residuals_lower = _as_residual_array(
            residuals_lower,
            name="residuals_lower",
        )
        self.predictions_upper = _as_prediction_array(
            predictions_upper,
            name="predictions_upper",
        )
        self.predictions_lower = _as_prediction_array(
            predictions_lower,
            name="predictions_lower",
        )

        if self.predictions_upper.shape != self.predictions_lower.shape:
            raise ValueError(
                "predictions_upper and predictions_lower must have matching shapes, "
                f"got {self.predictions_upper.shape} and {self.predictions_lower.shape}."
            )

        upper_scores = np.abs(self.residuals_upper)
        lower_scores = np.abs(self.residuals_lower)

        if self.symmetric:
            if upper_scores.shape != lower_scores.shape or not np.allclose(upper_scores, lower_scores):
                raise ValueError(
                    "symmetric=True expects residuals_upper and residuals_lower "
                    "to match after applying absolute values."
                )
            if not np.allclose(self.predictions_upper, self.predictions_lower):
                raise ValueError(
                    "symmetric=True expects predictions_upper and predictions_lower to match."
                )

            self.base_ = ConformalBase(
                residuals_upper=upper_scores,
                residuals_lower=upper_scores,
                predictions_upper=self.predictions_upper,
                predictions_lower=self.predictions_lower,
                alpha=self.alpha,
            )
            return

        self.base_ = ConformalBase(
            residuals_upper=upper_scores,
            residuals_lower=lower_scores,
            predictions_upper=self.predictions_upper,
            predictions_lower=self.predictions_lower,
            alpha=self.alpha,
        )

    @classmethod
    def from_point_predictions(
        cls,
        *,
        residuals: ArrayLike,
        predictions: ArrayLike,
        alpha: float,
        symmetric: bool = True,
        residuals_lower: ArrayLike | None = None,
    ) -> "ConformalPrediction":
        if symmetric:
            return cls(
                residuals_upper=residuals,
                residuals_lower=residuals,
                predictions_upper=predictions,
                predictions_lower=predictions,
                alpha=alpha,
                symmetric=True,
            )

        if residuals_lower is None:
            raise ValueError("residuals_lower must be provided when symmetric=False.")

        return cls(
            residuals_upper=residuals,
            residuals_lower=residuals_lower,
            predictions_upper=predictions,
            predictions_lower=predictions,
            alpha=alpha,
            symmetric=False,
        )
