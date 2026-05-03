from __future__ import annotations

import unittest

import numpy as np

from conformal_prediction import (
    ConformalBase,
    ConformalPrediction,
    ConformalizedQuantileRegression,
)


class ConformalBaseTests(unittest.TestCase):
    def test_side_specific_margins_are_broadcast_to_prediction_shape(self) -> None:
        conformal = ConformalBase(
            residuals_upper=np.array([1.0, 2.0, 3.0]),
            residuals_lower=np.array([0.5, 1.5, 2.5]),
            predictions_upper=np.array([10.0, 11.0]),
            predictions_lower=np.array([8.0, 9.0]),
            alpha=0.2,
        )

        margin_upper, margin_lower = conformal.predict_margins()
        calibrated_lower, calibrated_upper = conformal.predict_interval()

        np.testing.assert_allclose(margin_upper, np.array([3.0, 3.0]))
        np.testing.assert_allclose(margin_lower, np.array([2.5, 2.5]))
        np.testing.assert_allclose(calibrated_lower, np.array([5.5, 6.5]))
        np.testing.assert_allclose(calibrated_upper, np.array([13.0, 14.0]))

    def test_empty_residual_side_returns_zero_margin(self) -> None:
        conformal = ConformalBase(
            residuals_upper=np.array([]),
            residuals_lower=np.array([1.0, 2.0]),
            predictions_upper=np.array([3.0, 4.0]),
            predictions_lower=np.array([1.0, 2.0]),
            alpha=0.1,
        )

        margin_upper, margin_lower = conformal.predict_margins()

        np.testing.assert_allclose(margin_upper, np.zeros(2))
        np.testing.assert_allclose(margin_lower, np.array([2.0, 2.0]))


class ConformalPredictionTests(unittest.TestCase):
    def test_symmetric_conformal_prediction_uses_common_absolute_residuals(self) -> None:
        conformal = ConformalPrediction(
            residuals_upper=np.array([1.0, -2.0, 0.5, 3.0]),
            residuals_lower=np.array([1.0, -2.0, 0.5, 3.0]),
            predictions_upper=np.array([10.0, 20.0]),
            predictions_lower=np.array([10.0, 20.0]),
            alpha=0.25,
            symmetric=True,
        )

        margin_upper, margin_lower = conformal.predict_margins()
        calibrated_lower, calibrated_upper = conformal.predict_interval()

        np.testing.assert_allclose(margin_upper, np.array([3.0, 3.0]))
        np.testing.assert_allclose(margin_lower, np.array([3.0, 3.0]))
        np.testing.assert_allclose(calibrated_lower, np.array([7.0, 17.0]))
        np.testing.assert_allclose(calibrated_upper, np.array([13.0, 23.0]))

    def test_asymmetric_conformal_prediction_uses_separate_upper_and_lower_scores(self) -> None:
        conformal = ConformalPrediction(
            residuals_upper=np.array([2.0, 1.0]),
            residuals_lower=np.array([-0.5, -1.5, -1.0]),
            predictions_upper=np.array([10.0, 20.0]),
            predictions_lower=np.array([10.0, 20.0]),
            alpha=0.34,
            symmetric=False,
        )

        margin_upper, margin_lower = conformal.predict_margins()
        calibrated_lower, calibrated_upper = conformal.predict_interval()

        np.testing.assert_allclose(margin_upper, np.array([2.0, 2.0]))
        np.testing.assert_allclose(margin_lower, np.array([1.5, 1.5]))
        np.testing.assert_allclose(calibrated_lower, np.array([8.5, 18.5]))
        np.testing.assert_allclose(calibrated_upper, np.array([12.0, 22.0]))


class ConformalizedQuantileRegressionTests(unittest.TestCase):
    def test_cqr_uses_shared_max_score_for_both_sides(self) -> None:
        conformal = ConformalizedQuantileRegression(
            residuals_lower=np.array([2.0, 2.0, 2.0, 2.0]),
            residuals_upper=np.array([-3.0, -3.0, -3.0, -3.0]),
            predictions_lower=np.array([10.0, 20.0]),
            predictions_upper=np.array([15.0, 25.0]),
            alpha=0.1,
        )

        margin_upper, margin_lower = conformal.predict_margins()
        calibrated_lower, calibrated_upper = conformal.predict_interval()

        np.testing.assert_allclose(conformal.shared_residuals, np.array([-2.0, -2.0, -2.0, -2.0]))
        np.testing.assert_allclose(margin_upper, np.array([-2.0, -2.0]))
        np.testing.assert_allclose(margin_lower, np.array([-2.0, -2.0]))
        np.testing.assert_allclose(calibrated_lower, np.array([12.0, 22.0]))
        np.testing.assert_allclose(calibrated_upper, np.array([13.0, 23.0]))


if __name__ == "__main__":
    unittest.main()
