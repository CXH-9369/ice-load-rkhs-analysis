from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from kernel_logistic_regression import (  # noqa: E402
    fit_kernel_logistic_regression,
    objective_and_gradient,
    rate_balanced_weights,
    specimen_balanced_weights,
)
from rkhs_kernels import rbf_kernel  # noqa: E402


class KernelLogisticRegressionTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(23)
        vectors = rng.normal(size=(7, 3)).astype(np.float32)
        kernel = rbf_kernel(vectors).kernel
        labels = np.asarray([0, 1, 0, 1, 1, 0, 1], dtype=float)
        weights = np.asarray([1, 2, 1, 3, 2, 1, 2], dtype=float)
        weights /= weights.sum()
        parameters = rng.normal(scale=0.2, size=8)
        value, gradient = objective_and_gradient(
            parameters, kernel, labels, weights, 0.3
        )
        self.assertTrue(np.isfinite(value))
        epsilon = 1.0e-6
        numerical = np.empty_like(gradient)
        for index in range(len(parameters)):
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            plus_value, _ = objective_and_gradient(plus, kernel, labels, weights, 0.3)
            minus_value, _ = objective_and_gradient(minus, kernel, labels, weights, 0.3)
            numerical[index] = (plus_value - minus_value) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-6)

    def test_optimizer_reduces_objective(self) -> None:
        x = np.asarray([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]], dtype=np.float32)
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        kernel = rbf_kernel(x).kernel
        model = fit_kernel_logistic_regression(kernel, labels, regularization=0.05)
        self.assertTrue(model.optimizer_success)
        self.assertLess(model.final_objective, model.initial_objective)
        probabilities = model.predict_proba(kernel)
        self.assertGreater(float(np.mean(probabilities[labels == 1])), float(np.mean(probabilities[labels == 0])))

    def test_specimen_balanced_weights_equalize_specimen_totals(self) -> None:
        specimens = np.asarray(["a", "a", "a", "b", "c", "c"])
        weights = specimen_balanced_weights(specimens)
        totals = [float(np.sum(weights[specimens == specimen])) for specimen in ["a", "b", "c"]]
        np.testing.assert_allclose(totals, np.full(3, 1.0 / 3.0))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)

    def test_rate_balanced_weights_equalize_rate_groups_and_specimens(self) -> None:
        specimens = np.asarray(["a", "a", "b", "c", "c", "c", "d"])
        groups = np.asarray(["low", "low", "low", "high", "high", "high", "high"])
        weights = rate_balanced_weights(specimens, groups)
        self.assertAlmostEqual(float(np.sum(weights[groups == "low"])), 0.5)
        self.assertAlmostEqual(float(np.sum(weights[groups == "high"])), 0.5)
        self.assertAlmostEqual(float(np.sum(weights[specimens == "a"])), 0.25)
        self.assertAlmostEqual(float(np.sum(weights[specimens == "b"])), 0.25)
        self.assertAlmostEqual(float(np.sum(weights[specimens == "c"])), 0.25)
        self.assertAlmostEqual(float(np.sum(weights[specimens == "d"])), 0.25)

    def test_single_class_is_rejected(self) -> None:
        kernel = np.eye(4, dtype=np.float32)
        with self.assertRaises(ValueError):
            fit_kernel_logistic_regression(kernel, np.zeros(4))


if __name__ == "__main__":
    unittest.main()
