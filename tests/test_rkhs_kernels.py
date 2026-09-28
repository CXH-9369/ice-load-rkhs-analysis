from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rkhs_kernels import (  # noqa: E402
    composite_kernel,
    endpoint_scalar_kernel,
    fit_robust_standardizer,
    kernel_diagnostics,
    loading_rate_kernel,
    pressure_field_kernel,
    rbf_from_squared_distance,
    rbf_kernel,
    squared_euclidean_blockwise,
)


class RKHSCpuKernelTests(unittest.TestCase):
    def test_blockwise_distance_matches_direct_calculation(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(17, 9)).astype(np.float32)
        expected = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
        actual = squared_euclidean_blockwise(x, chunk_size=4)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(actual, actual.T)
        np.testing.assert_array_equal(np.diag(actual), np.zeros(len(x), dtype=np.float32))

    def test_cross_distance_shape_and_values(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.normal(size=(9, 4)).astype(np.float32)
        y = rng.normal(size=(6, 4)).astype(np.float32)
        expected = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
        actual = squared_euclidean_blockwise(x, y, chunk_size=3)
        self.assertEqual(actual.shape, (9, 6))
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_square_cross_rbf_does_not_force_unit_diagonal(self) -> None:
        x = np.asarray([[0.0], [1.0]], dtype=np.float32)
        y = np.asarray([[3.0], [4.0]], dtype=np.float32)
        distance = squared_euclidean_blockwise(x, y)
        kernel = rbf_from_squared_distance(
            distance, 1.0, unit_diagonal=False
        )
        self.assertLess(float(kernel[0, 0]), 1.0)
        self.assertLess(float(kernel[1, 1]), 1.0)

    def test_rbf_is_symmetric_unit_diagonal_and_psd(self) -> None:
        rng = np.random.default_rng(13)
        result = rbf_kernel(rng.normal(size=(21, 6)).astype(np.float32), chunk_size=5)
        diagnostics = kernel_diagnostics(result.kernel)
        self.assertTrue(diagnostics["symmetric"])
        self.assertTrue(diagnostics["psd_with_numerical_tolerance"])
        self.assertLessEqual(diagnostics["diagonal_maximum_error_from_one"], 1e-7)
        self.assertGreaterEqual(diagnostics["minimum_value"], 0.0)
        self.assertLessEqual(diagnostics["maximum_value"], 1.0)

    def test_training_standardizer_excludes_held_out_outlier(self) -> None:
        values = np.asarray([[[0.0]], [[1.0]], [[2.0]], [[1_000_000.0]]], dtype=np.float32)
        valid = np.ones_like(values, dtype=bool)
        fitted = fit_robust_standardizer(values, valid, training_indices=[0, 1, 2])
        self.assertAlmostEqual(float(fitted.median[0]), 1.0)
        self.assertLess(float(fitted.scale[0]), 10.0)

    def test_constant_loading_rate_produces_all_ones(self) -> None:
        result = loading_rate_kernel(
            np.full(12, -3.9, dtype=np.float32),
            np.asarray(["specimen-a"] * 12),
            training_indices=np.arange(12),
        )
        self.assertTrue(result.degenerate)
        np.testing.assert_array_equal(result.kernel, np.ones((12, 12), dtype=np.float32))

    def test_diagnostics_identify_rank_one_constant_kernel(self) -> None:
        diagnostics = kernel_diagnostics(np.ones((12, 12), dtype=np.float32))
        self.assertEqual(diagnostics["numerical_rank_at_tolerance"], 1)
        self.assertTrue(diagnostics["singular_at_tolerance"])
        self.assertAlmostEqual(float(diagnostics["effective_rank"]), 1.0, places=6)
        self.assertAlmostEqual(
            float(diagnostics["positive_spectrum_condition_number"]), 1.0, places=6
        )

    def test_loading_rate_scale_uses_unique_training_specimens(self) -> None:
        rates = np.asarray([0.0, 0.0, 0.0, 2.0, 4.0], dtype=np.float32)
        specimen_ids = np.asarray(["a", "a", "a", "b", "c"])
        result = loading_rate_kernel(rates, specimen_ids, training_indices=np.arange(5))
        # Unique specimen-rate distances are 2, 4 and 2, whose median is 2.
        self.assertAlmostEqual(result.length_scale, 2.0, places=6)

    def test_pressure_kernel_is_psd_and_block_size_invariant(self) -> None:
        rng = np.random.default_rng(17)
        pressure = rng.normal(size=(10, 4, 3, 8, 8)).astype(np.float32)
        overlap = np.ones((8, 8), dtype=np.float32)
        first = pressure_field_kernel(pressure, overlap, chunk_size=3)
        second = pressure_field_kernel(pressure, overlap, chunk_size=16)
        np.testing.assert_allclose(first.kernel, second.kernel, rtol=2e-5, atol=2e-5)
        self.assertTrue(kernel_diagnostics(first.kernel)["psd_with_numerical_tolerance"])

    def test_composite_kernel_is_psd(self) -> None:
        rng = np.random.default_rng(19)
        kernels = [
            rbf_kernel(rng.normal(size=(14, 5)).astype(np.float32)).kernel
            for _ in range(4)
        ]
        result = composite_kernel(
            kernels[0],
            kernels[1],
            kernels[2],
            kernels[3],
            beta_mechanics=0.3,
            beta_pressure=0.4,
            beta_interaction=0.3,
            rho=0.5,
        )
        self.assertTrue(kernel_diagnostics(result)["psd_with_numerical_tolerance"])

    def test_square_cross_composite_does_not_force_unit_diagonal(self) -> None:
        base = np.full((2, 2), 0.25, dtype=np.float32)
        result = composite_kernel(
            base,
            base,
            base,
            base,
            beta_mechanics=1.0,
            beta_pressure=0.0,
            beta_interaction=0.0,
            rho=1.0,
            unit_diagonal=False,
        )
        np.testing.assert_allclose(np.diag(result), [0.015625, 0.015625])

    def test_invalid_composite_weights_are_rejected(self) -> None:
        kernel = np.eye(3, dtype=np.float32)
        with self.assertRaises(ValueError):
            composite_kernel(
                kernel,
                kernel,
                kernel,
                kernel,
                beta_mechanics=0.5,
                beta_pressure=0.5,
                beta_interaction=0.5,
                rho=0.5,
            )


if __name__ == "__main__":
    unittest.main()
