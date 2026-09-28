from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_event_stage_bootstrap_analysis import (  # noqa: E402
    benjamini_hochberg,
    cliffs_delta,
    exact_permutation_p_cliffs_delta,
    pressure_distribution,
    weighted_support_iou,
)
from run_incremental_rkhs_sensitivity import (  # noqa: E402
    exact_mmd_test_fixed_bandwidth,
    incremental_vectors,
)


class EventStageBootstrapTests(unittest.TestCase):
    def test_cliffs_delta_direction_and_complete_separation(self) -> None:
        low = np.asarray([0.0, 1.0, 2.0])
        high = np.asarray([3.0, 4.0, 5.0, 6.0])
        self.assertEqual(cliffs_delta(low, high), 1.0)
        self.assertEqual(cliffs_delta(high, low), -1.0)

    def test_exact_permutation_cliffs_delta_is_two_sided(self) -> None:
        low = np.asarray([0.0, 1.0, 2.0])
        high = np.arange(3.0, 11.0)
        self.assertAlmostEqual(exact_permutation_p_cliffs_delta(low, high), 2.0 / 165.0)

    def test_benjamini_hochberg_preserves_original_order(self) -> None:
        p = np.asarray([0.04, 0.001, 0.02, np.nan])
        q = benjamini_hochberg(p)
        np.testing.assert_allclose(q[:3], np.asarray([0.04, 0.003, 0.03]))
        self.assertTrue(np.isnan(q[3]))

    def test_pressure_distribution_is_normalized_and_overlap_weighted(self) -> None:
        field = np.zeros((32, 32), dtype=float)
        overlap = np.ones((32, 32), dtype=float)
        field[0, 0] = 1.0
        field[0, 1] = 1.0
        overlap[0, 1] = 0.5
        distribution = pressure_distribution(field, overlap)
        self.assertAlmostEqual(float(distribution.sum()), 1.0)
        self.assertAlmostEqual(float(distribution[0]), 2.0 / 3.0)
        self.assertAlmostEqual(float(distribution[1]), 1.0 / 3.0)

    def test_identical_support_has_unit_iou(self) -> None:
        field = np.zeros((32, 32), dtype=float)
        field[4:8, 6:10] = 100.0
        overlap = np.ones((32, 32), dtype=float)
        self.assertEqual(weighted_support_iou(field, field, overlap), 1.0)

    def test_incremental_vectors_subtract_specimen_baseline(self) -> None:
        grid = np.asarray([0.10, 0.20, 0.30])
        trajectories = np.asarray(
            [
                [[1.0, 2.0], [3.0, 5.0], [6.0, 9.0]],
                [[10.0, 20.0], [13.0, 25.0], [16.0, 29.0]],
            ]
        )
        vectors = incremental_vectors(trajectories, grid, 0.10, evaluation_minimum=0.10)
        expected = np.asarray(
            [
                [0.0, 0.0, 2.0, 3.0, 5.0, 7.0],
                [0.0, 0.0, 3.0, 5.0, 6.0, 9.0],
            ]
        )
        np.testing.assert_allclose(vectors, expected)

    def test_fixed_bandwidth_mmd_returns_exact_finite_result(self) -> None:
        first = np.asarray([[0.0], [0.1], [0.2]])
        second = np.arange(1.0, 9.0)[:, None]
        values = np.vstack([first, second])
        statistic, p_value, permutations, bandwidth = exact_mmd_test_fixed_bandwidth(values, 3)
        self.assertGreater(statistic, 0.0)
        self.assertGreater(p_value, 0.0)
        self.assertLessEqual(p_value, 0.05)
        self.assertEqual(permutations, 165)
        self.assertGreater(bandwidth, 0.0)


if __name__ == "__main__":
    unittest.main()
