from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_contact_regime_conditioned_analysis import (  # noqa: E402
    permutation_spearman,
    spearman_rho,
)
from run_outlier_mechanism_audit import robust_topology_score  # noqa: E402


class ContactRegimeConditionedTests(unittest.TestCase):
    def test_spearman_detects_monotone_and_reversed_order(self) -> None:
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(spearman_rho(values, values), 1.0)
        self.assertAlmostEqual(spearman_rho(values, values[::-1]), -1.0)

    def test_small_sample_spearman_uses_exact_two_sided_permutation(self) -> None:
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        rho, p_value, permutations, method = permutation_spearman(
            values,
            values,
            np.random.default_rng(1),
            repeats=10,
        )
        self.assertAlmostEqual(rho, 1.0)
        self.assertAlmostEqual(p_value, 2.0 / 24.0)
        self.assertEqual(permutations, 24)
        self.assertEqual(method, "exact")

    def test_robust_topology_score_is_finite_and_centered_case_is_smallest(self) -> None:
        import pandas as pd

        columns = {
            "damage_minus_transition__pressure_area_ge_50kpa_fraction_of_face": [0.0, 1.0, 2.0],
            "damage_minus_transition__pressure_effective_area_fraction_of_face": [0.0, 1.0, 2.0],
            "damage_minus_transition__pressure_normalized_entropy": [0.0, 1.0, 2.0],
            "damage_minus_transition__pressure_top_10pct_cell_load_fraction": [0.0, 1.0, 2.0],
        }
        scores = robust_topology_score(pd.DataFrame(columns))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertEqual(int(np.argmin(scores)), 1)


if __name__ == "__main__":
    unittest.main()
