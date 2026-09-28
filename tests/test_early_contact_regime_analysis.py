from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_early_contact_regime_analysis import (  # noqa: E402
    adjusted_rand_index,
    eta_squared,
    kmedoids_exhaustive,
    silhouette_samples,
)


class EarlyContactRegimeTests(unittest.TestCase):
    def test_kmedoids_recovers_two_separated_groups(self) -> None:
        coordinates = np.asarray([[0.0], [0.1], [0.2], [3.0], [3.1], [3.2]])
        distance = np.abs(coordinates - coordinates.T)
        labels, medoids, cost = kmedoids_exhaustive(distance, 2, minimum_cluster_size=3)
        self.assertEqual(len(medoids), 2)
        self.assertLess(cost, 1.0)
        self.assertEqual(len(np.unique(labels[:3])), 1)
        self.assertEqual(len(np.unique(labels[3:])), 1)
        self.assertNotEqual(labels[0], labels[-1])

    def test_adjusted_rand_is_invariant_to_label_names(self) -> None:
        first = np.asarray([0, 0, 1, 1, 2, 2])
        renamed = np.asarray([2, 2, 0, 0, 1, 1])
        self.assertAlmostEqual(adjusted_rand_index(first, renamed), 1.0)

    def test_silhouette_and_eta_squared_detect_complete_separation(self) -> None:
        coordinates = np.asarray([[0.0], [0.1], [0.2], [3.0], [3.1], [3.2]])
        distance = np.abs(coordinates - coordinates.T)
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        self.assertGreater(float(np.mean(silhouette_samples(distance, labels))), 0.9)
        values = np.asarray([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
        self.assertAlmostEqual(eta_squared(values, labels), 1.0)


if __name__ == "__main__":
    unittest.main()
