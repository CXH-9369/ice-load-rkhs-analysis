import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_loso_fold import (  # noqa: E402
    SelectedSpecimen,
    binary_auc,
    deterministic_event_aware_sample,
    fit_pressure_rms_per_sample,
    interval_label,
    prepare_pressure_vectors_per_sample,
    select_training_threshold,
)


class LOSOFoldTests(unittest.TestCase):
    def test_interval_policy_excludes_partial_uncertain_interval(self):
        self.assertEqual(interval_label(8.0, 10.0, 11.0, 1.0), 0)
        self.assertIsNone(interval_label(9.5, 10.0, 11.0, 1.0))
        self.assertEqual(interval_label(10.0, 10.0, 11.0, 1.0), 1)

    def test_sample_specific_pressure_masks_have_unit_vector_energy(self):
        fields = np.zeros((2, 2, 3, 2, 2), dtype=np.float32)
        fields[:, :, 0] = 2.0
        fields[:, :, 1:] = 3.0
        weights = np.asarray(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        rms = fit_pressure_rms_per_sample(fields, weights)
        pressure, gradient = prepare_pressure_vectors_per_sample(fields, weights, rms)
        np.testing.assert_allclose(rms, [2.0, 3.0], rtol=1e-6)
        np.testing.assert_allclose(np.sum(pressure**2, axis=1), [1.0, 1.0], rtol=1e-6)
        np.testing.assert_allclose(np.sum(gradient**2, axis=1), [1.0, 1.0], rtol=1e-6)

    def test_event_aware_sampling_keeps_all_positives_and_is_deterministic(self):
        def specimen(name, labels):
            return SelectedSpecimen(
                specimen_id=name,
                rate_group="test",
                event_interval_s=(1.0, 1.0),
                tensor_indices=np.arange(len(labels), dtype=np.int32),
                labels=np.asarray(labels, dtype=np.int32),
                arrays={},
            )

        specimens = [
            specimen("a", [0, 0, 0, 0, 1]),
            specimen("b", [0, 0, 0, 1]),
        ]
        first = deterministic_event_aware_sample(specimens, maximum=6, seed=7)
        second = deterministic_event_aware_sample(specimens, maximum=6, seed=7)
        self.assertEqual(sum(map(len, first.values())), 6)
        np.testing.assert_array_equal(first["a"], second["a"])
        np.testing.assert_array_equal(first["b"], second["b"])
        self.assertIn(4, first["a"])
        self.assertIn(3, first["b"])

    def test_auc_and_threshold_on_separated_scores(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
        scores = np.asarray([0.1, 0.3, 0.7, 0.9])
        self.assertEqual(binary_auc(labels, scores), 1.0)
        threshold, score = select_training_threshold(
            labels, scores, np.full(4, 0.25)
        )
        self.assertGreaterEqual(threshold, 0.3)
        self.assertLessEqual(threshold, 0.7)
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
