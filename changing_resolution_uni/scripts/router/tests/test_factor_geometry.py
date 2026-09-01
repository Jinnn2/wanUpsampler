from __future__ import annotations

import unittest

import numpy as np

from changing_resolution_uni.scripts.router import (
    analyze_steps40_50_factor_geometry as audit,
)


class FactorGeometryTest(unittest.TestCase):
    def test_step_standardize_uses_train_statistics(self) -> None:
        train = np.asarray([[[1.0], [10.0]], [[3.0], [14.0]]], dtype=np.float32)
        validation = np.asarray([[[5.0], [18.0]]], dtype=np.float32)
        train_z, validation_z = audit.step_standardize(train, validation)
        np.testing.assert_allclose(train_z[:, 0, 0], [-1.0, 1.0])
        np.testing.assert_allclose(validation_z[0, :, 0], [3.0, 3.0])

    def test_balanced_threshold_finds_both_directions(self) -> None:
        x = np.asarray([0.0, 1.0, 2.0, 3.0])
        threshold, direction, score = audit.fit_balanced_threshold(
            x, np.asarray([0, 0, 1, 1])
        )
        self.assertEqual(direction, "ge")
        self.assertEqual(score, 1.0)
        prediction = audit.threshold_prediction(x, threshold, direction)
        np.testing.assert_array_equal(prediction, [False, False, True, True])

        threshold, direction, score = audit.fit_balanced_threshold(
            x, np.asarray([1, 1, 0, 0])
        )
        self.assertEqual(direction, "le")
        self.assertEqual(score, 1.0)
        prediction = audit.threshold_prediction(x, threshold, direction)
        np.testing.assert_array_equal(prediction, [True, True, False, False])

    def test_utility_targets_returns_acceptable_interval(self) -> None:
        trajectories = [
            {
                "qualities": np.asarray([0.8, 0.801, 0.799], dtype=np.float32),
                "costs": np.asarray([0.3, 0.2, 0.1], dtype=np.float32),
            }
        ]
        targets = audit.utility_targets(trajectories, [0.0], harm_epsilon=0.002)
        self.assertEqual(int(targets["oracle_index"][0, 0]), 1)
        self.assertEqual(int(targets["earliest_acceptable_index"][0, 0]), 0)
        self.assertEqual(int(targets["latest_acceptable_index"][0, 0]), 2)


if __name__ == "__main__":
    unittest.main()
