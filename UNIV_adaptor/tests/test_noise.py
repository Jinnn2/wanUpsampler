from __future__ import annotations

import unittest

import numpy as np

from UNIV_adaptor.noise import coordinate_gaussian_numpy
from UNIV_adaptor.transition import dvg_rounded_anchors


class CoordinateNoiseTest(unittest.TestCase):
    def test_is_deterministic(self):
        first = coordinate_gaussian_numpy((2, 3, 4, 6), seed=42)
        second = coordinate_gaussian_numpy((2, 3, 4, 6), seed=42)
        np.testing.assert_array_equal(first, second)

    def test_low_grid_matches_reference_anchors(self):
        target_shape = (1, 5, 6, 8)
        low_shape = (1, 3, 4, 4)
        target = coordinate_gaussian_numpy(target_shape, seed=7)
        low = coordinate_gaussian_numpy(
            low_shape,
            seed=7,
            reference_shape=target_shape,
        )
        t_indices = dvg_rounded_anchors(3, 5)
        h_indices = dvg_rounded_anchors(4, 6)
        w_indices = dvg_rounded_anchors(4, 8)
        expected = target[np.ix_([0], t_indices, h_indices, w_indices)]
        np.testing.assert_array_equal(low, expected)

    def test_coordinate_noise_uses_same_half_tie_anchor_rule_as_dvg(self):
        target = coordinate_gaussian_numpy((1, 6, 2, 2), seed=11)
        low = coordinate_gaussian_numpy(
            (1, 3, 2, 2),
            seed=11,
            reference_shape=(1, 6, 2, 2),
        )
        np.testing.assert_array_equal(low, target[:, (0, 3, 5)])

    def test_seed_changes_noise(self):
        first = coordinate_gaussian_numpy((1, 2, 2, 2), seed=1)
        second = coordinate_gaussian_numpy((1, 2, 2, 2), seed=2)
        self.assertFalse(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
