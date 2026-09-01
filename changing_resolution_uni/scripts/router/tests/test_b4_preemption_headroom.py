from __future__ import annotations

import unittest

import numpy as np

from changing_resolution_uni.scripts.router import (
    audit_steps40_50_b4_preemption_headroom as audit,
)


class B4PreemptionHeadroomTest(unittest.TestCase):
    def test_allowed_masks_respect_direction_and_radius(self) -> None:
        anchor = np.asarray([[2, 3]], dtype=np.int64)
        lower = audit.build_allowed_mask(anchor, 5, "lower", 1)
        higher = audit.build_allowed_mask(anchor, 5, "higher", 2)
        np.testing.assert_array_equal(lower[0, 0], [False, True, True, False, False])
        np.testing.assert_array_equal(higher[0, 1], [False, False, False, True, True])

    def test_individual_actions_prefer_anchor_on_tie(self) -> None:
        gains = np.asarray(
            [[[[0.0, 0.2, 0.0]], [[0.0, 0.0, 0.0]], [[0.3, 0.0, 0.0]]]],
            dtype=np.float32,
        )
        anchor = np.asarray([[2]], dtype=np.int64)
        allowed = np.ones((1, 1, 3), dtype=bool)
        chosen = audit.choose_best_actions(gains, allowed, anchor)
        np.testing.assert_array_equal(chosen[:, :, 0], [[1, 2, 0]])

    def test_common_action_enforces_seed_count(self) -> None:
        gains = np.asarray(
            [
                [
                    [[0.3, 0.0, 0.0]],
                    [[0.2, 0.0, 0.0]],
                    [[-0.1, 0.0, 0.0]],
                ]
            ],
            dtype=np.float32,
        )
        anchor = np.asarray([[1]], dtype=np.int64)
        allowed = np.ones((1, 1, 3), dtype=bool)
        majority = audit.choose_common_actions(
            gains, allowed, anchor, harm_epsilon=0.01, minimum_positive_seeds=2
        )
        all_three = audit.choose_common_actions(
            gains, allowed, anchor, harm_epsilon=0.01, minimum_positive_seeds=3
        )
        self.assertEqual(int(majority[0, 0]), 0)
        self.assertEqual(int(all_three[0, 0]), 1)


if __name__ == "__main__":
    unittest.main()
