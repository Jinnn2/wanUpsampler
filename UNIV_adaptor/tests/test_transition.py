from __future__ import annotations

import unittest

import numpy as np

from UNIV_adaptor.transition import (
    WanDVGAnchorTransition,
    dvg_anchor_plan,
    dvg_resize_axis,
    dvg_resize_latent,
    dvg_rounded_anchors,
)


class DVGAnchorConformanceTest(unittest.TestCase):
    def test_paper_example_uses_rounded_source_anchors(self):
        self.assertEqual(
            dvg_rounded_anchors(14, 21),
            (0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20),
        )

    def test_positive_half_ties_round_up(self):
        self.assertEqual(dvg_rounded_anchors(3, 6), (0, 3, 5))

    def test_arbitrary_k_to_n_matches_equations_11_and_12(self):
        for source_length in range(1, 18):
            for target_length in range(source_length, 35):
                with self.subTest(K=source_length, N=target_length):
                    anchors = dvg_rounded_anchors(source_length, target_length)
                    source = np.arange(source_length, dtype=np.float64)
                    actual = dvg_resize_axis(source, target_length, axis=0)

                    self.assertEqual(anchors[0], 0)
                    if source_length > 1:
                        self.assertEqual(anchors[-1], target_length - 1)
                    np.testing.assert_array_equal(actual[list(anchors)], source)

                    plan = dvg_anchor_plan(source_length, target_length)
                    expected = np.asarray(
                        [
                            (1.0 - beta) * source[lower]
                            + beta * source[upper]
                            for lower, upper, beta in zip(
                                plan.lower_source,
                                plan.upper_source,
                                plan.beta,
                            )
                        ]
                    )
                    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_three_axis_latent_reconstruction_preserves_all_source_anchors(self):
        source = np.arange(2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)
        target_shape = (1, 5, 7, 9)
        resized = dvg_resize_latent(source, target_shape)
        t = dvg_rounded_anchors(2, 5)
        h = dvg_rounded_anchors(3, 7)
        w = dvg_rounded_anchors(4, 9)
        restored_anchors = resized[np.ix_([0], t, h, w)]
        np.testing.assert_array_equal(restored_anchors, source)

    def test_dvg_transition_reports_latent_baseline(self):
        source = np.zeros((2, 3, 4, 5), dtype=np.float32)
        result = WanDVGAnchorTransition().lift(
            source,
            target_latent_shape=(2, 5, 8, 9),
        )
        self.assertEqual(result.baseline, "dvg_latent_anchor")
        self.assertEqual(result.clean_hr.shape, (2, 5, 8, 9))
        self.assertTrue(result.spatial_restore_applied)
        self.assertTrue(result.temporal_restore_applied)
        self.assertIsNone(result.decoded_frames)


if __name__ == "__main__":
    unittest.main()
