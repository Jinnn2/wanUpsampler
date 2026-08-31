from __future__ import annotations

import unittest

import numpy as np

from changing_resolution_uni.scripts.router import (
    select_b4_residual_correction as residual,
)


class B4ResidualCorrectionTest(unittest.TestCase):
    def test_zero_scale_exactly_preserves_margin_bits(self) -> None:
        margin = np.asarray([-1.0, -0.1, 0.0, 0.2], dtype=np.float32)
        prediction = np.asarray([2.0, -2.0, 1.0, -1.0], dtype=np.float32)
        corrected, gate, applied = residual.apply_residual_correction(
            margin,
            prediction,
            correction_scale=0.0,
            gate_threshold=0.0,
            residual_clip=2.0,
        )
        self.assertTrue(np.array_equal(corrected, margin))
        self.assertFalse(gate.any())
        np.testing.assert_array_equal(applied, np.zeros_like(margin))

    def test_gate_only_changes_low_confidence_rows_and_clips_residual(self) -> None:
        margin = np.asarray([-2.0, -0.25, 0.1, 1.5], dtype=np.float32)
        prediction = np.asarray([9.0, 9.0, -9.0, -9.0], dtype=np.float32)
        corrected, gate, applied = residual.apply_residual_correction(
            margin,
            prediction,
            correction_scale=0.2,
            gate_threshold=0.5,
            residual_clip=2.0,
        )
        np.testing.assert_array_equal(gate, [False, True, True, False])
        np.testing.assert_allclose(applied, [0.0, 0.4, -0.4, 0.0])
        np.testing.assert_allclose(corrected, [-2.0, 0.15, -0.3, 1.5])

    def test_factor_selection_is_strict_and_compact(self) -> None:
        names = [
            "x0.temporal_gradient_abs_mean",
            "x0.temporal_second_abs_mean",
            "trajectory.delta_rms_per_sigma",
            *[
                f"trajectory.delta_rms_per_sigma.channel_{index:02d}"
                for index in range(16)
            ],
            "trajectory.cosine.channel_00",
            "x0.mean",
        ]
        groups = residual.select_factor_indices(names)
        self.assertEqual(groups["trajectory_delta_rms_per_sigma"].size, 17)
        self.assertEqual(groups["x0_temporal"].size, 2)
        self.assertEqual(groups["combined"].size, 19)
        self.assertEqual(groups["schedule_control"].size, 0)
        self.assertNotIn(len(names) - 1, groups["combined"])

    def test_factor_selection_refuses_incomplete_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "sixteen channel"):
            residual.select_factor_indices(
                [
                    "x0.temporal_gradient_abs_mean",
                    "x0.temporal_second_abs_mean",
                    "trajectory.delta_rms_per_sigma",
                ]
            )

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            residual.apply_residual_correction(
                np.zeros(2, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                correction_scale=0.1,
                gate_threshold=0.5,
                residual_clip=2.0,
            )


if __name__ == "__main__":
    unittest.main()
