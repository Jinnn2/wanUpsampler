from __future__ import annotations

import unittest

from UNIV_adaptor.core import UniversalAction
from UNIV_adaptor.schedule import resolve_schedule, uniform_topk_steps


class ScheduleTest(unittest.TestCase):
    def test_expected_720p_half_grid_schedule(self):
        schedule = resolve_schedule(
            UniversalAction(0.512, 0.5, 0.5, 0.8),
            reference_nfe=50,
            target_latent_shape=(16, 21, 90, 156),
        )
        self.assertEqual(schedule.low_latent_shape, (16, 11, 46, 80))
        self.assertEqual(schedule.switch_step, 40)
        self.assertEqual(len(schedule.lr_compute_steps), 20)
        self.assertEqual(schedule.lr_compute_steps[0], 0)
        self.assertEqual(schedule.lr_compute_steps[-1], 39)
        self.assertEqual(len(schedule.lr_cache_steps), 20)
        self.assertEqual(schedule.hr_compute_steps, tuple(range(40, 50)))
        self.assertEqual(schedule.total_full_dit_evaluations, 30)
        self.assertEqual(schedule.low_video_frames, 41)
        self.assertEqual(schedule.target_video_frames, 81)

    def test_endpoint_switch_has_no_hr_suffix(self):
        schedule = resolve_schedule(
            UniversalAction(2 / 3, 0.8, 0.4, 1.0),
            reference_nfe=50,
            target_latent_shape=(16, 21, 90, 156),
        )
        self.assertEqual(schedule.switch_step, 50)
        self.assertEqual(len(schedule.lr_compute_steps), 20)
        self.assertEqual(schedule.hr_compute_steps, ())
        self.assertEqual(schedule.low_latent_shape[1], 17)

    def test_exact_half_spatial_ratio_rounds_up_at_tie(self):
        schedule = resolve_schedule(
            UniversalAction(0.5, 1.0, 1.0, 0.8),
            reference_nfe=50,
            target_latent_shape=(16, 21, 90, 156),
        )
        self.assertEqual(schedule.low_latent_shape[-2:], (46, 78))

    def test_uniform_topk_is_exact_and_endpoint_preserving(self):
        for prefix in range(2, 51):
            for count in range(2, prefix + 1):
                steps = uniform_topk_steps(prefix, count)
                self.assertEqual(len(steps), count)
                self.assertEqual(len(set(steps)), count)
                self.assertEqual(steps[0], 0)
                self.assertEqual(steps[-1], prefix - 1)
                self.assertEqual(tuple(sorted(steps)), steps)

    def test_invalid_switch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "switch_ratio"):
            resolve_schedule(
                UniversalAction(0.5, 0.5, 0.5, 0.7),
                reference_nfe=50,
                target_latent_shape=(16, 21, 90, 156),
            )


if __name__ == "__main__":
    unittest.main()
