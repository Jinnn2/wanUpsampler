from __future__ import annotations

import unittest

import numpy as np

from changing_resolution_uni.scripts.router import candidate_step_subset as subset


class CandidateStepSubsetTest(unittest.TestCase):
    def test_steps_40_to_50_select_expected_source_rows(self) -> None:
        source = np.asarray([30, 35, *range(40, 51)], dtype=np.int64)
        indices, selected = subset.resolve_candidate_subset(source, list(range(40, 51)))
        np.testing.assert_array_equal(indices, np.arange(2, 13))
        np.testing.assert_array_equal(selected, np.arange(40, 51))

    def test_subset_updates_every_candidate_aligned_field(self) -> None:
        trajectory = {
            "features": np.arange(5 * 2).reshape(5, 2),
            "sigmas": np.arange(5),
            "qualities": np.arange(5) + 10,
            "costs": np.arange(5) + 20,
            "latencies": np.arange(5) + 30,
            "dimensions": np.arange(5 * 3).reshape(5, 3),
            "prompt_id": 7,
        }
        subset.subset_trajectory_candidates(
            [trajectory], np.asarray([2, 3, 4], dtype=np.int64)
        )
        self.assertEqual(trajectory["features"].shape, (3, 2))
        np.testing.assert_array_equal(trajectory["qualities"], [12, 13, 14])
        np.testing.assert_array_equal(trajectory["latencies"], [32, 33, 34])
        self.assertEqual(trajectory["dimensions"].shape, (3, 3))

    def test_subset_must_retain_forced_final_step(self) -> None:
        source = np.asarray([30, 35, 40, 41, 50], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "forced final step 50"):
            subset.resolve_candidate_subset(source, [40, 41])

    def test_subset_rejects_unknown_or_unordered_steps(self) -> None:
        source = np.asarray([30, 35, 40, 41, 50], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "absent"):
            subset.resolve_candidate_subset(source, [40, 42, 50])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            subset.resolve_candidate_subset(source, [41, 40, 50])


if __name__ == "__main__":
    unittest.main()
