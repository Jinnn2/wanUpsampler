from __future__ import annotations

import unittest

import numpy as np

from changing_resolution_uni.scripts.router import (
    analyze_b4_conditional_state_capacity as audit,
)


class B4ConditionalStateCapacityTest(unittest.TestCase):
    def test_b4_context_records_anchor_and_local_pair(self) -> None:
        probabilities = np.asarray([0.1, 0.6, 0.2, 0.1], dtype=np.float32)
        context = audit.b4_context(probabilities, current=1, radius=2)
        self.assertEqual(context.shape, (14,))
        self.assertAlmostEqual(float(context[8]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(context[10]), 0.0, places=6)
        self.assertAlmostEqual(float(context[11]), 0.6, places=6)
        self.assertAlmostEqual(float(context[12]), 0.2, places=6)

    def test_within_prompt_centering_removes_prompt_mean(self) -> None:
        rows = audit.LocalRows(
            control=np.zeros((6, 1), dtype=np.float32),
            state=np.asarray(
                [[1.0], [2.0], [3.0], [10.0], [12.0], [14.0]], dtype=np.float32
            ),
            target=np.asarray([1.0, 2.0, 3.0, 4.0, 8.0, 12.0], dtype=np.float32),
            prompt_ids=np.asarray([1, 1, 1, 2, 2, 2]),
            trajectory_ids=np.arange(6),
            lambda_ids=np.zeros(6, dtype=np.int64),
            step_ids=np.zeros(6, dtype=np.int64),
            offsets=np.zeros(6, dtype=np.int64),
        )
        state, target, prompts = audit.centered_within_prompt_seed_rows(
            rows, rows.state
        )
        for prompt in np.unique(prompts):
            self.assertAlmostEqual(
                float(state[prompts == prompt].mean()), 0.0, places=6
            )
            self.assertAlmostEqual(
                float(target[prompts == prompt].mean()), 0.0, places=6
            )

    def test_shuffle_preserves_values_within_prompt_key(self) -> None:
        state = np.arange(6, dtype=np.float32).reshape(-1, 1)
        prompt_ids = np.asarray([1, 1, 1, 2, 2, 2])
        lambda_ids = np.zeros(6, dtype=np.int64)
        step_ids = np.zeros(6, dtype=np.int64)
        shuffled = audit.shuffle_rows_within_prompt_key(
            state,
            prompt_ids,
            lambda_ids,
            step_ids,
            np.random.default_rng(7),
        )
        for prompt in (1, 2):
            indices = np.flatnonzero(prompt_ids == prompt)
            self.assertEqual(sorted(state[indices, 0]), sorted(shuffled[indices, 0]))


if __name__ == "__main__":
    unittest.main()
