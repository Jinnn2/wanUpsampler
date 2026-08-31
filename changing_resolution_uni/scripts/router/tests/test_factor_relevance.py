from __future__ import annotations

import unittest

import numpy as np
import torch

from changing_resolution_uni.scripts.router import analyze_factor_relevance as audit


class FactorRelevanceTest(unittest.TestCase):
    def test_suffix_margin_preserves_first_argmax(self) -> None:
        logits = torch.log(torch.tensor([[0.40, 0.35, 0.25]]))
        margins = audit.suffix_margin_from_logits(logits).numpy()
        chosen = audit.first_nonnegative_margin(margins)
        self.assertEqual(int(chosen[0]), 0)

    def test_policy_margin_uses_first_nonnegative_candidate(self) -> None:
        margins = np.asarray(
            [
                [[-1.0, 0.2, 2.0, 30.0]],
                [[-1.0, -0.2, -0.1, 30.0]],
            ],
            dtype=np.float32,
        )
        chosen = audit.first_nonnegative_margin(margins)
        np.testing.assert_array_equal(chosen[:, 0], [1, 3])

    def test_centered_correlation_finds_within_prompt_seed_signal(self) -> None:
        prompt_ids = np.repeat(np.arange(4), 3)
        seed_signal = np.tile(np.asarray([-1.0, 0.0, 1.0]), 4)
        state = np.zeros((12, 2, 2), dtype=np.float32)
        state[:, :, 0] = seed_signal[:, None]
        state[:, :, 1] = np.repeat(np.arange(4), 3)[:, None]
        target = np.zeros((12, 2, 2), dtype=np.float32)
        target[:] = seed_signal[:, None, None]
        _, within = audit.centered_correlations(state, target, prompt_ids)
        self.assertGreater(float(within[0]), 0.99)
        self.assertAlmostEqual(float(within[1]), 0.0, places=6)

    def test_within_step_shuffle_preserves_shape_and_step_values(self) -> None:
        state = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
        shuffled = audit.shuffle_state_within_step(state, np.random.default_rng(7))
        self.assertEqual(shuffled.shape, state.shape)
        for step in range(3):
            self.assertEqual(
                sorted(map(tuple, shuffled[:, step])),
                sorted(map(tuple, state[:, step])),
            )


if __name__ == "__main__":
    unittest.main()
