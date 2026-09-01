from __future__ import annotations

import unittest

import numpy as np
import torch

from changing_resolution_uni.scripts.router import train_fixed_guard_router as guard


class FixedGuardRouterTest(unittest.TestCase):
    def test_guard_falls_back_and_moves_one_step(self) -> None:
        costs = np.asarray([0.4, 0.3, 0.2], dtype=np.float32)
        predictions = np.asarray([0.0, 0.0], dtype=np.float32)
        self.assertEqual(guard.fixed_guard_choice(predictions, costs, 1, 0.1, 0.1), 1)
        early = np.asarray([0.02, 0.0], dtype=np.float32)
        self.assertEqual(guard.fixed_guard_choice(early, costs, 1, 0.1, 0.001), 0)
        late = np.asarray([0.0, -0.02], dtype=np.float32)
        self.assertEqual(guard.fixed_guard_choice(late, costs, 1, 0.1, 0.001), 2)

    def test_prompt_control_ignores_state(self) -> None:
        model = guard.TinyFixedGuard(state_dim=12, dropout=0.0, use_state=False)
        model.eval()
        prompt = torch.ones(2, 4096)
        schedule = torch.ones(2, 10)
        first = model(prompt, torch.zeros(2, 12), schedule)
        second = model(prompt, torch.full((2, 12), 99.0), schedule)
        self.assertTrue(torch.equal(first, second))

    def test_causal_signal_shape(self) -> None:
        raw = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        signals = guard.build_causal_signals(raw)
        self.assertEqual(signals.shape, (2, 3, 12))
        np.testing.assert_array_equal(signals[:, 0, 1::3], 0.0)


if __name__ == "__main__":
    unittest.main()
