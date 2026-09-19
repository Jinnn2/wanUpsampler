from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from UNIV_adaptor.scripts.router.train_sparse_prompt_state_router import (
    action_interaction_basis,
    action_main_basis,
    fit_partial_ridge,
    interaction_features,
    lightx2v_python_env,
    predict_partial_ridge,
    proxy_features,
    shuffled_state_map,
)


class SparsePromptStateRouterTest(unittest.TestCase):
    def test_proxy_features_distinguish_temporal_change(self):
        static = np.full((4, 8, 8), 100, dtype=np.uint8)
        moving = static.copy()
        moving[1:, :, 4:] = 220
        static_values, names = proxy_features(static)
        moving_values, moving_names = proxy_features(moving)
        self.assertEqual(names, moving_names)
        self.assertEqual(static_values.shape, (21,))
        temporal = names.index("temporal_abs_mean")
        edge_motion = names.index("edge_motion_abs_mean")
        self.assertEqual(static_values[temporal], 0.0)
        self.assertGreater(moving_values[temporal], 0.0)
        self.assertGreater(moving_values[edge_motion], 0.0)

    def test_action_basis_and_partial_ridge_recover_context_interaction(self):
        levels = np.asarray(
            [
                [s, t, c]
                for _ in range(5)
                for s in (0, 1)
                for t in (0, 1)
                for c in (0, 1)
            ],
            dtype=np.float64,
        )
        state = np.repeat(np.linspace(-1.0, 1.0, 5), 8)[:, None]
        main = np.stack([action_main_basis(row) for row in levels])
        context = interaction_features(state, levels)
        target = 0.02 * levels[:, 0] + 0.05 * state[:, 0] * (2 * levels[:, 2] - 1)
        baseline = fit_partial_ridge(main, None, target, None)
        fused = fit_partial_ridge(main, context, target, 1e-4)
        baseline_mse = np.mean(
            (predict_partial_ridge(baseline, main, None) - target) ** 2
        )
        fused_mse = np.mean((predict_partial_ridge(fused, main, context) - target) ** 2)
        self.assertLess(fused_mse, baseline_mse * 0.01)
        self.assertEqual(
            action_interaction_basis(np.asarray([0, 1, 0])).tolist(),
            [1.0, -1.0, 1.0, -1.0],
        )

    def test_shuffle_preserves_prompt_and_rotates_seed_state(self):
        rows = []
        state = {}
        for prompt in ("p0", "p1"):
            for seed in (42, 100, 2024):
                group = f"{prompt}_b{seed}"
                state[group] = np.asarray([float(seed)])
                rows.append({"prompt_key": prompt, "group_id": group})
        shuffled = shuffled_state_map(rows, state)
        for prompt in ("p0", "p1"):
            original = {
                float(state[f"{prompt}_b{seed}"][0]) for seed in (42, 100, 2024)
            }
            rotated = {
                float(shuffled[f"{prompt}_b{seed}"][0]) for seed in (42, 100, 2024)
            }
            self.assertEqual(original, rotated)
            self.assertTrue(
                all(
                    not np.array_equal(
                        state[f"{prompt}_b{seed}"], shuffled[f"{prompt}_b{seed}"]
                    )
                    for seed in (42, 100, 2024)
                )
            )

    def test_lightx2v_environment_prepends_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "lightx2v").mkdir()
            environment = lightx2v_python_env(repo)
            self.assertEqual(
                Path(environment["PYTHONPATH"].split(os.pathsep)[0]), repo.resolve()
            )
        with self.assertRaisesRegex(FileNotFoundError, "LightX2V Python package"):
            lightx2v_python_env(repo / "missing")


if __name__ == "__main__":
    unittest.main()
