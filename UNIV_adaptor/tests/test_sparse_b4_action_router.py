from __future__ import annotations

import unittest

import numpy as np

from UNIV_adaptor.scripts.router.train_sparse_b4_action_router import (
    crossfit_soft_models,
    fit_full_predict,
    fit_soft_action_model,
    policy_diagnostics,
    predict_soft_action,
    prompt_mean_residual_decomposition,
    select_by_policy,
    soft_policy_loss_and_gradient,
)


def row(prompt: str, seed: int, action: str, level: int, target: float) -> dict:
    return {
        "prompt_key": prompt,
        "group_id": f"{prompt}_b{seed}",
        "observation_id": f"{prompt}_b{seed}__{action}",
        "action_id": action,
        "base_seed": seed,
        "prompt": f"video prompt {prompt}",
        "levels": np.asarray([level, 0, 0], dtype=np.float64),
        "main": np.asarray([1.0, level], dtype=np.float64),
        "target": target,
    }


class SparseB4ActionRouterTest(unittest.TestCase):
    def test_prompt_mean_residual_decomposition_centers_each_cell(self):
        rows = []
        for prompt, offset in (("p0", 0.0), ("p1", 0.2)):
            for seed, noise in ((42, -0.03), (100, 0.0), (2024, 0.03)):
                rows.append(row(prompt, seed, "SA_000", 0, offset + noise))
                rows.append(row(prompt, seed, "SA_100", 1, offset + 0.1 + noise))
        means, residuals, diagnostics = prompt_mean_residual_decomposition(rows)
        self.assertEqual(len(means), 4)
        self.assertEqual(diagnostics["seeds_per_prompt_action"], 3)
        for prompt in ("p0", "p1"):
            for action in ("SA_000", "SA_100"):
                values = [
                    item["target"]
                    for item in residuals
                    if item["prompt_key"] == prompt and item["action_id"] == action
                ]
                self.assertAlmostEqual(float(np.mean(values)), 0.0, places=12)

    def test_soft_action_optimizer_recovers_group_preferences(self):
        features = []
        utilities = []
        groups = []
        for index, context in enumerate((-1.0, -0.5, 0.5, 1.0)):
            # Two actions: the first follows +context and the second -context.
            features.extend(([1.0, context], [-1.0, -context]))
            utilities.extend((0.2 * context, -0.2 * context))
            groups.extend((f"g{index}", f"g{index}"))
        x = np.asarray(features)
        y = np.asarray(utilities)
        model = fit_soft_action_model(
            x,
            y,
            groups,
            temperature=0.05,
            alpha=0.01,
            penalty_start=1,
            max_iterations=200,
        )
        prediction = predict_soft_action(model, x)
        self.assertLess(np.mean((prediction - y) ** 2), 1e-4)
        self.assertLess(
            policy_diagnostics(
                [
                    {
                        "group_id": group,
                        "target": target,
                    }
                    for group, target in zip(groups, y)
                ],
                prediction,
                temperature=0.05,
            )["policy_regret"],
            1e-6,
        )

    def test_analytic_soft_gradient_matches_finite_difference(self):
        x = np.asarray([[1.0, 0.2], [-0.4, 1.0], [0.5, -0.5], [0.1, 0.8]])
        y = np.asarray([0.1, -0.02, -0.03, 0.08])
        groups = np.asarray([[0, 1], [2, 3]])
        weights = np.asarray([0.03, -0.04])
        loss, gradient = soft_policy_loss_and_gradient(
            weights,
            x,
            y,
            groups,
            temperature=0.05,
            alpha=0.2,
            penalty_start=1,
        )
        epsilon = 1e-6
        numeric = []
        for index in range(len(weights)):
            delta = np.zeros_like(weights)
            delta[index] = epsilon
            plus = soft_policy_loss_and_gradient(
                weights + delta,
                x,
                y,
                groups,
                temperature=0.05,
                alpha=0.2,
                penalty_start=1,
            )[0]
            minus = soft_policy_loss_and_gradient(
                weights - delta,
                x,
                y,
                groups,
                temperature=0.05,
                alpha=0.2,
                penalty_start=1,
            )[0]
            numeric.append((plus - minus) / (2 * epsilon))
        self.assertTrue(np.allclose(gradient, numeric, rtol=1e-5, atol=1e-6))
        self.assertTrue(np.isfinite(loss))

    def test_selection_uses_policy_regret_before_row_mse(self):
        rows = [
            {"group_id": "g0", "target": 0.10},
            {"group_id": "g0", "target": 0.09},
            {"group_id": "g1", "target": -0.10},
            {"group_id": "g1", "target": 0.01},
        ]
        low_mse_wrong_policy = np.asarray([0.08, 0.09, 0.02, 0.0])
        higher_mse_right_policy = np.asarray([0.20, 0.10, -0.20, 0.02])
        selected, diagnostics = select_by_policy(
            rows,
            {1.0: low_mse_wrong_policy, 10.0: higher_mse_right_policy},
            temperature=0.05,
        )
        self.assertEqual(selected, 10.0)
        by_alpha = {item["alpha"]: item for item in diagnostics}
        self.assertLess(by_alpha[1.0]["row_mse"], by_alpha[10.0]["row_mse"])
        self.assertGreater(
            by_alpha[1.0]["policy_regret"], by_alpha[10.0]["policy_regret"]
        )

    def test_prompt_mean_crossfit_uses_fold_local_text_features(self):
        rows = []
        state = {}
        for prompt_index in range(6):
            prompt = f"p{prompt_index}"
            for seed, noise in ((42, -0.01), (100, 0.0), (2024, 0.01)):
                group = f"{prompt}_b{seed}"
                state[group] = np.asarray([float(seed % 7)])
                rows.append(row(prompt, seed, "SA_000", 0, noise))
                rows.append(
                    row(
                        prompt,
                        seed,
                        "SA_100",
                        1,
                        0.02 * (prompt_index - 2) + noise,
                    )
                )
        means, _, _ = prompt_mean_residual_decomposition(rows)
        predictions = crossfit_soft_models(
            means,
            rows,
            family="prompt_only",
            prompt_mode="tfidf",
            prompt_vectors=None,
            state=state,
            shuffled_state=state,
            folds=3,
            seed=7,
            max_features=32,
            alphas=[0.1],
            temperature=0.05,
            max_iterations=50,
            include_main=True,
        )[0.1]
        self.assertEqual(predictions.shape, (len(rows),))
        self.assertTrue(np.isfinite(predictions).all())

    def test_state_full_fit_standardizes_from_train_rows(self):
        train_rows = []
        fresh_rows = []
        state = {}
        for prompt_index in range(5):
            prompt = f"train{prompt_index}"
            for seed in (42, 100, 2024):
                group = f"{prompt}_b{seed}"
                state[group] = np.asarray([prompt_index + seed / 3000.0])
                train_rows.append(row(prompt, seed, "SA_000", 0, -0.01))
                train_rows.append(row(prompt, seed, "SA_100", 1, 0.02))
        for prompt_index in range(2):
            prompt = f"fresh{prompt_index}"
            for seed in (42, 100, 2024):
                group = f"{prompt}_b{seed}"
                state[group] = np.asarray([10.0 + prompt_index + seed / 3000.0])
                fresh_rows.append(row(prompt, seed, "SA_000", 0, 0.0))
                fresh_rows.append(row(prompt, seed, "SA_100", 1, 0.0))
        prediction, model = fit_full_predict(
            train_rows,
            fresh_rows,
            family="state_only",
            prompt_mode="tfidf",
            prompt_vectors=None,
            state=state,
            shuffled_state=state,
            max_features=16,
            alpha=0.1,
            temperature=0.05,
            max_iterations=50,
            include_main=False,
        )
        self.assertEqual(prediction.shape, (len(fresh_rows),))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(np.isfinite(model["loss"]))


if __name__ == "__main__":
    unittest.main()
