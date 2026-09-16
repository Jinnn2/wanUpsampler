from __future__ import annotations

import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import numpy as np
    import torch

    from changing_resolution_uni.scripts.router.summarize_continuous_budget_prior import (
        mean_ci,
        metric_delta,
    )
    from changing_resolution_uni.scripts.router.train_continuous_budget_prior import (
        ContinuousBudgetRegressor,
        b4_temperature_compatibility,
        budget_targets,
        calibrate_budget_grid,
        continuous_budget_loss,
        evaluate_all_policies,
        matched_fixed_mixture_spec,
        nearest_budget_index,
        scalar_action_distribution,
        train_selected_fixed_index,
    )
    from changing_resolution_uni.scripts.router.model_router import (
        SoftDistillationMLPRouter,
    )


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class ContinuousBudgetPriorTest(unittest.TestCase):
    def test_hard_target_allows_legacy_b4_without_temperature_metadata(self) -> None:
        result = b4_temperature_compatibility(
            {},
            requested_tau=0.02,
            target_type="hard_oracle",
            require_match=False,
        )
        self.assertIsNone(result["checkpoint_soft_target_tau"])
        self.assertFalse(result["matches"])
        self.assertFalse(result["continuous_target_uses_temperature"])

    def test_strict_temperature_check_reports_both_values(self) -> None:
        with self.assertRaisesRegex(ValueError, r"checkpoint=0\.01, requested=0\.02"):
            b4_temperature_compatibility(
                {"soft_target_tau": 0.01},
                requested_tau=0.02,
                target_type="soft_expected",
                require_match=True,
            )

    def test_summary_delta_direction_is_positive_when_candidate_is_better(self) -> None:
        self.assertAlmostEqual(metric_delta(0.01, 0.03, "policy_regret"), 0.02)
        self.assertAlmostEqual(metric_delta(0.84, 0.82, "realized_vbench5"), 0.02)

    def test_prompt_bootstrap_averages_training_seeds_inside_prompt(self) -> None:
        point, low, high = mean_ci(
            {1: [0.2, 0.4], 2: [0.6, 0.8]},
            samples=100,
            rng=np.random.default_rng(7),
        )
        self.assertAlmostEqual(point, 0.5)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)

    def test_model_outputs_one_bounded_budget_per_prompt(self) -> None:
        model = ContinuousBudgetRegressor(
            in_dim=8,
            hidden_dims=(4,),
            dropout=0.0,
            output_min=0.2,
            output_max=0.6,
        )
        output = model(torch.randn(3, 8))
        self.assertEqual(tuple(output.shape), (3,))
        self.assertTrue(torch.all(output >= 0.2))
        self.assertTrue(torch.all(output <= 0.6))

    def test_scalar_backbone_matches_b4_hidden_parameter_shapes(self) -> None:
        scalar = ContinuousBudgetRegressor(in_dim=8, hidden_dims=(6, 4), dropout=0.0)
        b4 = SoftDistillationMLPRouter(
            in_dim=8, hidden_dims=[6, 4], num_classes=3, dropout=0.0
        )
        self.assertEqual(
            [tuple(value.shape) for value in scalar.mlp.parameters()],
            [tuple(value.shape) for value in b4.mlp.parameters()],
        )

    def test_utility_loss_moves_scalar_toward_higher_utility_budget(self) -> None:
        predicted = torch.tensor([0.5], requires_grad=True)
        grid = torch.tensor([0.2, 0.4, 0.6])
        batch = {
            "utilities": torch.tensor([[0.9, 0.5, 0.1]]),
            "target_step_idx": torch.tensor([0]),
            "soft_utility_target": torch.tensor([[1.0, 0.0, 0.0]]),
        }
        probabilities = scalar_action_distribution(predicted, grid, 0.04)
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(1)))
        loss, components = continuous_budget_loss(
            predicted,
            batch,
            grid,
            loss_type="utility_expected",
            target_type="hard_oracle",
            huber_beta=0.02,
            budget_temperature=0.04,
            utility_temperature=0.02,
            regression_weight=0.1,
        )
        loss.backward()
        self.assertGreater(float(predicted.grad), 0.0)
        self.assertGreater(float(components["expected_regret"]), 0.0)

    def test_matched_fixed_mixture_brackets_mean_cost(self) -> None:
        spec = matched_fixed_mixture_spec(35.0, torch.tensor([70.0, 40.0, 20.0]))
        self.assertEqual(spec["lower_index"], 2)
        self.assertEqual(spec["upper_index"], 1)
        self.assertAlmostEqual(spec["upper_weight"], 0.75)

    def test_nearest_budget_uses_nonuniform_cost_coordinates(self) -> None:
        grid = torch.tensor([0.72, 0.51, 0.33, 0.20])
        predicted = torch.tensor([0.47, 0.28, 0.69])
        self.assertEqual(
            nearest_budget_index(predicted, grid).tolist(),
            [1, 2, 0],
        )

    def test_hard_and_soft_budget_targets(self) -> None:
        grid = torch.tensor([0.7, 0.4, 0.2])
        batch = {
            "utilities": torch.ones(2, 3),
            "target_step_idx": torch.tensor([0, 2]),
            "soft_utility_target": torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]]),
        }
        hard = budget_targets(batch, grid, "hard_oracle")
        soft = budget_targets(batch, grid, "soft_expected")
        self.assertTrue(torch.allclose(hard, torch.tensor([0.7, 0.2])))
        self.assertTrue(torch.allclose(soft, torch.tensor([0.55, 0.25])))

    def test_budget_grid_is_train_median_latency_ratio(self) -> None:
        loader = [
            {
                "latencies": torch.tensor([[50.0, 20.0], [60.0, 30.0]]),
                "native_latency": torch.tensor([100.0, 100.0]),
            },
            {
                "latencies": torch.tensor([[70.0, 40.0]]),
                "native_latency": torch.tensor([100.0]),
            },
        ]
        grid = calibrate_budget_grid(loader)
        self.assertTrue(torch.allclose(grid, torch.tensor([0.6, 0.3])))

    def test_fixed_budget_is_selected_from_train_utility_only(self) -> None:
        loader = [
            {"utilities": torch.tensor([[0.2, 0.5, 0.4], [0.1, 0.6, 0.2]])},
            {"utilities": torch.tensor([[0.4, 0.3, 0.1]])},
        ]
        self.assertEqual(train_selected_fixed_index(loader), 1)

    def test_policy_evaluation_includes_projection_and_scalar_nearest(self) -> None:
        class FixedBudget(torch.nn.Module):
            def forward(self, pooled_t5: torch.Tensor) -> torch.Tensor:
                return torch.tensor([0.21, 0.68])[: pooled_t5.shape[0]]

        class FixedB4(torch.nn.Module):
            def forward(self, pooled_t5: torch.Tensor) -> dict[str, torch.Tensor]:
                probs = torch.tensor([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]])[
                    : pooled_t5.shape[0]
                ]
                return {
                    "discrete_probs": probs,
                    "pred_step_idx": probs.argmax(dim=1),
                }

        dimensions = {
            name: torch.tensor([[0.7, 0.8, 0.9], [0.9, 0.8, 0.7]])
            for name in (
                "subject_consistency",
                "background_consistency",
                "motion_smoothness",
                "aesthetic_quality",
                "imaging_quality",
            )
        }
        loader = [
            {
                "prompt_id": torch.tensor([10, 11]),
                "pooled_t5": torch.zeros(2, 4),
                "target_step_idx": torch.tensor([2, 0]),
                "soft_utility_target": torch.tensor([[0.0, 0.1, 0.9], [0.9, 0.1, 0.0]]),
                "utilities": torch.tensor([[0.70, 0.80, 0.90], [0.90, 0.80, 0.70]]),
                "vbench5": torch.tensor([[0.70, 0.80, 0.90], [0.90, 0.80, 0.70]]),
                "latencies": torch.tensor([[70.0, 40.0, 20.0], [70.0, 40.0, 20.0]]),
                "native_latency": torch.tensor([100.0, 100.0]),
                "seed_oracle_utility": torch.tensor([0.91, 0.91]),
                "vbench_dimensions": dimensions,
            }
        ]
        summaries, rows, mixture, fixed_curve = evaluate_all_policies(
            continuous_model=FixedBudget(),
            b4_model=FixedB4(),
            loader=loader,
            candidate_steps=[40, 45, 50],
            budget_grid=torch.tensor([0.7, 0.4, 0.2]),
            fixed_index=1,
            target_type="soft_expected",
            device=torch.device("cpu"),
            split="validation",
        )
        self.assertEqual(len(summaries), 6)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len(fixed_curve), 3)
        continuous = [
            row for row in rows if row["model_type"] == "continuous_budget_nearest"
        ]
        self.assertEqual([row["chosen_step"] for row in continuous], [50, 40])
        projected = [row for row in rows if row["model_type"] == "b4_projected_nearest"]
        self.assertEqual([row["chosen_step"] for row in projected], [45, 40])
        self.assertAlmostEqual(mixture["target_mean_latency_seconds"], 45.0)
        matched = [row for row in rows if row["model_type"] == "matched_fixed_mixture"]
        self.assertAlmostEqual(
            sum(row["realized_latency_sec"] for row in matched) / len(matched),
            45.0,
        )


if __name__ == "__main__":
    unittest.main()
