from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import torch

    from UNIV_adaptor.scripts.router.train_combined_v3_b4_control import (
        B4SoftUtilityRouter,
        b4_distillation_loss,
        expand_variable_lambda_batch,
        normalize_lambda,
        paired_bootstrap,
        soft_utility_targets,
        train_one_run,
        training_tensors,
    )

    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is unavailable")
class CombinedV3B4ControlTest(unittest.TestCase):
    def test_b4_hidden_backbone_matches_original_pattern(self) -> None:
        model = B4SoftUtilityRouter(8, 3, lambda_conditioned=False)
        names = [type(layer).__name__ for layer in model.mlp]
        self.assertEqual(
            names,
            [
                "Linear",
                "LayerNorm",
                "SiLU",
                "Dropout",
                "Linear",
                "LayerNorm",
                "SiLU",
                "Dropout",
            ],
        )
        self.assertEqual(tuple(model(torch.zeros(2, 8)).shape), (2, 3))

    def test_soft_target_uses_quality_minus_lambda_cost(self) -> None:
        quality = torch.tensor([[0.7, 0.8, 0.9]])
        cost = torch.tensor([0.1, 0.4, 0.9])
        low_lambda = soft_utility_targets(quality, cost, 0.01, temperature=0.02)
        high_lambda = soft_utility_targets(quality, cost, 0.5, temperature=0.02)
        self.assertEqual(int(low_lambda.argmax(dim=1)), 2)
        self.assertEqual(int(high_lambda.argmax(dim=1)), 0)
        self.assertAlmostEqual(float(low_lambda.sum()), 1.0, places=6)

    def test_wasserstein_uses_cost_order(self) -> None:
        logits = torch.log(torch.tensor([[0.05, 0.90, 0.05]]))
        targets = torch.tensor([[0.90, 0.05, 0.05]])
        _, _, catalog_emd = b4_distillation_loss(
            logits,
            targets,
            emd_weight=0.5,
            cost_order=torch.tensor([0, 1, 2]),
        )
        _, _, cost_emd = b4_distillation_loss(
            logits,
            targets,
            emd_weight=0.5,
            cost_order=torch.tensor([0, 2, 1]),
        )
        self.assertNotAlmostEqual(float(catalog_emd), float(cost_emd))

    def test_variable_lambda_training_expands_each_prompt(self) -> None:
        embeddings = torch.zeros(2, 8)
        quality = torch.tensor([[0.7, 0.8], [0.6, 0.9]])
        dataset = training_tensors(
            embeddings,
            quality,
            torch.tensor([0.2, 0.6]),
            [0.01, 0.1],
            temperature=0.02,
            lambda_conditioned=True,
            lambda_min=0.01,
            lambda_max=0.1,
        )
        self.assertEqual(len(dataset), 2)
        repeated, features, targets = expand_variable_lambda_batch(
            dataset.tensors[0],
            dataset.tensors[1],
            torch.tensor([0.2, 0.6]),
            [0.01, 0.1],
            temperature=0.02,
            lambda_min=0.01,
            lambda_max=0.1,
        )
        self.assertEqual(tuple(repeated.shape), (4, 8))
        self.assertTrue(torch.allclose(features, torch.tensor([-1.0, 1.0, -1.0, 1.0])))
        self.assertTrue(torch.allclose(targets.sum(dim=1), torch.ones(4)))

    def test_paired_bootstrap_reports_per_lambda_and_macro(self) -> None:
        rows = []
        for prompt_id in range(4):
            for lambda_value in (0.01, 0.02):
                for method, regret in (("fixed", 0.02), ("b4", 0.01)):
                    rows.append(
                        {
                            "global_prompt_id": prompt_id,
                            "lambda": lambda_value,
                            "method": method,
                            "policy_regret": regret,
                            "realized_utility": 1.0 - regret,
                            "realized_vbench5": 0.8,
                            "normalized_cost": 0.3,
                        }
                    )
        per_lambda, macro = paired_bootstrap(
            rows,
            reference_method="fixed",
            candidate_methods=["b4"],
            bootstrap_samples=100,
            bootstrap_seed=7,
        )
        regret_rows = [row for row in per_lambda if row["metric"] == "policy_regret"]
        self.assertEqual(len(regret_rows), 2)
        self.assertTrue(all(row["improvement_mean"] > 0.0 for row in regret_rows))
        macro_regret = next(row for row in macro if row["metric"] == "policy_regret")
        self.assertAlmostEqual(macro_regret["macro_improvement_mean"], 0.01)

    def test_lambda_normalization_spans_minus_one_to_one(self) -> None:
        values = normalize_lambda(
            torch.tensor([0.01, 0.055, 0.1]), lambda_min=0.01, lambda_max=0.1
        )
        np.testing.assert_allclose(values.numpy(), [-1.0, 0.0, 1.0], atol=1e-6)

    def test_fixed_and_variable_training_smoke(self) -> None:
        train_embeddings = torch.randn(4, 4096)
        validation_embeddings = torch.randn(2, 4096)
        train_quality = torch.tensor(
            [
                [0.70, 0.75, 0.80],
                [0.80, 0.74, 0.70],
                [0.71, 0.79, 0.76],
                [0.75, 0.73, 0.81],
            ]
        )
        validation_quality = torch.tensor([[0.72, 0.77, 0.80], [0.81, 0.75, 0.71]])
        cost = torch.tensor([0.2, 0.4, 0.7])
        args = SimpleNamespace(
            dropout=0.1,
            epochs=2,
            batch_size=2,
            num_workers=0,
            lr=1e-3,
            weight_decay=1e-4,
            soft_target_tau=0.02,
            emd_weight=0.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            out_root = Path(directory)
            common = {
                "seed": 42,
                "train_embeddings": train_embeddings,
                "train_quality": train_quality,
                "validation_embeddings": validation_embeddings,
                "validation_quality": validation_quality,
                "normalized_cost": cost,
                "lambdas": [0.01, 0.1],
                "action_ids": ["A", "B", "C"],
                "args": args,
                "device": torch.device("cpu"),
                "cost_order": torch.argsort(cost),
                "provenance": {"dataset": "fixture", "test_accessed": False},
                "out_root": out_root,
            }
            fixed = train_one_run(
                model_type="b4_fixed_lambda_bank",
                lambda_value=0.01,
                **common,
            )
            variable = train_one_run(
                model_type="b4_variable_lambda",
                lambda_value=None,
                **common,
            )
            self.assertTrue(Path(fixed["checkpoint"]).is_file())
            self.assertTrue(Path(variable["checkpoint"]).is_file())
            resumed = train_one_run(
                model_type="b4_fixed_lambda_bank",
                lambda_value=0.01,
                **common,
            )
            self.assertEqual(resumed["checkpoint_sha256"], fixed["checkpoint_sha256"])


if __name__ == "__main__":
    unittest.main()
