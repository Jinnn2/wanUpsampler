from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from UNIV_adaptor.scripts.router.train_combined_v3_prompt_prior_v2 import (
    ATTENTION,
    BIAS_ONLY,
    FIXED_MODEL,
    POOLED,
    PromptStore,
    StructuredQualityPrior,
    action_feature_matrix,
    best_cost_matched_mixture,
    cross_validate,
    deranged_indices,
    deterministic_folds,
    evaluation_rows,
    mean_policy_regret,
    train_epochs,
    utility_aligned_loss,
)


def catalog() -> list[dict]:
    return [
        {
            "artifact_id": "V2_B30",
            "requested_action": {
                "spatial_ratio": 0.625,
                "temporal_ratio": 0.67,
                "lr_nfe_ratio": 0.85,
                "switch_ratio": 0.9,
            },
            "proxy_compute_density": 0.30,
        },
        {
            "artifact_id": "LB30_LR25_S0300_HR04",
            "execution_action": {
                "spatial_ratio": 0.75,
                "temporal_ratio": 0.8,
                "true_lr_steps": 25,
                "renoise_sigma": 0.3,
                "hr_steps": 4,
            },
            "proxy_compute_density": 0.305,
        },
        {
            "artifact_id": "V2_B70",
            "requested_action": {
                "spatial_ratio": 0.875,
                "temporal_ratio": 0.8,
                "lr_nfe_ratio": 1.0,
                "switch_ratio": 0.8,
            },
            "proxy_compute_density": 0.70,
        },
    ]


class PromptPriorV2CoreTests(unittest.TestCase):
    def test_action_features_keep_two_b30_allocations_distinct(self) -> None:
        features, metadata = action_feature_matrix(
            catalog(), np.asarray([0.20, 0.23, 0.55], dtype=np.float64)
        )
        self.assertEqual(features.shape, (3, 10))
        self.assertFalse(np.allclose(features[0], features[1]))
        self.assertEqual(metadata["feature_names"][0], "spatial_ratio")
        self.assertEqual(metadata["raw_features"][0][5], 0.0)
        self.assertGreater(metadata["raw_features"][1][5], 0.0)

    def test_model_starts_at_global_prior_and_bounds_residual(self) -> None:
        action_features = torch.randn(3, 10)
        global_quality = torch.tensor([0.80, 0.81, 0.82])
        model = StructuredQualityPrior(
            16,
            action_features,
            global_quality,
            architecture=POOLED,
            hidden_dim=8,
            action_hidden_dim=4,
            dropout=0.0,
            max_residual=0.02,
        )
        initial = model(torch.randn(5, 16))
        torch.testing.assert_close(initial, global_quality.expand(5, -1))
        with torch.no_grad():
            model.interaction.weight.fill_(10.0)
        prediction = model(torch.randn(5, 16))
        residual = prediction - global_quality
        self.assertLessEqual(float(residual.detach().abs().max()), 0.020001)

    def test_attention_pool_accepts_variable_length_mask(self) -> None:
        model = StructuredQualityPrior(
            8,
            torch.randn(3, 4),
            torch.tensor([0.8, 0.8, 0.8]),
            architecture=ATTENTION,
            hidden_dim=8,
            action_hidden_dim=4,
            dropout=0.0,
        )
        tokens = torch.randn(2, 5, 8)
        mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]]
        )
        self.assertEqual(model(tokens, mask).shape, (2, 3))

    def test_utility_loss_is_finite_and_differentiable(self) -> None:
        predicted = (torch.randn(7, 3) * 0.01 + 0.8).requires_grad_()
        truth = torch.randn(7, 3) * 0.01 + 0.8
        cost = torch.tensor([0.2, 0.3, 0.6])
        loss, components = utility_aligned_loss(
            predicted,
            truth,
            cost,
            [0.03, 0.07],
            soft_target_tau=0.02,
            emd_weight=0.5,
            pairwise_weight=0.25,
            quality_weight=1.0,
            pairwise_tau=0.02,
            quality_huber_beta=0.02,
            cost_order=torch.argsort(cost),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(components), {"kl", "emd", "pairwise", "quality"})
        loss.backward()
        self.assertIsNotNone(predicted.grad)

    def test_folds_cover_each_prompt_once_and_derangement_has_no_fixed_point(
        self,
    ) -> None:
        folds = deterministic_folds(list(range(23)), 5, 17)
        observed = np.concatenate(folds)
        np.testing.assert_array_equal(np.sort(observed), np.arange(23))
        shuffled = deranged_indices(np.arange(23), 19)
        self.assertTrue(np.all(shuffled != np.arange(23)))

    def test_cost_matched_mixture_uses_train_quality_frontier(self) -> None:
        quality = np.asarray([[0.70, 0.80, 0.82], [0.72, 0.78, 0.84]], dtype=np.float64)
        cost = np.asarray([0.1, 0.3, 0.7], dtype=np.float64)
        mixture = best_cost_matched_mixture(quality, cost, 0.5)
        left, right = int(mixture["left"]), int(mixture["right"])
        weight = float(mixture["right_weight"])
        matched = (1.0 - weight) * cost[left] + weight * cost[right]
        self.assertAlmostEqual(matched, 0.5)

    def test_small_pooled_training_reduces_train_regret(self) -> None:
        torch.manual_seed(4)
        rng = np.random.default_rng(4)
        count = 48
        dim = 16
        action_count = 3
        embeddings = rng.normal(size=(count, dim)).astype(np.float32)
        quality = np.full((count, action_count), 0.8, dtype=np.float32)
        preferred = (embeddings[:, 0] > 0).astype(np.int64)
        quality[np.arange(count), preferred] += 0.04
        samples = [
            {
                "global_prompt_id": index,
                "embedding": embeddings[index],
            }
            for index in range(count)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for index in range(count):
                path = root / f"{index}.npz"
                np.savez_compressed(
                    path,
                    seq_embedding=embeddings[index][None, :].astype(np.float16),
                    attention_mask=np.ones(1, dtype=np.int64),
                )
                paths[index] = path
            store = PromptStore(samples, paths)
            action_features = torch.randn(action_count, 5)
            cost = torch.tensor([0.2, 0.2, 0.7])
            model = StructuredQualityPrior(
                dim,
                action_features,
                torch.from_numpy(quality).mean(dim=0),
                architecture=POOLED,
                hidden_dim=8,
                action_hidden_dim=4,
                dropout=0.0,
                max_residual=0.05,
            )
            before = mean_policy_regret(
                model(torch.from_numpy(embeddings)).detach().numpy(),
                quality,
                cost.numpy(),
                [0.01],
            )
            args = argparse.Namespace(
                lr=0.01,
                weight_decay=0.0,
                min_lr=0.001,
                batch_size=16,
                data_seed=11,
                lambdas=[0.01],
                soft_target_tau=0.02,
                emd_weight=0.0,
                pairwise_weight=0.5,
                quality_weight=1.0,
                pairwise_tau=0.02,
                quality_huber_beta=0.02,
            )
            train_epochs(
                model=model,
                store=store,
                quality=torch.from_numpy(quality),
                train_indices=np.arange(count, dtype=np.int64),
                feature_indices=np.arange(count, dtype=np.int64),
                epochs=20,
                args=args,
                normalized_cost=cost,
                cost_order=torch.argsort(cost),
                device=torch.device("cpu"),
            )
            after = mean_policy_regret(
                model(torch.from_numpy(embeddings)).detach().numpy(),
                quality,
                cost.numpy(),
                [0.01],
            )
            self.assertLess(after, before)

    def test_cross_validation_fold_results_resume_without_validation_data(self) -> None:
        torch.manual_seed(8)
        rng = np.random.default_rng(8)
        samples = [
            {
                "global_prompt_id": index,
                "embedding": rng.normal(size=8).astype(np.float32),
            }
            for index in range(12)
        ]
        store = PromptStore(samples, {index: Path("unused") for index in range(12)})
        quality = torch.from_numpy(
            (rng.normal(size=(12, 3)) * 0.01 + 0.8).astype(np.float32)
        )
        cost = torch.tensor([0.2, 0.3, 0.6])
        args = argparse.Namespace(
            cv_folds=2,
            fold_seed=17,
            max_epochs=2,
            batch_size=6,
            hidden_dim=8,
            action_hidden_dim=4,
            dropout=0.0,
            max_residual=0.03,
            lr=0.001,
            min_lr=0.0001,
            weight_decay=0.0,
            lambdas=[0.03, 0.07],
            soft_target_tau=0.02,
            emd_weight=0.5,
            pairwise_weight=0.25,
            quality_weight=1.0,
            pairwise_tau=0.02,
            quality_huber_beta=0.02,
            data_seed=19,
        )
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(
                architecture=POOLED,
                train_seed=42,
                store=store,
                quality=quality,
                action_features=torch.randn(3, 5),
                normalized_cost=cost,
                cost_order=torch.argsort(cost),
                args=args,
                device=torch.device("cpu"),
                shuffled=False,
                out_root=Path(directory),
                provenance={"dataset_sha256": "test"},
            )
            first_epoch, first_history = cross_validate(**kwargs)
            second_epoch, second_history = cross_validate(**kwargs)
            self.assertEqual(first_epoch, second_epoch)
            self.assertEqual(first_history, second_history)
            self.assertEqual(
                len(list(Path(directory).glob("cv/**/*.json"))),
                2,
            )

    def test_evaluation_includes_external_b4_and_cost_matched_mixture(self) -> None:
        train_quality = np.asarray(
            [[0.78, 0.80, 0.81], [0.79, 0.81, 0.80]], dtype=np.float32
        )
        validation_quality = np.asarray(
            [[0.77, 0.82, 0.80], [0.83, 0.79, 0.81]], dtype=np.float32
        )

        def sample(index: int, quality: np.ndarray) -> dict:
            return {
                "global_prompt_id": index,
                "prompt_sha256": f"prompt-{index}",
                "seed_count": 1,
                "candidate_quality": quality,
                "dimensions": {
                    name: quality.astype(np.float64)
                    for name in (
                        "subject_consistency",
                        "background_consistency",
                        "motion_smoothness",
                        "aesthetic_quality",
                        "imaging_quality",
                    )
                },
            }

        summaries, rows, _, mixture = evaluation_rows(
            train_samples=[sample(i, value) for i, value in enumerate(train_quality)],
            validation_samples=[
                sample(10 + i, value) for i, value in enumerate(validation_quality)
            ],
            action_ids=["a", "b", "c"],
            normalized_cost=np.asarray([0.1, 0.3, 0.6]),
            lambdas=[0.07],
            predictions={POOLED: validation_quality},
            primary_method=POOLED,
            per_seed_predictions={},
            soft_target_tau=0.02,
            external_choices={FIXED_MODEL: {0.07: np.asarray([1, 1])}},
        )
        methods = {row["method"] for row in summaries}
        self.assertIn(BIAS_ONLY, methods)
        self.assertIn(FIXED_MODEL, methods)
        self.assertIn(f"cost_matched_mixture_for_{POOLED}", methods)
        self.assertEqual(len(mixture), 1)
        self.assertTrue(rows)


if __name__ == "__main__":
    unittest.main()
