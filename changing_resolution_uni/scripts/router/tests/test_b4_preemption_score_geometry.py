from __future__ import annotations

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    from changing_resolution_uni.scripts.router import (
        audit_b4_preemption_score_geometry as audit,
    )
except ModuleNotFoundError:
    audit = None


@unittest.skipIf(audit is None, "PyTorch or score-audit dependencies are not installed")
class B4PreemptionScoreGeometryTest(unittest.TestCase):
    def test_balanced_threshold_preserves_score_direction(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        scores = np.asarray([-2.0, -1.0, 0.5, 1.0], dtype=np.float32)
        threshold = audit.select_balanced_threshold(labels, scores)
        self.assertEqual(audit.balanced_accuracy_at(labels, scores, threshold), 1.0)

    def test_score_examples_keep_anchor_only_policy_groups(self) -> None:
        steps = np.arange(40, 44)
        trajectories = []
        for prompt_id in (1, 2):
            trajectories.append(
                {
                    "prompt_id": prompt_id,
                    "seed": 42 + prompt_id,
                    "features": np.zeros((4, 4), dtype=np.float32),
                    "sigmas": np.linspace(0.9, 0.1, 4, dtype=np.float32),
                    "qualities": np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
                    "costs": np.zeros(4, dtype=np.float32),
                    "calibrated_latencies": np.asarray(
                        [4.0, 3.0, 2.0, 1.0], dtype=np.float32
                    ),
                }
            )
        signals = np.zeros((2, 4, 12), dtype=np.float32)
        probabilities = np.zeros((2, 1, 4), dtype=np.float32)
        probabilities[0, 0, 0] = 1.0
        probabilities[1, 0, 3] = 1.0
        inputs, shuffled, rows, groups = audit.build_score_examples(
            "train",
            trajectories,
            signals,
            signals,
            probabilities,
            [0.0],
            steps,
            np.ones(4, dtype=np.float32),
            radius=3,
            temperature=0.001,
            harm_epsilon=0.001,
        )
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(rows), 3)
        self.assertEqual(inputs.shape, shuffled.shape)

    def test_policy_summary_counts_anchor_only_groups(self) -> None:
        rows = [
            {
                "trajectory_index": 1,
                "lambda_index": 0,
                "candidate_index": 0,
                "state_score": 1.0,
                "current_utility": 0.9,
                "b4_utility": 0.6,
                "current_quality": 0.9,
                "b4_quality": 0.6,
                "current_latency_sec": 4.0,
                "b4_latency_sec": 1.0,
                "offset_from_b4": -3,
            }
        ]
        groups = [
            {
                "trajectory_index": 0,
                "lambda_index": 0,
                "prompt_id": 1,
                "b4_utility": 0.9,
                "oracle_utility": 0.9,
            },
            {
                "trajectory_index": 1,
                "lambda_index": 0,
                "prompt_id": 2,
                "b4_utility": 0.6,
                "oracle_utility": 0.9,
            },
        ]
        result = audit.summarize_policy(
            rows,
            groups,
            "state_score",
            threshold=0.0,
            harm_epsilon=0.001,
            bootstrap_samples=20,
            rng=np.random.default_rng(3),
        )
        self.assertEqual(result["trajectory_lambda_count"], 2)
        self.assertAlmostEqual(result["decision_change_rate"], 0.5)
        self.assertAlmostEqual(result["mean_utility_gain_vs_b4"], 0.15)


if __name__ == "__main__":
    unittest.main()
