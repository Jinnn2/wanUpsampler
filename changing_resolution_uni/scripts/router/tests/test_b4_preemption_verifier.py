from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import torch

    from changing_resolution_uni.scripts.router import (
        train_b4_preemption_verifier as verifier,
    )
except ModuleNotFoundError:
    torch = None
    verifier = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class B4PreemptionVerifierTest(unittest.TestCase):
    def test_causal_signals_use_only_current_and_past_steps(self) -> None:
        raw = np.asarray([[[1.0], [3.0], [7.0], [100.0]]], dtype=np.float32)
        signals = verifier.build_causal_signals(raw)
        np.testing.assert_allclose(signals[0, :, 0], [1.0, 3.0, 7.0, 100.0])
        np.testing.assert_allclose(signals[0, :, 1], [0.0, 2.0, 4.0, 93.0])
        np.testing.assert_allclose(signals[0, :, 2], [0.0, 0.0, 3.0, 48.5])
        changed_future = raw.copy()
        changed_future[0, 3, 0] = -500.0
        changed = verifier.build_causal_signals(changed_future)
        np.testing.assert_allclose(changed[0, :3], signals[0, :3])

    def test_step_normalizer_uses_per_absolute_step_statistics(self) -> None:
        signals = np.asarray(
            [
                [[1.0, 10.0], [100.0, 1000.0]],
                [[3.0, 14.0], [104.0, 1008.0]],
            ],
            dtype=np.float32,
        )
        mean, std = verifier.fit_step_normalizer(signals)
        normalized = verifier.normalize_signals(signals, mean, std)
        np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=1e-6)
        self.assertFalse(np.allclose(mean[0], mean[1]))

    def test_restricted_suffix_margin_stops_at_b4_anchor(self) -> None:
        utility = np.asarray([0.8, 0.7, 0.75, 0.6, 1.0], dtype=np.float32)
        margin = verifier.restricted_suffix_margin(utility, 0, 3)
        self.assertAlmostEqual(margin, 0.05, places=6)
        with self.assertRaises(ValueError):
            verifier.restricted_suffix_margin(utility, 3, 3)

    def test_training_examples_are_limited_to_b4_minus_three(self) -> None:
        candidate_steps = np.arange(40, 45)
        trajectory = {
            "qualities": np.asarray([0.8, 0.7, 0.75, 0.6, 0.5], dtype=np.float32),
            "costs": np.zeros(5, dtype=np.float32),
            "sigmas": np.linspace(0.9, 0.1, 5, dtype=np.float32),
        }
        signals = np.zeros((1, 5, 12), dtype=np.float32)
        probabilities = np.zeros((1, 1, 5), dtype=np.float32)
        probabilities[0, 0, 3] = 1.0
        inputs, targets, margins = verifier.build_training_examples(
            [trajectory],
            signals,
            probabilities,
            [0.0],
            candidate_steps,
            np.ones(5, dtype=np.float32),
            radius=3,
            temperature=0.1,
        )
        self.assertEqual(inputs.shape[0], 3)
        expected = 1.0 / (1.0 + math.exp(-0.5))
        self.assertAlmostEqual(float(targets[0]), expected, places=6)
        self.assertAlmostEqual(float(margins[0]), 0.05, places=6)

    def test_margin_diagnostics_expose_threshold_frontier(self) -> None:
        diagnostics = verifier.margin_diagnostics(
            np.asarray([-1.0, 0.2, 1.2], dtype=np.float32),
            np.asarray([0.1, 0.5, 0.9], dtype=np.float32),
            [0.0, 1.0],
        )
        self.assertGreater(diagnostics["soft_margin_loss"], 0.0)
        self.assertAlmostEqual(diagnostics["logit_ge_0p0_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(diagnostics["logit_ge_1p0_rate"], 1.0 / 3.0)

    def test_state_audit_reports_raw_structural_zeros(self) -> None:
        raw = np.asarray([[0.0, 1.0], [0.0, 3.0]], dtype=np.float32)
        normalized = np.asarray([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
        rows = verifier.state_feature_audit_rows(
            "train", raw, normalized, ["delta", "value"]
        )
        self.assertEqual(rows[0]["raw_exact_zero_rate"], 1.0)
        self.assertEqual(rows[0]["normalized_exact_zero_rate"], 0.0)

    def test_sequential_choice_uses_first_eligible_preemption(self) -> None:
        chosen = verifier.sequential_choice(
            np.asarray([0.2, 1.4, 3.0]), [3, 4, 5], anchor=6, threshold=1.0
        )
        self.assertEqual(chosen, 4)
        fallback = verifier.sequential_choice(
            np.asarray([0.2, 0.4]), [4, 5], anchor=6, threshold=1.0
        )
        self.assertEqual(fallback, 6)

    def test_control_model_ignores_state_prefix(self) -> None:
        model = verifier.SparsePreemptionVerifier(
            input_dim=20,
            state_dim=12,
            hidden_dim=16,
            dropout=0.0,
            use_state=False,
        ).eval()
        first = torch.zeros(2, 20)
        second = first.clone()
        second[:, :12] = 100.0
        with torch.no_grad():
            np.testing.assert_allclose(model(first).numpy(), model(second).numpy())

    def test_shuffle_preserves_each_absolute_step_multiset(self) -> None:
        signals = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
        shuffled = verifier.shuffled_validation_signals(signals, seed=7)
        for step in range(signals.shape[1]):
            original = sorted(map(tuple, signals[:, step]))
            observed = sorted(map(tuple, shuffled[:, step]))
            self.assertEqual(original, observed)


if __name__ == "__main__":
    unittest.main()
