from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from changing_resolution_uni.scripts.router.model_router import (
        RelativeQualityCurveMLPRouter,
        SoftDistillationMLPRouter,
    )
    from changing_resolution_uni.scripts.router.train_router import (
        choose_model_step,
        train_single_model,
    )


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class QualityCurveRouterTest(unittest.TestCase):
    def test_b4_q_preserves_b4_parameterization(self) -> None:
        b4 = SoftDistillationMLPRouter(in_dim=16, hidden_dims=[8, 4], num_classes=3)
        b4_q = RelativeQualityCurveMLPRouter(
            in_dim=16, hidden_dims=[8, 4], num_classes=3
        )
        self.assertEqual(
            {name: tuple(value.shape) for name, value in b4.state_dict().items()},
            {name: tuple(value.shape) for name, value in b4_q.state_dict().items()},
        )
        output = b4_q(torch.ones(2, 16))
        self.assertEqual(tuple(output["quality_deltas"].shape), (2, 3))

    def test_quality_curve_selection_combines_quality_and_normalized_latency(
        self,
    ) -> None:
        output = {"quality_deltas": torch.tensor([[0.03, 0.02, 0.0]])}
        latencies = torch.tensor([[100.0, 70.0, 40.0]])
        native_latency = torch.tensor([200.0])
        chosen = choose_model_step(
            output,
            latencies,
            native_latency,
            primary_lambda=0.08,
        )
        self.assertEqual(chosen.tolist(), [1])

    def test_direct_router_selection_remains_unchanged(self) -> None:
        output = {"pred_step_idx": torch.tensor([2, 0])}
        chosen = choose_model_step(
            output,
            torch.ones(2, 3),
            torch.ones(2),
            primary_lambda=0.08,
        )
        self.assertEqual(chosen.tolist(), [2, 0])

    def test_quality_curve_training_smoke(self) -> None:
        candidate_steps = [40, 45, 50]
        samples = []
        for prompt_id in range(4):
            vbench = torch.tensor(
                [0.80 + 0.01 * prompt_id, 0.82, 0.81 - 0.005 * prompt_id]
            )
            latencies = torch.tensor([80.0, 60.0, 40.0])
            native_latency = torch.tensor(200.0)
            utilities = vbench - 0.08 * latencies / native_latency
            target_idx = int(torch.argmax(utilities))
            samples.append(
                {
                    "pooled_t5": torch.randn(4096),
                    "relative_quality_target": vbench - vbench[-1],
                    "target_step_idx": torch.tensor(target_idx),
                    "target_step": torch.tensor(candidate_steps[target_idx]),
                    "utilities": utilities,
                    "vbench5": vbench,
                    "latencies": latencies,
                    "native_latency": native_latency,
                    "seed_oracle_utility": utilities.max(),
                }
            )
        loader = torch.utils.data.DataLoader(samples, batch_size=2, shuffle=False)
        args = SimpleNamespace(
            epochs=1,
            lr=1e-3,
            weight_decay=1e-4,
            quality_curve_beta=0.01,
            primary_lambda=0.08,
        )
        model, metrics, best_epoch = train_single_model(
            "mlp_quality_curve",
            loader,
            loader,
            candidate_steps,
            args,
            torch.device("cpu"),
        )
        self.assertIsInstance(model, RelativeQualityCurveMLPRouter)
        self.assertEqual(best_epoch, 1)
        self.assertGreaterEqual(metrics["policy_regret"], 0.0)


if __name__ == "__main__":
    unittest.main()
