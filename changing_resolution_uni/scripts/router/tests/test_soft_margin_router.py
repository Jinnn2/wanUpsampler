from __future__ import annotations

import unittest

import numpy as np
import torch

from changing_resolution_uni.scripts.router import train_soft_margin_router as soft
from changing_resolution_uni.scripts.router import train_variable_lambda_router as base


class SoftMarginRouterTest(unittest.TestCase):
    def test_suffix_margin_and_soft_targets_are_continuous(self) -> None:
        qualities = np.asarray([0.80, 0.801, 0.82], dtype=np.float32)
        costs = np.asarray([0.5, 0.4, 0.2], dtype=np.float32)
        margin, probability, weight = soft.soft_margin_targets(
            qualities, costs, lambda_value=0.0, temperature=0.02
        )
        self.assertLess(float(margin[0]), float(margin[1]))
        self.assertGreater(float(probability[1]), float(probability[0]))
        self.assertGreater(float(probability[0]), 0.0)
        self.assertLess(float(probability[0]), 1.0)
        self.assertEqual(float(probability[-1]), 1.0)
        self.assertEqual(float(weight[0]), 1.0)
        self.assertEqual(float(weight[-1]), 0.0)
        self.assertTrue(np.isfinite(margin).all())

    def test_step_normalizer_removes_per_candidate_mean(self) -> None:
        trajectories = [
            {"features": np.asarray([[1.0, 10.0], [100.0, 1000.0]], dtype=np.float32)},
            {"features": np.asarray([[3.0, 14.0], [104.0, 1008.0]], dtype=np.float32)},
        ]
        mean, std = soft.fit_step_state_normalizer(trajectories)
        normalized = np.stack(
            [(trajectory["features"] - mean) / std for trajectory in trajectories]
        )
        np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=1e-6)
        self.assertFalse(np.allclose(mean[0], mean[1]))

    def test_trajectory_dataset_does_not_flatten_candidate_steps(self) -> None:
        trajectories = [
            {
                "pooled_t5": np.ones(4096, dtype=np.float32),
                "features": np.ones((3, 2), dtype=np.float32),
                "sigmas": np.asarray([0.6, 0.3, 0.1], dtype=np.float32),
                "qualities": np.asarray([0.80, 0.81, 0.82], dtype=np.float32),
                "costs": np.asarray([0.7, 0.5, 0.3], dtype=np.float32),
            },
            {
                "pooled_t5": np.ones(4096, dtype=np.float32),
                "features": np.full((3, 2), 2.0, dtype=np.float32),
                "sigmas": np.asarray([0.6, 0.3, 0.1], dtype=np.float32),
                "qualities": np.asarray([0.81, 0.82, 0.83], dtype=np.float32),
                "costs": np.asarray([0.7, 0.5, 0.3], dtype=np.float32),
            },
        ]
        dataset = soft.SoftMarginTrajectoryDataset(
            trajectories,
            [0.01, 0.08],
            np.asarray([30, 40, 50]),
            np.zeros((3, 2), dtype=np.float32),
            np.ones((3, 2), dtype=np.float32),
            np.asarray([0.7, 0.5, 0.3], dtype=np.float32),
            margin_temperature=0.02,
        )
        self.assertEqual(len(dataset), 4)
        item = dataset[0]
        self.assertEqual(tuple(item["state"].shape), (3, 2))
        self.assertEqual(tuple(item["soft_stop_target"].shape), (3,))

    def test_zero_residual_exactly_reproduces_b4_argmax(self) -> None:
        prior = base.VariableLambdaB4Prior(candidate_count=3, dropout=0.0)
        with torch.no_grad():
            for parameter in prior.parameters():
                parameter.zero_()
            prior.head[-1].bias.copy_(torch.log(torch.tensor([0.40, 0.35, 0.25])))
        model = soft.CausalSoftMarginRouter(
            b4_prior=prior,
            state_dim=2,
            dropout=0.0,
            hidden_dim=8,
            residual_logit_limit=4.0,
            use_state=True,
        )
        output = model(
            torch.ones(1, 4096),
            torch.ones(1, 3, 2),
            torch.ones(1, 3, len(base.SCHEDULE_FEATURE_NAMES)),
            torch.tensor([0.08]),
        )
        self.assertTrue(torch.equal(output["residual_logit"], torch.zeros(1, 3)))
        chosen = soft.first_margin_stop(
            output["online_margin"].detach().numpy(), risk_margin=0.0
        )
        self.assertEqual(int(chosen[0]), 0)
        self.assertEqual(
            int(output["b4_probabilities"].argmax(dim=1).item()), int(chosen[0])
        )

    def test_causal_router_does_not_leak_future_state(self) -> None:
        torch.manual_seed(7)
        prior = base.VariableLambdaB4Prior(candidate_count=3, dropout=0.0)
        model = soft.CausalSoftMarginRouter(
            b4_prior=prior,
            state_dim=2,
            dropout=0.0,
            hidden_dim=8,
            residual_logit_limit=4.0,
            use_state=True,
        )
        with torch.no_grad():
            model.residual_head.weight.fill_(0.1)
        first_state = torch.zeros(1, 3, 2)
        second_state = first_state.clone()
        second_state[:, 2] = 100.0
        arguments = (
            torch.ones(1, 4096),
            torch.ones(1, 3, len(base.SCHEDULE_FEATURE_NAMES)),
            torch.tensor([0.08]),
        )
        first = model(arguments[0], first_state, arguments[1], arguments[2])[
            "online_margin"
        ]
        second = model(arguments[0], second_state, arguments[1], arguments[2])[
            "online_margin"
        ]
        torch.testing.assert_close(first[:, :2], second[:, :2])

    def test_control_has_matched_architecture_and_ignores_state(self) -> None:
        torch.manual_seed(11)
        prior = base.VariableLambdaB4Prior(candidate_count=3, dropout=0.0)
        torch.manual_seed(29)
        control = soft.CausalSoftMarginRouter(
            b4_prior=prior,
            state_dim=2,
            dropout=0.0,
            hidden_dim=8,
            residual_logit_limit=4.0,
            use_state=False,
        )
        torch.manual_seed(29)
        state = soft.CausalSoftMarginRouter(
            b4_prior=prior,
            state_dim=2,
            dropout=0.0,
            hidden_dim=8,
            residual_logit_limit=4.0,
            use_state=True,
        )
        self.assertEqual(
            {name: tuple(value.shape) for name, value in control.state_dict().items()},
            {name: tuple(value.shape) for name, value in state.state_dict().items()},
        )
        for name, value in control.state_dict().items():
            torch.testing.assert_close(value, state.state_dict()[name])
        arguments = (
            torch.ones(1, 4096),
            torch.ones(1, 3, len(base.SCHEDULE_FEATURE_NAMES)),
            torch.tensor([0.08]),
        )
        first = control(arguments[0], torch.zeros(1, 3, 2), arguments[1], arguments[2])[
            "online_margin"
        ]
        second = control(
            arguments[0], torch.full((1, 3, 2), 99.0), arguments[1], arguments[2]
        )["online_margin"]
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
