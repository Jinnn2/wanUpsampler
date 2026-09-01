from __future__ import annotations

import unittest

import numpy as np

from changing_resolution_uni.scripts.router import (
    summarize_b4_preemption_verifier as summary,
)


def make_row(
    model_type: str,
    threshold: str | float,
    prompt_id: int,
    realized_utility: float,
) -> dict[str, object]:
    return {
        "run_id": "seed_42",
        "prompt_id": prompt_id,
        "seed": 1000 + prompt_id,
        "lambda": 0.08,
        "model_type": model_type,
        "risk_threshold": threshold,
        "realized_utility": realized_utility,
    }


class B4PreemptionSummaryTest(unittest.TestCase):
    def test_paired_interval_averages_at_prompt_level(self) -> None:
        reference = [
            make_row("b4_offline", "baseline", 1, 0.5),
            make_row("b4_offline", "baseline", 2, 0.4),
        ]
        candidate = [
            make_row("preemption_state", 1.0, 1, 0.7),
            make_row("preemption_state", 1.0, 2, 0.5),
        ]
        point, low, high = summary.paired_interval(
            reference,
            candidate,
            "realized_utility",
            "higher",
            samples=200,
            rng=np.random.default_rng(7),
        )
        self.assertAlmostEqual(point, 0.15)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)

    def test_validate_coverage_rejects_missing_threshold_variant(self) -> None:
        rows = [make_row("b4_offline", "baseline", 1, 0.5)]
        for model_type in (
            "preemption_control",
            "preemption_state",
            "preemption_state_shuffled",
        ):
            rows.append(make_row(model_type, 1.0, 1, 0.5))
        summary.validate_coverage(rows, [1.0])
        rows.pop()
        with self.assertRaises(ValueError):
            summary.validate_coverage(rows, [1.0])


if __name__ == "__main__":
    unittest.main()
