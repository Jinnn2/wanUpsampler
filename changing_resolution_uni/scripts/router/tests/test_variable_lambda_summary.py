from __future__ import annotations

import unittest

import numpy as np

from changing_resolution_uni.scripts.router import (
    summarize_variable_lambda_runs as summarize,
)


class VariableLambdaSummaryTest(unittest.TestCase):
    def test_secondary_b4_reference_uses_candidate_better_direction(self) -> None:
        rows = []
        for prompt_id in (1000, 1001):
            for model_type, regret, quality, latency, harmful in (
                ("b4_offline", 0.20, 0.80, 80.0, 1.0),
                ("b4_prompt_state", 0.10, 0.90, 70.0, 0.0),
            ):
                rows.append(
                    {
                        "run_id": "seed_42",
                        "prompt_id": prompt_id,
                        "seed": 42 + prompt_id,
                        "lambda": 0.08,
                        "model_type": model_type,
                        "policy_regret": regret,
                        "realized_utility": 1.0 - regret,
                        "realized_vbench5": quality,
                        "realized_latency_sec": latency,
                        "speedup_vs_native": 200.0 / latency,
                        "harmful_stop": harmful,
                    }
                )
        paired = summarize.paired_rows_against_reference(
            rows=rows,
            model_types=["b4_offline", "b4_prompt_state"],
            lambdas=[0.08],
            run_ids=["seed_42"],
            reference_model="b4_offline",
            bootstrap_samples=100,
            rng=np.random.default_rng(2027),
        )
        macro_regret = next(
            row
            for row in paired
            if row["candidate_model"] == "b4_prompt_state"
            and row["lambda"] == "macro"
            and row["metric"] == "policy_regret"
        )
        self.assertEqual(macro_regret["positive_means"], "candidate_better")
        self.assertAlmostEqual(macro_regret["mean_delta"], 0.1)
        self.assertGreater(macro_regret["ci95_low"], 0.0)


if __name__ == "__main__":
    unittest.main()
