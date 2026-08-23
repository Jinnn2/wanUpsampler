from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from changing_resolution_uni.scripts.router import summarize_lambda_router_runs


class SummarizeLambdaRouterRunsTest(unittest.TestCase):
    def test_combines_runs_and_selects_lowest_regret_learned_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for lambda_value, regrets in ((0.01, (0.2, 0.1)), (0.02, (0.05, 0.08))):
                run_dir = root / f"lambda_{str(lambda_value).replace('.', '')}"
                run_dir.mkdir()
                payload = {
                    "primary_lambda": lambda_value,
                    "results": [
                        {
                            "Method": "Learned: Linear Probe (B1)",
                            "policy_regret": regrets[0],
                        },
                        {
                            "Method": "Learned: Soft Distillation MLP (B4)",
                            "policy_regret": regrets[1],
                        },
                    ],
                }
                (run_dir / "router_benchmark_summary.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            with mock.patch.object(
                sys, "argv", ["summarize", "--runs-root", str(root)]
            ):
                summarize_lambda_router_runs.main()

            with (root / "lambda_best_learned_models.csv").open(
                encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["Method"], "Learned: Soft Distillation MLP (B4)")
            self.assertEqual(rows[1]["Method"], "Learned: Linear Probe (B1)")


if __name__ == "__main__":
    unittest.main()
