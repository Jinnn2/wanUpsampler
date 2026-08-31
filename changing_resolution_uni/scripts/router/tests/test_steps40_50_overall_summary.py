from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from changing_resolution_uni.scripts.router import (
    summarize_steps40_50_overall as overall,
)
from changing_resolution_uni.scripts.router import (
    summarize_variable_lambda_runs as summary,
)


def prediction(
    *, model_type: str = "b4_offline", chosen_step: int = 45, regret: float = 0.01
) -> dict[str, object]:
    return {
        "run_id": "seed_42",
        "prompt_id": 1000,
        "seed": 1042,
        "lambda": 0.08,
        "model_type": model_type,
        "chosen_step": str(chosen_step),
        "oracle_step": "46",
        "best_fixed_step": "45",
        "policy_regret": regret,
        "best_fixed_regret": 0.01,
        "realized_utility": 0.79,
        "oracle_utility": 0.80,
        "realized_vbench5": 0.80,
        "realized_latency_sec": 70.0,
        "speedup_vs_native": 4.5,
        "harmful_stop": 1.0,
    }


class Steps4050OverallSummaryTest(unittest.TestCase):
    def test_identical_b4_predictions_are_accepted(self) -> None:
        left = [prediction()]
        right = [prediction()]
        overall.validate_frozen_b4_equivalence(left, right)

    def test_b4_choice_difference_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "discrete prediction differs"):
            overall.validate_frozen_b4_equivalence(
                [prediction()], [prediction(chosen_step=46)]
            )

    def test_b4_metric_difference_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "metric differs"):
            overall.validate_frozen_b4_equivalence(
                [prediction()], [prediction(regret=0.02)]
            )

    def test_coverage_requires_every_model_on_same_rows(self) -> None:
        rows = [
            prediction(model_type="b4_offline"),
            prediction(model_type="soft_margin_state"),
        ]
        overall.validate_coverage(rows, ["b4_offline", "soft_margin_state"])
        rows.append(
            {
                **prediction(model_type="soft_margin_state"),
                "prompt_id": 1001,
            }
        )
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            overall.validate_coverage(rows, ["b4_offline", "soft_margin_state"])

    def write_suite(self, root: Path, model_types: list[str], *, soft: bool) -> None:
        for train_seed in (42, 100, 2024):
            run = root / f"seed_{train_seed}"
            run.mkdir(parents=True)
            histories = {}
            for model_type in model_types:
                history_path = run / f"{model_type}_history.csv"
                history_path.write_text("epoch\n0\n", encoding="utf-8")
                histories[model_type] = {
                    "path": history_path.name,
                    "sha256": summary.sha256_file(history_path),
                }
            prediction_path = run / "predictions.csv"
            rows = []
            for model_index, model_type in enumerate(model_types):
                row = prediction(
                    model_type=model_type, regret=0.01 + model_index * 0.001
                )
                row.update(
                    {
                        "split": "validation",
                        "Method": model_type,
                        "normalized_cost": 0.4,
                        "raw_manifest_latency_sec_diagnostic": 70.0,
                        "raw_manifest_speedup_diagnostic": 4.5,
                    }
                )
                rows.append(row)
            with prediction_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            training = {
                "epochs": 30,
                "batch_size_trajectories" if soft else "batch_size": 64,
                "lr": 0.0003,
                "weight_decay": 0.0001,
                "dropout": 0.1,
                "b4_temperature": 0.02,
                "b4_emd_weight": 0.5,
            }
            run_summary = {
                "evaluation_protocol": summary.EVALUATION_PROTOCOL,
                "evaluation_split": "validation",
                "test_accessed": False,
                "train_seed": train_seed,
                "model_types": model_types,
                "train_lambdas": [0.08],
                "eval_lambdas": [0.08],
                "primary_lambda": 0.08,
                "harm_epsilon": 0.001,
                "decision_parameter": "risk_margin" if soft else "risk_threshold",
                "risk_margin": 0.0,
                "risk_threshold": 0.5,
                "feature_groups": ["trajectory_delta"],
                "selected_feature_count": 36,
                "dataset_manifest_sha256": "a" * 64,
                "source_candidate_steps": [30, 35, *range(40, 51)],
                "candidate_steps": list(range(40, 51)),
                "training": training,
                "cost_profile": [0.5] * 11,
                "train_prompts": 1000,
                "validation_prompts": 200,
                "latency_profile": {"sha256": "b" * 64},
                "artifacts": {
                    "predictions": prediction_path.name,
                    "checkpoints": {},
                    "training_histories": histories,
                },
            }
            (run / "run_summary.json").write_text(
                json.dumps(run_summary), encoding="utf-8"
            )

    def test_end_to_end_overall_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            residual = root / "residual"
            soft = root / "soft"
            out = root / "overall"
            self.write_suite(
                residual,
                ["b4_offline", "b4_residual_prompt", "b4_residual_state"],
                soft=False,
            )
            self.write_suite(
                soft,
                ["b4_offline", "soft_margin_control", "soft_margin_state"],
                soft=True,
            )
            argv = [
                "summarize",
                "--residual-runs-root",
                str(residual),
                "--soft-runs-root",
                str(soft),
                "--out-dir",
                str(out),
                "--bootstrap-samples",
                "20",
            ]
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    overall.main()
            report = json.loads(
                (out / "overall_selection.json").read_text(encoding="utf-8")
            )
            self.assertTrue(report["b4_predictions_identical_across_suites"])
            self.assertEqual(len(report["model_types"]), 5)
            self.assertFalse(report["test_accessed"])


if __name__ == "__main__":
    unittest.main()
