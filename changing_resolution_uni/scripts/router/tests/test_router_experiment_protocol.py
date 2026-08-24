from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from changing_resolution_uni.scripts.router import (
    bootstrap_confirmation_test,
    summarize_multiseed_selection,
)

PREDICTION_FIELDS = [
    "split",
    "Method",
    "method_role",
    "model_type",
    "prompt_id",
    "target_step",
    "chosen_step",
    "policy_regret",
    "realized_utility",
    "oracle_utility",
    "seed_oracle_utility",
    "regret_to_seed_oracle",
    "realized_vbench5",
    "realized_latency_sec",
    "native_latency_sec",
    "speedup_vs_native",
    "step_abs_error",
    "top1_correct",
    "top3_correct",
    "router_overhead_sec",
    "realized_subject_consistency",
    "realized_background_consistency",
    "realized_motion_smoothness",
    "realized_aesthetic_quality",
    "realized_imaging_quality",
]


def prediction(
    split: str,
    method: str,
    role: str,
    model_type: str,
    prompt_id: int,
    regret: float,
) -> dict[str, object]:
    return {
        "split": split,
        "Method": method,
        "method_role": role,
        "model_type": model_type,
        "prompt_id": prompt_id,
        "target_step": 48,
        "chosen_step": 48,
        "policy_regret": regret,
        "realized_utility": 0.8 - regret,
        "oracle_utility": 0.8,
        "seed_oracle_utility": 0.81,
        "regret_to_seed_oracle": 0.01 + regret,
        "realized_vbench5": 0.81 - regret,
        "realized_latency_sec": 30.0 + regret,
        "native_latency_sec": 180.0,
        "speedup_vs_native": 6.0,
        "step_abs_error": 0.0,
        "top1_correct": 1,
        "top3_correct": 1,
        "router_overhead_sec": 0.0001 if role == "learned" else 0.0,
        "realized_subject_consistency": 0.81 - regret,
        "realized_background_consistency": 0.81 - regret,
        "realized_motion_smoothness": 0.81 - regret,
        "realized_aesthetic_quality": 0.81 - regret,
        "realized_imaging_quality": 0.81 - regret,
    }


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class RouterExperimentProtocolTest(unittest.TestCase):
    def test_multiseed_selection_uses_validation_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (42, 100, 2024):
                run = root / f"seed_{seed}"
                run.mkdir()
                summary = {
                    "primary_lambda": 0.08,
                    "evaluation_stage": "selection",
                    "evaluation_split": "validation",
                    "test_accessed": False,
                    "meta": {
                        "train_seed": seed,
                        "split_seed": 7,
                        "quality_profile": "strict_vbench5_v1",
                        "latency_profile": "measured",
                        "candidate_steps": [47, 48, 50],
                    },
                }
                (run / "router_validation_summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )
                rows = []
                for prompt_id in (1, 2, 3):
                    rows.extend(
                        [
                            prediction(
                                "validation",
                                "Fixed Step 48 (Best Fixed)",
                                "best_fixed",
                                "",
                                prompt_id,
                                0.20,
                            ),
                            prediction(
                                "validation",
                                "Learned: Linear Probe (B1)",
                                "learned",
                                "linear_probe",
                                prompt_id,
                                0.10,
                            ),
                            prediction(
                                "validation",
                                "Learned: Soft Distillation MLP (B4)",
                                "learned",
                                "mlp_distill",
                                prompt_id,
                                0.05,
                            ),
                        ]
                    )
                write_predictions(run / "router_validation_predictions.csv", rows)

            argv = [
                "summarize",
                "--runs-root",
                str(root),
                "--bootstrap-samples",
                "100",
                "--reference-model",
                "linear_probe",
            ]
            with mock.patch.object(sys, "argv", argv):
                summarize_multiseed_selection.main()
            selection = json.loads(
                (root / "selection" / "architecture_selection.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(selection["selected_model_type"], "mlp_distill")
            self.assertFalse(selection["test_accessed"])
            self.assertEqual(selection["run_count"], 3)
            self.assertEqual(selection["reference_model"], "linear_probe")
            with (root / "selection" / "multiseed_reference_paired_deltas.csv").open(
                encoding="utf-8"
            ) as handle:
                direct = list(csv.DictReader(handle))
            regret = next(row for row in direct if row["metric"] == "policy_regret")
            self.assertEqual(regret["reference_model"], "linear_probe")
            self.assertEqual(regret["candidate_model"], "mlp_distill")
            self.assertAlmostEqual(float(regret["mean_delta"]), 0.05)

    def test_confirmation_bootstrap_requires_formal_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = {
                "evaluation_stage": "confirmation",
                "evaluation_split": "test",
                "test_accessed": True,
                "meta": {"formal_evidence": True, "measured_latency_only": True},
            }
            (root / "router_benchmark_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            (root / "test_access_guard.json").write_text(
                json.dumps({"completed_at_utc": "2026-01-01T00:00:00+00:00"}),
                encoding="utf-8",
            )
            rows = []
            for prompt_id in (1, 2, 3):
                rows.extend(
                    [
                        prediction(
                            "test",
                            "Fixed Step 48 (Best Fixed)",
                            "best_fixed",
                            "",
                            prompt_id,
                            0.20,
                        ),
                        prediction(
                            "test",
                            "Learned: Soft Distillation MLP (B4)",
                            "learned",
                            "mlp_distill",
                            prompt_id,
                            0.05,
                        ),
                    ]
                )
            write_predictions(root / "router_test_predictions.csv", rows)
            argv = [
                "bootstrap",
                "--run-dir",
                str(root),
                "--bootstrap-samples",
                "100",
            ]
            with mock.patch.object(sys, "argv", argv):
                bootstrap_confirmation_test.main()
            report = json.loads(
                (root / "confirmation_bootstrap_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(report["formal_evidence"])
            with (root / "confirmation_test_paired_deltas.csv").open(
                encoding="utf-8"
            ) as handle:
                paired = list(csv.DictReader(handle))
            regret = next(row for row in paired if row["metric"] == "policy_regret")
            self.assertAlmostEqual(float(regret["mean_delta"]), 0.15)


if __name__ == "__main__":
    unittest.main()
