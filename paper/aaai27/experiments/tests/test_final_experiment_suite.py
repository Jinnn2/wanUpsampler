from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from paper.aaai27.experiments.aggregate_human_review import summarize_prompt_majorities
from paper.aaai27.experiments.compile_vbench_paired_statistics import per_video_values
from paper.aaai27.experiments.collect_results import inspect_factorial_config
from paper.aaai27.experiments.prepare_blind_review import comparison_pairs, review_paths
from paper.aaai27.experiments.run_final_quality_efficiency import build_cases as build_efficiency_cases
from paper.aaai27.experiments.run_final_quality_efficiency import write_config as write_efficiency_config
from paper.aaai27.experiments.run_step40_strength_factorial import build_cases as build_strength_cases


class Step40StrengthSuiteTest(unittest.TestCase):
    def test_names_every_strength_and_exposes_custom_review_pairs(self) -> None:
        cases = build_strength_cases(40, [0.5, 0.75, 1.0])
        self.assertEqual(len(cases), 8)
        self.assertIn("step40_lora_s0p75_stage2", {case.name for case in cases})
        manifest = {
            "review_pairs": [
                {
                    "comparison": "lora_s0p75_with_stage2",
                    "step": 40,
                    "left_case": "step40_base_stage2",
                    "right_case": "step40_lora_s0p75_stage2",
                }
            ]
        }
        self.assertEqual(
            comparison_pairs(40, manifest),
            [("lora_s0p75_with_stage2", "step40_base_stage2", "step40_lora_s0p75_stage2")],
        )


class HumanPromptStatisticsTest(unittest.TestCase):
    def test_prompt_majority_sign_test_and_fleiss_kappa(self) -> None:
        prompt_votes = {
            ("lora", "temporal_stability", "0"): Counter({"base": 3}),
            ("lora", "temporal_stability", "1"): Counter({"lora": 2, "base": 1}),
            ("lora", "temporal_stability", "2"): Counter({"tie": 2, "lora": 1}),
        }
        rows, agreement = summarize_prompt_majorities(prompt_votes, {"lora": {"base", "lora"}})
        by_case = {row["preferred_case"]: row for row in rows}
        self.assertEqual(by_case["base"]["prompt_majorities"], 1)
        self.assertEqual(by_case["lora"]["prompt_majorities"], 1)
        self.assertEqual(by_case["tie"]["prompt_majorities"], 1)
        self.assertEqual(by_case["base"]["two_sided_sign_test_p"], 1.0)
        self.assertEqual(agreement[0]["ratings_per_item"], 3)
        self.assertIsInstance(agreement[0]["fleiss_kappa"], float)

    def test_named_review_paths_do_not_overwrite_default_package(self) -> None:
        root = Path("root")
        default = review_paths(root, "default")
        step45 = review_paths(root, "step45")
        self.assertNotEqual(default, step45)
        self.assertEqual(step45[0], root / "review/step45")


class FinalQualityEfficiencySuiteTest(unittest.TestCase):
    def test_protocol_uses_canonical_50_lr_steps_plus_one_extra_hr(self) -> None:
        args = SimpleNamespace(step40_strength=0.75, step45_strength=0.75)
        cases = {case.name: case for case in build_efficiency_cases(args)}
        self.assertEqual(cases["full_hr50"].total_evaluations, 50)
        self.assertEqual(cases["talh40"].hr_evaluations, 10)
        self.assertEqual(cases["talh45"].hr_evaluations, 5)
        self.assertEqual(cases["full_lr50_stage2_1hr"].lr_evaluations, 50)
        self.assertEqual(cases["full_lr50_stage2_1hr"].total_evaluations, 51)

    def test_writes_full_lr_refinement_and_full_hr_configs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                step40_strength=1.0,
                step45_strength=0.75,
                num_frames=81,
                guide_scale=6.0,
                stage2_checkpoint=str(root / "stage2.pt"),
                stage2_train_config=str(root / "stage2.yaml"),
                stage2_use_ema=False,
                final_refine_shift_increment=1.0,
                lora40_checkpoint=str(root / "lora40.safetensors"),
                lora45_checkpoint=str(root / "lora45.safetensors"),
            )
            for artifact in (
                args.stage2_checkpoint,
                args.stage2_train_config,
                args.lora40_checkpoint,
                args.lora45_checkpoint,
            ):
                Path(artifact).touch()
            cases = {case.name: case for case in build_efficiency_cases(args)}
            full_hr_path = root / "full_hr.json"
            full_lr_path = root / "full_lr.json"
            write_efficiency_config(full_hr_path, args, cases["full_hr50"])
            write_efficiency_config(full_lr_path, args, cases["full_lr50_stage2_1hr"])
            full_hr = json.loads(full_hr_path.read_text(encoding="utf-8"))
            full_lr = json.loads(full_lr_path.read_text(encoding="utf-8"))
            self.assertNotIn("changing_resolution", full_hr)
            self.assertEqual(full_lr["changing_resolution_steps"], [50])
            self.assertEqual(full_lr["wan_final_refine_shift_increment"], 1.0)
            full_hr_issues, _ = inspect_factorial_config(full_hr_path, asdict(cases["full_hr50"]))
            full_lr_issues, _ = inspect_factorial_config(
                full_lr_path, asdict(cases["full_lr50_stage2_1hr"])
            )
            self.assertEqual(full_hr_issues, [])
            self.assertEqual(full_lr_issues, [])


class VBenchPairedStatisticsTest(unittest.TestCase):
    def test_extracts_per_video_values_and_normalizes_imaging_quality(self) -> None:
        payload = {
            "cases": {
                "case": {
                    "numeric_metrics": {
                        "result.imaging_quality.1.0.video_results": 75.0,
                        "result.imaging_quality.1.1.video_results": 50.0,
                    }
                }
            }
        }
        self.assertEqual(per_video_values(payload, "case", "imaging_quality"), {0: 0.75, 1: 0.5})


if __name__ == "__main__":
    unittest.main()
