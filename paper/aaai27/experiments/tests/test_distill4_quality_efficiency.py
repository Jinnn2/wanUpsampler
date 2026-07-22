from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from changing_resolution_distill.rgb_super_resolution import BicubicRGBSuperResolver
from paper.aaai27.experiments.benchmark_warm_quality_efficiency import (
    DISTILL4_BATCH_RUNNER,
    batch_runner_for_manifest,
    display_name,
)
from paper.aaai27.experiments.run_distill4_quality_efficiency import (
    analysis_pairs,
    assign_cases_to_gpus,
    build_cases,
    inference_environment,
    write_config,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_4GPU_LAUNCHER = (
    REPO_ROOT
    / "changing_resolution_distill/scripts/eval/run_distill4_final_18case_4gpu.sh"
)


def suite_args(**overrides):
    values = {
        "case_groups": ["native", "handoff", "endpoint"],
        "endpoint_refinement_steps": [0, 1, 2, 4],
        "endpoint_resizers": ["stage2", "interp", "rgb"],
        "num_frames": 81,
        "guide_scale": 6.0,
        "dit_ckpt": "distill_model.pt",
        "renoise_mode": "random",
        "stage2_checkpoint": "stage2.pt",
        "stage2_train_config": "stage2.yaml",
        "stage2_use_ema": True,
        "rgb_sr_backend": "realesrgan",
        "rgb_sr_tile": 0,
        "rgb_sr_tile_pad": 10,
        "rgb_sr_pre_pad": 0,
        "rgb_sr_fp32": False,
        "realesrgan_x2_checkpoint": "RealESRGAN_x2plus.pth",
        "lora_checkpoint": "lora3.safetensors",
        "lora_strength": 0.75,
        "gpus": [0, 1, 2, 3],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class Distill4SuiteTest(unittest.TestCase):
    def test_main_suite_excludes_interp1_and_crosses_endpoint_budget_with_domain(
        self,
    ) -> None:
        cases = build_cases(suite_args())
        by_name = {case.name: case for case in cases}
        self.assertEqual(len(cases), 18)
        self.assertNotIn("interp1", by_name)
        self.assertEqual(by_name["interp2"].total_evaluations, 4)
        self.assertEqual(by_name["talh3"].lr_evaluations, 3)
        for resizer in ("stage2", "interp", "rgb"):
            for refinements in (0, 1, 2, 4):
                case = by_name[f"endpoint_{resizer}_{refinements}hr"]
                self.assertEqual(case.lr_evaluations, 4)
                self.assertEqual(case.hr_evaluations, refinements)
                self.assertEqual(case.total_evaluations, 4 + refinements)

    def test_endpoint_configs_preserve_distilled_suffix_and_lifting_domain(
        self,
    ) -> None:
        args = suite_args()
        cases = {case.name: case for case in build_cases(args)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb_path = root / "rgb.json"
            stage2_path = root / "stage2.json"
            talh_path = root / "talh.json"
            write_config(rgb_path, args, cases["endpoint_rgb_1hr"])
            write_config(stage2_path, args, cases["endpoint_stage2_4hr"])
            write_config(talh_path, args, cases["talh3"])
            rgb = json.loads(rgb_path.read_text(encoding="utf-8"))
            stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
            talh = json.loads(talh_path.read_text(encoding="utf-8"))

        self.assertEqual(rgb["denoising_step_list"], [1000, 750, 500, 250])
        self.assertEqual(rgb["changing_resolution_steps"], [4])
        self.assertEqual(rgb["wan_final_refine_steps"], 1)
        self.assertEqual(rgb["wan_rgb_sr_backend"], "realesrgan")
        self.assertNotIn("wan_clean_resizer_ckpt", rgb)
        self.assertEqual(stage2["wan_final_refine_steps"], 4)
        self.assertIn("wan_clean_resizer_ckpt", stage2)
        self.assertEqual(talh["changing_resolution_steps"], [3])
        self.assertEqual(talh["lora_active_steps"], [3])

    def test_pairs_include_early_vs_endpoint_and_rgb_domain(self) -> None:
        pairs = {row["comparison"] for row in analysis_pairs(build_cases(suite_args()))}
        self.assertIn("talh3_vs_endpoint_stage2_1hr", pairs)
        self.assertIn("talh3_vs_endpoint_rgb_1hr", pairs)
        self.assertIn("endpoint_interp_1hr_vs_endpoint_stage2_1hr", pairs)
        self.assertIn("endpoint_stage2_1hr_vs_endpoint_rgb_1hr", pairs)

    def test_four_gpu_schedule_assigns_every_case_once(self) -> None:
        cases = build_cases(suite_args())
        assignments = assign_cases_to_gpus(cases, [0, 1, 2, 3])
        self.assertEqual([assignment.gpu for assignment in assignments], [0, 1, 2, 3])
        scheduled = [
            case.name for assignment in assignments for case in assignment.cases
        ]
        self.assertCountEqual(scheduled, [case.name for case in cases])
        self.assertEqual(len(scheduled), len(set(scheduled)))
        loads = [assignment.estimated_cost for assignment in assignments]
        self.assertLessEqual(max(loads) - min(loads), 4.0)

    def test_gpu_worker_exposes_only_its_physical_gpu(self) -> None:
        environment = inference_environment(3)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "3")

    def test_final_launcher_pins_complete_18_case_four_gpu_suite(self) -> None:
        launcher = FINAL_4GPU_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("Exactly four comma-separated GPU ids", launcher)
        self.assertIn("--case-groups native handoff endpoint", launcher)
        self.assertIn("--endpoint-refinement-steps 0 1 2 4", launcher)
        self.assertIn("--endpoint-resizers stage2 interp rgb", launcher)
        self.assertIn('--gpus "${GPUS[@]}"', launcher)
        self.assertIn("expected=$((18 * LIMIT))", launcher)


class RGBSuperResolutionTest(unittest.TestCase):
    def test_bicubic_smoke_backend_uses_x2_then_exact_center_crop(self) -> None:
        video = torch.rand(2, 8, 10, 3)
        output = BicubicRGBSuperResolver(scale=2.0).resize(
            video, target_height=14, target_width=18
        )
        self.assertEqual(tuple(output.shape), (2, 14, 18, 3))
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_warm_benchmark_routes_distill_manifest_to_distill_batch_runner(
        self,
    ) -> None:
        runner = batch_runner_for_manifest({"family": "distill4_quality_efficiency"})
        self.assertEqual(runner, DISTILL4_BATCH_RUNNER)
        self.assertEqual(display_name({"name": "endpoint_rgb_1hr"}), "Endpoint-RGB-1HR")


if __name__ == "__main__":
    unittest.main()
