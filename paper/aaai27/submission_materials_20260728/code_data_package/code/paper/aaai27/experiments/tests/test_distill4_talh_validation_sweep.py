from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from paper.aaai27.experiments.run_distill4_talh_validation_sweep import (
    assert_validation_prompts_are_disjoint,
    build_sweep,
    prepare,
    select_best,
)


class TALHValidationSweepTest(unittest.TestCase):
    def test_builds_eight_unique_points(self) -> None:
        points = build_sweep([0.25, 0.5, 0.75, 1.0], ["random", "resize_flow"])
        self.assertEqual(len(points), 8)
        self.assertEqual(len({point.name for point in points}), 8)

    def test_default_validation_prompts_are_disjoint(self) -> None:
        prompts = (
            Path(__file__).resolve().parents[1]
            / "distill4_talh_validation_prompts_8.txt"
        )
        assert_validation_prompts_are_disjoint(prompts)

    def test_prepare_writes_point_specific_strength_and_renoise(self) -> None:
        args = SimpleNamespace(
            num_frames=81,
            guide_scale=6.0,
            dit_ckpt="distill.pt",
            stage2_checkpoint="stage2.pt",
            stage2_train_config="stage2.yaml",
            stage2_use_ema=True,
            lora_checkpoint="lora.safetensors",
            prompt_offset=0,
            prompts="validation.txt",
            seed=16000,
            strengths=[0.25],
            renoise_modes=["resize_flow"],
            gpus=[0, 1, 2, 3],
        )
        point = build_sweep([0.25], ["resize_flow"])[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare(root, args, [point], ["validation prompt"])
            config = json.loads(
                (root / "configs" / f"{point.name}.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads(
                (root / "run_manifest.json").read_text(encoding="utf-8")
            )
        self.assertEqual(config["lora_configs"][0]["strength"], 0.25)
        self.assertEqual(config["wan_distill_bridge_renoise_mode"], "resize_flow")
        self.assertTrue(manifest["settings"]["prompt_sets_checked_disjoint"])

    def test_selection_uses_quality5_and_records_temporal_diagnostic(self) -> None:
        dimensions = [
            "subject_consistency",
            "background_consistency",
            "motion_smoothness",
            "aesthetic_quality",
            "imaging_quality",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics").mkdir(parents=True)
            cases = [
                {
                    "name": "low",
                    "lora_strength": 0.25,
                    "renoise_mode": "random",
                },
                {
                    "name": "high",
                    "lora_strength": 0.75,
                    "renoise_mode": "resize_flow",
                },
            ]
            (root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "selection_rule": "maximize",
                        "cases": cases,
                    }
                ),
                encoding="utf-8",
            )
            payload = {"cases": {}}
            for case, score in (("low", 0.8), ("high", 0.9)):
                numeric = {
                    f"result.{dimension}.0": score for dimension in dimensions
                }
                numeric["result.temporal_flickering.0"] = score - 0.1
                payload["cases"][case] = {"numeric_metrics": numeric}
            (root / "metrics/vbench_v1_custom.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            output = select_best(root)
            selection = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(selection["selected"]["case"], "high")
        self.assertAlmostEqual(
            selection["selected"]["temporal_flickering"], 0.8
        )


if __name__ == "__main__":
    unittest.main()
