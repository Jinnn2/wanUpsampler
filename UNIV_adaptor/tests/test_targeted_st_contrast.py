from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from UNIV_adaptor.scripts.data.run_targeted_st_contrast_generation import prepare
from UNIV_adaptor.scripts.data.run_targeted_st_temporal_calibration import (
    build_plan as build_calibration_plan,
    validate_protocol as validate_calibration_protocol,
)
from UNIV_adaptor.scripts.data.score_targeted_st_contrast import (
    DIMENSIONS,
    HIGH_MOTION,
    LOW_MOTION,
    SPATIAL,
    report,
)


ROOT = Path(__file__).resolve().parents[2]


class TargetedStContrastTest(unittest.TestCase):
    def test_prepare_plans_equal_density_48_video_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            manifest = prepare(
                SimpleNamespace(
                    prompts=str(ROOT / "prompts" / "univ_targeted_st_contrast_v1.txt"),
                    protocol=str(
                        ROOT
                        / "UNIV_adaptor"
                        / "configs"
                        / "univ_targeted_st_contrast_v1.json"
                    ),
                    template_config=str(
                        ROOT
                        / "UNIV_adaptor"
                        / "configs"
                        / "wan21_t2v_univ_rgb_720p.example.json"
                    ),
                    model_root=str(Path(directory) / "model"),
                    out_root=str(out),
                    job_chunk_size=4,
                )
            )
            plan = json.loads(
                (out / "targeted_st_plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(plan["counts"]["videos"], 48)
            self.assertEqual(plan["counts"]["prompt_seed_groups"], 24)
            self.assertEqual(len(manifest["cases"]), 2)
            self.assertEqual(len(manifest["jobs"]), 12)
            self.assertEqual(
                {
                    round(value["proxy_compute_density"], 12)
                    for value in plan["cases"].values()
                },
                {0.5},
            )

    def test_report_recovers_controlled_directional_contrast(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            rows = []
            scores = {}
            temporal_case = "TEMPORAL_ONLY_RT036_CAL"
            for prompt_index in range(8):
                prompt_group = LOW_MOTION if prompt_index < 4 else HIGH_MOTION
                expected = temporal_case if prompt_index < 4 else SPATIAL
                for base_seed in (42, 100, 2024):
                    group_id = f"g{prompt_index}_b{base_seed}"
                    for case_id in (SPATIAL, temporal_case):
                        stem = f"{group_id}__{case_id}"
                        rows.append(
                            {
                                "observation_id": stem,
                                "stem": stem,
                                "group_id": group_id,
                                "prompt_index": prompt_index,
                                "prompt_group": prompt_group,
                                "expected_preference": expected,
                                "prompt": f"prompt {prompt_index}",
                                "prompt_sha256": f"hash{prompt_index}",
                                "base_seed": base_seed,
                                "seed": base_seed + prompt_index,
                                "case_id": case_id,
                                "axis": "spatial" if case_id == SPATIAL else "temporal",
                                "proxy_compute_density": 0.5,
                                "pipeline_seconds": 100.0,
                                "transition": "dvg_latent_anchor",
                                "requested_action": {
                                    "spatial_ratio": 0.6,
                                    "temporal_ratio": 0.4,
                                    "lr_nfe_ratio": 1.0,
                                    "switch_ratio": 0.8,
                                },
                            }
                        )
                        preferred = case_id == expected
                        quality = 0.8 if preferred else 0.7
                        scores[stem] = {
                            dimension: quality if dimension in DIMENSIONS[:5] else 0.5
                            for dimension in DIMENSIONS
                        }
            analysis = report(
                rows,
                {"scores": scores, "payload_sha256": "fixture"},
                out,
                "fixture-input",
            )
            self.assertTrue(analysis["directional_hypothesis_pass"])
            self.assertTrue(analysis["measured_latency_matched_within_5pct"])
            self.assertAlmostEqual(analysis["difference_in_differences"], 0.2)
            self.assertTrue((out / "paired_st.csv").is_file())
            self.assertTrue((out / "report.md").is_file())

    def test_temporal_calibration_plan_reuses_24_and_generates_24(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = validate_calibration_protocol(
                json.loads(
                    (
                        ROOT
                        / "UNIV_adaptor"
                        / "configs"
                        / "univ_targeted_st_temporal_calibration_v2.json"
                    ).read_text(encoding="utf-8")
                )
            )
            rows = []
            for prompt_index in range(8):
                prompt_group = LOW_MOTION if prompt_index < 4 else HIGH_MOTION
                for base_seed in (42, 100, 2024):
                    video = root / f"s_{prompt_index}_{base_seed}.mp4"
                    video.write_bytes(b"spatial-fixture")
                    rows.append(
                        {
                            "group_id": f"g{prompt_index}_b{base_seed}",
                            "prompt_index": prompt_index,
                            "prompt_group": prompt_group,
                            "prompt": f"prompt {prompt_index}",
                            "prompt_sha256": f"hash{prompt_index}",
                            "base_seed": base_seed,
                            "seed": base_seed + prompt_index,
                            "case_id": SPATIAL,
                            "proxy_compute_density": 0.5,
                            "requested_action": {
                                "spatial_ratio": 0.6123724356957945,
                                "temporal_ratio": 1.0,
                                "lr_nfe_ratio": 1.0,
                                "switch_ratio": 0.8,
                            },
                            "resolved_schedule": {"fixture": True},
                            "transition": "dvg_latent_anchor",
                            "video_path": str(video),
                            "video_sha256": "f" * 64,
                            "pipeline_seconds": 200.0,
                        }
                    )
            plan = build_calibration_plan(
                protocol,
                {
                    "rows": rows,
                    "identity": {"dataset_sha256": "fixture"},
                    "prompt_indices": list(range(8)),
                    "base_seeds": [42, 100, 2024],
                },
            )
            self.assertEqual(plan["counts"]["generated_videos"], 24)
            self.assertEqual(plan["counts"]["reused_videos"], 24)
            self.assertEqual(plan["counts"]["videos"], 48)
            self.assertAlmostEqual(
                plan["temporal_case"]["proxy_compute_density"], 0.488
            )
            self.assertAlmostEqual(
                plan["temporal_case"]["resolved_schedule"]["actual_temporal_ratio"],
                0.35,
            )


if __name__ == "__main__":
    unittest.main()
