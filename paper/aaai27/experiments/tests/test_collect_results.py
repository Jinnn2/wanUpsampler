from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from paper.aaai27.experiments.collect_results import (
    inspect_factorial,
    load_csv_source,
    summarize_paired_metrics,
    summarize_timing,
)


class FactorialInspectionTest(unittest.TestCase):
    def test_exact_manifest_videos_and_config_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "lora.safetensors"
            stage2 = root / "stage2.pt"
            train_config = root / "train.yaml"
            for path in (checkpoint, stage2, train_config):
                path.write_bytes(b"artifact")
            manifest = {
                "family": "distill4",
                "seed_base": 9800,
                "prompt_offset": 0,
                "prompts": ["one", "two"],
                "cases": [{"name": "step3_lora_stage2", "step": 3, "handoff": "lora", "resizer": "stage2"}],
            }
            (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            config_dir = root / "configs"
            config_dir.mkdir()
            config = {
                "compare_name": "step3_lora_stage2",
                "changing_resolution_steps": [3],
                "lora_active_steps": [3],
                "lora_configs": [{"path": str(checkpoint), "strength": 0.75}],
                "wan_clean_resizer_ckpt": str(stage2),
                "wan_clean_resizer_train_config": str(train_config),
            }
            (config_dir / "step3_lora_stage2.json").write_text(json.dumps(config), encoding="utf-8")
            video_dir = root / "videos/step3_lora_stage2"
            video_dir.mkdir(parents=True)
            for index in range(2):
                (video_dir / f"step3_lora_stage2_{index:02d}_seed{9800 + index}.mp4").write_bytes(b"video")

            result = inspect_factorial(root, expected_family="distill4")

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["valid_total"], 2)
            self.assertEqual(result["expected_total"], 2)

    def test_unexpected_video_invalidates_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "family": "wan50",
                        "seed_base": 10,
                        "prompt_offset": 0,
                        "prompts": ["one"],
                        "cases": [{"name": "step40_base_interp", "step": 40, "handoff": "base", "resizer": "interp"}],
                    }
                ),
                encoding="utf-8",
            )
            config_dir = root / "configs"
            config_dir.mkdir()
            (config_dir / "step40_base_interp.json").write_text(
                json.dumps({"compare_name": "step40_base_interp", "changing_resolution_steps": [40]}),
                encoding="utf-8",
            )
            video_dir = root / "videos/step40_base_interp"
            video_dir.mkdir(parents=True)
            (video_dir / "step40_base_interp_00_seed10.mp4").write_bytes(b"valid")
            (video_dir / "old_seed_video.mp4").write_bytes(b"stale")

            result = inspect_factorial(root, expected_family="wan50")

            self.assertEqual(result["status"], "invalid")
            self.assertEqual(result["cases"]["step40_base_interp"]["extra"], ["old_seed_video.mp4"])


class TimingSummaryTest(unittest.TestCase):
    def test_aggregates_and_computes_speedup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "time_summary.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case", "repeat", "seed", "elapsed_sec"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"case": "direct_720p", "repeat": 0, "seed": 1, "elapsed_sec": 20},
                        {"case": "direct_720p", "repeat": 1, "seed": 2, "elapsed_sec": 22},
                        {"case": "stage3_bridge_720p", "repeat": 0, "seed": 1, "elapsed_sec": 10},
                        {"case": "stage3_bridge_720p", "repeat": 1, "seed": 2, "elapsed_sec": 11},
                    ]
                )

            summary = summarize_timing(load_csv_source(path))
            rows = {row["case"]: row for row in summary["rows"]}

            self.assertEqual(summary["status"], "complete")
            self.assertAlmostEqual(rows["direct_720p"]["mean_sec"], 21.0)
            self.assertAlmostEqual(rows["stage3_bridge_720p"]["speedup_vs_direct"], 2.0)


class PairedStatisticsTest(unittest.TestCase):
    def test_reports_oriented_improvement_and_exact_sign_test(self) -> None:
        source = {
            "path": "fixture.csv",
            "status": "complete",
            "columns": ["original_l1", "lora_l1"],
            "rows": [
                {"original_l1": "0.4", "lora_l1": "0.3"},
                {"original_l1": "0.5", "lora_l1": "0.4"},
                {"original_l1": "0.6", "lora_l1": "0.5"},
            ],
        }

        summary = summarize_paired_metrics(source, bootstrap_samples=100, seed=1)
        row = summary["rows"][0]

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(row["metric"], "l1")
        self.assertAlmostEqual(row["oriented_improvement_mean"], 0.1)
        self.assertEqual((row["wins"], row["losses"], row["ties"]), (3, 0, 0))
        self.assertAlmostEqual(row["two_sided_sign_test_p"], 0.25)


if __name__ == "__main__":
    unittest.main()
