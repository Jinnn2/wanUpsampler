from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper.aaai27.experiments.collect_quality_efficiency import (
    EXPECTED_CASES,
    collect_quality_efficiency,
    inventory_videos,
    inspect_summaries,
)


class QualityEfficiencyCollectionTest(unittest.TestCase):
    def test_complete_suite_is_collected_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "quality_efficiency_final_v2"
            output = root / "collection"
            prompts = [f"prompt {index}" for index in range(10)]
            cases = [{"name": name} for name in EXPECTED_CASES]
            cases[-1].update(
                {
                    "lr_evaluations": 5,
                    "mixed_evaluations": 6,
                    "hr_evaluations": 7,
                    "total_evaluations": 18,
                }
            )
            suite.mkdir()
            (suite / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "family": "wan50_quality_efficiency",
                        "seed_base": 9700,
                        "prompt_offset": 0,
                        "prompts": prompts,
                        "cases": cases,
                    }
                ),
                encoding="utf-8",
            )
            for case in EXPECTED_CASES:
                case_root = suite / "videos" / case
                case_root.mkdir(parents=True)
                for index in range(10):
                    (case_root / f"{case}_{index:02d}_seed{9700 + index}.mp4").write_bytes(
                        b"v" * 2048
                    )

            with patch(
                "paper.aaai27.experiments.collect_quality_efficiency.inspect_factorial",
                return_value={"status": "complete", "issues": []},
            ):
                result = collect_quality_efficiency(
                    suite_root=suite,
                    output_root=output,
                    project_root=root,
                    include_videos=False,
                    probe_videos=False,
                    require_metrics=False,
                    require_timing=False,
                    allow_incomplete=False,
                )

            validation = json.loads((result / "validation.json").read_text(encoding="utf-8"))
            self.assertEqual(validation["status"], "complete")
            self.assertEqual(validation["valid_videos"], 110)
            self.assertTrue((result / "suite/run_manifest.json").is_file())
            self.assertTrue((result / "video_inventory.csv").is_file())
            self.assertTrue((result / "collection_manifest.json").is_file())
            self.assertTrue((result / "SHA256SUMS").is_file())
            self.assertFalse((result / "suite/videos").exists())

    def test_video_inventory_rejects_partial_and_ignores_named_legacy_ralu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "seed_base": 9700,
                "prompt_offset": 0,
                "prompts": ["one", "two"],
                "cases": [{"name": "ralu_quality"}],
            }
            video_root = root / "videos/ralu_quality"
            video_root.mkdir(parents=True)
            (video_root / "ralu_quality_00_seed9700.mp4").write_bytes(b"v" * 2048)
            (video_root / "ralu_quality_01_seed9701.mp4").write_bytes(b"partial")
            (root / "videos/ralu_nt45").mkdir()

            rows, issues, obsolete = inventory_videos(root, manifest, probe_videos=False)
            by_name = {row["filename"]: row for row in rows}

            self.assertEqual(by_name["ralu_quality_00_seed9700.mp4"]["status"], "valid")
            self.assertEqual(by_name["ralu_quality_01_seed9701.mp4"]["status"], "undersized")
            self.assertTrue(any("undersized" in issue for issue in issues))
            self.assertEqual(obsolete, [str(root / "videos/ralu_nt45")])

    def test_summary_case_coverage_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "quality_efficiency.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case", "elapsed_mean_s"])
                writer.writeheader()
                writer.writerow({"case": "a", "elapsed_mean_s": "1"})

            inventory, issues = inspect_summaries(
                root,
                expected_cases={"a", "b"},
                require_metrics=False,
                require_timing=True,
            )

            self.assertEqual(inventory["quality_efficiency.csv"]["status"], "case_mismatch")
            self.assertFalse(any("quality_efficiency.csv case coverage mismatch" in issue for issue in issues))
            self.assertTrue(any("quality_efficiency_warm.csv" in issue for issue in issues))

    def test_standard_nested_warm_output_is_recognized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "warm_quality_efficiency/quality_efficiency_warm.csv"
            path.parent.mkdir(parents=True)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case", "elapsed_mean_s"])
                writer.writeheader()
                writer.writerow({"case": "a", "elapsed_mean_s": "1"})

            inventory, issues = inspect_summaries(
                root,
                expected_cases={"a"},
                require_metrics=False,
                require_timing=True,
            )

            warm = inventory["quality_efficiency_warm.csv"]
            self.assertEqual(warm["status"], "present")
            self.assertEqual(Path(warm["path"]), path)
            self.assertFalse(any("quality_efficiency_warm.csv" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
