from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from paper.aaai27.experiments.aggregate_human_review import merge, read_csv, summarize, validate_completed
from paper.aaai27.experiments.collect_results import inspect_human_review
from paper.aaai27.experiments.prepare_blind_review import make_blind_id
from paper.aaai27.experiments.run_vbench_factorials import collect_results, prepare_inputs


class VBenchFactorialTest(unittest.TestCase):
    def test_prepares_prompt_maps_and_collects_every_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "family": "distill4",
                "seed_base": 9800,
                "prompt_offset": 0,
                "prompts": ["prompt one", "prompt two"],
                "cases": [{"name": "step3_base_interp"}, {"name": "step3_lora_stage2"}],
            }
            (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            for case in manifest["cases"]:
                video_dir = root / "videos" / case["name"]
                video_dir.mkdir(parents=True)
                for index in range(2):
                    (video_dir / f"{case['name']}_{index:02d}_seed{9800 + index}.mp4").write_bytes(b"x" * 1024)
            maps = prepare_inputs(root, manifest)
            self.assertEqual(len(maps), 2)
            mapping = json.loads(maps["step3_base_interp"].read_text(encoding="utf-8"))
            self.assertEqual(list(mapping.values()), manifest["prompts"])
            for case in manifest["cases"]:
                raw = root / "metrics/vbench_raw" / case["name"]
                raw.mkdir(parents=True)
                (raw / "result.json").write_text(json.dumps({"motion_smoothness": [0.8, []]}), encoding="utf-8")
            output = collect_results(root, manifest, ["motion_smoothness"], None)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["cases"]), {"step3_base_interp", "step3_lora_stage2"})


class HumanReviewTest(unittest.TestCase):
    def test_multistep_blind_ids_do_not_collide_and_single_step_ids_stay_stable(self) -> None:
        step40 = make_blind_id(1, "wan50", 0, 10, "comparison", 40, multi_step=True)
        step45 = make_blind_id(1, "wan50", 0, 10, "comparison", 45, multi_step=True)
        distill_a = make_blind_id(1, "distill4", 0, 10, "comparison", 3, multi_step=False)
        distill_b = make_blind_id(1, "distill4", 0, 10, "comparison", 99, multi_step=False)
        self.assertNotEqual(step40, step45)
        self.assertEqual(distill_a, distill_b)

    def test_merges_three_raters_and_unblinds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "review").mkdir(parents=True)
            (root / "_private").mkdir()
            (root / "run_manifest.json").write_text(json.dumps({"family": "distill4"}), encoding="utf-8")
            fields = [
                "blind_id",
                "video_A",
                "video_B",
                "overall_winner_A_B_tie",
                "confidence_1_to_5",
                "severe_failure_A_B_neither",
                "notes",
            ]
            ballot = {field: "" for field in fields}
            ballot.update({"blind_id": "blind1", "video_A": "A.mp4", "video_B": "B.mp4"})
            self._write(root / "review/human_ratings.csv", fields, [ballot])
            self._write(
                root / "_private/human_review_key.csv",
                ["blind_id", "family", "comparison", "case_A", "case_B"],
                [{"blind_id": "blind1", "family": "distill4", "comparison": "talh", "case_A": "base", "case_B": "talh"}],
            )
            specs = []
            for index, winner in enumerate(("B", "B", "tie"), start=1):
                row = dict(ballot)
                row.update(
                    {
                        "overall_winner_A_B_tie": winner,
                        "confidence_1_to_5": "5",
                        "severe_failure_A_B_neither": "neither",
                    }
                )
                path = root / f"rater{index}.csv"
                self._write(path, fields, [row])
                specs.append(f"r{index}={path}")
            completed = merge(root, specs)
            rows, completed_fields = read_csv(completed)
            validate_completed(root, rows, completed_fields, 3)
            csv_path, json_path = summarize(root, rows)
            self.assertTrue(csv_path.is_file())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            preferences = {row["preferred_case"]: row["votes"] for row in payload["preferences"]}
            self.assertEqual(preferences, {"talh": 2, "tie": 1})
            self.assertEqual(inspect_human_review(root)["completed_status"], "complete")

    @staticmethod
    def _write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
