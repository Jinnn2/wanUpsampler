from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic
from UNIV_adaptor.scripts.validation.score_hr_refinement_ablation import (
    CASES, DIMENSIONS, comparison_rows, evaluate, load_inputs, stage_inputs,
)


def fixture(root):
    cases = []
    for name, steps in CASES.items():
        video = root / f"{name}.mp4"
        video.write_bytes((name.encode() + b"-video") * 256)
        cases.append({
            "id": name, "hr_steps": steps, "hr_seconds": float(steps),
            "video_path": f"/original/server/path/{name}.mp4",
            "video_sha256": sha256_file(video), "boundary_sha256": "a" * 64,
            "hr_schedule": {"sigmas": [.6 * (1 - i / steps) for i in range(steps)] + [0.]},
        })
    summary = {
        "schema": "univ_hr_refinement_ablation_results_v1", "complete": True,
        "prompt": "A fox walks through snow.", "seed": 42, "cases": cases,
    }
    write_json_atomic(root / "comparison_summary.json", summary)
    return summary


def scores():
    return {name: {dimension: .8 for dimension in DIMENSIONS} for name in CASES}


class HRRefinementEvaluationTest(unittest.TestCase):
    def test_relocated_outputs_and_extra_montage_are_handled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            (root / "montage.mp4").write_bytes(b"montage")
            summary, cases = load_inputs(root)
            metrics = root / "metrics/hr_refinement"
            inputs, mapping = stage_inputs(cases, summary["prompt"], metrics)
            self.assertEqual({path.stem for path in inputs.glob("*.mp4")}, set(CASES))
            prompt_map = json.loads(mapping.read_text(encoding="utf-8"))
            self.assertEqual(set(prompt_map.values()), {summary["prompt"]})
            self.assertEqual(len(prompt_map), 4)
            self.assertTrue(all(Path(path).parent == inputs for path in prompt_map))
            self.assertEqual(stage_inputs(cases, summary["prompt"], metrics), (inputs, mapping))

    def test_wrong_video_or_boundary_cannot_be_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = fixture(root)
            changed = copy.deepcopy(summary)
            changed["cases"][1]["boundary_sha256"] = "b" * 64
            write_json_atomic(root / "comparison_summary.json", changed)
            with self.assertRaisesRegex(ValueError, "share one HR"):
                load_inputs(root)
            write_json_atomic(root / "comparison_summary.json", summary)
            with (root / "HR02.mp4").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "video hash"):
                load_inputs(root)

    def test_incomplete_generation_and_nonpositive_timing_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = fixture(root)
            summary["complete"] = False
            write_json_atomic(root / "comparison_summary.json", summary)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_inputs(root)
            summary["complete"] = True
            summary["cases"][0]["hr_seconds"] = 0
            write_json_atomic(root / "comparison_summary.json", summary)
            with self.assertRaisesRegex(ValueError, "timing"):
                load_inputs(root)

    def test_signed_deltas_and_speedups_use_hr10(self):
        cases = [{"id": name, "hr_steps": steps, "hr_seconds": float(steps)} for name, steps in CASES.items()]
        value = scores()
        value["HR02"]["imaging_quality"] = .78
        value["HR02"]["dynamic_degree"] = 0
        rows = comparison_rows(cases, value)
        self.assertAlmostEqual(rows[-1]["delta_imaging_quality_pp"], -2)
        self.assertEqual(rows[-1]["hr_speedup_vs_hr10"], 5)
        self.assertEqual(rows[0]["delta_imaging_quality_pp"], 0)
        self.assertFalse(any("mean" in key or "overall" in key for key in rows[0]))
        del value["HR04"]["motion_smoothness"]
        with self.assertRaisesRegex(ValueError, "incomplete VBench"):
            comparison_rows(cases, value)

    def test_dispatches_one_batch_and_writes_per_video_reports(self):
        module = "changing_resolution_uni.scripts.data.batch_vbench_score_dataset"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            args = SimpleNamespace(out_dir=str(root), vbench_root=str(root / "VBench"),
                                   vbench_python="python", vbench_commit="", mode="run", force=False)
            with patch(module + ".inspect_vbench_checkout", return_value={"git_commit": "test-commit"}), \
                 patch(module + ".score_case_directory", return_value=SimpleNamespace(
                     scores=scores(), provenance={"request_sha256": "content-bound"})) as scorer:
                evaluate(args)
            scorer.assert_called_once()
            self.assertEqual(scorer.call_args.args[5], DIMENSIONS)
            self.assertEqual(scorer.call_args.args[8], 1)
            metrics = root / "metrics/hr_refinement"
            output = json.loads((metrics / "vbench_scores.json").read_text(encoding="utf-8"))
            self.assertEqual(len(output["rows"]), 4)
            self.assertEqual(output["diagnostic_dimensions"], ["dynamic_degree"])
            self.assertTrue((metrics / "comparison.csv").is_file())
            self.assertIn("not ground truth", (metrics / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
