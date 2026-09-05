from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic
from UNIV_adaptor.scripts.validation.run_mrflow_refinement_ablation import (
    CONTROL_ID,
    DEFAULT_HR_STEPS,
    DEFAULT_SIGMAS,
    build_cases,
)
from UNIV_adaptor.scripts.validation.score_hr_refinement_ablation import DIMENSIONS
from UNIV_adaptor.scripts.validation.score_mrflow_refinement_ablation import (
    comparison_rows,
    evaluate,
    load_inputs,
)


def fixture(root: Path):
    cases = []
    clean_lr = "a" * 64
    clean_hr = "b" * 64
    noise = "c" * 64
    starts = {0.0: clean_hr, .12: "d" * 64, .2: "e" * 64, .3: "f" * 64}
    for planned in build_cases(DEFAULT_SIGMAS, DEFAULT_HR_STEPS, root):
        case_id = planned["id"]
        video = root / f"{case_id}.mp4"
        video.write_bytes((case_id.encode("ascii") + b"x" * 2048))
        steps = planned["hr_steps"]
        cases.append({
            "id": case_id,
            "refine_sigma": planned["refine_sigma"],
            "hr_steps": steps,
            "total_nfe": 50 + steps,
            "video_path": str(video),
            "video_sha256": sha256_file(video),
            "clean_lr_sha256": clean_lr,
            "clean_hr_sha256": clean_hr,
            "hr_noise_sha256": noise,
            "branch_start_sha256": starts[planned["refine_sigma"]],
            "hr_schedule": {
                "grid_policy": "transition_only" if steps == 0 else "direct_sigma_linear",
                "start_sigma": planned["refine_sigma"],
                "hr_steps": steps,
                "sigmas": planned["planned_sigmas"],
                "model_timesteps": planned["planned_model_timesteps"],
                "compute_indices": list(range(steps)),
            },
            "hr_seconds": float(steps),
            "candidate_denoise_seconds": 60.0 + steps,
            "pipeline_seconds_this_branch": 1.0,
        })
    summary = {
        "schema": "univ_mrflow_refinement_results_v1",
        "complete": True,
        "prompt": "fox",
        "seed": 42,
        "sigmas": list(DEFAULT_SIGMAS),
        "hr_steps": list(DEFAULT_HR_STEPS),
        "run_order": [row["id"] for row in cases],
        "cases": cases,
    }
    write_json_atomic(root / "comparison_summary.json", summary)
    return summary


def scores(case_ids):
    return {
        case_id: {dimension: .8 + index / 1000 for index, dimension in enumerate(DIMENSIONS)}
        for case_id in case_ids
    }


class MrFlowRefinementEvaluationTest(unittest.TestCase):
    def test_loads_full_matrix_and_shared_start_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            summary, cases = load_inputs(root)
            self.assertEqual(len(cases), 10)
            self.assertEqual(cases[0]["id"], CONTROL_ID)
            self.assertEqual(cases[0]["hr_seconds"], 0)
            for sigma in summary["sigmas"]:
                self.assertEqual(len({
                    row["branch_start_sha256"]
                    for row in cases if row["refine_sigma"] == sigma
                }), 1)

    def test_rejects_changed_video_grid_and_sigma_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = fixture(root)
            changed = copy.deepcopy(original)
            changed["cases"][1]["hr_schedule"]["sigmas"][0] += .01
            write_json_atomic(root / "comparison_summary.json", changed)
            with self.assertRaisesRegex(ValueError, "direct-sigma"):
                load_inputs(root)
            write_json_atomic(root / "comparison_summary.json", original)
            with (root / f"{original['cases'][1]['id']}.mp4").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "video hash"):
                load_inputs(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            changed = fixture(root)
            changed["cases"][2]["branch_start_sha256"] = "9" * 64
            write_json_atomic(root / "comparison_summary.json", changed)
            with self.assertRaisesRegex(ValueError, "starting tensor"):
                load_inputs(root)

    def test_rows_use_transition_only_quality_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            _, cases = load_inputs(root)
            values = scores([row["id"] for row in cases])
            values[cases[1]["id"]]["imaging_quality"] = .85
            rows = comparison_rows(cases, values)
            baseline = next(row for row in rows if row["case"] == CONTROL_ID)
            changed = next(row for row in rows if row["case"] == cases[1]["id"])
            self.assertEqual(baseline["delta_imaging_quality_pp"], 0)
            self.assertAlmostEqual(changed["delta_imaging_quality_pp"], 4.6)
            self.assertEqual(changed["total_nfe"], 51)

    def test_dispatches_one_vbench_batch_and_writes_reports(self):
        module = "changing_resolution_uni.scripts.data.batch_vbench_score_dataset"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = fixture(root)
            values = scores(summary["run_order"])
            args = SimpleNamespace(
                out_dir=str(root),
                vbench_root=str(root / "VBench"),
                vbench_python="python",
                vbench_commit="",
                mode="run",
                force=False,
            )
            with patch(module + ".inspect_vbench_checkout", return_value={"git_commit": "test"}), \
                 patch(module + ".score_case_directory", return_value=SimpleNamespace(
                     scores=values, provenance={"request_sha256": "bound"}
                 )) as scorer:
                evaluate(args)
            scorer.assert_called_once()
            metrics = root / "metrics/mrflow_refinement"
            payload = json.loads((metrics / "vbench_scores.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["rows"]), 10)
            self.assertTrue((metrics / "comparison.csv").is_file())
            report = (metrics / "comparison.md").read_text(encoding="utf-8")
            self.assertIn("transition-only control", report)
            self.assertIn("S0120_HR01", report)


if __name__ == "__main__":
    unittest.main()
