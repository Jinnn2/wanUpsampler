from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "run_clean_360p_stage2_oracle_metrics.py"
SPEC = importlib.util.spec_from_file_location("oracle_metrics", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_PATH}")
oracle_metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle_metrics)


class OracleMetricsTest(unittest.TestCase):
    def test_collects_vbench5_and_builds_sample_level_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "oracle"
            self._write_oracle_fixture(root)
            inventory = oracle_metrics.load_inventory(
                root,
                strict_protocol=False,
                min_video_bytes=1024,
            )
            oracle_metrics.prepare_inputs(root, inventory)

            scores = {
                "native_hr": [0.90, 0.90],
                "step40": [0.89, 0.84],
                "step50": [0.87, 0.85],
            }
            for case_name, case in inventory["cases"].items():
                raw_root = root / "metrics" / "oracle_vbench_raw_quality5" / case_name
                recorded = raw_root / "recorded_eval_results.json"
                oracle_metrics.write_json(
                    recorded,
                    self._vbench_payload(case, scores[case_name]),
                )
                oracle_metrics.write_json(
                    raw_root / "zzzz_distractor_eval_results.json",
                    self._vbench_payload(case, [0.10, 0.10]),
                )
                oracle_metrics.write_json(
                    raw_root / "run_record.json",
                    {
                        "schema": "wan_taa_free_oracle_vbench_run_v1",
                        "profile": "quality5",
                        "case": case_name,
                        "dimensions": oracle_metrics.QUALITY5_DIMENSIONS,
                        "input_signature": oracle_metrics.case_signature(
                            case, oracle_metrics.QUALITY5_DIMENSIONS
                        ),
                        "result_path": str(recorded.resolve()),
                        "result_sha256": oracle_metrics.sha256_file(recorded),
                    },
                )

            outputs = oracle_metrics.collect_metrics(
                root,
                inventory,
                include_overall=False,
                overall_weight=0.0,
                max_quality_drop=0.02,
                latency_lambda=0.05,
                timing_source="branch",
            )
            payload = json.loads(outputs["canonical_json"].read_text(encoding="utf-8"))
            candidates = {
                (row["sample_id"], row["candidate_step"]): row
                for row in payload["candidate_per_sample"]
            }
            labels = {row["sample_id"]: row for row in payload["labels"]}

            self.assertAlmostEqual(
                candidates[("0000_seed9700", 40)]["imaging_quality"], 0.89
            )
            self.assertAlmostEqual(candidates[("0000_seed9700", 40)]["vbench5"], 0.89)
            self.assertEqual(labels["0000_seed9700"]["quality_floor_step"], 40)
            self.assertFalse(
                labels["0000_seed9700"]["quality_floor_fallback_to_max_quality"]
            )
            self.assertEqual(labels["0000_seed9700"]["weighted_utility_step"], 40)
            self.assertEqual(labels["0000_seed9700"]["pareto_steps"], "40 50")

            self.assertEqual(labels["0001_seed9701"]["quality_floor_step"], 50)
            self.assertTrue(
                labels["0001_seed9701"]["quality_floor_fallback_to_max_quality"]
            )
            self.assertEqual(labels["0001_seed9701"]["weighted_utility_step"], 50)
            self.assertEqual(labels["0001_seed9701"]["pareto_steps"], "50")
            self.assertTrue(
                all(
                    Path(source).name == "recorded_eval_results.json"
                    for source in payload["quality_result_sources"].values()
                )
            )

            expected_csv_rows = {
                "candidate_per_sample_csv": 4,
                "native_per_sample_csv": 2,
                "candidate_summary_csv": 2,
                "oracle_labels_csv": 2,
            }
            for key, expected_rows in expected_csv_rows.items():
                with outputs[key].open(encoding="utf-8", newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), expected_rows)

            overall_scores = {
                "native_hr": [0.95, 0.95],
                "step40": [0.90, 0.82],
                "step50": [0.88, 0.86],
            }
            for case_name, case in inventory["cases"].items():
                raw_root = root / "metrics" / "oracle_vbench_raw_overall" / case_name
                recorded = raw_root / "recorded_eval_results.json"
                oracle_metrics.write_json(
                    recorded,
                    self._vbench_payload(
                        case,
                        overall_scores[case_name],
                        dimensions=oracle_metrics.OVERALL_DIMENSIONS,
                    ),
                )
                oracle_metrics.write_json(
                    raw_root / "run_record.json",
                    {
                        "schema": "wan_taa_free_oracle_vbench_run_v1",
                        "profile": "overall",
                        "case": case_name,
                        "dimensions": oracle_metrics.OVERALL_DIMENSIONS,
                        "input_signature": oracle_metrics.case_signature(
                            case, oracle_metrics.OVERALL_DIMENSIONS
                        ),
                        "result_path": str(recorded.resolve()),
                        "result_sha256": oracle_metrics.sha256_file(recorded),
                    },
                )
            overall_outputs = oracle_metrics.collect_metrics(
                root,
                inventory,
                include_overall=True,
                overall_weight=0.20,
                max_quality_drop=0.02,
                latency_lambda=0.05,
                timing_source="branch",
            )
            overall_payload = json.loads(
                overall_outputs["canonical_json"].read_text(encoding="utf-8")
            )
            overall_candidates = {
                (row["sample_id"], row["candidate_step"]): row
                for row in overall_payload["candidate_per_sample"]
            }
            self.assertAlmostEqual(
                overall_candidates[("0000_seed9700", 40)]["overall_consistency"],
                0.90,
            )
            self.assertAlmostEqual(
                overall_candidates[("0000_seed9700", 40)]["selection_quality"],
                0.892,
            )

    @staticmethod
    def _write_oracle_fixture(root: Path) -> None:
        steps = [40, 50]
        oracle_metrics.write_json(
            root / "protocol.json",
            {
                "schema": "wan_taa_free_oracle_protocol_v1",
                "execution_mode": "branch",
                "candidate_steps": steps,
                "infer_steps": 50,
                "prompt_count": 2,
                "taa_enabled": False,
            },
        )
        timings = [
            {40: 6.0, 50: 4.0, "native_hr": 10.0},
            {40: 8.0, 50: 5.0, "native_hr": 10.0},
        ]
        for index, sample_timings in enumerate(timings):
            seed = 9700 + index
            sample_id = f"{index:04d}_seed{seed}"
            branches = []
            for step in steps:
                video = (
                    root
                    / "videos"
                    / f"step{step:02d}"
                    / (f"{sample_id}_step{step:02d}.mp4")
                )
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b"v" * 2048)
                branches.append(
                    {
                        "candidate_step": step,
                        "lr_evaluations": step,
                        "hr_evaluations": 50 - step,
                        "estimated_warm_pipeline_seconds": sample_timings[step],
                    }
                )
            native_video = (
                root / "videos" / "native_hr" / (f"{sample_id}_native_hr.mp4")
            )
            native_video.parent.mkdir(parents=True, exist_ok=True)
            native_video.write_bytes(b"v" * 2048)
            oracle_metrics.write_json(
                root / "manifests" / f"{sample_id}.json",
                {
                    "schema": "wan_taa_free_oracle_v1",
                    "execution_mode": "branch",
                    "prompt_index": index,
                    "prompt": f"prompt {index}",
                    "seed": seed,
                    "candidate_steps": steps,
                    "taa_enabled": False,
                    "branches": branches,
                    "native_hr": {
                        "lr_evaluations": 0,
                        "hr_evaluations": 50,
                        "warm_pipeline_seconds": sample_timings["native_hr"],
                    },
                },
            )

    @staticmethod
    def _vbench_payload(
        case: dict,
        scores: list[float],
        *,
        dimensions: list[str] | None = None,
    ) -> dict:
        payload = {}
        for dimension in dimensions or oracle_metrics.QUALITY5_DIMENSIONS:
            scale = 100.0 if dimension == "imaging_quality" else 1.0
            payload[dimension] = [
                sum(scores) / len(scores) * scale,
                [
                    {
                        "video_path": str(video),
                        "video_results": score * scale,
                    }
                    for video, score in zip(case["videos"], scores)
                ],
            ]
        return payload


if __name__ == "__main__":
    unittest.main()
