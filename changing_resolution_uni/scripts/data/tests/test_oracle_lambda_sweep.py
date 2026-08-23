from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from changing_resolution_uni.scripts.data.sweep_oracle_lambda import (
    inclusive_decimal_grid,
    load_prompt_arrays,
    sha256,
    sweep_distributions,
)
from changing_resolution_uni.scripts.data.oracle_record_schema import (
    FORMAL_STEPS,
    QUALITY5_DIMENSIONS,
)


class OracleLambdaSweepTest(unittest.TestCase):
    def test_inclusive_grid_has_one_hundred_points(self) -> None:
        values = inclusive_decimal_grid("0.001", "0.100", "0.001")
        self.assertEqual(len(values), 100)
        self.assertEqual(values[0], 0.001)
        self.assertEqual(values[-1], 0.1)

    def test_lambda_moves_distribution_toward_lower_latency(self) -> None:
        prompts = {
            0: {
                "qualities": np.asarray([[0.91, 0.90], [0.91, 0.90]]),
                "normalized_latencies": np.asarray([[0.9, 0.2], [0.9, 0.2]]),
                "seed_count": 2,
            },
            1: {
                "qualities": np.asarray([[0.92, 0.89], [0.92, 0.89]]),
                "normalized_latencies": np.asarray([[0.9, 0.2], [0.9, 0.2]]),
                "seed_count": 2,
            },
        }
        summary, distribution = sweep_distributions(
            prompts,
            candidate_steps=[30, 50],
            lambdas=[0.001, 0.1],
            near_tie_threshold=0.001,
        )
        counts = {
            (row["lambda"], row["step"]): row["count"] for row in distribution
        }
        self.assertEqual(counts[(0.001, 30)], 2)
        self.assertEqual(counts[(0.1, 50)], 2)
        self.assertEqual(summary[0]["endpoint_fraction"], 1.0)
        self.assertEqual(summary[1]["label_changes_from_previous_lambda"], 2)

    def test_loader_enforces_prompt_offset_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records_dir = root / "records"
            records_dir.mkdir()
            record_files = []
            record_hashes = {}
            for seed in (49, 107):
                name = f"p000007_s{seed}.json"
                record_files.append(name)
                record = {
                    "prompt_id": 7,
                    "seed": seed,
                    "prompt_text": "prompt 7",
                    "native_vbench5": 0.92,
                    "native_latency_seconds": 200.0,
                    "native_dimensions": {
                        dimension: 0.92 for dimension in QUALITY5_DIMENSIONS
                    },
                    "native_latency_source": "warm_pipeline_seconds",
                    "candidates": [
                        {
                            "step": step,
                            "vbench5": 0.90,
                            "latency_seconds": 100.0 - index,
                            "latency_source": "estimated_warm_pipeline_seconds",
                            "dimensions": {
                                dimension: 0.90
                                for dimension in QUALITY5_DIMENSIONS
                            },
                        }
                        for index, step in enumerate(FORMAL_STEPS)
                    ],
                    "scoring_provenance": {
                        "schema": "strict_vbench5_record_provenance_v1",
                        "quality_dimensions": QUALITY5_DIMENSIONS,
                        "diagnostic_dimensions": [],
                        "quality_aggregation": "arithmetic_mean_raw_vbench5_float64",
                        "vbench": {
                            "git_commit": "1" * 40,
                            "tracked_dirty": False,
                            "evaluate_py_sha256": "2" * 64,
                        },
                        "cases": {
                            case: {
                                "request_sha256": "a" * 64,
                                "result_sha256": "b" * 64,
                                "full_info_sha256": "c" * 64,
                                "run_manifest_path": "/strict/run/score_run_manifest.json",
                            }
                            for case in [
                                "native_hr",
                                *(f"step{step}" for step in FORMAL_STEPS),
                            ]
                        },
                    },
                }
                record_path = records_dir / name
                record_path.write_text(json.dumps(record), encoding="utf-8")
                record_hashes[name] = sha256(record_path)
            (root / "dataset_manifest.json").write_text(
                json.dumps(
                    {
                        "is_complete": True,
                        "quality_profile": "strict_vbench5_v1",
                        "expected_prompts": 1,
                        "expected_base_seeds": [42, 100],
                        "seed_policy": "prompt_offset",
                        "candidate_steps": FORMAL_STEPS,
                        "record_files": record_files,
                        "record_sha256": record_hashes,
                    }
                ),
                encoding="utf-8",
            )

            prompts, metadata = load_prompt_arrays(root)
            self.assertEqual(sorted(prompts), [7])
            self.assertEqual(prompts[7]["seed_count"], 2)
            self.assertEqual(metadata["record_count"], 2)
            self.assertEqual(
                metadata["latency_source_counts"],
                {"estimated_warm_pipeline_seconds": 2 * len(FORMAL_STEPS)},
            )


if __name__ == "__main__":
    unittest.main()
