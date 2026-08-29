from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from changing_resolution_uni.scripts.data.oracle_record_schema import (
    QUALITY5_DIMENSIONS,
)
from changing_resolution_uni.scripts.router import build_train_latency_profile as builder


STEPS = [30, 50]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_record(prompt_id: int, native: float) -> dict:
    seed = 42 + prompt_id
    cases = {
        name: {
            "request_sha256": "a" * 64,
            "result_sha256": "b" * 64,
            "full_info_sha256": "c" * 64,
            "run_manifest_path": "/strict/score_run_manifest.json",
        }
        for name in ("native_hr", "step30", "step50")
    }
    return {
        "prompt_id": prompt_id,
        "seed": seed,
        "prompt_text": f"prompt {prompt_id}",
        "native_vbench5": 0.83,
        "native_latency_seconds": native,
        "native_dimensions": {name: 0.83 for name in QUALITY5_DIMENSIONS},
        "native_latency_source": "warm_pipeline_seconds",
        "candidates": [
            {
                "step": step,
                "vbench5": 0.81 + index * 0.01,
                "latency_seconds": native * ratio,
                "latency_source": "estimated_warm_pipeline_seconds",
                "dimensions": {
                    name: 0.81 + index * 0.01 for name in QUALITY5_DIMENSIONS
                },
            }
            for index, (step, ratio) in enumerate(zip(STEPS, (0.4, 0.2)))
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
            "cases": cases,
        },
    }


class TrainLatencyProfileTest(unittest.TestCase):
    def test_builds_locked_mean_of_ratios_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scored = root / "scored_train"
            records = scored / "records"
            records.mkdir(parents=True)
            names = []
            hashes = {}
            for prompt_id, native in ((0, 200.0), (1, 225.0)):
                name = f"p{prompt_id:06d}_s{42 + prompt_id}.json"
                path = records / name
                path.write_text(json.dumps(strict_record(prompt_id, native)), encoding="utf-8")
                names.append(name)
                hashes[name] = sha256(path)
            manifest = {
                "schema": "prompt_conditioned_scored_oracle_dataset_v3",
                "quality_profile": "strict_vbench5_v1",
                "quality_dimensions": QUALITY5_DIMENSIONS,
                "candidate_steps": STEPS,
                "record_files": names,
                "record_sha256": hashes,
                "is_complete": True,
            }
            (scored / "dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            output = root / "profile.json"
            argv = [
                "profile",
                "--scored-train-dir",
                str(scored),
                "--output",
                str(output),
                "--hardware-label",
                "H100",
                "--expected-prompts",
                "2",
                "--bootstrap-samples",
                "20",
            ]
            with mock.patch.object(sys, "argv", argv):
                builder.main()
            profile = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(profile["hardware_label"], "H100")
            self.assertEqual(profile["selected_normalized_cost_profile"], [0.4, 0.2])
            self.assertEqual(profile["source_prompt_count"], 2)
            self.assertEqual(profile["stability"]["max_even_odd_absolute_difference"], 0.0)
            self.assertTrue(profile["monotonic_nonincreasing"])
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(FileExistsError):
                    builder.main()


if __name__ == "__main__":
    unittest.main()
