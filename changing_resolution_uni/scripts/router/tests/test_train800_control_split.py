from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from changing_resolution_uni.scripts.router import (
    prepare_train800_control_split as control,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Train800ControlSplitTest(unittest.TestCase):
    def build_source(self, root: Path, *, duplicate_prompt: bool = False) -> Path:
        source = root / "source_states"
        feature_dir = source / "features" / "train"
        t5_dir = source / "t5"
        feature_dir.mkdir(parents=True)
        t5_dir.mkdir()
        rows = []
        for prompt_id in range(10):
            feature = feature_dir / f"p{prompt_id:06d}_s{prompt_id + 42}.npz"
            t5 = t5_dir / f"prompt_{prompt_id:06d}.npz"
            feature.write_bytes(f"feature {prompt_id}".encode())
            t5.write_bytes(f"t5 {prompt_id}".encode())
            prompt_text = (
                "prompt 0"
                if duplicate_prompt and prompt_id == 1
                else f"prompt {prompt_id}"
            )
            rows.append(
                {
                    "split": "train",
                    "prompt_id": prompt_id,
                    "seed": prompt_id + 42,
                    "base_seed": 42,
                    "prompt_text": prompt_text,
                    "feature_file": str(feature.relative_to(source)),
                    "t5_embedding_path": str(t5),
                    "record_path": f"/records/{prompt_id}.json",
                    "record_sha256": str(prompt_id) * 64,
                    "sample_manifest_path": f"/manifests/{prompt_id}.json",
                }
            )
        train_index = source / "train_trajectories.jsonl"
        train_index.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        manifest = {
            "schema": control.DATASET_SCHEMA,
            "generation_root": "/generation",
            "generation_plan": "/generation/generation_plan.json",
            "generation_plan_sha256": "a" * 64,
            "candidate_steps": [30, 50],
            "quality_dimensions": ["quality"],
            "feature_names": ["feature"],
            "feature_groups": {"group": [0]},
            "feature_count": 1,
            "lambda_dependent_features": False,
            "selected_splits": ["train", "validation"],
            "test_accessed": False,
            "splits": {
                "train": {
                    "trajectory_count": 10,
                    "index_file": train_index.name,
                    "index_sha256": sha256(train_index),
                },
                # The derived builder must not open this source validation index.
                "validation": {
                    "trajectory_count": 999,
                    "index_file": "missing_validation_trajectories.jsonl",
                    "index_sha256": "b" * 64,
                },
            },
            "scored_sources": {
                "train": {
                    "manifest": "/scored/train/dataset_manifest.json",
                    "manifest_sha256": "c" * 64,
                    "quality_profile": "strict_vbench5_v1",
                }
            },
            "latency_profile": {
                "schema": "train_calibrated_latency_profile_v1",
                "path": "/latency/train_h100.json",
                "sha256": "d" * 64,
                "hardware_label": "H100",
                "source_split": "train",
                "aggregation": "mean_of_per_trajectory_normalized_costs",
            },
            "is_complete": True,
        }
        (source / "dataset_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return source

    def run_builder(self, source: Path, output: Path) -> None:
        argv = [
            "prepare-control",
            "--source-dataset-dir",
            str(source),
            "--output-dir",
            str(output),
            "--validation-count",
            "2",
            "--expected-source-prompts",
            "10",
            "--expected-base-seed",
            "42",
            "--split-salt",
            "fixture-control-v1",
        ]
        with mock.patch.object(sys, "argv", argv):
            control.main()

    def test_hash_split_is_deterministic_and_skips_source_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.build_source(root)
            first = root / "first"
            second = root / "second"
            self.run_builder(source, first)
            self.run_builder(source, second)

            first_manifest = json.loads(
                (first / "dataset_manifest.json").read_text(encoding="utf-8")
            )
            second_manifest = json.loads(
                (second / "dataset_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                first_manifest["splits"]["validation"]["prompt_ids"],
                second_manifest["splits"]["validation"]["prompt_ids"],
            )
            self.assertEqual(first_manifest["splits"]["train"]["prompt_count"], 8)
            self.assertEqual(first_manifest["splits"]["validation"]["prompt_count"], 2)
            self.assertFalse(first_manifest["test_accessed"])
            self.assertFalse(
                first_manifest["derivation"]["source_validation_index_accessed"]
            )
            train_ids = set(first_manifest["splits"]["train"]["prompt_ids"])
            validation_ids = set(first_manifest["splits"]["validation"]["prompt_ids"])
            self.assertFalse(train_ids & validation_ids)
            self.assertEqual(train_ids | validation_ids, set(range(10)))

            rows = control.read_jsonl(first / "validation_trajectories.jsonl")
            self.assertTrue(all(row["split"] == "validation" for row in rows))
            self.assertTrue(all(len(row["feature_sha256"]) == 64 for row in rows))
            self.assertTrue(all(len(row["t5_embedding_sha256"]) == 64 for row in rows))

    def test_duplicate_normalized_prompt_text_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.build_source(root, duplicate_prompt=True)
            with self.assertRaisesRegex(ValueError, "prompt-leaking split"):
                self.run_builder(source, root / "output")


if __name__ == "__main__":
    unittest.main()
