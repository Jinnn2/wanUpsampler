from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from changing_resolution_uni.scripts.data.oracle_record_schema import (
    QUALITY5_DIMENSIONS,
)
from changing_resolution_uni.scripts.router import (
    prepare_1500_variable_lambda_states as prepare,
)
from changing_resolution_uni.scripts.router import (
    summarize_variable_lambda_runs as summarize,
)
from changing_resolution_uni.scripts.router import (
    train_variable_lambda_router as train,
)


STEPS = [30, 50]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def strict_record(prompt_id: int, seed: int) -> dict:
    cases = {
        name: {
            "request_sha256": "a" * 64,
            "result_sha256": "b" * 64,
            "full_info_sha256": "c" * 64,
            "run_manifest_path": "/strict/score_run_manifest.json",
        }
        for name in ("native_hr", "step30", "step50")
    }
    qualities = [0.81 + 0.001 * prompt_id, 0.82]
    latencies = [80.0, 40.0]
    return {
        "prompt_id": prompt_id,
        "seed": seed,
        "prompt_text": f"prompt {prompt_id}",
        "native_vbench5": 0.83,
        "native_latency_seconds": 200.0,
        "native_dimensions": {name: 0.83 for name in QUALITY5_DIMENSIONS},
        "native_latency_source": "warm_pipeline_seconds",
        "candidates": [
            {
                "step": step,
                "vbench5": quality,
                "latency_seconds": latency,
                "latency_source": "estimated_warm_pipeline_seconds",
                "dimensions": {name: quality for name in QUALITY5_DIMENSIONS},
            }
            for step, quality, latency in zip(STEPS, qualities, latencies)
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


def write_scored_dataset(root: Path, records: list[tuple[str, str]]) -> None:
    records_dir = root / "records"
    records_dir.mkdir(parents=True)
    names = []
    hashes = {}
    for name, content in records:
        path = records_dir / name
        path.write_text(content, encoding="utf-8")
        names.append(name)
        hashes[name] = sha256(path)
    manifest = {
        "schema": "prompt_conditioned_scored_oracle_dataset_v3",
        "quality_profile": "strict_vbench5_v1",
        "quality_dimensions": QUALITY5_DIMENSIONS,
        "is_complete": True,
        "record_files": names,
        "record_sha256": hashes,
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_raw_trajectory(
    physical_root: Path,
    prompt_id: int,
    base_seed: int,
) -> None:
    actual_seed = base_seed + prompt_id
    seed_dir = (
        physical_root / "_parts" / "part_00" / "raw_samples" / f"seed_{base_seed}"
    )
    sample_id = f"{prompt_id:04d}_seed{actual_seed}"
    branches = []
    for index, step in enumerate(STEPS):
        latent_dir = seed_dir / "latents" / f"step{step:02d}"
        latent_dir.mkdir(parents=True, exist_ok=True)
        path = latent_dir / f"{sample_id}_step{step:02d}.pt"
        x0 = torch.full(prepare.EXPECTED_LATENT_SHAPE, 0.1 * (index + 1))
        x_t = x0 + 0.02 * (2 - index)
        torch.save(
            {
                "schema": prepare.LATENT_SCHEMA,
                "sample_id": sample_id,
                "prompt": f"prompt {prompt_id}",
                "seed": actual_seed,
                "candidate_step": step,
                "step_index_zero_based": step - 1,
                "infer_steps": 50,
                "sigma": 0.4 if step == 30 else 0.05,
                "taa_enabled": False,
                "x_t_lr": x_t,
                "x0_pred_lr": x0,
            },
            path,
        )
        branches.append({"candidate_step": step, "latent_path": str(path)})
    manifest_dir = seed_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"{sample_id}.json").write_text(
        json.dumps(
            {
                "schema": "wan_taa_free_oracle_v1",
                "prompt_index": prompt_id,
                "prompt": f"prompt {prompt_id}",
                "seed": actual_seed,
                "candidate_steps": STEPS,
                "branches": branches,
            }
        ),
        encoding="utf-8",
    )
    t5_dir = physical_root / "t5_embeddings"
    t5_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        t5_dir / f"prompt_{prompt_id:06d}.npz",
        pooled_embedding=np.full(4096, prompt_id + 1, dtype=np.float16),
    )


class VariableLambdaRouterTest(unittest.TestCase):
    def build_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        generation = root / "oracle_dataset_1500_8gpu"
        generation.mkdir()
        plan = {
            "schema": prepare.PLAN_SCHEMA,
            "candidate_steps": STEPS,
            "splits": {
                "train": {
                    "prompt_offset": 0,
                    "prompt_count": 1,
                    "seeds": [42],
                    "physical_dataset": "train",
                },
                "validation": {
                    "prompt_offset": 1,
                    "prompt_count": 1,
                    "seeds": [42],
                    "physical_dataset": "eval",
                },
                "test": {
                    "prompt_offset": 2,
                    "prompt_count": 1,
                    "seeds": [42],
                    "physical_dataset": "eval",
                },
            },
            "artifacts": {"latent_schema": prepare.LATENT_SCHEMA},
        }
        (generation / "generation_plan.json").write_text(
            json.dumps(plan), encoding="utf-8"
        )
        write_raw_trajectory(generation / "train", 0, 42)
        write_raw_trajectory(generation / "eval", 1, 42)
        # This malformed test manifest proves selection preparation filters by
        # filename before reading held-out content.
        test_manifest_dir = (
            generation
            / "eval"
            / "_parts"
            / "part_00"
            / "raw_samples"
            / "seed_42"
            / "manifests"
        )
        (test_manifest_dir / "0002_seed44.json").write_text(
            "not test-accessible json", encoding="utf-8"
        )
        scored_train = root / "scored" / "train"
        scored_eval = root / "scored" / "eval"
        write_scored_dataset(
            scored_train,
            [("p000000_s42.json", json.dumps(strict_record(0, 42)))],
        )
        write_scored_dataset(
            scored_eval,
            [
                ("p000001_s43.json", json.dumps(strict_record(1, 43))),
                ("p000002_s44.json", "not test-accessible json"),
            ],
        )
        return generation, scored_train, scored_eval

    def test_prepare_train_validation_without_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, scored_train, scored_eval = self.build_fixture(root)
            output = root / "states"
            argv = [
                "prepare",
                "--generation-root",
                str(generation),
                "--scored-train-dir",
                str(scored_train),
                "--scored-eval-dir",
                str(scored_eval),
                "--output-dir",
                str(output),
                "--splits",
                "train",
                "validation",
                "--progress-every",
                "1",
                "--torch-threads",
                "1",
            ]
            with (
                mock.patch.object(prepare, "FORMAL_STEPS", STEPS),
                mock.patch.object(sys, "argv", argv),
            ):
                prepare.main()
            manifest = json.loads(
                (output / "dataset_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["test_accessed"])
            self.assertEqual(manifest["splits"]["train"]["trajectory_count"], 1)
            self.assertEqual(manifest["splits"]["validation"]["trajectory_count"], 1)
            self.assertGreater(manifest["feature_count"], 100)
            self.assertFalse((output / "test_trajectories.jsonl").exists())
            train_feature = next((output / "features" / "train").glob("*.npz"))
            with np.load(train_feature, allow_pickle=False) as payload:
                features = np.asarray(payload["features"])
            has_previous_index = manifest["feature_names"].index(
                "trajectory.has_previous"
            )
            self.assertEqual(float(features[0, has_previous_index]), 0.0)
            self.assertEqual(float(features[1, has_previous_index]), 1.0)

    def test_variable_lambda_training_and_multiseed_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, scored_train, scored_eval = self.build_fixture(root)
            states = root / "states"
            prepare_argv = [
                "prepare",
                "--generation-root",
                str(generation),
                "--scored-train-dir",
                str(scored_train),
                "--scored-eval-dir",
                str(scored_eval),
                "--output-dir",
                str(states),
                "--splits",
                "train",
                "validation",
                "--torch-threads",
                "1",
            ]
            with (
                mock.patch.object(prepare, "FORMAL_STEPS", STEPS),
                mock.patch.object(sys, "argv", prepare_argv),
            ):
                prepare.main()
            run = root / "run"
            train_argv = [
                "train",
                "--dataset-dir",
                str(states),
                "--out-dir",
                str(run),
                "--model-type",
                "both",
                "--train-lambdas",
                "0.01",
                "0.08",
                "--eval-lambdas",
                "0.01",
                "0.08",
                "--primary-lambda",
                "0.08",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--device",
                "cpu",
            ]
            with mock.patch.object(sys, "argv", train_argv):
                train.main()
            summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["test_accessed"])
            self.assertEqual(summary["train_prompts"], 1)
            self.assertEqual(summary["validation_prompts"], 1)
            with (run / "validation_predictions.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)

            runs_root = root / "runs"
            for seed in (42, 100, 2024):
                destination = runs_root / f"seed_{seed}"
                shutil.copytree(run, destination)
                payload = json.loads(
                    (destination / "run_summary.json").read_text(encoding="utf-8")
                )
                payload["train_seed"] = seed
                (destination / "run_summary.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            summarize_argv = [
                "summarize",
                "--runs-root",
                str(runs_root),
                "--bootstrap-samples",
                "100",
            ]
            with mock.patch.object(sys, "argv", summarize_argv):
                summarize.main()
            selection = json.loads(
                (runs_root / "selection" / "architecture_selection.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(selection["test_accessed"])
            self.assertEqual(selection["run_count"], 3)
            self.assertTrue(
                (runs_root / "selection" / "paired_reference_deltas.csv").is_file()
            )

    def test_lambda_changes_stop_regret(self) -> None:
        qualities = np.asarray([0.81, 0.82], dtype=np.float32)
        costs = np.asarray([0.4, 0.2], dtype=np.float32)
        low_lambda = train.true_stop_regret(qualities, costs, 0.01)
        high_lambda = train.true_stop_regret(qualities, costs, 0.10)
        self.assertGreater(float(low_lambda[0]), 0.0)
        self.assertGreater(float(high_lambda[0]), float(low_lambda[0]))


if __name__ == "__main__":
    unittest.main()
