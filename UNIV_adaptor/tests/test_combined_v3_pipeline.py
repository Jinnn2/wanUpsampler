from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from UNIV_adaptor.combined_v3 import (
    COMBINED_RECORD_SCHEMA,
    QUALITY_DIMENSIONS,
    canonical_sha256,
    sha256_file,
)
from UNIV_adaptor.scripts.data.merge_scored_combined_v3 import merge
from UNIV_adaptor.scripts.data.score_combined_v3_dataset import (
    CASE_SCORE_SCHEMA,
    finalize,
    materialize_case,
    prepare,
    validate_score_manifest,
)

try:
    from UNIV_adaptor.scripts.router.train_combined_v3_budget_prior import (
        load_samples,
        train_latency_profile,
        validate_embedding_manifest,
        validate_merged_index,
    )

    TRAINING_IMPORTS_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    TRAINING_IMPORTS_AVAILABLE = False


class CombinedV3PipelineTest(unittest.TestCase):
    def _make_shard(self, root: Path, shard_name: str, prompts: list[str]) -> None:
        extension_body = {
            "shard": shard_name,
            "out_root": str(root.resolve()),
            "model_root": str((root.parent / "model").resolve()),
        }
        extension = {
            "schema": "univ_low_budget_extension_generation_manifest_v1",
            "manifest_sha256": canonical_sha256(extension_body),
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            **extension_body,
        }
        root.mkdir(parents=True)
        (root / "extension_manifest.json").write_text(
            json.dumps(extension), encoding="utf-8"
        )
        assignments = [("train", 0, 42)] + [
            ("validation", 1, seed + 1) for seed in (42, 100, 2024)
        ]
        for split, prompt_id, seed in assignments:
            prompt = prompts[prompt_id]
            trajectory = f"{split}_p{prompt_id:06d}_s{seed}"
            artifact_root = root / "artifacts" / trajectory
            native = self._artifact(artifact_root / "native.mp4", 10.0)
            candidates = []
            for index in range(9):
                action_id = f"A{index}"
                candidates.append(
                    {
                        "artifact_id": action_id,
                        "budget_id": action_id,
                        "display_budget": f"B{index}",
                        "action_schema": "fixture_action_v1",
                        "target_cost_ratio": 0.1 + index * 0.05,
                        **self._artifact(
                            artifact_root / f"{action_id}.mp4", 1.0 + index
                        ),
                    }
                )
            body = {
                "generation_status": "generated_unscored",
                "trajectory_key": trajectory,
                "split": split,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "prompt_sha256": canonical_sha256(prompt),
                "base_seed": seed - prompt_id,
                "seed": seed,
                "native_teacher": native,
                "budget_candidates": candidates,
                "candidate_count": 9,
                "source_records": {},
                "provenance": {},
            }
            record = {
                "schema": COMBINED_RECORD_SCHEMA,
                "record_sha256": canonical_sha256(body),
                **body,
            }
            output = root / "combined_records" / split / f"{trajectory}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(record), encoding="utf-8")

    @staticmethod
    def _artifact(path: Path, seconds: float) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("ascii") + b"x" * 1024)
        return {
            "video_path": str(path.resolve()),
            "video_sha256": sha256_file(path),
            "video_bytes": path.stat().st_size,
            "cost": {"pipeline_seconds": seconds},
        }

    def _prepare_scored_fixture(self, root: Path) -> tuple[Path, Path]:
        primary = root / "primary"
        reserve = root / "reserve"
        self._make_shard(primary, "primary", ["primary train", "primary val"])
        self._make_shard(reserve, "reserve", ["reserve train", "reserve val"])
        score_root = root / "scores"
        prepare(
            SimpleNamespace(
                shard=[("primary", primary), ("reserve", reserve)],
                out_root=str(score_root),
                splits=["train", "validation"],
                expected_train_records_per_shard=1,
                expected_validation_records_per_shard=3,
            )
        )
        manifest_path = score_root / "score_manifest.json"
        manifest = validate_score_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        first_video_dir, _ = materialize_case(manifest, manifest["cases"][0])
        first_source = Path(manifest["cases"][0]["rows"][0]["video_path"])
        self.assertTrue(
            os.path.samefile(next(first_video_dir.glob("*.mp4")), first_source)
        )
        vbench = {
            "git_commit": "a" * 40,
            "tracked_dirty": False,
            "tracked_dirty_paths": [],
            "evaluate_py_sha256": "b" * 64,
        }
        for case in manifest["cases"]:
            scores = {
                row["trajectory_key"]: {
                    **{
                        name: 0.5 + index * 0.001
                        for index, name in enumerate(QUALITY_DIMENSIONS)
                    },
                    "dynamic_degree": 1.0,
                }
                for row in case["rows"]
            }
            body = {
                "score_manifest_sha256": manifest["manifest_sha256"],
                "case_id": case["case_id"],
                "case_sha256": canonical_sha256(case),
                "action_id": case["action_id"],
                "split": case["split"],
                "shard_id": case["shard_id"],
                "record_count": case["record_count"],
                "quality_dimensions": list(QUALITY_DIMENSIONS),
                "diagnostic_dimensions": ["dynamic_degree"],
                "scores": scores,
                "vbench_provenance": {"vbench": vbench},
            }
            payload = {
                "schema": CASE_SCORE_SCHEMA,
                "bundle_sha256": canonical_sha256(body),
                **body,
            }
            path = score_root / "case_scores" / f"{case['case_id']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        finalize(SimpleNamespace(manifest=str(manifest_path)))
        return score_root, score_root / "scored_dataset_manifest.json"

    def test_score_finalize_merge_and_training_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            score_root, scored_manifest = self._prepare_scored_fixture(root)
            dataset_root = root / "dataset"
            merge(
                SimpleNamespace(
                    scored_manifest=str(scored_manifest),
                    output_root=str(dataset_root),
                    expected_train_prompts=2,
                    expected_validation_prompts=2,
                    validation_seeds=3,
                )
            )
            dataset = json.loads(
                (dataset_root / "dataset_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(dataset["prompts_by_split"], {"train": 2, "validation": 2})
            self.assertEqual(
                dataset["trajectories_by_split"], {"train": 2, "validation": 6}
            )
            self.assertEqual(dataset["video_reference_count"], 80)
            self.assertFalse(dataset["test_accessed"])
            self.assertEqual(
                len({row["global_prompt_id"] for row in dataset["prompts"]}), 4
            )
            self.assertEqual(
                len(list((score_root / "scored_records").glob("*/*/*.json"))), 8
            )

    @unittest.skipUnless(TRAINING_IMPORTS_AVAILABLE, "torch is unavailable")
    def test_training_input_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, scored_manifest = self._prepare_scored_fixture(root)
            dataset_root = root / "dataset"
            merge(
                SimpleNamespace(
                    scored_manifest=str(scored_manifest),
                    output_root=str(dataset_root),
                    expected_train_prompts=2,
                    expected_validation_prompts=2,
                    validation_seeds=3,
                )
            )
            dataset = validate_merged_index(
                json.loads(
                    (dataset_root / "dataset_index.json").read_text(encoding="utf-8")
                )
            )
            self._materialize_embeddings(dataset_root, dataset)
            _, embeddings = validate_embedding_manifest(dataset_root, dataset)
            samples, action_ids = load_samples(dataset, embeddings)
            latency = train_latency_profile(
                dataset, action_ids, hardware_label="fixture_gpu"
            )
            self.assertEqual(len(samples["train"]), 2)
            self.assertEqual(len(samples["validation"]), 2)
            self.assertEqual(action_ids, [f"A{index}" for index in range(9)])
            self.assertEqual(latency["record_count"], 2)
            self.assertEqual(set(latency["normalized_action_cost"]), set(action_ids))

    @staticmethod
    def _materialize_embeddings(dataset_root: Path, dataset: dict) -> None:
        t5_root = dataset_root / "t5_embeddings"
        t5_root.mkdir()
        model_root = Path(dataset["model_root"])
        model_root.mkdir(parents=True, exist_ok=True)
        checkpoint = model_root / "t5.pt"
        checkpoint.write_bytes(b"native-t5")
        entries = []
        for prompt in dataset["prompts"]:
            prompt_id = prompt["global_prompt_id"]
            npz_path = t5_root / f"prompt_{prompt_id:06d}.npz"
            metadata_path = t5_root / f"prompt_{prompt_id:06d}.json"
            np.savez_compressed(
                npz_path,
                pooled_embedding=np.full(4096, prompt_id, dtype=np.float16),
                seq_embedding=np.zeros((1, 4096), dtype=np.float16),
                input_ids=np.zeros(1, dtype=np.int64),
                attention_mask=np.ones(1, dtype=np.int64),
            )
            raw_hash = hashlib.sha256(prompt["prompt"].encode("utf-8")).hexdigest()
            metadata_path.write_text(
                json.dumps(
                    {
                        "prompt_id": prompt_id,
                        "prompt_text": prompt["prompt"],
                        "prompt_sha256": raw_hash,
                    }
                ),
                encoding="utf-8",
            )
            entries.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_text": prompt["prompt"],
                    "prompt_sha256": raw_hash,
                    "npz_file": str(npz_path.resolve()),
                    "npz_sha256": sha256_file(npz_path),
                    "json_file": str(metadata_path.resolve()),
                    "json_sha256": sha256_file(metadata_path),
                }
            )
        body = {
            "prompts_file": str((dataset_root / "prompts.txt").resolve()),
            "prompts_file_sha256": sha256_file(dataset_root / "prompts.txt"),
            "model_path": dataset["model_root"],
            "text_encoder_checkpoint": str(checkpoint),
            "text_encoder_checkpoint_sha256": sha256_file(checkpoint),
            "tokenizer_path": None,
            "backend": "wan_native",
            "required_backend": "wan_native",
            "extractor_sha256": sha256_file(
                Path(
                    "changing_resolution_uni/scripts/data/"
                    "extract_prompt_t5_embeddings.py"
                )
            ),
            "precision": "bf16",
            "max_seq_len": 512,
            "prompt_offset": 0,
            "limit": None,
            "total_prompts": len(entries),
            "complete": True,
            "prompts": entries,
        }
        manifest = {
            "schema": "prompt_t5_embeddings_manifest_v2",
            "manifest_sha256": canonical_sha256(body),
            **body,
        }
        (t5_root / "t5_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )


if __name__ == "__main__":
    unittest.main()
