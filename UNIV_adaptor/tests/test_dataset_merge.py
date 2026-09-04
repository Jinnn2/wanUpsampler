from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from UNIV_adaptor.data_protocol import (
    RECORD_SCHEMA,
    sha256_file,
    validate_trajectory_record,
    write_json_atomic,
)
from UNIV_adaptor.scripts.data.merge_prompt_budget_datasets import merge_datasets
from UNIV_adaptor.scripts.data.run_prompt_budget_generation import load_json, prepare
from UNIV_adaptor.scripts.data.select_prompt_shard import select_prompt_shard
from UNIV_adaptor.tests.test_data_protocol import protocol


class PromptShardSelectionTest(unittest.TestCase):
    def test_selection_is_deterministic_unique_and_excludes_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            primary = root / "primary.txt"
            source.write_text(
                "\n".join([f"prompt {index}" for index in range(30)] + ["prompt 3"])
                + "\n",
                encoding="utf-8",
            )
            primary.write_text("prompt 1\nprompt 2\n", encoding="utf-8")
            first, metadata = select_prompt_shard(
                source,
                exclusions=[primary],
                count=10,
                seed="reserve-v1",
            )
            second, _ = select_prompt_shard(
                source,
                exclusions=[primary],
                count=10,
                seed="reserve-v1",
            )
            self.assertEqual(first, second)
            self.assertEqual(len(first), len(set(first)))
            self.assertTrue(set(first).isdisjoint({"prompt 1", "prompt 2"}))
            self.assertEqual(metadata["source_duplicates"], 1)


class DatasetMergeTest(unittest.TestCase):
    def test_two_compatible_shards_merge_without_copying_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path, template_path = self._shared_inputs(root)
            first = self._prepare_shard(
                root / "primary",
                ["a", "b", "c", "d"],
                protocol_path,
                template_path,
                root / "model",
            )
            second = self._prepare_shard(
                root / "reserve",
                ["e", "f", "g", "h"],
                protocol_path,
                template_path,
                root / "model",
            )
            self._materialize_records(first)
            self._materialize_records(second)

            merged = merge_datasets(
                [("primary", first), ("reserve", second)],
                splits=["train", "validation", "test"],
                require_scores=False,
                verify_hashes=True,
            )
            self.assertEqual(merged["prompt_count"], 8)
            self.assertEqual(merged["trajectory_count"], 12)
            self.assertEqual(merged["video_count"], 72)
            self.assertEqual(
                len({record["record_uid"] for record in merged["records"]}),
                12,
            )
            self.assertEqual(
                len({prompt["global_prompt_id"] for prompt in merged["prompts"]}),
                8,
            )
            for record in merged["records"]:
                self.assertTrue(Path(record["record_path"]).is_file())
            first_record = load_json(merged["records"][0]["record_path"])
            Path(first_record["native_teacher"]["video_path"]).unlink()
            with self.assertRaisesRegex(RuntimeError, "missing or undersized"):
                merge_datasets(
                    [("primary", first), ("reserve", second)],
                    splits=["train", "validation", "test"],
                    require_scores=False,
                    verify_hashes=False,
                )

    def test_prompt_overlap_across_shards_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path, template_path = self._shared_inputs(root)
            first = self._prepare_shard(
                root / "primary",
                ["a", "b", "c", "same"],
                protocol_path,
                template_path,
                root / "model",
            )
            second = self._prepare_shard(
                root / "reserve",
                ["e", "f", "g", "same"],
                protocol_path,
                template_path,
                root / "model",
            )
            self._materialize_records(first)
            self._materialize_records(second)
            with self.assertRaisesRegex(RuntimeError, "prompt overlap"):
                merge_datasets(
                    [("primary", first), ("reserve", second)],
                    splits=["train"],
                    require_scores=False,
                    verify_hashes=False,
                )

    @staticmethod
    def _shared_inputs(root: Path) -> tuple[Path, Path]:
        protocol_path = root / "protocol.json"
        template_path = root / "template.json"
        protocol_path.write_text(json.dumps(protocol()), encoding="utf-8")
        template_path.write_text(
            json.dumps(
                {
                    "infer_steps": 50,
                    "target_video_length": 81,
                    "target_height": 720,
                    "target_width": 1248,
                    "feature_caching": "NoCaching",
                }
            ),
            encoding="utf-8",
        )
        return protocol_path, template_path

    @staticmethod
    def _prepare_shard(
        root: Path,
        prompts: list[str],
        protocol_path: Path,
        template_path: Path,
        model_root: Path,
    ) -> Path:
        prompt_path = root.parent / f"{root.name}_prompts.txt"
        prompt_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        prepare(
            SimpleNamespace(
                protocol=str(protocol_path),
                prompts=str(prompt_path),
                template_config=str(template_path),
                model_root=str(model_root),
                out_root=str(root),
                job_chunk_size=1,
                worker_count=8,
            )
        )
        return root

    @staticmethod
    def _materialize_records(root: Path) -> None:
        manifest = load_json(root / "generation_manifest.json")
        plan = load_json(manifest["plan_path"])
        for assignment in plan["assignments"]:
            artifacts = root / "fake_artifacts" / assignment["trajectory_key"]
            native = DatasetMergeTest._artifact(artifacts / "native.mp4", b"native")
            candidates = []
            for candidate in assignment["budget_candidates"]:
                artifact = DatasetMergeTest._artifact(
                    artifacts / f"{candidate['budget_id']}.mp4",
                    candidate["budget_id"].encode("ascii"),
                )
                candidates.append({**candidate, **artifact})
            record = {
                "schema": RECORD_SCHEMA,
                "generation_status": "generated_unscored",
                "plan_sha256": plan["plan_sha256"],
                "trajectory_key": assignment["trajectory_key"],
                "split": assignment["split"],
                "prompt_id": assignment["prompt_id"],
                "prompt": assignment["prompt"],
                "prompt_sha256": assignment["prompt_sha256"],
                "base_seed": assignment["base_seed"],
                "seed": assignment["seed"],
                "native_teacher": native,
                "budget_candidates": candidates,
                "provenance": {"manifest_sha256": manifest["manifest_sha256"]},
            }
            validate_trajectory_record(record, require_scores=False)
            write_json_atomic(
                root
                / "records"
                / assignment["split"]
                / f"{assignment['trajectory_key']}.json",
                record,
            )

    @staticmethod
    def _artifact(path: Path, prefix: bytes) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(prefix + b"x" * 1024)
        return {
            "video_path": str(path.resolve()),
            "video_sha256": sha256_file(path),
            "video_bytes": path.stat().st_size,
            "cost": {"pipeline_seconds": 1.0},
        }


if __name__ == "__main__":
    unittest.main()
