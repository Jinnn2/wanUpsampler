from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from UNIV_adaptor.data_protocol import RECORD_SCHEMA as BASE_RECORD_SCHEMA
from UNIV_adaptor.low_budget_protocol import (
    EXPECTED_DISPLAY_BUDGETS,
    PLAN_SCHEMA,
    build_plan,
    validate_plan,
    validate_protocol,
)
from UNIV_adaptor.scripts.data.run_low_budget_extension import (
    finalize,
    job_complete,
    prepare as prepare_extension,
    validate_manifest,
)
from UNIV_adaptor.scripts.data.run_prompt_budget_generation import (
    prepare as prepare_base,
)
from UNIV_adaptor.tests.test_data_protocol import protocol as base_protocol


REPO_ROOT = Path(__file__).resolve().parents[2]


def extension_protocol(*, small: bool = False) -> dict:
    path = REPO_ROOT / "UNIV_adaptor/configs/univ_low_budget_extension.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if small:
        value["splits"] = [
            {
                "name": "train",
                "prompt_count": 2,
                "base_seeds": [42],
                "collection_mode": "low_budget_extension",
            },
            {
                "name": "validation",
                "prompt_count": 1,
                "base_seeds": [42, 100],
                "collection_mode": "low_budget_extension",
            },
            {
                "name": "test",
                "prompt_count": 1,
                "base_seeds": [42, 100],
                "collection_mode": "low_budget_extension",
            },
        ]
    return value


class LowBudgetProtocolTest(unittest.TestCase):
    def test_checked_in_protocol_resolves_four_actions_and_3600_candidates(self):
        protocol = extension_protocol()
        plan = build_plan(protocol, [f"prompt {index}" for index in range(500)])
        self.assertEqual(plan["schema"], PLAN_SCHEMA)
        self.assertEqual(len(plan["assignments"]), 900)
        self.assertEqual(
            sum(len(row["low_budget_candidates"]) for row in plan["assignments"]),
            3600,
        )
        candidates = plan["assignments"][0]["low_budget_candidates"]
        self.assertEqual(
            tuple(row["display_budget"] for row in candidates),
            EXPECTED_DISPLAY_BUDGETS,
        )
        self.assertEqual(
            [row["resolved_schedule"]["low_latent_shape"] for row in candidates],
            [
                [16, 11, 46, 78],
                [16, 12, 46, 78],
                [16, 17, 50, 86],
                [16, 17, 68, 118],
            ],
        )
        self.assertEqual(
            candidates[0]["planned_hr_schedule"]["sigmas"], [0.2, 0.1, 0.0]
        )
        self.assertTrue(
            all(len(row["planned_lr_schedule"]["sigmas"]) == 26 for row in candidates)
        )
        validate_plan(plan)

    def test_protocol_rejects_budget_name_collision_and_cost_drift(self):
        value = extension_protocol()
        value["budget_presets"][3]["artifact_id"] = "B30"
        with self.assertRaisesRegex(ValueError, "LB prefix"):
            validate_protocol(value)
        value = extension_protocol()
        value["budget_presets"][0]["action"]["hr_steps"] = 4
        with self.assertRaisesRegex(ValueError, "misses target"):
            validate_protocol(value)

    def test_plan_hash_detects_mutation(self):
        plan = build_plan(extension_protocol(small=True), ["a", "b", "c", "d"])
        changed = copy.deepcopy(plan)
        changed["assignments"][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_plan(changed)


class LowBudgetGenerationTest(unittest.TestCase):
    @staticmethod
    def _prepare_fixture(root: Path):
        prompts = root / "prompts.txt"
        prompts.write_text("a\nb\nc\nd\n", encoding="utf-8")
        template = root / "template.json"
        template.write_text(
            json.dumps(
                {
                    "infer_steps": 50,
                    "target_video_length": 81,
                    "target_height": 720,
                    "target_width": 1248,
                    "feature_caching": "NoCaching",
                    "sample_shift": 8.0,
                }
            ),
            encoding="utf-8",
        )
        base_protocol_path = root / "base_protocol.json"
        base_protocol_path.write_text(json.dumps(base_protocol()), encoding="utf-8")
        model_root = root / "model"
        base_root = root / "base"
        base_manifest = prepare_base(
            SimpleNamespace(
                protocol=str(base_protocol_path),
                prompts=str(prompts),
                template_config=str(template),
                model_root=str(model_root),
                out_root=str(base_root),
                job_chunk_size=1,
                worker_count=8,
            )
        )
        extension_path = root / "extension_protocol.json"
        extension_path.write_text(
            json.dumps(extension_protocol(small=True)), encoding="utf-8"
        )
        out_root = root / "extension"
        extension_manifest = prepare_extension(
            SimpleNamespace(
                protocol=str(extension_path),
                prompts=str(prompts),
                template_config=str(template),
                model_root=str(model_root),
                base_dataset_root=str(base_root),
                out_root=str(out_root),
                job_chunk_size=1,
                worker_count=8,
            )
        )
        return base_root, base_manifest, out_root, extension_manifest

    def test_prepare_binds_base_dataset_and_builds_four_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            base_root, base_manifest, _, manifest = self._prepare_fixture(
                Path(directory)
            )
            validate_manifest(manifest)
            self.assertEqual(len(manifest["cases"]), 4)
            self.assertEqual(len(manifest["jobs"]), 24)
            self.assertEqual(
                manifest["base_dataset"]["plan_sha256"],
                base_manifest["plan_sha256"],
            )
            self.assertEqual(
                Path(manifest["base_dataset"]["root"]), base_root.resolve()
            )
            config = json.loads(
                Path(manifest["cases"][0]["config_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(config["univ_mrflow_lr_steps"], 25)
            self.assertFalse(config["univ_mrflow_reuse_endpoint"])
            self.assertEqual(config["univ_mrflow_endpoint_state_dtype"], "fp16")

    def test_finalize_joins_five_base_and_four_low_budget_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_root, base_manifest, out_root, manifest = self._prepare_fixture(root)
            base_plan = json.loads(
                Path(base_manifest["plan_path"]).read_text(encoding="utf-8")
            )
            for assignment in base_plan["assignments"]:
                if assignment["split"] != "train":
                    continue
                artifact = {
                    "video_path": "/data/base.mp4",
                    "video_sha256": "d" * 64,
                    "cost": {"pipeline_seconds": 1.0},
                }
                record = {
                    "schema": BASE_RECORD_SCHEMA,
                    "generation_status": "generated_unscored",
                    "plan_sha256": base_plan["plan_sha256"],
                    **{
                        key: assignment[key]
                        for key in (
                            "trajectory_key",
                            "split",
                            "prompt_id",
                            "prompt",
                            "prompt_sha256",
                            "base_seed",
                            "seed",
                        )
                    },
                    "native_teacher": artifact,
                    "budget_candidates": [
                        {**candidate, **artifact}
                        for candidate in assignment["budget_candidates"]
                    ],
                    "provenance": {"fixture": True},
                }
                path = (
                    base_root
                    / "records"
                    / "train"
                    / f"{assignment['trajectory_key']}.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record), encoding="utf-8")

            for job in manifest["jobs"]:
                if job["split"] != "train":
                    continue
                index = job["prompt_offset"]
                seed = job["base_seed"] + index
                output_dir = Path(job["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                output = output_dir / f"{job['case_id']}_{index:02d}_seed{seed}.mp4"
                output.write_bytes(b"v" * 1024)
                endpoint = output.with_suffix(output.suffix + ".endpoint.pt")
                endpoint.write_bytes(b"s" * 1024)
                sidecar = output.with_suffix(output.suffix + ".univ.json")
                sidecar.write_text(
                    json.dumps(
                        {
                            "schema": "wan_univ_mrflow_ablation_v1",
                            "seed": seed,
                            "artifact_id": job["case_id"],
                            "endpoint_state": {
                                "schema": "univ_mrflow_clean_transition_v1",
                                "path": str(endpoint),
                                "seed": seed,
                                "clean_lr_sha256": "a" * 64,
                                "clean_hr_sha256": "b" * 64,
                                "hr_noise_sha256": "c" * 64,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                timing = Path(job["timing_path"])
                timing.parent.mkdir(parents=True, exist_ok=True)
                timing.write_text(
                    "\n".join(
                        json.dumps(row)
                        for row in (
                            {"kind": "initialization"},
                            {
                                "kind": "video",
                                "prompt_index": index,
                                "seed": seed,
                                "output": str(output),
                                "pipeline_elapsed_s": 2.0,
                                "segment_elapsed_s": 1.5,
                            },
                        )
                    )
                    + "\n",
                    encoding="utf-8",
                )

            finalize(
                SimpleNamespace(
                    manifest=str(out_root / "extension_manifest.json"),
                    out_root=str(out_root),
                    splits=["train"],
                )
            )
            combined = sorted((out_root / "combined_records/train").glob("*.json"))
            self.assertEqual(len(combined), 2)
            record = json.loads(combined[0].read_text(encoding="utf-8"))
            self.assertEqual(record["candidate_count"], 9)
            self.assertEqual(len(record["budget_candidates"]), 9)
            self.assertEqual(
                [row["artifact_id"] for row in record["budget_candidates"][-4:]],
                [
                    "LB10_LR25_S0200_HR02",
                    "LB15_LR25_S0300_HR04",
                    "LB20_LR25_S0300_HR04",
                    "LB30_LR25_S0300_HR04",
                ],
            )
            self.assertEqual(
                record["budget_candidates"][-1]["endpoint_state"]["seed"],
                record["seed"],
            )

    def test_job_completion_requires_endpoint_with_matching_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "videos"
            output_dir.mkdir()
            output = output_dir / "LB10_00_seed42.mp4"
            output.write_bytes(b"x" * 1024)
            endpoint = output.with_suffix(output.suffix + ".endpoint.pt")
            endpoint.write_bytes(b"s" * 1024)
            sidecar = output.with_suffix(output.suffix + ".univ.json")
            sidecar.write_text(
                json.dumps(
                    {
                        "schema": "wan_univ_mrflow_ablation_v1",
                        "seed": 42,
                        "artifact_id": "LB10",
                        "endpoint_state": {
                            "schema": "univ_mrflow_clean_transition_v1",
                            "path": str(endpoint),
                            "seed": 42,
                            "clean_lr_sha256": "a" * 64,
                            "clean_hr_sha256": "b" * 64,
                            "hr_noise_sha256": "c" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            timing = root / "timing.jsonl"
            rows = [
                {"kind": "initialization"},
                {
                    "kind": "video",
                    "prompt_index": 0,
                    "seed": 42,
                    "output": str(output),
                },
            ]
            timing.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            job = {
                "timing_path": str(timing),
                "prompt_count": 1,
                "prompt_offset": 0,
                "base_seed": 42,
                "output_dir": str(output_dir),
                "case_id": "LB10",
            }
            self.assertTrue(job_complete(job))
            runtime = json.loads(sidecar.read_text(encoding="utf-8"))
            runtime["endpoint_state"]["seed"] = 43
            sidecar.write_text(json.dumps(runtime), encoding="utf-8")
            self.assertFalse(job_complete(job))


if __name__ == "__main__":
    unittest.main()
