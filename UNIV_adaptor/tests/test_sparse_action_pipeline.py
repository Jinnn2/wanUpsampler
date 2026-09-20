from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from UNIV_adaptor.data_protocol import (
    RECORD_SCHEMA as PHASE2_RECORD_SCHEMA,
    build_collection_plan,
    canonical_sha256,
    sha256_file,
    write_json_atomic,
)
from UNIV_adaptor.scripts.data.run_prompt_budget_generation import (
    MANIFEST_SCHEMA as PHASE2_MANIFEST_SCHEMA,
)
from UNIV_adaptor.scripts.data.run_matched_star_generation import (
    build_plan as build_matched_star_plan,
    prepare as prepare_matched_star,
)
from UNIV_adaptor.scripts.data.run_sparse_action_generation import (
    finalize,
    load_json,
    prepare,
    select_existing_train,
)
from UNIV_adaptor.scripts.data.score_sparse_action_dataset import (
    DIMENSIONS,
    collect,
    report,
)
from UNIV_adaptor.sparse_action_protocol import (
    assign_probe_ids,
    design_balance,
    expected_counts,
    validate_sparse_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def small_protocol() -> dict:
    value = json.loads(
        (REPO_ROOT / "UNIV_adaptor/configs/univ_sparse_action_phase3.json").read_text(
            encoding="utf-8"
        )
    )
    value["existing_train_prompt_count"] = 2
    value["fresh_development_prompt_count"] = 1
    return value


def make_artifact(path: Path, token: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((token.encode("utf-8") + b"_") * 1024)
    return {
        "video_path": str(path.resolve()),
        "video_sha256": sha256_file(path),
        "video_bytes": path.stat().st_size,
        "cost": {
            "pipeline_seconds": 10.0,
            "segment_seconds": 9.0,
            "peak_allocated_gib": 1.0,
        },
    }


def make_phase2_source(root: Path) -> Path:
    source = root / "phase2"
    protocol = load_json(
        REPO_ROOT / "UNIV_adaptor/configs/univ_prompt_budget_phase2.json"
    )
    protocol["splits"] = [
        {
            "name": "train",
            "prompt_count": 2,
            "base_seeds": [42],
            "collection_mode": "full_budget_curve",
        },
        {
            "name": "validation",
            "prompt_count": 1,
            "base_seeds": [42, 100, 2024],
            "collection_mode": "full_budget_curve",
        },
        {
            "name": "test",
            "prompt_count": 1,
            "base_seeds": [42],
            "collection_mode": "full_budget_curve",
        },
    ]
    plan = build_collection_plan(
        protocol,
        ["source train zero", "source train one", "source validation", "source test"],
    )
    write_json_atomic(source / "collection_plan.json", plan)
    manifest_body = {
        "plan_sha256": plan["plan_sha256"],
        "protocol_sha256": plan["protocol_sha256"],
        "test_fixture": True,
    }
    manifest = {
        "schema": PHASE2_MANIFEST_SCHEMA,
        "manifest_sha256": canonical_sha256(manifest_body),
        "created_at_utc": "2026-09-19T00:00:00+00:00",
        **manifest_body,
    }
    write_json_atomic(source / "generation_manifest.json", manifest)
    for assignment in plan["assignments"]:
        if assignment["split"] != "train":
            continue
        native = make_artifact(
            source / "videos" / f"{assignment['trajectory_key']}__native.mp4",
            f"{assignment['trajectory_key']}_native",
        )
        candidates = []
        for candidate in assignment["budget_candidates"]:
            artifact = make_artifact(
                source
                / "videos"
                / f"{assignment['trajectory_key']}__{candidate['budget_id']}.mp4",
                f"{assignment['trajectory_key']}_{candidate['budget_id']}",
            )
            candidates.append({**candidate, **artifact})
        record = {
            "schema": PHASE2_RECORD_SCHEMA,
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
            "provenance": {"test_fixture": True},
        }
        write_json_atomic(
            source / "records" / "train" / f"{assignment['trajectory_key']}.json",
            record,
        )
    return source


def complete_sparse_jobs(manifest: dict) -> None:
    for job in manifest["jobs"]:
        job_input = load_json(job["input_path"])
        timing_rows = [
            {
                "kind": "initialization",
                "job_id": job["job_id"],
                "elapsed_s": 1.0,
            }
        ]
        for row in job_input["rows"]:
            output = Path(row["output"])
            artifact = make_artifact(output, row["observation_id"])
            sidecar = output.with_suffix(output.suffix + ".univ.json")
            sidecar.write_text("{}\n", encoding="utf-8")
            timing_rows.append(
                {
                    "kind": "video",
                    "observation_id": row["observation_id"],
                    "seed": row["seed"],
                    "output": str(output.resolve()),
                    "pipeline_elapsed_s": artifact["cost"]["pipeline_seconds"],
                    "segment_elapsed_s": artifact["cost"]["segment_seconds"],
                    "peak_allocated_gib": 1.0,
                }
            )
        timing_path = Path(job["timing_path"])
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(
            "".join(json.dumps(row) + "\n" for row in timing_rows),
            encoding="utf-8",
        )


class SparseActionPipelineTest(unittest.TestCase):
    def test_checked_in_protocol_has_exact_non_exhaustive_budget(self):
        protocol = validate_sparse_protocol(
            load_json(REPO_ROOT / "UNIV_adaptor/configs/univ_sparse_action_phase3.json")
        )
        self.assertEqual(
            expected_counts(protocol),
            {
                "existing_prompts": 69,
                "fresh_prompts": 20,
                "generated_videos_upper_bound": 999,
                "reused_videos_lower_bound": 69,
                "scored_videos": 1068,
                "prompt_seed_groups": 267,
            },
        )
        assignments = assign_probe_ids(
            protocol, [f"prompt_{index}" for index in range(89)]
        )
        self.assertTrue(all(len(value) == 3 for value in assignments.values()))
        balance = design_balance(protocol, assignments)
        self.assertLessEqual(
            max(balance["action_prompt_counts"].values())
            - min(balance["action_prompt_counts"].values()),
            1,
        )
        for counts in balance["dimension_marginal_counts"].values():
            self.assertLessEqual(abs(counts["0"] - counts["1"]), 1)
        for counts in balance["pairwise_level_counts"].values():
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_prepare_finalize_collect_and_relative_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_phase2_source(root)
            protocol_path = root / "sparse_protocol.json"
            write_json_atomic(protocol_path, small_protocol())
            existing, source_identity, _ = select_existing_train(
                validate_sparse_protocol(small_protocol()), source
            )
            self.assertFalse(source_identity["scores_accessed"])
            self.assertTrue(
                all("SA_001" in row["reusable_actions"] for row in existing)
            )
            fresh_path = root / "fresh.txt"
            fresh_path.write_text(
                "one genuinely fresh development prompt\n", encoding="utf-8"
            )
            template_path = root / "template.json"
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
            out = root / "sparse_output"
            args = SimpleNamespace(
                protocol=str(protocol_path),
                source_phase2_root=str(source),
                fresh_prompts=str(fresh_path),
                template_config=str(template_path),
                model_root=str(root / "model"),
                out_root=str(out),
                job_chunk_size=2,
                worker_count=8,
            )
            manifest = prepare(args)
            plan = load_json(out / "sparse_action_plan.json")
            self.assertLessEqual(plan["counts"]["generated_videos"], 34)
            self.assertGreaterEqual(plan["counts"]["reused_videos"], 2)
            self.assertEqual(plan["counts"]["scored_videos"], 36)
            self.assertTrue(plan["source_phase2"]["scores_accessed"] is False)
            self.assertEqual(
                {group["cohort"] for group in plan["groups"]},
                {"existing_train", "fresh_development"},
            )

            complete_sparse_jobs(manifest)

            dataset = finalize(
                SimpleNamespace(
                    manifest=str(out / "generation_manifest.json"),
                    out_root=str(out),
                )
            )
            self.assertEqual(dataset["counts"]["scored_videos"], 36)
            rows, identity = collect(out)
            self.assertEqual(len(rows), 36)
            self.assertEqual(len({row["group_id"] for row in rows}), 9)
            self.assertGreaterEqual(
                sum(row["artifact_mode"] == "reuse" for row in rows), 2
            )

            scores = {
                row["stem"]: {dimension: 0.8 for dimension in DIMENSIONS}
                for row in rows
            }
            score_body = {
                "input_sha256": canonical_sha256(identity),
                "dimensions": list(DIMENSIONS),
                "scores": scores,
                "provenance": {dimension: {"test": True} for dimension in DIMENSIONS},
            }
            payload = {
                "schema": "univ_sparse_action_vbench_scores_v1",
                "payload_sha256": canonical_sha256(score_body),
                **score_body,
            }
            metrics = out / "metrics_test"
            metrics.mkdir()
            scored = report(rows, payload, metrics, score_body["input_sha256"])
            self.assertFalse(scored["lambda_bound"])
            self.assertFalse(scored["hard_oracle_labels_created"])
            self.assertEqual(scored["relative_pair_count"], 27)
            self.assertTrue((metrics / "relative_quality_pairs.csv").is_file())

    def test_matched_star_reuses_exact_phase3_actions_and_adds_train_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_phase2_source(root)
            phase3_protocol_path = root / "phase3_protocol.json"
            write_json_atomic(phase3_protocol_path, small_protocol())
            fresh_path = root / "phase3_fresh.txt"
            fresh_path.write_text("phase3 fresh prompt\n", encoding="utf-8")
            template_path = root / "template.json"
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
            phase3_out = root / "phase3"
            manifest = prepare(
                SimpleNamespace(
                    protocol=str(phase3_protocol_path),
                    source_phase2_root=str(source),
                    fresh_prompts=str(fresh_path),
                    fresh_prompt_offset=0,
                    template_config=str(template_path),
                    model_root=str(root / "model"),
                    out_root=str(phase3_out),
                    job_chunk_size=4,
                    worker_count=8,
                )
            )
            complete_sparse_jobs(manifest)
            finalize(
                SimpleNamespace(
                    manifest=str(phase3_out / "generation_manifest.json"),
                    out_root=str(phase3_out),
                )
            )

            star_protocol = load_json(
                REPO_ROOT / "UNIV_adaptor/configs/univ_matched_star_phase4.json"
            )
            star_protocol["source_sparse_prompt_count"] = 3
            star_protocol["new_train_prompt_count"] = 1
            star_protocol["existing_train_prompt_count"] = 3
            star_protocol["fresh_development_prompt_count"] = 1
            new_prompts = root / "new_prompts.txt"
            new_prompts.write_text("one new matched star prompt\n", encoding="utf-8")
            plan = build_matched_star_plan(
                star_protocol,
                source_root=phase3_out,
                prompts_path=new_prompts,
                prompt_offset=0,
            )
            self.assertEqual(plan["counts"]["prompt_seed_groups"], 12)
            self.assertEqual(plan["counts"]["scored_videos"], 48)
            self.assertEqual(plan["counts"]["reused_videos"], 12)
            self.assertEqual(plan["counts"]["generated_videos"], 36)
            source_groups = [
                group
                for group in plan["groups"]
                if not group["prompt_key"].startswith("star_train_")
            ]
            self.assertTrue(
                all(
                    next(
                        row
                        for row in group["actions"]
                        if row["action_id"] == "REFERENCE"
                    )["artifact_mode"]
                    == "reuse"
                    for group in source_groups
                )
            )
            self.assertTrue(
                all(
                    next(
                        row
                        for row in group["actions"]
                        if row["action_id"] in {"STAR_S", "STAR_T"}
                    )["artifact_mode"]
                    == "generate"
                    for group in source_groups
                )
            )
            catalog = plan["action_catalog"]
            self.assertEqual(
                catalog["STAR_S"]["requested_action"]["lr_nfe_ratio"], 0.55
            )
            self.assertEqual(
                catalog["STAR_T"]["requested_action"]["lr_nfe_ratio"], 0.55
            )

            star_protocol_path = root / "star_protocol.json"
            write_json_atomic(star_protocol_path, star_protocol)
            star_out = root / "star_output"
            star_manifest = prepare_matched_star(
                SimpleNamespace(
                    protocol=str(star_protocol_path),
                    source_phase3_root=str(phase3_out),
                    new_prompts=str(new_prompts),
                    new_prompt_offset=0,
                    template_config=str(template_path),
                    model_root=str(root / "model"),
                    out_root=str(star_out),
                    job_chunk_size=4,
                    worker_count=8,
                )
            )
            complete_sparse_jobs(star_manifest)
            star_dataset = finalize(
                SimpleNamespace(
                    manifest=str(star_out / "generation_manifest.json"),
                    out_root=str(star_out),
                )
            )
            self.assertEqual(star_dataset["counts"]["generated_videos"], 36)
            self.assertEqual(star_dataset["counts"]["reused_videos"], 12)
            star_rows, _ = collect(star_out)
            self.assertEqual(len(star_rows), 48)
            self.assertEqual(
                {row["action_id"] for row in star_rows},
                {"REFERENCE", "STAR_S", "STAR_T", "STAR_C"},
            )


if __name__ == "__main__":
    unittest.main()
