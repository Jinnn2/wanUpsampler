"""Reuse targeted S videos and generate a faster temporal calibration arm."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.core import UniversalAction  # noqa: E402
from UNIV_adaptor.data_protocol import (  # noqa: E402
    CandidateAction,
    canonical_sha256,
    proxy_compute_density,
    sha256_file,
    write_json_atomic,
)
from UNIV_adaptor.scripts.data.run_prompt_budget_generation import (  # noqa: E402
    MANIFEST_SCHEMA,
    artifact_payload,
    base_runtime_config,
    build_jobs,
    generate_job,
    job_complete,
    load_json,
    materialize_immutable_inputs,
    rebalance_jobs,
    selected_jobs,
    source_identity,
    validate_manifest,
    validate_template,
)
from UNIV_adaptor.scripts.data.run_targeted_st_contrast_generation import (  # noqa: E402
    DATASET_SCHEMA,
    RECORD_SCHEMA,
)
from UNIV_adaptor.scripts.data.score_targeted_st_contrast import collect  # noqa: E402


PROTOCOL_SCHEMA = "univ_targeted_st_temporal_calibration_protocol_v1"
CALIBRATION_PLAN_SCHEMA = "univ_targeted_st_temporal_calibration_plan_v1"


def validate_protocol(value: dict[str, Any]) -> dict[str, Any]:
    protocol = json.loads(json.dumps(value))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported targeted temporal calibration protocol")
    if int(protocol.get("reference_nfe", 0)) != 50:
        raise ValueError("temporal calibration requires reference_nfe=50")
    shape = tuple(int(value) for value in protocol.get("target_latent_shape", []))
    if len(shape) != 4:
        raise ValueError("target_latent_shape must contain four integers")
    protocol["target_latent_shape"] = list(shape)
    source_case = str(protocol.get("source_spatial_case_id", "")).strip()
    case = protocol.get("temporal_case")
    if not source_case or not isinstance(case, dict):
        raise ValueError("calibration protocol lacks source spatial or temporal case")
    case_id = str(case.get("id", "")).strip()
    if not case_id or case_id == source_case or case.get("axis") != "temporal":
        raise ValueError("calibrated temporal case id/axis is invalid")
    raw = case.get("action")
    if not isinstance(raw, dict):
        raise ValueError("calibrated temporal action must be an object")
    action = UniversalAction(
        spatial_ratio=float(raw["spatial_ratio"]),
        temporal_ratio=float(raw["temporal_ratio"]),
        lr_nfe_ratio=float(raw["lr_nfe_ratio"]),
        switch_ratio=float(raw["switch_ratio"]),
    )
    action.validate()
    if (
        action.spatial_ratio != 1.0
        or action.lr_nfe_ratio != 1.0
        or action.temporal_ratio != 0.36
        or action.switch_ratio != 0.8
    ):
        raise ValueError(
            "calibration action must be the locked temporal-only rt=0.36 arm"
        )
    case["id"] = case_id
    case["action"] = {
        "spatial_ratio": action.spatial_ratio,
        "temporal_ratio": action.temporal_ratio,
        "lr_nfe_ratio": action.lr_nfe_ratio,
        "switch_ratio": action.switch_ratio,
    }
    protocol["source_spatial_case_id"] = source_case
    return protocol


def load_source(source_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    rows, identity = collect(source_root.resolve())
    spatial_rows = [
        row for row in rows if row["case_id"] == protocol["source_spatial_case_id"]
    ]
    if len(spatial_rows) * 2 != len(rows):
        raise ValueError("source dataset lacks one reusable spatial row per group")
    prompt_indices = sorted({row["prompt_index"] for row in spatial_rows})
    base_seeds = sorted({row["base_seed"] for row in spatial_rows})
    if prompt_indices != list(range(8)) or base_seeds != [42, 100, 2024]:
        raise ValueError("source dataset is not the locked 8-prompt/3-seed contrast")
    if len(spatial_rows) != len(prompt_indices) * len(base_seeds):
        raise ValueError("source spatial coverage is incomplete")
    manifest = load_json(source_root / "generation_manifest.json")
    prompts_path = Path(manifest["prompts_file"])
    if (
        not prompts_path.is_file()
        or sha256_file(prompts_path) != manifest["prompts_file_sha256"]
    ):
        raise ValueError("source prompt file is missing or changed")
    return {
        "rows": sorted(spatial_rows, key=lambda row: row["group_id"]),
        "identity": identity,
        "prompt_indices": prompt_indices,
        "base_seeds": base_seeds,
        "prompts_path": prompts_path.resolve(),
        "prompts_sha256": manifest["prompts_file_sha256"],
    }


def build_plan(protocol: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    action = UniversalAction(**protocol["temporal_case"]["action"])
    temporal = CandidateAction(
        action=action,
        transition=protocol["transition"],
        proxy_density=proxy_compute_density(action),
    ).as_dict(
        reference_nfe=protocol["reference_nfe"],
        target_latent_shape=tuple(protocol["target_latent_shape"]),
    )
    temporal = {
        "case_id": protocol["temporal_case"]["id"],
        "axis": "temporal",
        **temporal,
    }
    groups = []
    for row in source["rows"]:
        expected = (
            temporal["case_id"]
            if row["prompt_group"] == "low_motion_high_detail_small_objects"
            else protocol["source_spatial_case_id"]
        )
        groups.append(
            {
                "group_id": row["group_id"],
                "prompt_index": row["prompt_index"],
                "prompt_group": row["prompt_group"],
                "expected_preference": expected,
                "prompt": row["prompt"],
                "prompt_sha256": row["prompt_sha256"],
                "base_seed": row["base_seed"],
                "seed": row["seed"],
                "spatial_reuse": {
                    "case_id": row["case_id"],
                    "axis": "spatial",
                    "proxy_compute_density": row["proxy_compute_density"],
                    "requested_action": row["requested_action"],
                    "resolved_schedule": row["resolved_schedule"],
                    "transition": row["transition"],
                    "artifact": {
                        "video_path": row["video_path"],
                        "video_sha256": row["video_sha256"],
                        "video_bytes": Path(row["video_path"]).stat().st_size,
                        "cost": {"pipeline_seconds": row["pipeline_seconds"]},
                    },
                },
                "temporal_case_id": temporal["case_id"],
            }
        )
    immutable = json.loads(json.dumps(protocol))
    body = {
        "protocol": immutable,
        "protocol_sha256": canonical_sha256(immutable),
        "source_dataset": source["identity"],
        "temporal_case": temporal,
        "groups": groups,
        "counts": {
            "prompts": len(source["prompt_indices"]),
            "prompt_seed_groups": len(groups),
            "generated_videos": len(groups),
            "reused_videos": len(groups),
            "videos": 2 * len(groups),
        },
    }
    return {
        "schema": CALIBRATION_PLAN_SCHEMA,
        "plan_sha256": canonical_sha256(body),
        **body,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    protocol = validate_protocol(load_json(args.protocol))
    source = load_source(Path(args.source_root), protocol)
    template = load_json(args.template_config)
    validate_template(template, protocol)
    plan = build_plan(protocol, source)
    out_root = Path(args.out_root).resolve()
    plan_path = out_root / "targeted_st_plan.json"
    manifest_path = out_root / "generation_manifest.json"
    descriptor = plan["temporal_case"]
    config = {
        **base_runtime_config(template),
        "univ_action": descriptor["requested_action"],
        "univ_cache_mode": "residual",
        "univ_transition_baseline": descriptor["transition"],
        "univ_enable_transition_diagnostics": False,
        "univ_native_hr_state_path": "",
        "univ_native_hr_state_key": "state",
    }
    config_path = out_root / "configs" / f"{descriptor['case_id']}.json"
    cases = [
        {
            "case_id": descriptor["case_id"],
            "kind": "budget",
            "model_cls": "wan2.1_univ_pipeline",
            "config_path": str(config_path),
            "config_sha256": canonical_sha256(config),
            "expected_weight": float(descriptor["proxy_compute_density"]),
        }
    ]
    job_protocol = {
        "splits": [
            {
                "name": "development",
                "prompt_count": len(source["prompt_indices"]),
                "base_seeds": source["base_seeds"],
            }
        ]
    }
    jobs = build_jobs(
        job_protocol,
        cases,
        out_root=out_root,
        chunk_size=args.job_chunk_size,
        worker_count=8,
    )
    body = {
        "protocol_sha256": plan["protocol_sha256"],
        "preset_status": "locked_targeted_st_temporal_calibration_v1",
        "plan_sha256": plan["plan_sha256"],
        "plan_path": str(plan_path),
        "prompts_file": str(source["prompts_path"]),
        "prompts_file_sha256": source["prompts_sha256"],
        "template_config": str(Path(args.template_config).resolve()),
        "template_config_sha256": sha256_file(args.template_config),
        "model_root": str(Path(args.model_root).resolve()),
        "source": source_identity(),
        "source_targeted_dataset": source["identity"],
        "calibration_driver_sha256": sha256_file(Path(__file__).resolve()),
        "job_chunk_size": args.job_chunk_size,
        "worker_count": 8,
        "cases": cases,
        "jobs": jobs,
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_sha256": canonical_sha256(body),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        **body,
    }
    if manifest_path.is_file():
        previous = validate_manifest(load_json(manifest_path))
        if previous["manifest_sha256"] != manifest["manifest_sha256"]:
            raise RuntimeError("calibration protocol changed; use a new OUT_ROOT")
        manifest = previous
    materialize_immutable_inputs(
        plan_path=plan_path,
        plan=plan,
        prepared_configs={str(config_path): config},
    )
    if not manifest_path.is_file():
        write_json_atomic(manifest_path, manifest)
    print(json.dumps({**plan["counts"], "jobs": len(jobs)}, indent=2))
    return manifest


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    manifest = validate_manifest(load_json(args.manifest))
    jobs = selected_jobs(manifest, ["development"])
    incomplete = [job["job_id"] for job in jobs if not job_complete(manifest, job)]
    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} calibration jobs are incomplete: {incomplete[:8]}"
        )
    timing = {}
    for job in jobs:
        for line in Path(job["timing_path"]).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("kind") == "video":
                key = (int(row["prompt_index"]), int(row["seed"]))
                if key in timing:
                    raise RuntimeError(f"duplicate calibration timing row: {key}")
                timing[key] = row
    plan = load_json(manifest["plan_path"])
    if plan.get("schema") != CALIBRATION_PLAN_SCHEMA:
        raise ValueError("unsupported calibration plan")
    out_root = Path(args.out_root).resolve()
    records = []
    temporal = plan["temporal_case"]
    for group in plan["groups"]:
        generated = artifact_payload(timing[(group["prompt_index"], group["seed"])])
        body = {
            **{
                key: group[key]
                for key in (
                    "group_id",
                    "prompt_index",
                    "prompt_group",
                    "expected_preference",
                    "prompt",
                    "prompt_sha256",
                    "base_seed",
                    "seed",
                )
            },
            "plan_sha256": plan["plan_sha256"],
            "artifacts": [
                group["spatial_reuse"],
                {
                    "case_id": temporal["case_id"],
                    "axis": "temporal",
                    "action_key": temporal["action_key"],
                    "proxy_compute_density": temporal["proxy_compute_density"],
                    "requested_action": temporal["requested_action"],
                    "resolved_schedule": temporal["resolved_schedule"],
                    "transition": temporal["transition"],
                    "artifact": generated,
                },
            ],
        }
        record = {
            "schema": RECORD_SCHEMA,
            "record_sha256": canonical_sha256(body),
            **body,
        }
        path = out_root / "records" / "development" / f"{group['group_id']}.json"
        if path.is_file() and load_json(path) != record:
            raise RuntimeError(
                f"refusing to replace changed calibration record: {path}"
            )
        if not path.is_file():
            write_json_atomic(path, record)
        records.append(
            {
                "group_id": group["group_id"],
                "record_path": str(path),
                "record_file_sha256": sha256_file(path),
                "record_sha256": record["record_sha256"],
            }
        )
    body = {
        "plan_path": str(Path(manifest["plan_path"]).resolve()),
        "plan_file_sha256": sha256_file(manifest["plan_path"]),
        "plan_sha256": plan["plan_sha256"],
        "generation_manifest_path": str(Path(args.manifest).resolve()),
        "generation_manifest_file_sha256": sha256_file(args.manifest),
        "generation_manifest_sha256": manifest["manifest_sha256"],
        "counts": plan["counts"],
        "records": records,
    }
    dataset = {
        "schema": DATASET_SCHEMA,
        "dataset_sha256": canonical_sha256(body),
        **body,
    }
    path = out_root / "targeted_st_dataset.json"
    if path.is_file() and load_json(path) != dataset:
        raise RuntimeError(f"refusing to replace changed calibration dataset: {path}")
    if not path.is_file():
        write_json_atomic(path, dataset)
    print(
        f"Finalized {plan['counts']['generated_videos']} generated + "
        f"{plan['counts']['reused_videos']} reused videos: {path}"
    )
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare"):
        target = sub.add_parser(name)
        target.add_argument("--source-root", required=True)
        target.add_argument("--protocol", required=True)
        target.add_argument("--template-config", required=True)
        target.add_argument("--model-root", required=True)
        if name == "prepare":
            target.add_argument("--out-root", required=True)
            target.add_argument("--job-chunk-size", type=int, default=2)
    list_parser = sub.add_parser("list-jobs")
    list_parser.add_argument("--manifest", required=True)
    list_parser.add_argument("--worker-slot", type=int)
    list_parser.add_argument("--limit", type=int, default=0)
    job_parser = sub.add_parser("generate-job")
    job_parser.add_argument("--manifest", required=True)
    job_parser.add_argument("--job-id", required=True)
    job_parser.add_argument("--wan-python", required=True)
    job_parser.add_argument("--lightx2v-repo", required=True)
    job_parser.add_argument("--realesrgan-repo", default="")
    job_parser.add_argument("--negative-prompt", default="")
    job_parser.add_argument("--resume", action="store_true")
    finalize_parser = sub.add_parser("finalize")
    finalize_parser.add_argument("--manifest", required=True)
    finalize_parser.add_argument("--out-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "check":
        protocol = validate_protocol(load_json(args.protocol))
        source = load_source(Path(args.source_root), protocol)
        validate_template(load_json(args.template_config), protocol)
        print(json.dumps(build_plan(protocol, source)["counts"], indent=2))
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "list-jobs":
        manifest = validate_manifest(load_json(args.manifest))
        jobs = rebalance_jobs(selected_jobs(manifest, ["development"]), worker_count=8)
        if args.worker_slot is not None:
            jobs = [job for job in jobs if job["worker_slot"] == args.worker_slot]
        if args.limit > 0:
            jobs = jobs[: args.limit]
        for job in jobs:
            print(job["job_id"])
    elif args.command == "generate-job":
        generate_job(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
