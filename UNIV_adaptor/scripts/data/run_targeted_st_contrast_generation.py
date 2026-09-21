"""Generate the targeted equal-density spatial-versus-temporal contrast."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
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


PROTOCOL_SCHEMA = "univ_targeted_st_contrast_protocol_v1"
PLAN_SCHEMA = "univ_targeted_st_contrast_plan_v1"
RECORD_SCHEMA = "univ_targeted_st_contrast_record_v1"
DATASET_SCHEMA = "univ_targeted_st_contrast_dataset_v1"


def load_prompts(path: str | Path) -> list[str]:
    prompts = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts or len(prompts) != len(set(prompts)):
        raise ValueError("targeted prompts must be non-empty and unique")
    return prompts


def validate_protocol(value: dict[str, Any], *, prompt_count: int) -> dict[str, Any]:
    protocol = json.loads(json.dumps(value))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported targeted S/T protocol schema")
    if int(protocol.get("reference_nfe", 0)) != 50:
        raise ValueError("targeted S/T protocol requires reference_nfe=50")
    shape = tuple(int(value) for value in protocol.get("target_latent_shape", []))
    if len(shape) != 4:
        raise ValueError("target_latent_shape must contain four integers")
    protocol["target_latent_shape"] = list(shape)
    seeds = [int(seed) for seed in protocol.get("base_seeds", [])]
    if len(seeds) != 3 or len(seeds) != len(set(seeds)):
        raise ValueError("targeted S/T protocol requires three unique base seeds")
    protocol["base_seeds"] = seeds
    target_density = float(protocol.get("target_proxy_compute_density", 0.0))
    if not 0.0 < target_density < 1.0:
        raise ValueError("target proxy density must be in (0, 1)")
    protocol["target_proxy_compute_density"] = target_density

    group_by_index: dict[int, str] = {}
    expected_by_group: dict[str, str] = {}
    for group in protocol.get("prompt_groups", []):
        group_id = str(group.get("id", "")).strip()
        expected = str(group.get("expected_preference", "")).strip()
        if not group_id or not expected:
            raise ValueError("every prompt group requires id and expected_preference")
        expected_by_group[group_id] = expected
        for index in group.get("prompt_indices", []):
            index = int(index)
            if index in group_by_index:
                raise ValueError(f"prompt index belongs to multiple groups: {index}")
            group_by_index[index] = group_id
    if set(group_by_index) != set(range(prompt_count)):
        raise ValueError("prompt groups must partition every prompt index exactly once")

    cases = protocol.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError("targeted S/T protocol requires exactly two cases")
    case_ids = []
    axes = []
    for case in cases:
        case_id = str(case.get("id", "")).strip()
        axis = str(case.get("axis", "")).strip()
        action = UniversalAction(
            **{key: float(raw) for key, raw in case.get("action", {}).items()}
        )
        action.validate()
        density = proxy_compute_density(action)
        if not math.isclose(density, target_density, abs_tol=1e-12):
            raise ValueError(
                f"{case_id} proxy density {density} differs from target {target_density}"
            )
        case["id"] = case_id
        case["axis"] = axis
        case["action"] = {
            "spatial_ratio": action.spatial_ratio,
            "temporal_ratio": action.temporal_ratio,
            "lr_nfe_ratio": action.lr_nfe_ratio,
            "switch_ratio": action.switch_ratio,
        }
        case_ids.append(case_id)
        axes.append(axis)
    if len(case_ids) != len(set(case_ids)) or set(axes) != {"spatial", "temporal"}:
        raise ValueError("cases must be unique spatial and temporal interventions")
    if set(expected_by_group.values()) - set(case_ids):
        raise ValueError("prompt group expects an unknown case")
    protocol["prompt_group_by_index"] = {
        str(index): group_by_index[index] for index in range(prompt_count)
    }
    return protocol


def build_plan(protocol: dict[str, Any], prompts: list[str]) -> dict[str, Any]:
    cases = {}
    for case in protocol["cases"]:
        action = UniversalAction(**case["action"])
        candidate = CandidateAction(
            action=action,
            transition=protocol["transition"],
            proxy_density=proxy_compute_density(action),
        ).as_dict(
            reference_nfe=protocol["reference_nfe"],
            target_latent_shape=tuple(protocol["target_latent_shape"]),
        )
        cases[case["id"]] = {"axis": case["axis"], **candidate}
    groups = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_sha256 = canonical_sha256(prompt)
        prompt_group = protocol["prompt_group_by_index"][str(prompt_index)]
        expected = next(
            group["expected_preference"]
            for group in protocol["prompt_groups"]
            if group["id"] == prompt_group
        )
        for base_seed in protocol["base_seeds"]:
            seed = base_seed + prompt_index
            groups.append(
                {
                    "group_id": f"targeted_p{prompt_index:03d}_b{base_seed}",
                    "prompt_index": prompt_index,
                    "prompt_group": prompt_group,
                    "expected_preference": expected,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha256,
                    "base_seed": base_seed,
                    "seed": seed,
                    "case_ids": list(cases),
                }
            )
    immutable = json.loads(json.dumps(protocol))
    body = {
        "protocol": immutable,
        "protocol_sha256": canonical_sha256(immutable),
        "prompts_sha256": canonical_sha256(prompts),
        "cases": cases,
        "groups": groups,
        "counts": {
            "prompts": len(prompts),
            "prompt_seed_groups": len(groups),
            "cases_per_group": len(cases),
            "videos": len(groups) * len(cases),
        },
    }
    return {"schema": PLAN_SCHEMA, "plan_sha256": canonical_sha256(body), **body}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompts(args.prompts)
    protocol = validate_protocol(load_json(args.protocol), prompt_count=len(prompts))
    template = load_json(args.template_config)
    validate_template(template, protocol)
    plan = build_plan(protocol, prompts)
    out_root = Path(args.out_root).resolve()
    plan_path = out_root / "targeted_st_plan.json"
    manifest_path = out_root / "generation_manifest.json"
    base = base_runtime_config(template)
    prepared_configs = {}
    cases = []
    for case_id, descriptor in plan["cases"].items():
        config = {
            **base,
            "univ_action": descriptor["requested_action"],
            "univ_cache_mode": "residual",
            "univ_transition_baseline": descriptor["transition"],
            "univ_enable_transition_diagnostics": False,
            "univ_native_hr_state_path": "",
            "univ_native_hr_state_key": "state",
        }
        config_path = out_root / "configs" / f"{case_id}.json"
        prepared_configs[str(config_path)] = config
        cases.append(
            {
                "case_id": case_id,
                "kind": "budget",
                "model_cls": "wan2.1_univ_pipeline",
                "config_path": str(config_path),
                "config_sha256": canonical_sha256(config),
                "expected_weight": float(descriptor["proxy_compute_density"]),
            }
        )
    job_protocol = {
        "splits": [
            {
                "name": "development",
                "prompt_count": len(prompts),
                "base_seeds": protocol["base_seeds"],
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
        "preset_status": "locked_targeted_st_contrast_v1",
        "plan_sha256": plan["plan_sha256"],
        "plan_path": str(plan_path),
        "prompts_file": str(Path(args.prompts).resolve()),
        "prompts_file_sha256": sha256_file(args.prompts),
        "template_config": str(Path(args.template_config).resolve()),
        "template_config_sha256": sha256_file(args.template_config),
        "model_root": str(Path(args.model_root).resolve()),
        "source": source_identity(),
        "targeted_driver_sha256": sha256_file(Path(__file__).resolve()),
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
            raise RuntimeError("targeted protocol changed; use a new OUT_ROOT")
        manifest = previous
    materialize_immutable_inputs(
        plan_path=plan_path,
        plan=plan,
        prepared_configs=prepared_configs,
    )
    if not manifest_path.is_file():
        write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "plan_sha256": manifest["plan_sha256"],
                "prompts": len(prompts),
                "prompt_seed_groups": len(plan["groups"]),
                "videos": plan["counts"]["videos"],
                "jobs": len(jobs),
            },
            indent=2,
        )
    )
    return manifest


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    manifest = validate_manifest(load_json(args.manifest))
    jobs = selected_jobs(manifest, ["development"])
    incomplete = [job["job_id"] for job in jobs if not job_complete(manifest, job)]
    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} targeted jobs are incomplete: {incomplete[:8]}"
        )
    timing = {}
    for job in jobs:
        for line in Path(job["timing_path"]).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("kind") != "video":
                continue
            key = (job["case_id"], int(row["prompt_index"]), int(row["seed"]))
            if key in timing:
                raise RuntimeError(f"duplicate targeted timing row: {key}")
            timing[key] = row
    plan = load_json(manifest["plan_path"])
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("plan_sha256") != manifest["plan_sha256"]
    ):
        raise ValueError("targeted plan/manifest mismatch")
    out_root = Path(args.out_root).resolve()
    records = []
    for group in plan["groups"]:
        artifacts = []
        for case_id in group["case_ids"]:
            row = timing[(case_id, group["prompt_index"], group["seed"])]
            artifacts.append(
                {
                    "case_id": case_id,
                    **plan["cases"][case_id],
                    "artifact": artifact_payload(row),
                }
            )
        body = {
            **{key: value for key, value in group.items() if key != "case_ids"},
            "plan_sha256": plan["plan_sha256"],
            "artifacts": artifacts,
        }
        record = {
            "schema": RECORD_SCHEMA,
            "record_sha256": canonical_sha256(body),
            **body,
        }
        path = out_root / "records" / "development" / f"{group['group_id']}.json"
        if path.is_file() and load_json(path) != record:
            raise RuntimeError(f"refusing to replace changed targeted record: {path}")
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
        raise RuntimeError(f"refusing to replace changed targeted dataset: {path}")
    if not path.is_file():
        write_json_atomic(path, dataset)
    print(
        f"Finalized {len(records)} groups and {plan['counts']['videos']} videos: {path}"
    )
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    prepare_parser = sub.add_parser("prepare")
    for target in (check, prepare_parser):
        target.add_argument("--protocol", required=True)
        target.add_argument("--prompts", required=True)
        target.add_argument("--template-config", required=True)
        target.add_argument("--model-root", required=True)
    prepare_parser.add_argument("--out-root", required=True)
    prepare_parser.add_argument("--job-chunk-size", type=int, default=4)
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
    args = parser.parse_args()
    if args.command == "prepare" and args.job_chunk_size < 1:
        parser.error("job-chunk-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "check":
        prompts = load_prompts(args.prompts)
        protocol = validate_protocol(
            load_json(args.protocol), prompt_count=len(prompts)
        )
        validate_template(load_json(args.template_config), protocol)
        print(json.dumps(build_plan(protocol, prompts)["counts"], indent=2))
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
