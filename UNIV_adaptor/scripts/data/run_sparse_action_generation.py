from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.data_protocol import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    validate_collection_plan,
    validate_trajectory_record,
)
from UNIV_adaptor.sparse_action_protocol import (  # noqa: E402
    PLAN_SCHEMA,
    RECORD_SCHEMA,
    action_catalog,
    assign_probe_ids,
    design_balance,
    expected_counts,
    validate_sparse_plan,
    validate_sparse_protocol,
    validate_sparse_record,
    write_json_atomic,
)
from UNIV_adaptor.scripts.data.run_prompt_budget_generation import (  # noqa: E402
    base_runtime_config,
    validate_manifest as validate_phase2_manifest,
    validate_template,
)


MANIFEST_SCHEMA = "univ_sparse_action_generation_manifest_v1"
DATASET_SCHEMA = "univ_sparse_action_dataset_manifest_v1"


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_prompts(
    path: str | Path, *, offset: int = 0, limit: int | None = None
) -> list[str]:
    prompts = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        raise ValueError("fresh prompt source must be non-empty")
    if offset < 0 or (limit is not None and limit < 1):
        raise ValueError("fresh prompt offset/limit must be non-negative/positive")
    selected = prompts[offset:] if limit is None else prompts[offset : offset + limit]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("selected fresh prompts must be non-empty and unique")
    return selected


def select_existing_train(
    protocol: dict[str, Any],
    source_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    source_manifest = validate_phase2_manifest(
        load_json(source_root / "generation_manifest.json")
    )
    source_plan = load_json(source_root / "collection_plan.json")
    validate_collection_plan(source_plan)
    for key in ("plan_sha256", "protocol_sha256"):
        if source_manifest[key] != source_plan[key]:
            raise ValueError(f"Phase2 source manifest/plan mismatch: {key}")

    reuse_seed = protocol["reuse_reference_base_seed"]
    candidates = [
        row
        for row in source_plan["assignments"]
        if row["split"] == "train" and int(row["base_seed"]) == reuse_seed
    ]
    if len(candidates) < protocol["existing_train_prompt_count"]:
        raise ValueError("Phase2 source does not contain enough train prompts")
    ranked = sorted(
        candidates,
        key=lambda row: canonical_sha256(
            [protocol["selection_salt"], row["prompt_sha256"]]
        ),
    )
    selected = ranked[: protocol["existing_train_prompt_count"]]
    result = []
    catalog = action_catalog(protocol)
    expected_reference = protocol["reference_action"]
    source_action_id = protocol["reference_source_action_id"]
    for assignment in selected:
        record_path = (
            source_root / "records" / "train" / f"{assignment['trajectory_key']}.json"
        )
        record = load_json(record_path)
        validate_trajectory_record(
            record,
            expected_plan_sha256=source_plan["plan_sha256"],
            require_scores=False,
        )
        for key in (
            "trajectory_key",
            "split",
            "prompt_id",
            "prompt",
            "prompt_sha256",
            "base_seed",
            "seed",
        ):
            if record[key] != assignment[key]:
                raise ValueError(
                    f"Phase2 source record identity mismatch: {record_path}: {key}"
                )
        matches = [
            row
            for row in record["budget_candidates"]
            if row["budget_id"] == source_action_id
        ]
        if len(matches) != 1:
            raise ValueError(f"missing Phase2 reference action in {record_path}")
        source_candidate = matches[0]
        if source_candidate["requested_action"] != expected_reference:
            raise ValueError("Phase2 reference action differs from sparse protocol")
        reusable_actions = {}
        for action_id, action in catalog.items():
            action_matches = [
                candidate
                for candidate in record["budget_candidates"]
                if candidate["requested_action"] == action["requested_action"]
                and candidate["transition"] == action["transition"]
            ]
            if len(action_matches) > 1:
                raise ValueError(
                    f"multiple Phase2 candidates match sparse action {action_id}"
                )
            if not action_matches:
                continue
            matched = action_matches[0]
            artifact = copy_artifact(matched)
            verify_artifact(artifact)
            reusable_actions[action_id] = {
                "artifact": artifact,
                "source_action_id": matched["budget_id"],
                "source_record_path": str(record_path.resolve()),
                "source_record_sha256": sha256_file(record_path),
                "source_trajectory_key": assignment["trajectory_key"],
            }
        if "REFERENCE" not in reusable_actions:
            raise ValueError("Phase2 reference action was not reusable")
        result.append(
            {
                "prompt_key": f"existing_p{int(assignment['prompt_id']):06d}",
                "cohort": "existing_train",
                "prompt": assignment["prompt"],
                "prompt_sha256": assignment["prompt_sha256"],
                "seed_offset": int(assignment["prompt_id"]),
                "source_prompt_id": int(assignment["prompt_id"]),
                "reusable_actions": reusable_actions,
            }
        )
    source_identity = {
        "root": str(source_root.resolve()),
        "generation_manifest_sha256": source_manifest["manifest_sha256"],
        "generation_manifest_file_sha256": sha256_file(
            source_root / "generation_manifest.json"
        ),
        "plan_sha256": source_plan["plan_sha256"],
        "plan_file_sha256": sha256_file(source_root / "collection_plan.json"),
        "record_splits_accessed": ["train"],
        "prompt_overlap_scopes": ["train", "validation"],
        "scores_accessed": False,
        "selection": "prompt_hash_rank_only",
    }
    reserved_prompt_hashes = {
        row["prompt_sha256"]
        for row in source_plan["assignments"]
        if row["split"] in {"train", "validation"}
    }
    return result, source_identity, reserved_prompt_hashes


def build_plan(
    protocol_value: dict[str, Any],
    *,
    source_root: Path,
    fresh_prompts_path: Path,
    fresh_prompt_offset: int = 0,
) -> dict[str, Any]:
    protocol = validate_sparse_protocol(protocol_value)
    existing, source_identity, reserved_prompt_hashes = select_existing_train(
        protocol, source_root
    )
    required_fresh = protocol["fresh_development_prompt_count"]
    fresh_prompts = load_prompts(
        fresh_prompts_path,
        offset=fresh_prompt_offset,
        limit=required_fresh,
    )
    if len(fresh_prompts) != required_fresh:
        raise ValueError(
            f"fresh prompt file must contain exactly {required_fresh} prompts, "
            f"got {len(fresh_prompts)}"
        )
    overlap = reserved_prompt_hashes & {
        canonical_sha256(prompt) for prompt in fresh_prompts
    }
    if overlap:
        raise ValueError("fresh prompts overlap Phase2 train/validation prompts")
    fresh = [
        {
            "prompt_key": f"fresh_p{index:06d}",
            "cohort": "fresh_development",
            "prompt": prompt,
            "prompt_sha256": canonical_sha256(prompt),
            "seed_offset": protocol["fresh_seed_offset"] + index,
            "source_prompt_id": None,
            "reusable_actions": {},
        }
        for index, prompt in enumerate(fresh_prompts)
    ]
    prompts = existing + fresh
    prompt_keys = [row["prompt_key"] for row in prompts]
    probe_assignments = assign_probe_ids(protocol, prompt_keys)
    catalog = action_catalog(protocol)
    groups = []
    for prompt in prompts:
        probe_ids = probe_assignments[prompt["prompt_key"]]
        for base_seed in protocol["base_seeds"]:
            actual_seed = int(base_seed) + int(prompt["seed_offset"])
            group_id = f"{prompt['prompt_key']}_b{base_seed}"
            actions = []
            for action_id in ["REFERENCE", *probe_ids]:
                observation_id = f"{group_id}__{action_id}"
                reuse = (
                    prompt["cohort"] == "existing_train"
                    and base_seed == protocol["reuse_reference_base_seed"]
                    and action_id in prompt["reusable_actions"]
                )
                action_row = {
                    "observation_id": observation_id,
                    "action_id": action_id,
                    "artifact_mode": "reuse" if reuse else "generate",
                }
                if reuse:
                    action_row.update(prompt["reusable_actions"][action_id])
                actions.append(action_row)
            groups.append(
                {
                    "group_id": group_id,
                    "prompt_key": prompt["prompt_key"],
                    "cohort": prompt["cohort"],
                    "prompt": prompt["prompt"],
                    "prompt_sha256": prompt["prompt_sha256"],
                    "source_prompt_id": prompt["source_prompt_id"],
                    "seed_offset": prompt["seed_offset"],
                    "base_seed": base_seed,
                    "seed": actual_seed,
                    "actions": actions,
                }
            )
    immutable_protocol = json.loads(json.dumps(protocol))
    generated_count = sum(
        row["artifact_mode"] == "generate"
        for group in groups
        for row in group["actions"]
    )
    reused_count = sum(
        row["artifact_mode"] == "reuse" for group in groups for row in group["actions"]
    )
    nominal = expected_counts(protocol)
    actual_counts = {
        "existing_prompts": protocol["existing_train_prompt_count"],
        "fresh_prompts": protocol["fresh_development_prompt_count"],
        "generated_videos": generated_count,
        "reused_videos": reused_count,
        "scored_videos": generated_count + reused_count,
        "prompt_seed_groups": len(groups),
        "generated_videos_upper_bound": nominal["generated_videos_upper_bound"],
    }
    body = {
        "protocol_sha256": canonical_sha256(immutable_protocol),
        "protocol": immutable_protocol,
        "source_phase2": source_identity,
        "fresh_prompts_file": str(fresh_prompts_path.resolve()),
        "fresh_prompts_file_sha256": sha256_file(fresh_prompts_path),
        "fresh_prompt_offset": fresh_prompt_offset,
        "fresh_prompt_count": required_fresh,
        "action_catalog": catalog,
        "design_balance": design_balance(protocol, probe_assignments),
        "counts": actual_counts,
        "groups": groups,
    }
    plan = {"schema": PLAN_SCHEMA, "plan_sha256": canonical_sha256(body), **body}
    validate_sparse_plan(plan)
    return plan


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    protocol = validate_sparse_protocol(load_json(args.protocol))
    template = load_json(args.template_config)
    validate_template(template, protocol)
    out_root = Path(args.out_root).resolve()
    plan = build_plan(
        protocol,
        source_root=Path(args.source_phase2_root).resolve(),
        fresh_prompts_path=Path(args.fresh_prompts).resolve(),
        fresh_prompt_offset=int(getattr(args, "fresh_prompt_offset", 0)),
    )
    plan_path = out_root / "sparse_action_plan.json"
    catalog = plan["action_catalog"]
    base = base_runtime_config(template)
    cases = []
    prepared_configs = {}
    for action_id, action in catalog.items():
        config = dict(base)
        config.update(
            {
                "univ_action": action["requested_action"],
                "univ_cache_mode": "residual",
                "univ_transition_baseline": action["transition"],
                "univ_enable_transition_diagnostics": False,
                "univ_native_hr_state_path": "",
                "univ_native_hr_state_key": "state",
            }
        )
        config_path = out_root / "configs" / f"{action_id}.json"
        prepared_configs[str(config_path)] = config
        cases.append(
            {
                "action_id": action_id,
                "config_path": str(config_path),
                "config_sha256": canonical_sha256(config),
                "model_cls": "wan2.1_univ_pipeline",
                "expected_weight": max(0.2, float(action["proxy_compute_density"])),
            }
        )
    jobs, job_inputs = build_jobs(
        plan,
        cases,
        out_root=out_root,
        chunk_size=args.job_chunk_size,
        worker_count=args.worker_count,
    )
    source = source_identity()
    body = {
        "plan_path": str(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "protocol_sha256": plan["protocol_sha256"],
        "template_config": str(Path(args.template_config).resolve()),
        "template_config_sha256": sha256_file(args.template_config),
        "model_root": str(Path(args.model_root).resolve()),
        "source": source,
        "worker_count": args.worker_count,
        "job_chunk_size": args.job_chunk_size,
        "cases": cases,
        "jobs": jobs,
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_sha256": canonical_sha256(body),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        **body,
    }
    manifest_path = out_root / "generation_manifest.json"
    if manifest_path.is_file():
        previous = validate_manifest(load_json(manifest_path))
        if previous["manifest_sha256"] != manifest["manifest_sha256"]:
            raise RuntimeError(
                f"generation protocol changed under {out_root}; use a new OUT_ROOT"
            )
        manifest = previous
    materialize_immutable(plan_path, plan, prepared_configs, job_inputs)
    if not manifest_path.is_file():
        write_json_atomic(manifest_path, manifest)
    print_summary(manifest, plan)
    return manifest


def build_jobs(
    plan: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    out_root: Path,
    chunk_size: int,
    worker_count: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    case_map = {case["action_id"]: case for case in cases}
    generated: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in plan["groups"]:
        for row in group["actions"]:
            if row["artifact_mode"] != "generate":
                continue
            observation_id = row["observation_id"]
            output = out_root / "videos" / row["action_id"] / f"{observation_id}.mp4"
            generated[row["action_id"]].append(
                {
                    "observation_id": observation_id,
                    "prompt_key": group["prompt_key"],
                    "prompt": group["prompt"],
                    "prompt_sha256": group["prompt_sha256"],
                    "base_seed": group["base_seed"],
                    "seed": group["seed"],
                    "action_id": row["action_id"],
                    "output": str(output),
                }
            )
    raw_jobs = []
    inputs = {}
    for action_id in sorted(generated):
        rows = sorted(generated[action_id], key=lambda item: item["observation_id"])
        case = case_map[action_id]
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            job_id = f"{action_id}_n{start:04d}_{start + len(chunk) - 1:04d}"
            input_path = out_root / "job_inputs" / f"{job_id}.json"
            input_payload = {
                "schema": "univ_sparse_action_job_input_v1",
                "job_id": job_id,
                "action_id": action_id,
                "rows": chunk,
            }
            input_sha256 = canonical_sha256(input_payload)
            inputs[str(input_path)] = input_payload
            raw_jobs.append(
                {
                    "job_id": job_id,
                    "action_id": action_id,
                    "model_cls": case["model_cls"],
                    "config_path": case["config_path"],
                    "config_sha256": case["config_sha256"],
                    "input_path": str(input_path),
                    "input_sha256": input_sha256,
                    "timing_path": str(out_root / "timings" / f"{job_id}.jsonl"),
                    "observation_count": len(chunk),
                    "expected_weight": len(chunk) * float(case["expected_weight"]),
                }
            )
    loads = [0.0] * worker_count
    for job in sorted(
        raw_jobs, key=lambda item: (-item["expected_weight"], item["job_id"])
    ):
        slot = min(range(worker_count), key=loads.__getitem__)
        job["worker_slot"] = slot
        loads[slot] += float(job["expected_weight"])
    return sorted(raw_jobs, key=lambda item: item["job_id"]), inputs


def materialize_immutable(
    plan_path: Path,
    plan: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    job_inputs: dict[str, dict[str, Any]],
) -> None:
    payloads = {str(plan_path): plan, **configs, **job_inputs}
    for path_text, payload in payloads.items():
        path = Path(path_text)
        if path.is_file():
            if load_json(path) != payload:
                raise RuntimeError(f"immutable generated input changed: {path}")
        else:
            write_json_atomic(path, payload)


def source_identity() -> dict[str, Any]:
    paths = (
        REPO_ROOT / "UNIV_adaptor/sparse_action_protocol.py",
        REPO_ROOT / "UNIV_adaptor/scripts/data/run_sparse_action_generation.py",
        REPO_ROOT / "UNIV_adaptor/scripts/bridge/run_wan_univ_sparse_batch.py",
        REPO_ROOT / "UNIV_adaptor/wan_runner.py",
        REPO_ROOT / "UNIV_adaptor/transition.py",
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "implementation_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths
        },
    }


def validate_manifest(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported sparse generation manifest: {value.get('schema')}"
        )
    body = {
        key: item
        for key, item in value.items()
        if key not in {"schema", "manifest_sha256", "created_at_utc"}
    }
    if canonical_sha256(body) != value.get("manifest_sha256"):
        raise ValueError("sparse generation manifest hash mismatch")
    return value


def selected_jobs(
    manifest: dict[str, Any], *, worker_slot: int | None = None
) -> list[dict[str, Any]]:
    jobs = manifest["jobs"]
    if worker_slot is not None:
        jobs = [job for job in jobs if int(job["worker_slot"]) == worker_slot]
    return jobs


def job_complete(job: dict[str, Any]) -> bool:
    timing_path = Path(job["timing_path"])
    input_path = Path(job["input_path"])
    if not timing_path.is_file() or not input_path.is_file():
        return False
    if canonical_sha256(load_json(input_path)) != job["input_sha256"]:
        return False
    expected = {row["observation_id"]: row for row in load_json(input_path)["rows"]}
    try:
        timing_rows = [
            json.loads(line)
            for line in timing_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    initialization = [row for row in timing_rows if row.get("kind") == "initialization"]
    videos = [row for row in timing_rows if row.get("kind") == "video"]
    if len(initialization) != 1 or len(videos) != len(expected):
        return False
    if {row.get("observation_id") for row in videos} != set(expected):
        return False
    for row in videos:
        planned = expected[row["observation_id"]]
        output = Path(row["output"]).resolve()
        if (
            int(row["seed"]) != int(planned["seed"])
            or output != Path(planned["output"]).resolve()
            or not output.is_file()
            or output.stat().st_size < 1024
        ):
            return False
        sidecar = output.with_suffix(output.suffix + ".univ.json")
        if not sidecar.is_file() or sidecar.stat().st_size == 0:
            return False
    return True


def generate_job(args: argparse.Namespace) -> None:
    manifest = validate_manifest(load_json(args.manifest))
    jobs = {job["job_id"]: job for job in manifest["jobs"]}
    if args.job_id not in jobs:
        raise ValueError(f"job is absent from manifest: {args.job_id}")
    job = jobs[args.job_id]
    if args.resume and job_complete(job):
        print(f"[resume] {job['job_id']}")
        return
    if canonical_sha256(load_json(job["input_path"])) != job["input_sha256"]:
        raise RuntimeError(f"job input changed: {job['input_path']}")
    if canonical_sha256(load_json(job["config_path"])) != job["config_sha256"]:
        raise RuntimeError(f"job config changed: {job['config_path']}")
    environment = dict(os.environ)
    environment["LIGHTX2V_REPO"] = str(Path(args.lightx2v_repo).resolve())
    python_roots = [environment["LIGHTX2V_REPO"], str(REPO_ROOT)]
    if args.realesrgan_repo:
        python_roots.insert(1, str(Path(args.realesrgan_repo).resolve()))
    if environment.get("PYTHONPATH"):
        python_roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_roots)
    command = [
        args.wan_python,
        str(REPO_ROOT / "UNIV_adaptor/scripts/bridge/run_wan_univ_sparse_batch.py"),
        "--job-json",
        job["input_path"],
        "--model_cls",
        job["model_cls"],
        "--model_path",
        manifest["model_root"],
        "--config_json",
        job["config_path"],
        "--timing-jsonl",
        job["timing_path"],
        "--target_video_length",
        "81",
        "--negative_prompt",
        args.negative_prompt,
    ]
    print(f"[generate] {job['job_id']}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    if not job_complete(job):
        raise RuntimeError(f"job finished without complete artifacts: {job['job_id']}")


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    manifest = validate_manifest(load_json(args.manifest))
    plan = validate_sparse_plan(load_json(manifest["plan_path"]))
    incomplete = [job["job_id"] for job in manifest["jobs"] if not job_complete(job)]
    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} sparse generation jobs are incomplete: {incomplete[:10]}"
        )
    timing = {}
    for job in manifest["jobs"]:
        for line in Path(job["timing_path"]).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("kind") != "video":
                continue
            observation_id = row["observation_id"]
            if observation_id in timing:
                raise RuntimeError(f"duplicate timing observation: {observation_id}")
            timing[observation_id] = row

    out_root = Path(args.out_root).resolve()
    record_index = []
    generated = reused = 0
    for group in plan["groups"]:
        record_actions = []
        for planned in group["actions"]:
            action = plan["action_catalog"][planned["action_id"]]
            if planned["artifact_mode"] == "reuse":
                artifact = planned["artifact"]
                verify_artifact(artifact)
                reused += 1
            else:
                artifact = artifact_from_timing(timing[planned["observation_id"]])
                generated += 1
            record_actions.append(
                {
                    "observation_id": planned["observation_id"],
                    "action_id": planned["action_id"],
                    "artifact_mode": planned["artifact_mode"],
                    "action": action,
                    "artifact": artifact,
                    "reuse_provenance": {
                        key: planned[key]
                        for key in (
                            "source_record_path",
                            "source_record_sha256",
                            "source_trajectory_key",
                            "source_action_id",
                        )
                        if key in planned
                    },
                }
            )
        record = {
            "schema": RECORD_SCHEMA,
            "generation_status": "generated_unscored",
            "plan_sha256": plan["plan_sha256"],
            "group_id": group["group_id"],
            "prompt_key": group["prompt_key"],
            "cohort": group["cohort"],
            "prompt": group["prompt"],
            "prompt_sha256": group["prompt_sha256"],
            "source_prompt_id": group["source_prompt_id"],
            "base_seed": group["base_seed"],
            "seed": group["seed"],
            "actions": record_actions,
            "provenance": {
                "generation_manifest_sha256": manifest["manifest_sha256"],
                "protocol_sha256": plan["protocol_sha256"],
                "scores_accessed": False,
                "lambda_bound": False,
            },
        }
        validate_sparse_record(record, expected_plan_sha256=plan["plan_sha256"])
        record_path = (
            out_root / "records" / group["cohort"] / f"{group['group_id']}.json"
        )
        if record_path.is_file() and load_json(record_path) != record:
            raise RuntimeError(
                f"refusing to replace different sparse record: {record_path}"
            )
        if not record_path.is_file():
            write_json_atomic(record_path, record)
        record_index.append(
            {
                "group_id": group["group_id"],
                "path": str(record_path),
                "file_sha256": sha256_file(record_path),
            }
        )
    counts = plan["counts"]
    if generated != counts["generated_videos"] or reused != counts["reused_videos"]:
        raise RuntimeError("finalized generated/reused counts differ from sparse plan")
    body = {
        "plan_path": manifest["plan_path"],
        "plan_file_sha256": sha256_file(manifest["plan_path"]),
        "plan_sha256": plan["plan_sha256"],
        "generation_manifest_path": str(Path(args.manifest).resolve()),
        "generation_manifest_file_sha256": sha256_file(args.manifest),
        "generation_manifest_sha256": manifest["manifest_sha256"],
        "protocol_sha256": plan["protocol_sha256"],
        "counts": counts,
        "action_catalog": plan["action_catalog"],
        "design_balance": plan["design_balance"],
        "records": record_index,
    }
    dataset = {
        "schema": DATASET_SCHEMA,
        "dataset_sha256": canonical_sha256(body),
        **body,
    }
    dataset_path = out_root / "sparse_dataset_manifest.json"
    if dataset_path.is_file() and load_json(dataset_path) != dataset:
        raise RuntimeError(
            f"refusing to replace different dataset manifest: {dataset_path}"
        )
    if not dataset_path.is_file():
        write_json_atomic(dataset_path, dataset)
    print(
        f"Finalized {len(record_index)} prompt-seed records: "
        f"generated={generated}, reused={reused}, scored inputs={generated + reused}"
    )
    print(dataset_path)
    return dataset


def validate_dataset_manifest(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"unsupported sparse dataset manifest: {value.get('schema')}")
    body = {
        key: item
        for key, item in value.items()
        if key not in {"schema", "dataset_sha256"}
    }
    if canonical_sha256(body) != value.get("dataset_sha256"):
        raise ValueError("sparse dataset manifest hash mismatch")
    return value


def artifact_from_timing(row: dict[str, Any]) -> dict[str, Any]:
    video = Path(row["output"]).resolve()
    artifact = {
        "video_path": str(video),
        "video_sha256": sha256_file(video),
        "video_bytes": video.stat().st_size,
        "cost": {
            "pipeline_seconds": float(row["pipeline_elapsed_s"]),
            "segment_seconds": float(row["segment_elapsed_s"]),
            "peak_allocated_gib": float(row.get("peak_allocated_gib", 0.0)),
        },
    }
    sidecar = video.with_suffix(video.suffix + ".univ.json")
    artifact["runtime_sidecar_path"] = str(sidecar)
    artifact["runtime_sidecar_sha256"] = sha256_file(sidecar)
    return artifact


def copy_artifact(source: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "video_path",
        "video_sha256",
        "video_bytes",
        "cost",
        "runtime_sidecar_path",
        "runtime_sidecar_sha256",
    )
    return {key: json.loads(json.dumps(source[key])) for key in fields if key in source}


def verify_artifact(artifact: dict[str, Any]) -> None:
    video = Path(artifact["video_path"]).resolve()
    if not video.is_file() or video.stat().st_size != int(artifact["video_bytes"]):
        raise ValueError(f"missing or changed video: {video}")
    if sha256_file(video) != artifact["video_sha256"]:
        raise ValueError(f"video SHA256 mismatch: {video}")
    sidecar_path = artifact.get("runtime_sidecar_path")
    sidecar_hash = artifact.get("runtime_sidecar_sha256")
    if bool(sidecar_path) != bool(sidecar_hash):
        raise ValueError("artifact has incomplete sidecar identity")
    if sidecar_path and sha256_file(sidecar_path) != sidecar_hash:
        raise ValueError(f"runtime sidecar SHA256 mismatch: {sidecar_path}")


def print_summary(manifest: dict[str, Any], plan: dict[str, Any]) -> None:
    worker_loads = [0.0] * int(manifest["worker_count"])
    for job in manifest["jobs"]:
        worker_loads[int(job["worker_slot"])] += float(job["expected_weight"])
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "plan_sha256": plan["plan_sha256"],
                "counts": plan["counts"],
                "job_count": len(manifest["jobs"]),
                "estimated_worker_loads": worker_loads,
                "design_balance": plan["design_balance"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sparse prompt-action observations without per-prompt enumeration"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--protocol", required=True)
    prepare_parser.add_argument("--source-phase2-root", required=True)
    prepare_parser.add_argument("--fresh-prompts", required=True)
    prepare_parser.add_argument("--fresh-prompt-offset", type=int, default=0)
    prepare_parser.add_argument("--template-config", required=True)
    prepare_parser.add_argument("--model-root", required=True)
    prepare_parser.add_argument("--out-root", required=True)
    prepare_parser.add_argument("--job-chunk-size", type=int, default=32)
    prepare_parser.add_argument("--worker-count", type=int, default=8)

    list_parser = subparsers.add_parser("list-jobs")
    list_parser.add_argument("--manifest", required=True)
    list_parser.add_argument("--worker-slot", type=int)
    list_parser.add_argument("--limit", type=int, default=0)

    generate_parser = subparsers.add_parser("generate-job")
    generate_parser.add_argument("--manifest", required=True)
    generate_parser.add_argument("--job-id", required=True)
    generate_parser.add_argument("--wan-python", required=True)
    generate_parser.add_argument("--lightx2v-repo", required=True)
    generate_parser.add_argument("--realesrgan-repo", default="")
    generate_parser.add_argument("--negative-prompt", default="")
    generate_parser.add_argument("--resume", action="store_true")

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--manifest", required=True)
    finalize_parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    if args.command == "prepare" and (
        args.job_chunk_size < 1 or args.worker_count != 8
    ):
        parser.error("job-chunk-size must be positive and worker-count must be 8")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "list-jobs":
        manifest = validate_manifest(load_json(args.manifest))
        jobs = selected_jobs(manifest, worker_slot=args.worker_slot)
        if args.limit > 0:
            jobs = jobs[: args.limit]
        for job in jobs:
            print(job["job_id"])
    elif args.command == "generate-job":
        generate_job(args)
    elif args.command == "finalize":
        finalize(args)


if __name__ == "__main__":
    main()
