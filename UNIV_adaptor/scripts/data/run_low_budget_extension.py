from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.data_protocol import (  # noqa: E402
    canonical_sha256,
    load_prompts,
    sha256_file,
    validate_collection_plan,
    validate_trajectory_record,
    write_json_atomic,
)
from UNIV_adaptor.low_budget_protocol import (  # noqa: E402
    COMBINED_RECORD_SCHEMA,
    RECORD_SCHEMA,
    build_plan,
    validate_plan,
    validate_protocol,
)
from UNIV_adaptor.scripts.data.run_prompt_budget_generation import (  # noqa: E402
    base_runtime_config,
    load_json,
    validate_manifest as validate_base_manifest,
    validate_template,
)


MANIFEST_SCHEMA = "univ_low_budget_extension_generation_manifest_v1"


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    protocol = validate_protocol(load_json(args.protocol))
    prompts = load_prompts(args.prompts)
    plan = build_plan(protocol, prompts)
    template = load_json(args.template_config)
    validate_template(template, protocol)
    if float(template.get("sample_shift", 0.0)) != protocol["sample_shift"]:
        raise ValueError("template sample_shift does not match extension protocol")
    out_root = Path(args.out_root).resolve()
    base_root = Path(args.base_dataset_root).resolve()
    if out_root == base_root:
        raise ValueError("extension OUT_ROOT must differ from the immutable base root")
    base_identity = validate_base_dataset(
        base_root,
        plan=plan,
        prompts_file=Path(args.prompts).resolve(),
        template_config=Path(args.template_config).resolve(),
        model_root=Path(args.model_root).resolve(),
    )

    plan_path = out_root / "extension_plan.json"
    manifest_path = out_root / "extension_manifest.json"
    base = base_runtime_config(template)
    cases = []
    prepared_configs: dict[str, dict[str, Any]] = {}
    endpoint_dtype = protocol["endpoint_state"]["archive_dtype"]
    for preset in protocol["budget_presets"]:
        action = preset["action"]
        artifact_id = preset["artifact_id"]
        candidate = next(
            row
            for row in plan["assignments"][0]["low_budget_candidates"]
            if row["artifact_id"] == artifact_id
        )
        config = dict(base)
        config.update(
            {
                "univ_action": {
                    "spatial_ratio": action["spatial_ratio"],
                    "temporal_ratio": action["temporal_ratio"],
                    "lr_nfe_ratio": 1.0,
                    "switch_ratio": 1.0,
                },
                "univ_cache_mode": "residual",
                "univ_transition_baseline": action["transition"],
                "univ_enable_transition_diagnostics": False,
                "univ_mrflow_lr_steps": action["true_lr_steps"],
                "univ_mrflow_refine_sigma": action["renoise_sigma"],
                "univ_mrflow_hr_steps": action["hr_steps"],
                "univ_mrflow_reuse_endpoint": False,
                "univ_mrflow_boundary_path": "",
                "univ_mrflow_endpoint_state_dtype": endpoint_dtype,
                "univ_low_budget_artifact_id": artifact_id,
                "univ_low_budget_action_key": candidate["action_key"],
            }
        )
        config_path = out_root / "configs" / f"{artifact_id}.json"
        prepared_configs[str(config_path)] = config
        cases.append(
            {
                "case_id": artifact_id,
                "display_budget": preset["display_budget"],
                "kind": "low_budget_endpoint",
                "model_cls": "wan2.1_univ_mrflow_budget",
                "config_path": str(config_path),
                "config_sha256": canonical_sha256(config),
                "endpoint_state_required": True,
                "expected_weight": max(0.1, float(candidate["proxy_compute_density"])),
            }
        )

    jobs = build_jobs(
        protocol,
        cases,
        out_root=out_root,
        chunk_size=args.job_chunk_size,
        worker_count=args.worker_count,
    )
    body = {
        "protocol_sha256": plan["protocol_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "plan_path": str(plan_path),
        "base_dataset": base_identity,
        "out_root": str(out_root),
        "prompts_file": str(Path(args.prompts).resolve()),
        "prompts_file_sha256": sha256_file(args.prompts),
        "template_config": str(Path(args.template_config).resolve()),
        "template_config_sha256": sha256_file(args.template_config),
        "model_root": str(Path(args.model_root).resolve()),
        "source": source_identity(),
        "job_chunk_size": args.job_chunk_size,
        "worker_count": args.worker_count,
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
            raise RuntimeError(
                f"low-budget extension changed under {out_root}; use a new OUT_ROOT"
            )
        manifest = previous
    materialize_inputs(plan_path, plan, prepared_configs)
    if not manifest_path.is_file():
        write_json_atomic(manifest_path, manifest)
    print_summary(manifest)
    return manifest


def validate_base_dataset(
    base_root: Path,
    *,
    plan: dict[str, Any],
    prompts_file: Path,
    template_config: Path,
    model_root: Path,
) -> dict[str, Any]:
    manifest_path = base_root / "generation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"base generation manifest is missing: {manifest_path}")
    manifest = validate_base_manifest(load_json(manifest_path))
    base_plan_path = Path(manifest["plan_path"])
    base_plan = load_json(base_plan_path)
    validate_collection_plan(base_plan)
    if sha256_file(prompts_file) != manifest["prompts_file_sha256"]:
        raise ValueError("extension prompt file differs from the base dataset")
    if sha256_file(template_config) != manifest["template_config_sha256"]:
        raise ValueError("extension template config differs from the base dataset")
    if Path(manifest["model_root"]).resolve() != model_root:
        raise ValueError("extension model root differs from the base dataset")
    base_protocol = base_plan["protocol"]
    extension_protocol = plan["protocol"]
    for field in ("reference_nfe", "target_latent_shape"):
        if base_protocol[field] != extension_protocol[field]:
            raise ValueError(f"extension {field} differs from the base dataset")
    base_assignments = {row["trajectory_key"]: row for row in base_plan["assignments"]}
    identity_fields = (
        "trajectory_key",
        "split",
        "prompt_id",
        "prompt",
        "prompt_sha256",
        "base_seed",
        "seed",
    )
    for assignment in plan["assignments"]:
        base_assignment = base_assignments.get(assignment["trajectory_key"])
        if base_assignment is None:
            raise ValueError(
                f"extension trajectory is absent from base plan: {assignment['trajectory_key']}"
            )
        if {key: assignment[key] for key in identity_fields} != {
            key: base_assignment[key] for key in identity_fields
        }:
            raise ValueError(
                f"extension identity differs from base: {assignment['trajectory_key']}"
            )
    return {
        "root": str(base_root),
        "generation_manifest_path": str(manifest_path),
        "generation_manifest_file_sha256": sha256_file(manifest_path),
        "generation_manifest_sha256": manifest["manifest_sha256"],
        "plan_path": str(base_plan_path),
        "plan_file_sha256": sha256_file(base_plan_path),
        "plan_sha256": base_plan["plan_sha256"],
    }


def materialize_inputs(
    plan_path: Path,
    plan: dict[str, Any],
    configs: dict[str, dict[str, Any]],
) -> None:
    if plan_path.is_file():
        if load_json(plan_path).get("plan_sha256") != plan["plan_sha256"]:
            raise RuntimeError(f"extension plan was modified: {plan_path}")
    else:
        write_json_atomic(plan_path, plan)
    for path_text, config in configs.items():
        path = Path(path_text)
        if path.is_file():
            if canonical_sha256(load_json(path)) != canonical_sha256(config):
                raise RuntimeError(f"extension config was modified: {path}")
        else:
            write_json_atomic(path, config)


def build_jobs(
    protocol: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    out_root: Path,
    chunk_size: int,
    worker_count: int,
) -> list[dict[str, Any]]:
    raw_jobs = []
    prompt_offset = 0
    for split in protocol["splits"]:
        split_start = prompt_offset
        split_stop = split_start + split["prompt_count"]
        for base_seed in split["base_seeds"]:
            for start in range(split_start, split_stop, chunk_size):
                count = min(chunk_size, split_stop - start)
                for case in cases:
                    raw_jobs.append(
                        {
                            "job_id": (
                                f"{split['name']}_{case['case_id']}_base{base_seed}_"
                                f"p{start:06d}_{start + count - 1:06d}"
                            ),
                            "split": split["name"],
                            "case_id": case["case_id"],
                            "display_budget": case["display_budget"],
                            "kind": case["kind"],
                            "model_cls": case["model_cls"],
                            "config_path": case["config_path"],
                            "config_sha256": case["config_sha256"],
                            "endpoint_state_required": True,
                            "base_seed": base_seed,
                            "prompt_offset": start,
                            "prompt_count": count,
                            "output_dir": str(
                                out_root / "videos" / split["name"] / case["case_id"]
                            ),
                            "timing_path": str(
                                out_root
                                / "timings"
                                / f"{split['name']}_{case['case_id']}_base{base_seed}_p{start:06d}_{start + count - 1:06d}.jsonl"
                            ),
                            "expected_weight": count * float(case["expected_weight"]),
                        }
                    )
        prompt_offset = split_stop
    worker_loads = [0.0] * worker_count
    for job in sorted(raw_jobs, key=lambda row: -row["expected_weight"]):
        slot = min(range(worker_count), key=worker_loads.__getitem__)
        job["worker_slot"] = slot
        worker_loads[slot] += float(job["expected_weight"])
    return sorted(raw_jobs, key=lambda row: row["job_id"])


def selected_jobs(
    manifest: dict[str, Any], splits: Iterable[str]
) -> list[dict[str, Any]]:
    requested = set(splits)
    available = {job["split"] for job in manifest["jobs"]}
    missing = requested - available
    if missing:
        raise ValueError(f"unknown splits: {sorted(missing)}")
    return [job for job in manifest["jobs"] if job["split"] in requested]


def endpoint_from_sidecar(
    sidecar: Path,
    *,
    expected_seed: int,
    expected_artifact_id: str | None = None,
) -> dict[str, Any] | None:
    try:
        runtime = load_json(sidecar)
        endpoint = runtime["endpoint_state"]
        path = Path(endpoint["path"]).resolve()
        digests = (
            endpoint.get("clean_lr_sha256"),
            endpoint.get("clean_hr_sha256"),
            endpoint.get("hr_noise_sha256"),
        )
        if (
            runtime.get("schema") != "wan_univ_mrflow_ablation_v1"
            or endpoint.get("schema") != "univ_mrflow_clean_transition_v1"
            or int(runtime["seed"]) != expected_seed
            or int(endpoint["seed"]) != expected_seed
            or (
                expected_artifact_id is not None
                and runtime.get("artifact_id") != expected_artifact_id
            )
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                for digest in digests
            )
        ):
            return None
        if not path.is_file() or path.stat().st_size < 1024:
            return None
        return endpoint
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def job_complete(job: dict[str, Any]) -> bool:
    timing_path = Path(job["timing_path"])
    if not timing_path.is_file():
        return False
    try:
        rows = [
            json.loads(line)
            for line in timing_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    if len([row for row in rows if row.get("kind") == "initialization"]) != 1:
        return False
    videos = [row for row in rows if row.get("kind") == "video"]
    if len(videos) != job["prompt_count"]:
        return False
    expected_indices = range(
        job["prompt_offset"], job["prompt_offset"] + job["prompt_count"]
    )
    if sorted(int(row["prompt_index"]) for row in videos) != list(expected_indices):
        return False
    for row in videos:
        index = int(row["prompt_index"])
        seed = int(job["base_seed"]) + index
        expected = (
            Path(job["output_dir"]) / f"{job['case_id']}_{index:02d}_seed{seed}.mp4"
        ).resolve()
        output = Path(row["output"]).resolve()
        sidecar = output.with_suffix(output.suffix + ".univ.json")
        if (
            int(row["seed"]) != seed
            or output != expected
            or not output.is_file()
            or output.stat().st_size < 1024
            or not sidecar.is_file()
            or endpoint_from_sidecar(
                sidecar,
                expected_seed=seed,
                expected_artifact_id=job["case_id"],
            )
            is None
        ):
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
    if sha256_file(manifest["prompts_file"]) != manifest["prompts_file_sha256"]:
        raise RuntimeError("prompt file changed after extension preparation")
    config = load_json(job["config_path"])
    if canonical_sha256(config) != job["config_sha256"]:
        raise RuntimeError(f"job config changed: {job['config_path']}")
    environment = dict(os.environ)
    environment["LIGHTX2V_REPO"] = str(Path(args.lightx2v_repo).resolve())
    roots = [environment["LIGHTX2V_REPO"], str(REPO_ROOT)]
    if environment.get("PYTHONPATH"):
        roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    target_frames = (
        4
        * (
            int(load_json(manifest["plan_path"])["protocol"]["target_latent_shape"][1])
            - 1
        )
        + 1
    )
    command = [
        args.wan_python,
        str(REPO_ROOT / "UNIV_adaptor/scripts/bridge/run_wan_univ_batch.py"),
        "--seed",
        str(job["base_seed"]),
        "--model_cls",
        job["model_cls"],
        "--model_path",
        manifest["model_root"],
        "--config_json",
        job["config_path"],
        "--prompts_file",
        manifest["prompts_file"],
        "--out_dir",
        job["output_dir"],
        "--name_prefix",
        job["case_id"],
        "--limit",
        str(job["prompt_count"]),
        "--prompt-offset",
        str(job["prompt_offset"]),
        "--timing-jsonl",
        job["timing_path"],
        "--timing-warmup",
        "0",
        "--target_video_length",
        str(target_frames),
        "--negative_prompt",
        args.negative_prompt,
    ]
    print(f"[generate] {job['job_id']}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    if not job_complete(job):
        raise RuntimeError(f"job finished without complete artifacts: {job['job_id']}")


def artifact_payload(row: dict[str, Any]) -> dict[str, Any]:
    video = Path(row["output"]).resolve()
    sidecar = video.with_suffix(video.suffix + ".univ.json")
    endpoint = endpoint_from_sidecar(sidecar, expected_seed=int(row["seed"]))
    if endpoint is None:
        raise RuntimeError(f"invalid endpoint sidecar: {sidecar}")
    endpoint_path = Path(endpoint["path"]).resolve()
    return {
        "video_path": str(video),
        "video_sha256": sha256_file(video),
        "video_bytes": video.stat().st_size,
        "runtime_sidecar_path": str(sidecar),
        "runtime_sidecar_sha256": sha256_file(sidecar),
        "endpoint_state": {
            **endpoint,
            "path": str(endpoint_path),
            "file_sha256": sha256_file(endpoint_path),
            "file_bytes": endpoint_path.stat().st_size,
        },
        "cost": {
            "pipeline_seconds": float(row["pipeline_elapsed_s"]),
            "segment_seconds": float(row["segment_elapsed_s"]),
            "peak_allocated_gib": float(row.get("peak_allocated_gib", 0.0)),
            "stage_seconds": row.get("univ_stage_timing_s", {}),
        },
    }


def finalize(args: argparse.Namespace) -> None:
    manifest = validate_manifest(load_json(args.manifest))
    out_root = Path(args.out_root).resolve()
    if out_root != Path(manifest["out_root"]).resolve():
        raise ValueError("finalize out root does not match the extension manifest")
    base_identity = manifest["base_dataset"]
    if out_root == Path(base_identity["root"]).resolve():
        raise ValueError("extension output cannot overwrite the immutable base root")
    for path_field, hash_field in (
        ("generation_manifest_path", "generation_manifest_file_sha256"),
        ("plan_path", "plan_file_sha256"),
    ):
        source_path = Path(base_identity[path_field])
        if sha256_file(source_path) != base_identity[hash_field]:
            raise RuntimeError(
                f"base dataset changed after extension planning: {source_path}"
            )
    jobs = selected_jobs(manifest, args.splits)
    incomplete = [job["job_id"] for job in jobs if not job_complete(job)]
    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} extension jobs are incomplete: {', '.join(incomplete[:10])}"
        )
    timing = {}
    for job in jobs:
        for line in Path(job["timing_path"]).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("kind") == "video":
                key = (
                    job["split"],
                    job["case_id"],
                    int(row["prompt_index"]),
                    int(row["seed"]),
                )
                if key in timing:
                    raise RuntimeError(f"duplicate extension video: {key}")
                timing[key] = row
    plan = validate_plan(load_json(manifest["plan_path"]))
    selected = set(args.splits)
    base_root = Path(base_identity["root"])
    count = 0
    for assignment in plan["assignments"]:
        if assignment["split"] not in selected:
            continue
        candidates = []
        for candidate in assignment["low_budget_candidates"]:
            key = (
                assignment["split"],
                candidate["artifact_id"],
                assignment["prompt_id"],
                assignment["seed"],
            )
            candidates.append({**candidate, **artifact_payload(timing[key])})
        base_record_path = (
            base_root
            / "records"
            / assignment["split"]
            / f"{assignment['trajectory_key']}.json"
        )
        if not base_record_path.is_file():
            raise RuntimeError(f"base trajectory record is missing: {base_record_path}")
        base_record = load_json(base_record_path)
        validate_trajectory_record(base_record, require_scores=False)
        for field in (
            "trajectory_key",
            "split",
            "prompt_id",
            "prompt",
            "prompt_sha256",
            "base_seed",
            "seed",
        ):
            if base_record.get(field) != assignment[field]:
                raise RuntimeError(
                    f"base record identity mismatch for {assignment['trajectory_key']}: {field}"
                )
        extension_record = {
            "schema": RECORD_SCHEMA,
            "generation_status": "generated_unscored",
            "plan_sha256": plan["plan_sha256"],
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
            "base_record": {
                "path": str(base_record_path.resolve()),
                "file_sha256": sha256_file(base_record_path),
                "schema": base_record["schema"],
            },
            "low_budget_candidates": candidates,
            "provenance": {
                "extension_manifest_sha256": manifest["manifest_sha256"],
                "protocol_sha256": manifest["protocol_sha256"],
                "observation_mode": "prompt_plus_endpoint",
                "trajectory_origin": "independent_step0",
            },
        }
        extension_path = (
            out_root
            / "records"
            / assignment["split"]
            / f"{assignment['trajectory_key']}.json"
        )
        write_json_atomic(extension_path, extension_record)

        legacy_candidates = []
        for candidate in base_record["budget_candidates"]:
            artifact_id = f"V2_{candidate['budget_id']}"
            legacy_candidates.append(
                {
                    **candidate,
                    "legacy_budget_id": candidate["budget_id"],
                    "budget_id": artifact_id,
                    "artifact_id": artifact_id,
                    "display_budget": candidate["budget_id"],
                    "action_schema": "univ_action_v2",
                    "source_record_path": str(base_record_path.resolve()),
                }
            )
        new_candidates = [
            {
                **candidate,
                "budget_id": candidate["artifact_id"],
                "action_schema": "univ_execution_action_v3",
                "source_record_path": str(extension_path.resolve()),
            }
            for candidate in candidates
        ]
        combined_body = {
            "generation_status": "generated_unscored",
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
            "native_teacher": base_record["native_teacher"],
            "budget_candidates": legacy_candidates + new_candidates,
            "candidate_count": len(legacy_candidates) + len(new_candidates),
            "source_records": {
                "base": str(base_record_path.resolve()),
                "extension": str(extension_path.resolve()),
            },
            "provenance": {
                "base_plan_sha256": manifest["base_dataset"]["plan_sha256"],
                "extension_plan_sha256": plan["plan_sha256"],
                "extension_manifest_sha256": manifest["manifest_sha256"],
            },
        }
        combined = {
            "schema": COMBINED_RECORD_SCHEMA,
            "record_sha256": canonical_sha256(combined_body),
            **combined_body,
        }
        combined_path = (
            out_root
            / "combined_records"
            / assignment["split"]
            / f"{assignment['trajectory_key']}.json"
        )
        write_json_atomic(combined_path, combined)
        count += 1
    print(f"Finalized {count} extension and combined trajectory records")


def source_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    paths = (
        REPO_ROOT / "UNIV_adaptor/low_budget_protocol.py",
        REPO_ROOT / "UNIV_adaptor/mrflow_ablation_runner.py",
        REPO_ROOT / "UNIV_adaptor/hr_refinement.py",
        REPO_ROOT / "UNIV_adaptor/scripts/bridge/run_wan_univ_batch.py",
        REPO_ROOT / "UNIV_adaptor/scripts/data/run_low_budget_extension.py",
    )
    return {
        "git_commit": commit,
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "implementation_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths
        },
    }


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported extension manifest: {manifest.get('schema')}")
    body = {
        key: value
        for key, value in manifest.items()
        if key not in {"schema", "manifest_sha256", "created_at_utc"}
    }
    if canonical_sha256(body) != manifest.get("manifest_sha256"):
        raise ValueError("extension manifest hash mismatch")
    return manifest


def print_summary(manifest: dict[str, Any]) -> None:
    by_split: dict[str, int] = {}
    for job in manifest["jobs"]:
        by_split[job["split"]] = by_split.get(job["split"], 0) + job["prompt_count"]
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "plan_sha256": manifest["plan_sha256"],
                "case_count": len(manifest["cases"]),
                "job_count": len(manifest["jobs"]),
                "videos_by_split": by_split,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate UNIV low-budget extension")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--protocol", required=True)
    prepare_parser.add_argument("--prompts", required=True)
    prepare_parser.add_argument("--template-config", required=True)
    prepare_parser.add_argument("--model-root", required=True)
    prepare_parser.add_argument("--base-dataset-root", required=True)
    prepare_parser.add_argument("--out-root", required=True)
    prepare_parser.add_argument("--job-chunk-size", type=int, default=25)
    prepare_parser.add_argument("--worker-count", type=int, default=8)

    list_parser = sub.add_parser("list-jobs")
    list_parser.add_argument("--manifest", required=True)
    list_parser.add_argument("--splits", nargs="+", required=True)
    list_parser.add_argument("--worker-slot", type=int)
    list_parser.add_argument("--limit", type=int, default=0)

    job_parser = sub.add_parser("generate-job")
    job_parser.add_argument("--manifest", required=True)
    job_parser.add_argument("--job-id", required=True)
    job_parser.add_argument("--wan-python", required=True)
    job_parser.add_argument("--lightx2v-repo", required=True)
    job_parser.add_argument("--negative-prompt", default="")
    job_parser.add_argument("--resume", action="store_true")

    finalize_parser = sub.add_parser("finalize")
    finalize_parser.add_argument("--manifest", required=True)
    finalize_parser.add_argument("--out-root", required=True)
    finalize_parser.add_argument("--splits", nargs="+", required=True)
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
        jobs = selected_jobs(manifest, args.splits)
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
