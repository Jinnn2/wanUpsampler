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
    RECORD_SCHEMA,
    build_collection_plan,
    canonical_sha256,
    load_prompts,
    sha256_file,
    validate_collection_plan,
    validate_protocol,
    validate_trajectory_record,
    write_json_atomic,
)


MANIFEST_SCHEMA = "univ_prompt_budget_generation_manifest_v1"


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def base_runtime_config(template: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in template.items()
        if not key.startswith(("univ_", "wan_rgb_"))
    }


def validate_template(template: dict[str, Any], protocol: dict[str, Any]) -> None:
    expected_shape = (
        16,
        (int(template["target_video_length"]) - 1) // 4 + 1,
        int(template["target_height"]) // 8,
        int(template["target_width"]) // 8,
    )
    if list(expected_shape) != protocol["target_latent_shape"]:
        raise ValueError(
            "template target shape does not match protocol: "
            f"template={expected_shape}, protocol={protocol['target_latent_shape']}"
        )
    if int(template["infer_steps"]) != protocol["reference_nfe"]:
        raise ValueError("template infer_steps does not match protocol reference_nfe")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    protocol = validate_protocol(load_json(args.protocol))
    prompts = load_prompts(args.prompts)
    plan = build_collection_plan(protocol, prompts)
    template = load_json(args.template_config)
    validate_template(template, protocol)
    out_root = Path(args.out_root).resolve()
    plan_path = out_root / "collection_plan.json"
    manifest_path = out_root / "generation_manifest.json"

    base = base_runtime_config(template)
    case_configs: list[dict[str, Any]] = []
    native_path = out_root / "configs" / "native_hr50.json"
    case_configs.append(
        {
            "case_id": "native_hr50",
            "kind": "native",
            "model_cls": "wan2.1_univ_native",
            "config_path": str(native_path),
            "config": base,
            "expected_weight": 1.0,
        }
    )
    for preset in protocol["budget_presets"]:
        config = dict(base)
        config.update(
            {
                "univ_action": preset["action"],
                "univ_cache_mode": "residual",
                "univ_transition_baseline": protocol["transition"],
                "univ_enable_transition_diagnostics": False,
                "univ_native_hr_state_path": "",
                "univ_native_hr_state_key": "state",
            }
        )
        case_id = preset["id"]
        config_path = out_root / "configs" / f"{case_id}.json"
        candidate = next(
            item
            for item in plan["assignments"][0]["budget_candidates"]
            if item["budget_id"] == case_id
        )
        case_configs.append(
            {
                "case_id": case_id,
                "kind": "budget",
                "model_cls": "wan2.1_univ_pipeline",
                "config_path": str(config_path),
                "config": config,
                "expected_weight": max(
                    0.25, float(candidate["proxy_compute_density"])
                ),
            }
        )

    cases: list[dict[str, Any]] = []
    prepared_configs: dict[str, dict[str, Any]] = {}
    for case in case_configs:
        config = case["config"]
        prepared_configs[case["config_path"]] = config
        cases.append(
            {
                key: value
                for key, value in case.items()
                if key != "config"
            }
            | {"config_sha256": canonical_sha256(config)}
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
        "preset_status": protocol["preset_status"],
        "plan_sha256": plan["plan_sha256"],
        "plan_path": str(plan_path),
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
        previous = load_json(manifest_path)
        if previous.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError(
                f"generation protocol changed under {out_root}; use a new OUT_ROOT"
            )
        manifest = previous
    materialize_immutable_inputs(
        plan_path=plan_path,
        plan=plan,
        prepared_configs=prepared_configs,
    )
    if not manifest_path.is_file():
        write_json_atomic(manifest_path, manifest)
    print_summary(manifest)
    return manifest


def materialize_immutable_inputs(
    *,
    plan_path: Path,
    plan: dict[str, Any],
    prepared_configs: dict[str, dict[str, Any]],
) -> None:
    if plan_path.is_file():
        previous = load_json(plan_path)
        if previous.get("plan_sha256") != plan["plan_sha256"]:
            raise RuntimeError(f"collection plan was modified: {plan_path}")
    else:
        write_json_atomic(plan_path, plan)
    for path_text, config in prepared_configs.items():
        path = Path(path_text)
        if path.is_file():
            if canonical_sha256(load_json(path)) != canonical_sha256(config):
                raise RuntimeError(f"generated case config was modified: {path}")
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
    raw_jobs: list[dict[str, Any]] = []
    prompt_offset = 0
    for split in protocol["splits"]:
        split_start = prompt_offset
        split_stop = split_start + split["prompt_count"]
        for base_seed in split["base_seeds"]:
            for start in range(split_start, split_stop, chunk_size):
                count = min(chunk_size, split_stop - start)
                for case in cases:
                    job_id = (
                        f"{split['name']}_{case['case_id']}_base{base_seed}_"
                        f"p{start:06d}_{start + count - 1:06d}"
                    )
                    raw_jobs.append(
                        {
                            "job_id": job_id,
                            "split": split["name"],
                            "case_id": case["case_id"],
                            "kind": case["kind"],
                            "model_cls": case["model_cls"],
                            "config_path": case["config_path"],
                            "config_sha256": case["config_sha256"],
                            "base_seed": base_seed,
                            "prompt_offset": start,
                            "prompt_count": count,
                            "output_dir": str(
                                out_root / "videos" / split["name"] / case["case_id"]
                            ),
                            "timing_path": str(out_root / "timings" / f"{job_id}.jsonl"),
                            "expected_weight": count * float(case["expected_weight"]),
                        }
                    )
        prompt_offset = split_stop

    worker_loads = [0.0] * worker_count
    for job in sorted(raw_jobs, key=lambda value: -value["expected_weight"]):
        worker_slot = min(range(worker_count), key=worker_loads.__getitem__)
        job["worker_slot"] = worker_slot
        worker_loads[worker_slot] += float(job["expected_weight"])
    return sorted(raw_jobs, key=lambda value: value["job_id"])


def selected_jobs(
    manifest: dict[str, Any], splits: Iterable[str]
) -> list[dict[str, Any]]:
    requested = set(splits)
    available = {job["split"] for job in manifest["jobs"]}
    missing = requested - available
    if missing:
        raise ValueError(f"unknown splits: {sorted(missing)}")
    return [job for job in manifest["jobs"] if job["split"] in requested]


def job_complete(manifest: dict[str, Any], job: dict[str, Any]) -> bool:
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
    initialization = [row for row in rows if row.get("kind") == "initialization"]
    videos = [row for row in rows if row.get("kind") == "video"]
    if len(initialization) != 1 or len(videos) != job["prompt_count"]:
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
            Path(job["output_dir"])
            / f"{job['case_id']}_{index:02d}_seed{seed}.mp4"
        ).resolve()
        output = Path(row["output"]).resolve()
        if int(row["seed"]) != seed or output != expected:
            return False
        if not output.is_file() or output.stat().st_size < 1024:
            return False
        if job["kind"] == "budget":
            sidecar = output.with_suffix(output.suffix + ".univ.json")
            if not sidecar.is_file() or sidecar.stat().st_size == 0:
                return False
    return True


def generate_job(args: argparse.Namespace) -> None:
    manifest = validate_manifest(load_json(args.manifest))
    if (
        manifest["preset_status"] == "frozen_for_pilot_cost_calibration"
        and not args.allow_pilot_presets
    ):
        raise RuntimeError(
            "budget presets have not been frozen after measured cost; "
            "use --allow-pilot-presets only for a bounded calibration smoke"
        )
    jobs = {job["job_id"]: job for job in manifest["jobs"]}
    if args.job_id not in jobs:
        raise ValueError(f"job is absent from manifest: {args.job_id}")
    job = jobs[args.job_id]
    if args.resume and job_complete(manifest, job):
        print(f"[resume] {job['job_id']}")
        return
    if sha256_file(manifest["prompts_file"]) != manifest["prompts_file_sha256"]:
        raise RuntimeError("prompts file changed after generation manifest preparation")
    config = load_json(job["config_path"])
    if canonical_sha256(config) != job["config_sha256"]:
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
        "81",
        "--negative_prompt",
        args.negative_prompt,
    ]
    print(f"[generate] {job['job_id']}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    if not job_complete(manifest, job):
        raise RuntimeError(f"job finished without complete artifacts: {job['job_id']}")


def finalize(args: argparse.Namespace) -> None:
    manifest = validate_manifest(load_json(args.manifest))
    jobs = selected_jobs(manifest, args.splits)
    incomplete = [job["job_id"] for job in jobs if not job_complete(manifest, job)]
    if incomplete:
        preview = ", ".join(incomplete[:10])
        raise RuntimeError(f"{len(incomplete)} jobs are incomplete: {preview}")
    timing: dict[tuple[str, str, int, int], dict[str, Any]] = {}
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
                    raise RuntimeError(f"duplicate generated video record: {key}")
                timing[key] = row

    plan = load_json(manifest["plan_path"])
    validate_collection_plan(plan)
    selected_split_set = set(args.splits)
    record_count = 0
    for assignment in plan["assignments"]:
        if assignment["split"] not in selected_split_set:
            continue
        common = (
            assignment["split"],
            assignment["prompt_id"],
            assignment["seed"],
        )
        native_row = timing[(common[0], "native_hr50", common[1], common[2])]
        candidates = []
        for candidate in assignment["budget_candidates"]:
            row = timing[(common[0], candidate["budget_id"], common[1], common[2])]
            candidates.append({**candidate, **artifact_payload(row)})
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
            "native_teacher": artifact_payload(native_row),
            "budget_candidates": candidates,
            "provenance": {
                "generation_manifest_sha256": manifest["manifest_sha256"],
                "protocol_sha256": manifest["protocol_sha256"],
                "trajectory_origin": "independent_step0",
                "observation_mode": "prompt_only",
            },
        }
        validate_trajectory_record(
            record,
            expected_plan_sha256=plan["plan_sha256"],
            require_scores=False,
        )
        path = (
            Path(args.out_root)
            / "records"
            / assignment["split"]
            / f"{assignment['trajectory_key']}.json"
        )
        write_json_atomic(path, record)
        record_count += 1
    print(f"Finalized {record_count} generated-unscored trajectory records")


def artifact_payload(row: dict[str, Any]) -> dict[str, Any]:
    video = Path(row["output"]).resolve()
    payload: dict[str, Any] = {
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
    if sidecar.is_file():
        payload["runtime_sidecar_path"] = str(sidecar)
        payload["runtime_sidecar_sha256"] = sha256_file(sidecar)
    return payload


def source_identity() -> dict[str, Any]:
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
    implementation_paths = (
        REPO_ROOT / "UNIV_adaptor/data_protocol.py",
        REPO_ROOT / "UNIV_adaptor/wan_runner.py",
        REPO_ROOT / "UNIV_adaptor/transition.py",
        REPO_ROOT / "UNIV_adaptor/scripts/bridge/run_wan_univ_batch.py",
        REPO_ROOT / "UNIV_adaptor/scripts/data/run_prompt_budget_generation.py",
    )
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "implementation_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in implementation_paths
        },
    }


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported generation manifest schema: {manifest.get('schema')}")
    body = {
        key: value
        for key, value in manifest.items()
        if key not in {"schema", "manifest_sha256", "created_at_utc"}
    }
    if canonical_sha256(body) != manifest.get("manifest_sha256"):
        raise ValueError("generation manifest hash mismatch")
    return manifest


def print_summary(manifest: dict[str, Any]) -> None:
    split_jobs: dict[str, int] = {}
    worker_loads = [0.0] * int(manifest["worker_count"])
    for job in manifest["jobs"]:
        split_jobs[job["split"]] = split_jobs.get(job["split"], 0) + 1
        worker_loads[job["worker_slot"]] += float(job["expected_weight"])
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "plan_sha256": manifest["plan_sha256"],
                "case_count": len(manifest["cases"]),
                "job_count": len(manifest["jobs"]),
                "jobs_by_split": split_jobs,
                "estimated_worker_loads": worker_loads,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate UNIV prompt-budget curves")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--protocol", required=True)
    prepare_parser.add_argument("--prompts", required=True)
    prepare_parser.add_argument("--template-config", required=True)
    prepare_parser.add_argument("--model-root", required=True)
    prepare_parser.add_argument("--out-root", required=True)
    prepare_parser.add_argument("--job-chunk-size", type=int, default=100)
    prepare_parser.add_argument("--worker-count", type=int, default=8)

    list_parser = subparsers.add_parser("list-jobs")
    list_parser.add_argument("--manifest", required=True)
    list_parser.add_argument("--splits", nargs="+", required=True)
    list_parser.add_argument("--worker-slot", type=int)
    list_parser.add_argument("--limit", type=int, default=0)

    job_parser = subparsers.add_parser("generate-job")
    job_parser.add_argument("--manifest", required=True)
    job_parser.add_argument("--job-id", required=True)
    job_parser.add_argument("--wan-python", required=True)
    job_parser.add_argument("--lightx2v-repo", required=True)
    job_parser.add_argument("--realesrgan-repo", default="")
    job_parser.add_argument("--negative-prompt", default="")
    job_parser.add_argument("--resume", action="store_true")
    job_parser.add_argument("--allow-pilot-presets", action="store_true")

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--manifest", required=True)
    finalize_parser.add_argument("--out-root", required=True)
    finalize_parser.add_argument("--splits", nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.job_chunk_size < 1 or args.worker_count != 8:
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
