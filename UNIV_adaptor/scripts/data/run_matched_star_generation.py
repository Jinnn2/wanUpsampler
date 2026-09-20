"""Prepare and run the Phase 4 reference-centered matched-star supplement."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.data_protocol import canonical_sha256, sha256_file  # noqa: E402
from UNIV_adaptor.scripts.data.run_prompt_budget_generation import (  # noqa: E402
    base_runtime_config,
    validate_template,
)
from UNIV_adaptor.scripts.data.run_sparse_action_generation import (  # noqa: E402
    MANIFEST_SCHEMA,
    build_jobs,
    copy_artifact,
    finalize,
    generate_job,
    load_json,
    materialize_immutable,
    print_summary,
    selected_jobs,
    validate_dataset_manifest,
    validate_manifest,
    verify_artifact,
)
from UNIV_adaptor.sparse_action_protocol import (  # noqa: E402
    PLAN_SCHEMA,
    action_catalog,
    design_balance,
    expected_counts,
    validate_sparse_plan,
    validate_sparse_protocol,
    validate_sparse_record,
    write_json_atomic,
)


TARGET_ACTIONS = ("REFERENCE", "STAR_S", "STAR_T", "STAR_C")


def prompt_lines(path: Path) -> list[tuple[int, str]]:
    result = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        prompt = raw.strip()
        if prompt and not raw.lstrip().startswith("#"):
            result.append((line_number, prompt))
    if not result:
        raise ValueError("new prompt source must be non-empty")
    return result


def source_sparse_groups(
    source_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_path = source_root / "sparse_dataset_manifest.json"
    dataset = validate_dataset_manifest(load_json(dataset_path))
    plan_path = Path(dataset["plan_path"]).resolve()
    if not plan_path.is_file() or sha256_file(plan_path) != dataset["plan_file_sha256"]:
        raise ValueError("source Phase3 plan identity changed")
    plan = validate_sparse_plan(load_json(plan_path))
    if plan["plan_sha256"] != dataset["plan_sha256"]:
        raise ValueError("source Phase3 dataset/plan mismatch")
    indexed = {item["group_id"]: item for item in dataset["records"]}
    if set(indexed) != {group["group_id"] for group in plan["groups"]}:
        raise ValueError("source Phase3 record index is incomplete")
    groups = []
    for group in plan["groups"]:
        item = indexed[group["group_id"]]
        record_path = Path(item["path"]).resolve()
        if sha256_file(record_path) != item["file_sha256"]:
            raise ValueError(f"source Phase3 record changed: {record_path}")
        record = validate_sparse_record(
            load_json(record_path), expected_plan_sha256=plan["plan_sha256"]
        )
        for field in (
            "group_id",
            "prompt_key",
            "cohort",
            "prompt",
            "prompt_sha256",
            "source_prompt_id",
            "base_seed",
            "seed",
        ):
            if record[field] != group[field]:
                raise ValueError(f"source Phase3 identity mismatch: {field}")
        reusable = []
        for action_row in record["actions"]:
            artifact = copy_artifact(action_row["artifact"])
            verify_artifact(artifact)
            reusable.append(
                {
                    "action_id": action_row["action_id"],
                    "action_key": action_row["action"]["action_key"],
                    "artifact": artifact,
                    "source_record_path": str(record_path),
                    "source_record_sha256": item["file_sha256"],
                    "source_trajectory_key": group["group_id"],
                }
            )
        groups.append({**group, "reusable": reusable})
    identity = {
        "root": str(source_root.resolve()),
        "dataset_manifest_path": str(dataset_path.resolve()),
        "dataset_manifest_file_sha256": sha256_file(dataset_path),
        "dataset_sha256": dataset["dataset_sha256"],
        "plan_path": str(plan_path),
        "plan_file_sha256": sha256_file(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "scores_accessed": False,
        "artifact_selection": "exact_action_key_match_only",
    }
    return groups, identity


def reuse_for_action(
    source_group: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any] | None:
    matches = [
        row
        for row in source_group["reusable"]
        if row["action_key"] == target["action_key"]
    ]
    if len(matches) > 1:
        raise ValueError(
            f"multiple source artifacts match {source_group['group_id']} {target['action_id']}"
        )
    return matches[0] if matches else None


def planned_actions(
    group_id: str,
    catalog: dict[str, dict[str, Any]],
    source_group: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows = []
    for action_id in TARGET_ACTIONS:
        observation_id = f"{group_id}__{action_id}"
        reuse = (
            reuse_for_action(source_group, catalog[action_id])
            if source_group is not None
            else None
        )
        row = {
            "observation_id": observation_id,
            "action_id": action_id,
            "artifact_mode": "reuse" if reuse else "generate",
        }
        if reuse:
            row.update(
                {
                    "artifact": reuse["artifact"],
                    "source_record_path": reuse["source_record_path"],
                    "source_record_sha256": reuse["source_record_sha256"],
                    "source_trajectory_key": reuse["source_trajectory_key"],
                    "source_action_id": reuse["action_id"],
                }
            )
        rows.append(row)
    return rows


def build_plan(
    protocol_value: dict[str, Any],
    *,
    source_root: Path,
    prompts_path: Path,
    prompt_offset: int,
) -> dict[str, Any]:
    protocol = validate_sparse_protocol(protocol_value)
    if protocol["artifact_reuse_scope"] != "bound_source_dataset_any_seed":
        raise ValueError("matched-star protocol must bind source-dataset reuse")
    source_groups, source_identity = source_sparse_groups(source_root)
    expected_source_prompts = int(protocol.get("source_sparse_prompt_count", 0))
    source_prompt_keys = {group["prompt_key"] for group in source_groups}
    if len(source_prompt_keys) != expected_source_prompts:
        raise ValueError(
            f"source sparse prompt count differs: {len(source_prompt_keys)} != {expected_source_prompts}"
        )
    expected_source_groups = expected_source_prompts * len(protocol["base_seeds"])
    if len(source_groups) != expected_source_groups:
        raise ValueError("source sparse prompt/seed coverage differs")
    source_base_seeds: dict[str, set[int]] = {}
    for group in source_groups:
        source_base_seeds.setdefault(group["prompt_key"], set()).add(
            int(group["base_seed"])
        )
    if any(
        seeds != set(protocol["base_seeds"]) for seeds in source_base_seeds.values()
    ):
        raise ValueError("source sparse base seeds differ from matched-star protocol")

    new_count = int(protocol.get("new_train_prompt_count", 0))
    available = prompt_lines(prompts_path)
    selected = available[prompt_offset : prompt_offset + new_count]
    if (
        len(selected) != new_count
        or len({prompt for _, prompt in selected}) != new_count
    ):
        raise ValueError("new training prompt slice has wrong count or duplicates")
    source_hashes = {group["prompt_sha256"] for group in source_groups}
    selected_hashes = {canonical_sha256(prompt) for _, prompt in selected}
    if source_hashes & selected_hashes:
        raise ValueError("new matched-star prompts overlap source Phase3 prompts")

    catalog = action_catalog(protocol)
    groups = []
    for source in source_groups:
        groups.append(
            {
                key: source[key]
                for key in (
                    "group_id",
                    "prompt_key",
                    "cohort",
                    "prompt",
                    "prompt_sha256",
                    "source_prompt_id",
                    "seed_offset",
                    "base_seed",
                    "seed",
                )
            }
            | {
                "actions": planned_actions(
                    source["group_id"], catalog, source_group=source
                )
            }
        )
    for source_line, prompt in selected:
        prompt_key = f"star_train_p{source_line:06d}"
        seed_offset = int(protocol["fresh_seed_offset"]) + source_line
        for base_seed in protocol["base_seeds"]:
            group_id = f"{prompt_key}_b{base_seed}"
            groups.append(
                {
                    "group_id": group_id,
                    "prompt_key": prompt_key,
                    "cohort": "existing_train",
                    "prompt": prompt,
                    "prompt_sha256": canonical_sha256(prompt),
                    "source_prompt_id": source_line,
                    "seed_offset": seed_offset,
                    "base_seed": int(base_seed),
                    "seed": int(base_seed) + seed_offset,
                    "actions": planned_actions(group_id, catalog, source_group=None),
                }
            )

    existing_prompts = {
        group["prompt_key"] for group in groups if group["cohort"] == "existing_train"
    }
    fresh_prompts = {
        group["prompt_key"]
        for group in groups
        if group["cohort"] == "fresh_development"
    }
    if len(existing_prompts) != protocol["existing_train_prompt_count"]:
        raise ValueError("matched-star existing-train prompt count differs")
    if len(fresh_prompts) != protocol["fresh_development_prompt_count"]:
        raise ValueError("matched-star fresh prompt count differs")
    generated = sum(
        row["artifact_mode"] == "generate"
        for group in groups
        for row in group["actions"]
    )
    reused = sum(
        row["artifact_mode"] == "reuse" for group in groups for row in group["actions"]
    )
    nominal = expected_counts(protocol)
    counts = {
        "existing_prompts": len(existing_prompts),
        "fresh_prompts": len(fresh_prompts),
        "generated_videos": generated,
        "reused_videos": reused,
        "scored_videos": generated + reused,
        "prompt_seed_groups": len(groups),
        "generated_videos_upper_bound": nominal["generated_videos_upper_bound"],
    }
    prompt_probes = {
        key: ("STAR_S", "STAR_T", "STAR_C")
        for key in sorted(existing_prompts | fresh_prompts)
    }
    body = {
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "source_phase3": source_identity,
        "new_train_prompts_file": str(prompts_path.resolve()),
        "new_train_prompts_file_sha256": sha256_file(prompts_path),
        "new_train_prompt_offset": prompt_offset,
        "new_train_prompt_count": new_count,
        "action_catalog": catalog,
        "design_balance": design_balance(protocol, prompt_probes),
        "counts": counts,
        "groups": sorted(groups, key=lambda row: row["group_id"]),
    }
    plan = {"schema": PLAN_SCHEMA, "plan_sha256": canonical_sha256(body), **body}
    validate_sparse_plan(plan)
    return plan


def implementation_identity() -> dict[str, Any]:
    paths = (
        REPO_ROOT / "UNIV_adaptor/sparse_action_protocol.py",
        Path(__file__).resolve(),
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


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    protocol = validate_sparse_protocol(load_json(args.protocol))
    template = load_json(args.template_config)
    validate_template(template, protocol)
    out_root = Path(args.out_root).resolve()
    plan = build_plan(
        protocol,
        source_root=Path(args.source_phase3_root).resolve(),
        prompts_path=Path(args.new_prompts).resolve(),
        prompt_offset=args.new_prompt_offset,
    )
    plan_path = out_root / "sparse_action_plan.json"
    catalog = plan["action_catalog"]
    base = base_runtime_config(template)
    cases = []
    prepared_configs = {}
    for action_id in TARGET_ACTIONS:
        action = catalog[action_id]
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
    body = {
        "plan_path": str(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "protocol_sha256": plan["protocol_sha256"],
        "template_config": str(Path(args.template_config).resolve()),
        "template_config_sha256": sha256_file(args.template_config),
        "model_root": str(Path(args.model_root).resolve()),
        "source": implementation_identity(),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--protocol", required=True)
    prepare_parser.add_argument("--source-phase3-root", required=True)
    prepare_parser.add_argument("--new-prompts", required=True)
    prepare_parser.add_argument("--new-prompt-offset", type=int, default=181)
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
        args.job_chunk_size < 1 or args.worker_count != 8 or args.new_prompt_offset < 0
    ):
        parser.error(
            "job-chunk-size must be positive, worker-count 8, and prompt offset non-negative"
        )
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
