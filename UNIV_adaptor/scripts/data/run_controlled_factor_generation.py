"""Generate the controlled 2x2 prompt-factor FULL/S/T/C dataset."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


PROTOCOL_SCHEMA = "univ_controlled_factor_protocol_v1"
PLAN_SCHEMA = "univ_controlled_factor_plan_v1"
RECORD_SCHEMA = "univ_controlled_factor_record_v1"
DATASET_SCHEMA = "univ_controlled_factor_dataset_v1"
SPLITS = ("train", "validation", "test")
FACTOR_LEVELS = {"motion_level": {"low", "high"}, "detail_level": {"low", "high"}}
AXES = {"full", "spatial", "temporal", "cache"}


def load_prompt_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid prompt JSONL line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"prompt JSONL line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError("controlled prompt JSONL is empty")
    return rows


def validate_prompts(
    raw_rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for expected_id, raw in enumerate(raw_rows):
        prompt_id = int(raw.get("prompt_id", -1))
        family_id = str(raw.get("family_id", "")).strip()
        split = str(raw.get("split", "")).strip()
        prompt = str(raw.get("prompt", "")).strip()
        motion = str(raw.get("motion_level", "")).strip()
        detail = str(raw.get("detail_level", "")).strip()
        if prompt_id != expected_id:
            raise ValueError(
                f"prompt ids must be contiguous and ordered; expected {expected_id}, got {prompt_id}"
            )
        if not family_id or not prompt or split not in SPLITS:
            raise ValueError(f"invalid prompt identity/split at prompt {prompt_id}")
        if motion not in FACTOR_LEVELS["motion_level"]:
            raise ValueError(f"invalid motion_level at prompt {prompt_id}")
        if detail not in FACTOR_LEVELS["detail_level"]:
            raise ValueError(f"invalid detail_level at prompt {prompt_id}")
        rows.append(
            {
                "prompt_id": prompt_id,
                "family_id": family_id,
                "split": split,
                "motion_level": motion,
                "detail_level": detail,
                "factor_cell": f"motion_{motion}__detail_{detail}",
                "prompt": prompt,
                "prompt_sha256": canonical_sha256(prompt),
            }
        )
    expected_count = int(protocol["expected_prompt_count"])
    if len(rows) != expected_count or len({row["prompt"] for row in rows}) != len(rows):
        raise ValueError("controlled prompts must have the locked count and unique text")
    expected_splits = {key: int(value) for key, value in protocol["expected_split_counts"].items()}
    observed_splits = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    if observed_splits != expected_splits:
        raise ValueError(f"prompt split counts mismatch: {observed_splits}")
    split_sequence = [row["split"] for row in rows]
    expected_sequence = [
        split
        for split in SPLITS
        for _ in range(expected_splits[split])
    ]
    if split_sequence != expected_sequence:
        raise ValueError("prompt splits must be contiguous train, validation, test")
    family_splits: dict[str, set[str]] = {}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        family_splits.setdefault(row["family_id"], set()).add(row["split"])
        by_family.setdefault(row["family_id"], []).append(row)
    if any(len(values) != 1 for values in family_splits.values()):
        raise ValueError("a semantic family cannot cross data splits")
    expected_cells = {
        (motion, detail)
        for motion in FACTOR_LEVELS["motion_level"]
        for detail in FACTOR_LEVELS["detail_level"]
    }
    for family, family_rows in by_family.items():
        cells = {(row["motion_level"], row["detail_level"]) for row in family_rows}
        if len(family_rows) != 4 or cells != expected_cells:
            raise ValueError(f"family {family} does not contain the complete 2x2 factorial")
    expected_families = {
        key: int(value) for key, value in protocol["expected_family_counts"].items()
    }
    observed_families = {
        split: len({row["family_id"] for row in rows if row["split"] == split})
        for split in SPLITS
    }
    if observed_families != expected_families:
        raise ValueError(f"prompt family counts mismatch: {observed_families}")
    return rows


def validate_protocol(value: dict[str, Any]) -> dict[str, Any]:
    protocol = json.loads(json.dumps(value))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported controlled-factor protocol schema")
    if int(protocol.get("reference_nfe", 0)) != 50:
        raise ValueError("controlled-factor protocol requires reference_nfe=50")
    shape = [int(item) for item in protocol.get("target_latent_shape", [])]
    if len(shape) != 4 or min(shape) <= 0:
        raise ValueError("target_latent_shape must have four positive integers")
    protocol["target_latent_shape"] = shape
    seeds = [int(seed) for seed in protocol.get("base_seeds", [])]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("controlled-factor protocol requires three unique base seeds")
    protocol["base_seeds"] = seeds
    if set(protocol.get("expected_split_counts", {})) != set(SPLITS):
        raise ValueError("expected_split_counts must define train/validation/test")
    if set(protocol.get("expected_family_counts", {})) != set(SPLITS):
        raise ValueError("expected_family_counts must define train/validation/test")
    cases = protocol.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("controlled-factor protocol requires FULL/S/T/C cases")
    ids: set[str] = set()
    axes: set[str] = set()
    for case in cases:
        case_id = str(case.get("id", "")).strip()
        axis = str(case.get("axis", "")).strip()
        if not case_id or case_id in ids or axis in axes:
            raise ValueError("controlled case ids and axes must be unique")
        action = UniversalAction(
            **{key: float(raw) for key, raw in case.get("action", {}).items()}
        )
        action.validate()
        if axis == "full":
            if action != UniversalAction(1.0, 1.0, 1.0, 1.0):
                raise ValueError("FULL case must be the uncompressed 50-step trajectory")
        else:
            changed = {
                "spatial": action.spatial_ratio != 1.0,
                "temporal": action.temporal_ratio != 1.0,
                "cache": action.lr_nfe_ratio != 1.0,
            }
            if changed != {name: name == axis for name in ("spatial", "temporal", "cache")}:
                raise ValueError(f"case {case_id} does not isolate its declared axis")
            if not 0.45 <= proxy_compute_density(action) <= 0.52:
                raise ValueError(f"case {case_id} is outside the locked half-density band")
        case["id"] = case_id
        case["axis"] = axis
        case["action"] = {
            "spatial_ratio": action.spatial_ratio,
            "temporal_ratio": action.temporal_ratio,
            "lr_nfe_ratio": action.lr_nfe_ratio,
            "switch_ratio": action.switch_ratio,
        }
        ids.add(case_id)
        axes.add(axis)
    if axes != AXES:
        raise ValueError(f"controlled cases must cover axes {sorted(AXES)}")
    return protocol


def build_plan(protocol: dict[str, Any], prompts: list[dict[str, Any]]) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {}
    for case in protocol["cases"]:
        action = UniversalAction(**case["action"])
        descriptor = CandidateAction(
            action=action,
            transition=protocol["transition"],
            proxy_density=proxy_compute_density(action),
        ).as_dict(
            reference_nfe=protocol["reference_nfe"],
            target_latent_shape=tuple(protocol["target_latent_shape"]),
        )
        cases[case["id"]] = {"axis": case["axis"], **descriptor}
    groups = []
    for prompt in prompts:
        for base_seed in protocol["base_seeds"]:
            groups.append(
                {
                    **prompt,
                    "group_id": f"cf_p{prompt['prompt_id']:03d}_b{base_seed}",
                    "base_seed": base_seed,
                    "seed": base_seed + prompt["prompt_id"],
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
            "families": len({row["family_id"] for row in prompts}),
            "prompt_seed_groups": len(groups),
            "cases_per_group": len(cases),
            "videos": len(groups) * len(cases),
        },
    }
    return {"schema": PLAN_SCHEMA, "plan_sha256": canonical_sha256(body), **body}


def materialize_plain_prompts(path: Path, prompts: list[dict[str, Any]]) -> None:
    text = "\n".join(row["prompt"] for row in prompts) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"materialized prompt text changed: {path}")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    protocol = validate_protocol(load_json(args.protocol))
    prompts = validate_prompts(load_prompt_rows(args.prompts_jsonl), protocol)
    template = load_json(args.template_config)
    validate_template(template, protocol)
    plan = build_plan(protocol, prompts)
    out_root = Path(args.out_root).resolve()
    plan_path = out_root / "controlled_factor_plan.json"
    manifest_path = out_root / "generation_manifest.json"
    plain_prompts = out_root / "immutable_inputs" / "prompts.txt"
    materialize_plain_prompts(plain_prompts, prompts)
    prepared_configs: dict[str, dict[str, Any]] = {}
    cases = []
    base = base_runtime_config(template)
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
    split_protocol = {
        "splits": [
            {
                "name": split,
                "prompt_count": int(protocol["expected_split_counts"][split]),
                "base_seeds": protocol["base_seeds"],
            }
            for split in SPLITS
        ]
    }
    jobs = build_jobs(
        split_protocol,
        cases,
        out_root=out_root,
        chunk_size=args.job_chunk_size,
        worker_count=8,
    )
    body = {
        "protocol_sha256": plan["protocol_sha256"],
        "preset_status": "locked_controlled_factor_v1",
        "plan_sha256": plan["plan_sha256"],
        "plan_path": str(plan_path),
        "prompts_file": str(plain_prompts),
        "prompts_file_sha256": sha256_file(plain_prompts),
        "prompt_metadata_file": str(Path(args.prompts_jsonl).resolve()),
        "prompt_metadata_file_sha256": sha256_file(args.prompts_jsonl),
        "template_config": str(Path(args.template_config).resolve()),
        "template_config_sha256": sha256_file(args.template_config),
        "model_root": str(Path(args.model_root).resolve()),
        "source": source_identity(),
        "controlled_driver_sha256": sha256_file(Path(__file__).resolve()),
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
            raise RuntimeError("controlled protocol changed; use a new OUT_ROOT")
        manifest = previous
    materialize_immutable_inputs(
        plan_path=plan_path, plan=plan, prepared_configs=prepared_configs
    )
    if not manifest_path.is_file():
        write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "plan_sha256": manifest["plan_sha256"],
                **plan["counts"],
                "jobs": len(jobs),
            },
            indent=2,
        )
    )
    return manifest


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"reuse destination differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def import_targeted_reuse(args: argparse.Namespace) -> None:
    """Import the exact first-eight S/T arms from calibrated targeted v2."""

    from UNIV_adaptor.scripts.data.score_targeted_st_contrast import collect

    manifest = validate_manifest(load_json(args.manifest))
    if manifest["job_chunk_size"] != 8:
        raise ValueError("targeted reuse requires job_chunk_size=8")
    rows, identity = collect(Path(args.reuse_root).resolve())
    reusable = [
        row
        for row in rows
        if row["prompt_index"] < 8
        and row["case_id"] in {"SPATIAL_ONLY_D050", "TEMPORAL_ONLY_RT036_CAL"}
    ]
    if len(reusable) != 48:
        raise ValueError(f"expected 48 reusable targeted videos, found {len(reusable)}")
    prompts = Path(manifest["prompts_file"]).read_text(encoding="utf-8").splitlines()
    jobs = {job["job_id"]: job for job in manifest["jobs"]}
    imported: dict[str, list[dict[str, Any]]] = {}
    for row in reusable:
        index = int(row["prompt_index"])
        if row["prompt"] != prompts[index] or int(row["seed"]) != int(row["base_seed"]) + index:
            raise ValueError(f"targeted reuse identity mismatch at prompt {index}")
        job_id = (
            f"train_{row['case_id']}_base{row['base_seed']}_p000000_000007"
        )
        if job_id not in jobs:
            raise ValueError(f"reuse job absent from controlled manifest: {job_id}")
        job = jobs[job_id]
        destination = (
            Path(job["output_dir"])
            / f"{row['case_id']}_{index:02d}_seed{row['seed']}.mp4"
        )
        source = Path(row["video_path"])
        _link_or_copy(source, destination)
        source_sidecar = source.with_suffix(source.suffix + ".univ.json")
        if not source_sidecar.is_file():
            raise ValueError(f"targeted reuse sidecar is missing: {source_sidecar}")
        _link_or_copy(
            source_sidecar, destination.with_suffix(destination.suffix + ".univ.json")
        )
        imported.setdefault(job_id, []).append(
            {
                "kind": "video",
                "prompt_index": index,
                "seed": int(row["seed"]),
                "output": str(destination.resolve()),
                "pipeline_elapsed_s": float(row["pipeline_seconds"]),
                "segment_elapsed_s": float(row["pipeline_seconds"]),
                "reuse_source_video_sha256": row["video_sha256"],
            }
        )
    for job_id, video_rows in imported.items():
        job = jobs[job_id]
        timing_path = Path(job["timing_path"])
        payload = [
            {
                "kind": "initialization",
                "reuse_source_dataset": identity,
                "reuse_role": "exact_prompt_seed_action_artifact_import",
            },
            *sorted(video_rows, key=lambda row: row["prompt_index"]),
        ]
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in payload)
        if timing_path.is_file() and timing_path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"reuse timing already differs: {timing_path}")
        if not timing_path.is_file():
            timing_path.parent.mkdir(parents=True, exist_ok=True)
            timing_path.write_text(text, encoding="utf-8")
        if not job_complete(manifest, job):
            raise RuntimeError(f"imported targeted job is incomplete: {job_id}")
    print(f"Imported {len(reusable)} exact targeted S/T videos into {len(imported)} jobs")


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    manifest = validate_manifest(load_json(args.manifest))
    jobs = selected_jobs(manifest, list(SPLITS))
    incomplete = [job["job_id"] for job in jobs if not job_complete(manifest, job)]
    if incomplete:
        raise RuntimeError(f"{len(incomplete)} controlled jobs are incomplete: {incomplete[:8]}")
    timing: dict[tuple[str, int, int], dict[str, Any]] = {}
    for job in jobs:
        for line in Path(job["timing_path"]).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("kind") != "video":
                continue
            key = (job["case_id"], int(row["prompt_index"]), int(row["seed"]))
            if key in timing:
                raise RuntimeError(f"duplicate controlled timing row: {key}")
            timing[key] = row
    plan = load_json(manifest["plan_path"])
    if plan.get("schema") != PLAN_SCHEMA or plan.get("plan_sha256") != manifest["plan_sha256"]:
        raise ValueError("controlled plan/manifest mismatch")
    out_root = Path(args.out_root).resolve()
    records = []
    for group in plan["groups"]:
        artifacts = []
        for case_id in group["case_ids"]:
            row = timing[(case_id, group["prompt_id"], group["seed"])]
            artifacts.append(
                {"case_id": case_id, **plan["cases"][case_id], "artifact": artifact_payload(row)}
            )
        body = {
            **{key: value for key, value in group.items() if key != "case_ids"},
            "plan_sha256": plan["plan_sha256"],
            "artifacts": artifacts,
        }
        record = {"schema": RECORD_SCHEMA, "record_sha256": canonical_sha256(body), **body}
        path = out_root / "records" / group["split"] / f"{group['group_id']}.json"
        if path.is_file() and load_json(path) != record:
            raise RuntimeError(f"refusing to replace changed controlled record: {path}")
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
    dataset = {"schema": DATASET_SCHEMA, "dataset_sha256": canonical_sha256(body), **body}
    path = out_root / "controlled_factor_dataset.json"
    if path.is_file() and load_json(path) != dataset:
        raise RuntimeError(f"refusing to replace changed controlled dataset: {path}")
    if not path.is_file():
        write_json_atomic(path, dataset)
    print(f"Finalized {len(records)} groups and {plan['counts']['videos']} videos: {path}")
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare"):
        target = sub.add_parser(name)
        target.add_argument("--protocol", required=True)
        target.add_argument("--prompts-jsonl", required=True)
        target.add_argument("--template-config", required=True)
        target.add_argument("--model-root", required=True)
        if name == "prepare":
            target.add_argument("--out-root", required=True)
            target.add_argument("--job-chunk-size", type=int, default=8)
    reuse = sub.add_parser("reuse-targeted")
    reuse.add_argument("--manifest", required=True)
    reuse.add_argument("--reuse-root", required=True)
    listing = sub.add_parser("list-jobs")
    listing.add_argument("--manifest", required=True)
    listing.add_argument("--worker-slot", type=int)
    listing.add_argument("--limit", type=int, default=0)
    job = sub.add_parser("generate-job")
    job.add_argument("--manifest", required=True)
    job.add_argument("--job-id", required=True)
    job.add_argument("--wan-python", required=True)
    job.add_argument("--lightx2v-repo", required=True)
    job.add_argument("--realesrgan-repo", default="")
    job.add_argument("--negative-prompt", default="")
    job.add_argument("--resume", action="store_true")
    final = sub.add_parser("finalize")
    final.add_argument("--manifest", required=True)
    final.add_argument("--out-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "check":
        protocol = validate_protocol(load_json(args.protocol))
        prompts = validate_prompts(load_prompt_rows(args.prompts_jsonl), protocol)
        validate_template(load_json(args.template_config), protocol)
        print(json.dumps(build_plan(protocol, prompts)["counts"], indent=2))
    elif args.command == "prepare":
        if args.job_chunk_size != 8:
            raise ValueError("controlled_factor_v1 locks job_chunk_size=8 for exact reuse")
        prepare(args)
    elif args.command == "reuse-targeted":
        import_targeted_reuse(args)
    elif args.command == "list-jobs":
        manifest = validate_manifest(load_json(args.manifest))
        jobs = rebalance_jobs(selected_jobs(manifest, list(SPLITS)), worker_count=8)
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
