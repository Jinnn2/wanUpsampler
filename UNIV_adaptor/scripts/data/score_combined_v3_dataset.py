from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.combined_v3 import (  # noqa: E402
    ALLOWED_SPLITS,
    QUALITY_DIMENSIONS,
    SCORED_DATASET_SCHEMA,
    SCORED_RECORD_SCHEMA,
    SCORE_MANIFEST_SCHEMA,
    action_catalog,
    load_json,
    quality_payload,
    validate_generated_record,
    validate_scored_record,
    verify_file,
)
from UNIV_adaptor.data_protocol import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    write_json_atomic,
)
from UNIV_adaptor.scripts.data.run_low_budget_extension import (  # noqa: E402
    validate_manifest as validate_extension_manifest,
)


CASE_SCORE_SCHEMA = "univ_combined_v3_case_scores_v1"
SAFE_DIAGNOSTICS = ("dynamic_degree", "overall_consistency", "temporal_flickering")


def parse_shard(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--shard must use SHARD_ID=/absolute/root")
    shard_id, root_text = value.split("=", 1)
    shard_id = shard_id.strip()
    if not shard_id or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in shard_id
    ):
        raise argparse.ArgumentTypeError("invalid shard id")
    return shard_id, Path(root_text).resolve()


def validate_score_manifest(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != SCORE_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported score manifest: {value.get('schema')}")
    body = {
        key: item
        for key, item in value.items()
        if key not in {"schema", "manifest_sha256", "created_at_utc"}
    }
    if canonical_sha256(body) != value.get("manifest_sha256"):
        raise ValueError("score manifest hash mismatch")
    if value.get("splits") != list(ALLOWED_SPLITS):
        raise ValueError("score manifest must contain train and validation only")
    return value


def generated_record_paths(root: Path, split: str) -> list[Path]:
    return sorted((root / "combined_records" / split).glob("*.json"))


def prepare(args: argparse.Namespace) -> None:
    shards = sorted(args.shard)
    if len(shards) < 1:
        raise ValueError("at least one shard is required")
    if len({name for name, _ in shards}) != len(shards):
        raise ValueError("shard ids must be unique")
    if len({str(root) for _, root in shards}) != len(shards):
        raise ValueError("shard roots must be unique")
    splits = list(args.splits)
    if splits != list(ALLOWED_SPLITS):
        raise ValueError("selection scoring must use exactly: train validation")

    catalog: list[dict[str, Any]] | None = None
    model_root: Path | None = None
    prompt_owners: dict[str, str] = {}
    shard_payloads = []
    cases: list[dict[str, Any]] = []
    out_root = Path(args.out_root).resolve()
    for shard_id, root in shards:
        extension_path = root / "extension_manifest.json"
        extension = validate_extension_manifest(load_json(extension_path))
        if Path(extension.get("out_root", "")).resolve() != root:
            raise ValueError(f"extension manifest root mismatch: {extension_path}")
        model_root_text = str(extension.get("model_root", "")).strip()
        if not model_root_text:
            raise ValueError(
                f"extension manifest has no Wan model root: {extension_path}"
            )
        current_model_root = Path(model_root_text).resolve()
        if model_root is None:
            model_root = current_model_root
        elif current_model_root != model_root:
            raise ValueError("Primary and Reserve use different Wan model roots")
        shard_records = []
        case_rows: dict[str, list[dict[str, Any]]] = {}
        for split in splits:
            paths = generated_record_paths(root, split)
            if not paths:
                raise FileNotFoundError(f"no combined records for {shard_id}/{split}")
            for path in paths:
                record = load_json(path)
                validate_generated_record(record)
                if record["split"] != split:
                    raise ValueError(f"record split mismatch: {path}")
                current_catalog = action_catalog(record)
                if catalog is None:
                    catalog = current_catalog
                elif canonical_sha256(current_catalog) != canonical_sha256(catalog):
                    raise ValueError(f"action catalog mismatch: {path}")
                owner = prompt_owners.get(record["prompt_sha256"])
                if owner is not None and owner != shard_id:
                    raise ValueError(
                        f"prompt overlap across shards: {record['prompt_sha256']}"
                    )
                prompt_owners[record["prompt_sha256"]] = shard_id
                file_sha = sha256_file(path)
                artifacts = [("native_hr", record["native_teacher"])] + [
                    (item["artifact_id"], item) for item in record["budget_candidates"]
                ]
                artifact_rows = {}
                for action_id, artifact in artifacts:
                    row = {
                        "trajectory_key": record["trajectory_key"],
                        "prompt": record["prompt"],
                        "prompt_sha256": record["prompt_sha256"],
                        "video_path": str(Path(artifact["video_path"]).resolve()),
                        "video_sha256": artifact["video_sha256"],
                    }
                    case_rows.setdefault(f"{split}::{action_id}", []).append(row)
                    artifact_rows[action_id] = {
                        "video_path": row["video_path"],
                        "video_sha256": row["video_sha256"],
                    }
                shard_records.append(
                    {
                        "trajectory_key": record["trajectory_key"],
                        "split": split,
                        "prompt_id": record["prompt_id"],
                        "prompt_sha256": record["prompt_sha256"],
                        "seed": record["seed"],
                        "record_path": str(path.resolve()),
                        "record_file_sha256": file_sha,
                        "record_sha256": record["record_sha256"],
                        "artifacts": artifact_rows,
                    }
                )
            expected_count = (
                args.expected_train_records_per_shard
                if split == "train"
                else args.expected_validation_records_per_shard
            )
            if len(paths) != expected_count:
                raise RuntimeError(
                    f"{shard_id}/{split} has {len(paths)} records; expected {expected_count}"
                )
        for local_key, rows in sorted(case_rows.items()):
            split, action_id = local_key.split("::", 1)
            cases.append(
                {
                    "case_id": f"{shard_id}__{split}__{action_id}",
                    "shard_id": shard_id,
                    "split": split,
                    "action_id": action_id,
                    "record_count": len(rows),
                    "rows": sorted(rows, key=lambda item: item["trajectory_key"]),
                }
            )
        shard_payloads.append(
            {
                "shard_id": shard_id,
                "root": str(root),
                "extension_manifest_path": str(extension_path.resolve()),
                "extension_manifest_file_sha256": sha256_file(extension_path),
                "extension_manifest_sha256": extension["manifest_sha256"],
                "model_root": str(current_model_root),
                "record_count": len(shard_records),
                "records": sorted(
                    shard_records,
                    key=lambda item: (
                        ALLOWED_SPLITS.index(item["split"]),
                        item["trajectory_key"],
                    ),
                ),
            }
        )
    if catalog is None:
        raise RuntimeError("no action catalog was discovered")
    if model_root is None:
        raise RuntimeError("no Wan model root was discovered")
    body = {
        "out_root": str(out_root),
        "splits": splits,
        "quality_dimensions": list(QUALITY_DIMENSIONS),
        "model_root": str(model_root),
        "action_catalog": catalog,
        "action_catalog_sha256": canonical_sha256(catalog),
        "shards": shard_payloads,
        "cases": cases,
        "case_count": len(cases),
        "trajectory_count": sum(item["record_count"] for item in shard_payloads),
        "video_count": sum(item["record_count"] for item in cases),
        "expected_records_per_shard": {
            "train": args.expected_train_records_per_shard,
            "validation": args.expected_validation_records_per_shard,
        },
        "implementation_sha256": {
            "UNIV_adaptor/combined_v3.py": sha256_file(
                REPO_ROOT / "UNIV_adaptor/combined_v3.py"
            ),
            "UNIV_adaptor/scripts/data/score_combined_v3_dataset.py": sha256_file(
                Path(__file__).resolve()
            ),
        },
    }
    manifest = {
        "schema": SCORE_MANIFEST_SCHEMA,
        "manifest_sha256": canonical_sha256(body),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        **body,
    }
    path = out_root / "score_manifest.json"
    if path.is_file():
        previous = validate_score_manifest(load_json(path))
        if previous["manifest_sha256"] != manifest["manifest_sha256"]:
            raise RuntimeError(
                f"refusing to replace a different score manifest: {path}"
            )
    else:
        write_json_atomic(path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(path),
                "shards": len(shards),
                "trajectories": body["trajectory_count"],
                "videos": body["video_count"],
                "cases": body["case_count"],
            },
            indent=2,
        )
    )


def case_by_id(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [item for item in manifest["cases"] if item["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate case id: {case_id}")
    return matches[0]


def materialize_case(
    manifest: dict[str, Any], case: dict[str, Any]
) -> tuple[Path, Path]:
    out_root = Path(manifest["out_root"])
    video_dir = out_root / "staging" / case["case_id"] / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    prompt_map = {}
    expected_names = set()
    for row in case["rows"]:
        source = verify_file(
            row["video_path"], row["video_sha256"], label="scoring source video"
        )
        name = f"{row['trajectory_key']}.mp4"
        expected_names.add(name)
        link = video_dir / name
        if link.exists():
            if not os.path.samefile(link, source):
                raise RuntimeError(f"staging hard link differs from manifest: {link}")
        else:
            try:
                os.link(source, link)
            except OSError as exc:
                raise RuntimeError(
                    "cannot create zero-copy scoring hard link; keep SCORE_ROOT on "
                    f"the same filesystem as the videos: {link} -> {source}"
                ) from exc
        prompt_map[str(link.absolute())] = row["prompt"]
    actual_names = {path.name for path in video_dir.glob("*.mp4")}
    if actual_names != expected_names:
        raise RuntimeError(
            f"unexpected files in staging case {case['case_id']}: "
            f"extra={sorted(actual_names - expected_names)[:10]}"
        )
    prompt_path = video_dir.parent / "prompt_map.json"
    write_json_atomic(prompt_path, prompt_map)
    return video_dir, prompt_path


def validate_case_score(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != CASE_SCORE_SCHEMA:
        raise ValueError("unsupported case score bundle")
    body = {
        key: item
        for key, item in value.items()
        if key not in {"schema", "bundle_sha256"}
    }
    if canonical_sha256(body) != value.get("bundle_sha256"):
        raise ValueError("case score bundle hash mismatch")
    return value


def score_case(args: argparse.Namespace) -> None:
    manifest = validate_score_manifest(load_json(args.manifest))
    case = case_by_id(manifest, args.case_id)
    diagnostics = list(args.diagnostic_dimensions)
    if any(value not in SAFE_DIAGNOSTICS for value in diagnostics):
        raise ValueError(f"diagnostics must be selected from {SAFE_DIAGNOSTICS}")
    if len(diagnostics) != len(set(diagnostics)):
        raise ValueError("diagnostic dimensions must be unique")
    video_dir, prompt_map = materialize_case(manifest, case)

    from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import (
        inspect_vbench_checkout,
        score_case_directory,
    )

    identity = inspect_vbench_checkout(
        Path(args.vbench_root).resolve(),
        expected_commit=args.expected_vbench_commit or None,
    )
    bundle = score_case_directory(
        Path(args.vbench_root).resolve(),
        args.vbench_python,
        video_dir,
        prompt_map,
        Path(manifest["out_root"]) / "metrics" / "vbench" / case["case_id"],
        [*QUALITY_DIMENSIONS, *diagnostics],
        list(QUALITY_DIMENSIONS),
        diagnostics,
        args.ngpus,
        args.force_rescore,
        identity,
    )
    score_body = {
        "score_manifest_sha256": manifest["manifest_sha256"],
        "case_id": case["case_id"],
        "case_sha256": canonical_sha256(case),
        "action_id": case["action_id"],
        "split": case["split"],
        "shard_id": case["shard_id"],
        "record_count": case["record_count"],
        "quality_dimensions": list(QUALITY_DIMENSIONS),
        "diagnostic_dimensions": diagnostics,
        "scores": bundle.scores,
        "vbench_provenance": bundle.provenance,
    }
    payload = {
        "schema": CASE_SCORE_SCHEMA,
        "bundle_sha256": canonical_sha256(score_body),
        **score_body,
    }
    output = Path(manifest["out_root"]) / "case_scores" / f"{case['case_id']}.json"
    if output.is_file():
        previous = validate_case_score(load_json(output))
        if previous["bundle_sha256"] != payload["bundle_sha256"]:
            raise RuntimeError(f"refusing to replace different case scores: {output}")
    else:
        write_json_atomic(output, payload)
    print(f"Scored {case['case_id']}: {case['record_count']} videos")


def score_status(args: argparse.Namespace) -> None:
    manifest = validate_score_manifest(load_json(args.manifest))
    root = Path(manifest["out_root"])
    complete = 0
    for case in manifest["cases"]:
        path = root / "case_scores" / f"{case['case_id']}.json"
        try:
            bundle = validate_case_score(load_json(path))
            if bundle["score_manifest_sha256"] == manifest[
                "manifest_sha256"
            ] and bundle["case_sha256"] == canonical_sha256(case):
                complete += 1
        except (OSError, ValueError, KeyError):
            pass
    print(f"Cases scored: {complete}/{len(manifest['cases'])}")
    if complete == len(manifest["cases"]):
        print("Ready to finalize scored records")


def finalize(args: argparse.Namespace) -> None:
    manifest = validate_score_manifest(load_json(args.manifest))
    root = Path(manifest["out_root"])
    case_scores = {}
    case_score_index = []
    identities = []
    diagnostics: list[str] | None = None
    for case in manifest["cases"]:
        path = root / "case_scores" / f"{case['case_id']}.json"
        bundle = validate_case_score(load_json(path))
        if bundle["score_manifest_sha256"] != manifest["manifest_sha256"]:
            raise RuntimeError(f"case belongs to another score manifest: {path}")
        if bundle["case_sha256"] != canonical_sha256(case):
            raise RuntimeError(f"case definition changed after scoring: {path}")
        if set(bundle["scores"]) != {row["trajectory_key"] for row in case["rows"]}:
            raise RuntimeError(f"case score coverage mismatch: {path}")
        current_diagnostics = list(bundle["diagnostic_dimensions"])
        if diagnostics is None:
            diagnostics = current_diagnostics
        elif diagnostics != current_diagnostics:
            raise RuntimeError("case diagnostic dimensions differ")
        identities.append(bundle["vbench_provenance"]["vbench"])
        case_scores[case["case_id"]] = bundle
        case_score_index.append(
            {
                "case_id": case["case_id"],
                "path": str(path.resolve()),
                "file_sha256": sha256_file(path),
                "bundle_sha256": bundle["bundle_sha256"],
            }
        )
    if not identities or any(item != identities[0] for item in identities[1:]):
        raise RuntimeError("all cases must use one identical VBench checkout")
    diagnostics = diagnostics or []

    scored_index = []
    for shard in manifest["shards"]:
        verify_file(
            shard["extension_manifest_path"],
            shard["extension_manifest_file_sha256"],
            label="extension manifest",
        )
        for item in shard["records"]:
            source_path = verify_file(
                item["record_path"], item["record_file_sha256"], label="combined record"
            )
            source = load_json(source_path)
            validate_generated_record(source)
            if source["record_sha256"] != item["record_sha256"]:
                raise RuntimeError(f"combined record identity changed: {source_path}")

            def with_score(artifact: dict[str, Any], action_id: str) -> dict[str, Any]:
                case_id = f"{shard['shard_id']}__{item['split']}__{action_id}"
                scores = case_scores[case_id]["scores"][item["trajectory_key"]]
                return {**artifact, "quality": quality_payload(scores, diagnostics)}

            candidates = [
                with_score(candidate, candidate["artifact_id"])
                for candidate in source["budget_candidates"]
            ]
            body = {
                "generation_status": "scored_vbench5",
                **{
                    key: source[key]
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
                "native_teacher": with_score(source["native_teacher"], "native_hr"),
                "budget_candidates": candidates,
                "candidate_count": len(candidates),
                "source_record": {
                    "path": str(source_path),
                    "file_sha256": item["record_file_sha256"],
                    "record_sha256": source["record_sha256"],
                },
                "scoring_provenance": {
                    "score_manifest_sha256": manifest["manifest_sha256"],
                    "quality_profile": "strict_vbench5_mean_v1",
                    "quality_dimensions": list(QUALITY_DIMENSIONS),
                    "diagnostic_dimensions": diagnostics,
                    "vbench": identities[0],
                    "case_score_bundles": {
                        action_id: case_scores[
                            f"{shard['shard_id']}__{item['split']}__{action_id}"
                        ]["bundle_sha256"]
                        for action_id in [
                            "native_hr",
                            *[candidate["artifact_id"] for candidate in candidates],
                        ]
                    },
                },
            }
            scored = {
                "schema": SCORED_RECORD_SCHEMA,
                "record_sha256": canonical_sha256(body),
                **body,
            }
            validate_scored_record(scored)
            output = (
                root
                / "scored_records"
                / shard["shard_id"]
                / item["split"]
                / f"{item['trajectory_key']}.json"
            )
            if output.is_file():
                previous = load_json(output)
                validate_scored_record(previous)
                if previous["record_sha256"] != scored["record_sha256"]:
                    raise RuntimeError(
                        f"refusing to replace a different scored record: {output}"
                    )
            else:
                write_json_atomic(output, scored)
            scored_index.append(
                {
                    "shard_id": shard["shard_id"],
                    "split": item["split"],
                    "trajectory_key": item["trajectory_key"],
                    "prompt_sha256": item["prompt_sha256"],
                    "seed": item["seed"],
                    "path": str(output.resolve()),
                    "file_sha256": sha256_file(output),
                    "record_sha256": scored["record_sha256"],
                }
            )
    dataset_body = {
        "score_manifest_path": str(Path(args.manifest).resolve()),
        "score_manifest_file_sha256": sha256_file(args.manifest),
        "score_manifest_sha256": manifest["manifest_sha256"],
        "quality_profile": "strict_vbench5_mean_v1",
        "quality_dimensions": list(QUALITY_DIMENSIONS),
        "diagnostic_dimensions": diagnostics,
        "vbench": identities[0],
        "model_root": manifest["model_root"],
        "action_catalog": manifest["action_catalog"],
        "action_catalog_sha256": manifest["action_catalog_sha256"],
        "case_score_bundles": case_score_index,
        "selected_splits": list(ALLOWED_SPLITS),
        "test_accessed": False,
        "trajectory_count": len(scored_index),
        "records": sorted(
            scored_index,
            key=lambda row: (
                ALLOWED_SPLITS.index(row["split"]),
                row["shard_id"],
                row["trajectory_key"],
            ),
        ),
    }
    dataset = {
        "schema": SCORED_DATASET_SCHEMA,
        "dataset_sha256": canonical_sha256(dataset_body),
        **dataset_body,
    }
    output = root / "scored_dataset_manifest.json"
    if output.is_file():
        previous = load_json(output)
        if previous.get("dataset_sha256") != dataset["dataset_sha256"]:
            raise RuntimeError(
                f"refusing to replace a different scored dataset: {output}"
            )
    else:
        write_json_atomic(output, dataset)
    print(f"Finalized {len(scored_index)} immutable scored records: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument(
        "--shard", action="append", required=True, type=parse_shard
    )
    prepare_parser.add_argument("--out-root", required=True)
    prepare_parser.add_argument("--splits", nargs="+", default=list(ALLOWED_SPLITS))
    prepare_parser.add_argument(
        "--expected-train-records-per-shard", type=int, default=300
    )
    prepare_parser.add_argument(
        "--expected-validation-records-per-shard", type=int, default=300
    )

    list_parser = sub.add_parser("list-cases")
    list_parser.add_argument("--manifest", required=True)

    status_parser = sub.add_parser("status")
    status_parser.add_argument("--manifest", required=True)

    score_parser = sub.add_parser("score-case")
    score_parser.add_argument("--manifest", required=True)
    score_parser.add_argument("--case-id", required=True)
    score_parser.add_argument("--vbench-root", required=True)
    score_parser.add_argument("--vbench-python", default=sys.executable)
    score_parser.add_argument("--ngpus", type=int, default=8)
    score_parser.add_argument("--expected-vbench-commit", default="")
    score_parser.add_argument(
        "--diagnostic-dimensions", nargs="*", default=["dynamic_degree"]
    )
    score_parser.add_argument("--force-rescore", action="store_true")

    finalize_parser = sub.add_parser("finalize")
    finalize_parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "list-cases":
        manifest = validate_score_manifest(load_json(args.manifest))
        for case in manifest["cases"]:
            print(case["case_id"])
    elif args.command == "status":
        score_status(args)
    elif args.command == "score-case":
        if args.ngpus < 1:
            raise ValueError("ngpus must be positive")
        score_case(args)
    elif args.command == "finalize":
        finalize(args)


if __name__ == "__main__":
    main()
