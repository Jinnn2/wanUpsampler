"""Audit reusable Native-HR50 artifacts for a finalized matched-star dataset.

The audit is deliberately read-only with respect to every source dataset.  It
matches artifacts by prompt hash and actual generator seed, requires an
identical Native-HR runtime configuration, and optionally verifies video
SHA-256 before writing a standalone reuse map.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.data_protocol import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    write_json_atomic,
)


SCHEMA = "univ_native_hr_reuse_audit_v1"


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value


def json_fingerprint(path: Path) -> str:
    return canonical_sha256(load_json(path))


def resolve_lineage(
    phase4_root: Path,
    *,
    phase3_root: Path | None,
    phase2_root: Path | None,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    phase4_plan_path = phase4_root / "sparse_action_plan.json"
    phase4_plan = load_json(phase4_plan_path)
    source_phase3 = phase4_plan.get("source_phase3")
    if not isinstance(source_phase3, dict):
        raise ValueError("Phase4 plan has no source_phase3 identity")
    resolved_phase3 = (
        phase3_root.resolve()
        if phase3_root is not None
        else Path(str(source_phase3["root"])).resolve()
    )
    phase3_plan_path = resolved_phase3 / "sparse_action_plan.json"
    phase3_plan = load_json(phase3_plan_path)
    source_phase2 = phase3_plan.get("source_phase2")
    if not isinstance(source_phase2, dict):
        raise ValueError("Phase3 plan has no source_phase2 identity")
    resolved_phase2 = (
        phase2_root.resolve()
        if phase2_root is not None
        else Path(str(source_phase2["root"])).resolve()
    )
    return phase4_plan, phase4_plan_path, phase3_plan, resolved_phase2


def expected_groups(plan: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for group in plan.get("groups", []):
        key = (str(group["prompt_sha256"]), int(group["seed"]))
        if key in result:
            raise ValueError(f"duplicate Phase4 prompt/seed identity: {key}")
        result[key] = {
            "group_id": str(group["group_id"]),
            "prompt_key": str(group["prompt_key"]),
            "prompt_sha256": key[0],
            "base_seed": int(group["base_seed"]),
            "seed": key[1],
        }
    if not result:
        raise ValueError("Phase4 plan contains no groups")
    return result


def discover_roots(search_roots: list[Path], include_roots: list[Path]) -> list[Path]:
    candidates = {path.resolve() for path in include_roots}
    for search_root in search_roots:
        root = search_root.resolve()
        if (root / "records").is_dir():
            candidates.add(root)
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("univ_prompt_budget"):
                candidates.add(child.resolve())
    return sorted(candidates, key=str)


def native_config_identity(root: Path) -> tuple[str | None, str | None]:
    path = root / "configs" / "native_hr50.json"
    if not path.is_file():
        return None, None
    return str(path.resolve()), json_fingerprint(path)


def inspect_artifact(
    artifact: Any,
    *,
    verify_video_sha: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(artifact, dict):
        return None, "native_teacher_missing"
    required = ("video_path", "video_sha256", "video_bytes", "cost")
    missing = [key for key in required if key not in artifact]
    if missing:
        return None, f"native_teacher_missing_fields:{','.join(missing)}"
    video = Path(str(artifact["video_path"])).resolve()
    if not video.is_file():
        return None, f"video_missing:{video}"
    expected_bytes = int(artifact["video_bytes"])
    if video.stat().st_size != expected_bytes:
        return None, f"video_size_mismatch:{video}"
    expected_sha = str(artifact["video_sha256"])
    if len(expected_sha) != 64:
        return None, f"invalid_video_sha256:{video}"
    if verify_video_sha:
        observed_sha = sha256_file(video)
        if observed_sha != expected_sha:
            return None, f"video_sha256_mismatch:{video}"
    cost = artifact["cost"]
    if not isinstance(cost, dict) or float(cost.get("pipeline_seconds", 0.0)) <= 0:
        return None, f"invalid_pipeline_seconds:{video}"
    return {
        "video_path": str(video),
        "video_sha256": expected_sha,
        "video_bytes": expected_bytes,
        "cost": json.loads(json.dumps(cost)),
        "runtime_sidecar_path": artifact.get("runtime_sidecar_path"),
        "runtime_sidecar_sha256": artifact.get("runtime_sidecar_sha256"),
        "video_sha256_verified": verify_video_sha,
    }, None


def audit(
    *,
    phase4_root: Path,
    phase3_root: Path | None = None,
    phase2_root: Path | None = None,
    search_roots: list[Path] | None = None,
    include_roots: list[Path] | None = None,
    verify_video_sha: bool = False,
) -> dict[str, Any]:
    phase4_root = phase4_root.resolve()
    phase4_plan, phase4_plan_path, phase3_plan, bound_phase2 = resolve_lineage(
        phase4_root,
        phase3_root=phase3_root,
        phase2_root=phase2_root,
    )
    wanted = expected_groups(phase4_plan)
    baseline_config_path, baseline_config_sha = native_config_identity(bound_phase2)
    if baseline_config_sha is None:
        raise FileNotFoundError(
            f"bound Phase2 Native-HR config not found: {bound_phase2}"
        )

    roots = discover_roots(
        search_roots or [phase4_root.parent],
        [bound_phase2, *(include_roots or [])],
    )
    candidates: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    invalid: list[dict[str, Any]] = []
    root_summaries = []
    record_files_scanned = 0
    for root in roots:
        config_path, config_sha = native_config_identity(root)
        compatible = config_sha == baseline_config_sha
        matched = valid = 0
        records = sorted((root / "records").glob("**/*.json"))
        record_files_scanned += len(records)
        for record_path in records:
            try:
                record = load_json(record_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                invalid.append(
                    {
                        "source_record_path": str(record_path.resolve()),
                        "reason": f"record_read_error:{exc}",
                    }
                )
                continue
            if "prompt_sha256" not in record or "seed" not in record:
                continue
            key = (str(record["prompt_sha256"]), int(record["seed"]))
            if key not in wanted:
                continue
            matched += 1
            if not compatible:
                invalid.append(
                    {
                        "group_id": wanted[key]["group_id"],
                        "source_record_path": str(record_path.resolve()),
                        "source_root": str(root),
                        "reason": "native_config_mismatch",
                    }
                )
                continue
            artifact, reason = inspect_artifact(
                record.get("native_teacher"),
                verify_video_sha=verify_video_sha,
            )
            if artifact is None:
                invalid.append(
                    {
                        "group_id": wanted[key]["group_id"],
                        "source_record_path": str(record_path.resolve()),
                        "source_root": str(root),
                        "reason": reason,
                    }
                )
                continue
            valid += 1
            candidates[key].append(
                {
                    **wanted[key],
                    "artifact": artifact,
                    "source_root": str(root),
                    "source_record_path": str(record_path.resolve()),
                    "source_record_sha256": sha256_file(record_path),
                    "native_config_path": config_path,
                    "native_config_sha256": config_sha,
                    "bound_phase2_source": root == bound_phase2,
                }
            )
        root_summaries.append(
            {
                "root": str(root),
                "native_config_path": config_path,
                "native_config_sha256": config_sha,
                "config_matches_bound_phase2": compatible,
                "record_files": len(records),
                "identity_matches": matched,
                "valid_artifacts": valid,
            }
        )

    selected = []
    duplicates = []
    conflicts = []
    for key, rows in sorted(candidates.items()):
        bound = [row for row in rows if row["bound_phase2_source"]]
        ordered = sorted(rows, key=lambda row: (not row["bound_phase2_source"], row["source_root"]))
        chosen = bound[0] if bound else ordered[0]
        distinct_hashes = sorted({row["artifact"]["video_sha256"] for row in rows})
        if len(rows) > 1:
            duplicate = {
                "group_id": chosen["group_id"],
                "candidate_count": len(rows),
                "distinct_video_sha256": distinct_hashes,
                "selected_source_record_path": chosen["source_record_path"],
                "candidate_source_record_paths": [row["source_record_path"] for row in ordered],
            }
            duplicates.append(duplicate)
            if len(distinct_hashes) > 1:
                conflicts.append(duplicate)
        selected.append(chosen)

    selected_group_ids = {row["group_id"] for row in selected}
    missing = [
        group
        for group in sorted(wanted.values(), key=lambda row: row["group_id"])
        if group["group_id"] not in selected_group_ids
    ]
    bound_count = sum(row["bound_phase2_source"] for row in selected)
    body = {
        "phase4_root": str(phase4_root),
        "phase4_plan_path": str(phase4_plan_path.resolve()),
        "phase4_plan_file_sha256": sha256_file(phase4_plan_path),
        "phase4_plan_sha256": phase4_plan.get("plan_sha256"),
        "phase3_plan_sha256": phase3_plan.get("plan_sha256"),
        "bound_phase2_root": str(bound_phase2),
        "bound_native_config_path": baseline_config_path,
        "bound_native_config_sha256": baseline_config_sha,
        "video_sha256_verified": verify_video_sha,
        "counts": {
            "expected_groups": len(wanted),
            "roots_scanned": len(roots),
            "record_files_scanned": record_files_scanned,
            "reused_groups": len(selected),
            "bound_phase2_reused_groups": bound_count,
            "other_reused_groups": len(selected) - bound_count,
            "missing_groups": len(missing),
            "duplicate_groups": len(duplicates),
            "conflict_groups": len(conflicts),
            "invalid_candidates": len(invalid),
        },
        "roots": root_summaries,
        "selected": selected,
        "missing": missing,
        "duplicates": duplicates,
        "conflicts": conflicts,
        "invalid_candidates": invalid,
    }
    return {"schema": SCHEMA, "audit_sha256": canonical_sha256(body), **body}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-root", required=True)
    parser.add_argument("--phase3-root")
    parser.add_argument("--phase2-root")
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--include-root", action="append", default=[])
    parser.add_argument("--verify-video-sha", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    phase4_root = Path(args.phase4_root)
    report = audit(
        phase4_root=phase4_root,
        phase3_root=Path(args.phase3_root) if args.phase3_root else None,
        phase2_root=Path(args.phase2_root) if args.phase2_root else None,
        search_roots=[Path(path) for path in args.search_root]
        if args.search_root
        else None,
        include_roots=[Path(path) for path in args.include_root],
        verify_video_sha=args.verify_video_sha,
    )
    output = (
        Path(args.out).resolve()
        if args.out
        else phase4_root.resolve() / "native_hr_reuse_audit.json"
    )
    write_json_atomic(output, report)
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(output)


if __name__ == "__main__":
    main()
