#!/usr/bin/env python3
"""Audit scored oracle records and optionally quarantine invalid legacy files."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_uni.scripts.data.oracle_record_schema import (
    FORMAL_STEPS,
    OracleRecordError,
    validate_scored_record,
)


CANONICAL_NAME = re.compile(r"^p(?P<prompt>\d{6})_s(?P<seed>-?\d+)\.json$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit oracle records; invalid files are never deleted."
    )
    parser.add_argument(
        "--dataset_dir",
        default=str(REPO_ROOT / "data" / "changing_resolution_uni" / "oracle_dataset_1k"),
        help="Dataset root containing records/.",
    )
    parser.add_argument(
        "--quarantine_dir",
        default=None,
        help="Move invalid files to this recoverable directory after auditing.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any invalid record remains in records/.",
    )
    parser.add_argument(
        "--profile",
        choices=["router", "formal"],
        default="router",
        help=(
            "router accepts legacy scalar VBench-5; formal requires five dimensions, "
            "verified aggregation, scoring provenance, and traceable latency sources."
        ),
    )
    return parser.parse_args()


def audit_record(
    path: Path, *, require_dimensions: bool, require_provenance: bool = False
) -> tuple[tuple[int, int] | None, str | None]:
    match = CANONICAL_NAME.fullmatch(path.name)
    if match is None:
        return None, "filename is not canonical p{prompt_id:06d}_s{seed}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        normalized = validate_scored_record(
            data,
            candidate_steps=FORMAL_STEPS,
            require_dimensions=require_dimensions,
            require_provenance=require_provenance,
        )
    except (OSError, json.JSONDecodeError, OracleRecordError) as exc:
        return None, str(exc)
    key = (int(normalized["prompt_id"]), int(normalized["seed"]))
    filename_key = (int(match.group("prompt")), int(match.group("seed")))
    if key != filename_key:
        return key, f"filename key {filename_key} does not match record key {key}"
    return key, None


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    records_dir = dataset_dir / "records"
    if not records_dir.is_dir():
        raise FileNotFoundError(f"Records directory not found: {records_dir}")

    manifest_errors: list[str] = []
    if args.profile == "formal":
        manifest_path = dataset_dir / "dataset_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("quality_profile") != "strict_vbench5_v1":
                manifest_errors.append("quality_profile is not strict_vbench5_v1")
            record_files = manifest.get("record_files")
            record_hashes = manifest.get("record_sha256")
            if not isinstance(record_files, list) or not isinstance(record_hashes, dict):
                manifest_errors.append("record_files or record_sha256 is missing")
            else:
                actual_names = {path.name for path in records_dir.glob("*.json")}
                declared_names = {str(name) for name in record_files}
                if actual_names != declared_names:
                    manifest_errors.append(
                        "record file coverage mismatch; "
                        f"undeclared={sorted(actual_names - declared_names)[:20]}, "
                        f"missing={sorted(declared_names - actual_names)[:20]}"
                    )
                if set(record_hashes) != declared_names:
                    manifest_errors.append("record_sha256 keys do not match record_files")
                for name in sorted(actual_names & declared_names):
                    if sha256_file(records_dir / name) != record_hashes.get(name):
                        manifest_errors.append(f"record SHA256 mismatch: {name}")
                        if len(manifest_errors) >= 50:
                            break
        except (OSError, json.JSONDecodeError) as exc:
            manifest_errors.append(f"invalid dataset manifest: {exc}")

    invalid: list[tuple[Path, str]] = []
    seen: dict[tuple[int, int], Path] = {}
    valid = 0
    for path in sorted(records_dir.glob("*.json")):
        key, reason = audit_record(
            path,
            require_dimensions=args.profile == "formal",
            require_provenance=args.profile == "formal",
        )
        if reason is None and key is not None and key in seen:
            reason = f"duplicate prompt/seed also present in {seen[key].name}"
        if reason is not None:
            invalid.append((path, reason))
            continue
        seen[key] = path
        valid += 1

    print(f"Dataset: {dataset_dir}")
    print(f"Valid scored records: {valid}")
    print(f"Invalid or legacy records: {len(invalid)}")
    print(f"Manifest errors: {len(manifest_errors)}")
    for reason in manifest_errors[:50]:
        print(f"  - dataset_manifest.json: {reason}")
    for path, reason in invalid[:50]:
        print(f"  - {path.name}: {reason}")
    if len(invalid) > 50:
        print(f"  ... and {len(invalid) - 50} more")

    if args.quarantine_dir and invalid:
        quarantine_dir = Path(args.quarantine_dir).resolve()
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        for path, _ in invalid:
            destination = quarantine_dir / path.name
            if destination.exists():
                destination = quarantine_dir / (
                    f"{path.stem}_{path.stat().st_mtime_ns}{path.suffix}"
                )
            shutil.move(str(path), str(destination))
        print(f"Quarantined {len(invalid)} files to: {quarantine_dir}")
        invalid = []

    if args.strict and (invalid or manifest_errors):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
