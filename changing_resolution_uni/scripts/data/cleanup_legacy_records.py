#!/usr/bin/env python3
"""Audit scored oracle records and optionally quarantine invalid legacy files."""
from __future__ import annotations

import argparse
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
        help="router accepts scalar VBench-5; formal also requires all five dimensions.",
    )
    return parser.parse_args()


def audit_record(
    path: Path, *, require_dimensions: bool
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

    invalid: list[tuple[Path, str]] = []
    seen: dict[tuple[int, int], Path] = {}
    valid = 0
    for path in sorted(records_dir.glob("*.json")):
        key, reason = audit_record(
            path, require_dimensions=args.profile == "formal"
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

    if args.strict and invalid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
