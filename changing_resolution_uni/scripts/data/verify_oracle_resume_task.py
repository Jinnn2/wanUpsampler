#!/usr/bin/env python3
"""Verify one resumable oracle micro-task against retained artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-root", type=Path, required=True)
    parser.add_argument("--prompt-offset", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--candidate-steps", type=int, nargs="+", required=True)
    parser.add_argument("--include-native-hr", type=int, choices=(0, 1), required=True)
    parser.add_argument("--require-latents", type=int, choices=(0, 1), default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--ignore-records",
        action="store_true",
        help="Verify retained generation artifacts without requiring packaged records.",
    )
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def valid_json(path: Path) -> dict[str, Any] | None:
    if not nonempty(path):
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.prompt_offset < 0 or args.limit < 1:
        raise SystemExit("prompt-offset must be non-negative and limit must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("seeds must be unique")
    if len(set(args.candidate_steps)) != len(args.candidate_steps):
        raise SystemExit("candidate-steps must be unique")

    root = args.part_root.resolve()
    counts = {
        "records": 0,
        "manifests": 0,
        "candidate_videos": 0,
        "native_videos": 0,
        "latents": 0,
    }
    expected = {
        "records": 0 if args.ignore_records else args.limit * len(args.seeds),
        "manifests": 0 if args.dry_run else args.limit * len(args.seeds),
        "candidate_videos": (
            0 if args.dry_run else args.limit * len(args.seeds) * len(args.candidate_steps)
        ),
        "native_videos": (
            0
            if args.dry_run or not args.include_native_hr
            else args.limit * len(args.seeds)
        ),
        "latents": (
            0
            if args.dry_run or not args.require_latents
            else args.limit * len(args.seeds) * len(args.candidate_steps)
        ),
    }
    invalid_records = 0

    for prompt_id in range(args.prompt_offset, args.prompt_offset + args.limit):
        for base_seed in args.seeds:
            if not args.ignore_records:
                record_path = root / "records" / f"p{prompt_id:06d}_s{base_seed}.json"
                record = valid_json(record_path)
                if (
                    record is not None
                    and int(record.get("prompt_id", -1)) == prompt_id
                    and int(record.get("seed", -1)) == base_seed
                    and record.get("status") != "manifest_not_found"
                    and (
                        args.dry_run
                        or isinstance(record.get("manifest"), dict)
                        or isinstance(record.get("candidates"), list)
                    )
                ):
                    counts["records"] += 1
                elif record_path.exists():
                    invalid_records += 1

            if args.dry_run:
                continue

            sample_seed = base_seed + prompt_id
            sample_id = f"{prompt_id:04d}_seed{sample_seed}"
            seed_root = root / "raw_samples" / f"seed_{base_seed}"
            manifest = valid_json(seed_root / "manifests" / f"{sample_id}.json")
            if manifest is not None:
                branch_steps = {
                    int(row["candidate_step"])
                    for row in manifest.get("branches", [])
                    if isinstance(row, dict) and "candidate_step" in row
                }
                manifest_complete = set(args.candidate_steps).issubset(branch_steps)
                if args.include_native_hr:
                    manifest_complete = manifest_complete and isinstance(
                        manifest.get("native_hr"), dict
                    )
                if manifest_complete:
                    counts["manifests"] += 1

            for step in args.candidate_steps:
                video = (
                    seed_root
                    / "videos"
                    / f"step{step:02d}"
                    / f"{sample_id}_step{step:02d}.mp4"
                )
                if nonempty(video):
                    counts["candidate_videos"] += 1
                if args.require_latents:
                    latent = (
                        seed_root
                        / "latents"
                        / f"step{step:02d}"
                        / f"{sample_id}_step{step:02d}.pt"
                    )
                    if nonempty(latent):
                        counts["latents"] += 1

            if args.include_native_hr:
                native = (
                    seed_root / "videos" / "native_hr" / f"{sample_id}_native_hr.mp4"
                )
                if nonempty(native):
                    counts["native_videos"] += 1

    complete = counts == expected
    report = {
        "schema": "oracle_resume_task_verification_v1",
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "part_root": str(root),
        "prompt_offset": args.prompt_offset,
        "limit": args.limit,
        "seeds": args.seeds,
        "candidate_steps": args.candidate_steps,
        "include_native_hr": bool(args.include_native_hr),
        "require_latents": bool(args.require_latents),
        "dry_run": args.dry_run,
        "records_checked": not args.ignore_records,
        "counts": counts,
        "expected": expected,
        "invalid_existing_records": invalid_records,
        "complete": complete,
    }
    if args.marker is not None:
        if complete:
            write_json_atomic(args.marker, report)
        else:
            args.marker.unlink(missing_ok=True)
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if complete else 3


if __name__ == "__main__":
    sys.exit(main())
