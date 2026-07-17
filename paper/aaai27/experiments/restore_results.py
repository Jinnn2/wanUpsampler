from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
MEDIA_SUFFIXES = {".avi", ".gif", ".mkv", ".mov", ".mp4", ".webm"}


def main() -> None:
    args = parse_args()
    export_root = Path(args.export_root).resolve()
    project_root = Path(args.project_root).resolve()
    plan = build_restore_plan(export_root, project_root)
    verify_checksums(export_root)
    total_bytes = sum(source.stat().st_size for source, _ in plan)
    print(f"Verified export files : {len(read_checksums(export_root))}")
    print(f"Result files to restore: {len(plan)}")
    print(f"Logical result bytes   : {total_bytes}")
    target_root = expected_target_root(project_root)
    print(f"Target root            : {target_root}")
    if not args.execute:
        print("Dry run only; add --execute to restore.")
        return
    restore_results(plan, target_root, hardlink=args.hardlink)
    print(f"Restored atomically    : {target_root}")


def build_restore_plan(export_root: Path, project_root: Path) -> list[tuple[Path, Path]]:
    inventory = load_json(export_root / "core/result_inventory.json")
    path_map = load_json(export_root / "provenance/path_map.json")
    recorded_root = Path(inventory.get("canonical_results_root", "")).resolve()
    target_root = expected_target_root(project_root)
    if recorded_root.name != "aaai27_experiments" or recorded_root.parent.name != "outputs":
        raise SystemExit(f"Unsafe canonical_results_root in inventory: {recorded_root}")
    entries: dict[Path, Path] = {}
    for item in path_map.get("files", []):
        original = Path(str(item.get("source", ""))).resolve()
        try:
            relative = original.relative_to(recorded_root)
        except ValueError:
            continue
        exported = safe_export_path(export_root, str(item.get("exported", "")))
        if not exported.is_file():
            raise SystemExit(f"Mapped export file is missing: {exported}")
        destination = target_root / relative
        previous = entries.get(destination)
        if previous is not None and sha256(previous) != sha256(exported):
            raise SystemExit(f"Conflicting exported copies for restored path: {destination}")
        entries[destination] = exported
    if not entries:
        raise SystemExit("No files under canonical_results_root were found in path_map.json")
    return sorted(((source, destination) for destination, source in entries.items()), key=lambda item: str(item[1]))


def restore_results(plan: list[tuple[Path, Path]], target_root: Path, *, hardlink: bool) -> None:
    if target_root.exists():
        raise SystemExit(f"Refusing to overwrite an existing result root: {target_root}")
    target_root.parent.mkdir(parents=True, exist_ok=True)
    raw_staging = tempfile.mkdtemp(prefix=".aaai27_experiments.restore.", dir=target_root.parent)
    staging = Path(raw_staging).resolve()
    if staging.parent != target_root.parent.resolve():
        raise SystemExit(f"Unsafe restore staging path: {staging}")
    try:
        for source, destination in plan:
            relative = destination.relative_to(target_root)
            staged = staging / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            if hardlink and source.suffix.lower() in MEDIA_SUFFIXES:
                os.link(source, staged)
            else:
                shutil.copy2(source, staged)
        staging.replace(target_root)
    except BaseException:
        if staging.is_dir() and staging.parent == target_root.parent.resolve():
            shutil.rmtree(staging)
        raise


def verify_checksums(export_root: Path) -> None:
    rows = read_checksums(export_root)
    failures = []
    for expected, relative in rows:
        path = safe_export_path(export_root, relative)
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256(path) != expected:
            failures.append(f"checksum mismatch: {relative}")
    if failures:
        preview = "; ".join(failures[:10])
        raise SystemExit(f"Export checksum verification failed ({len(failures)} file(s)): {preview}")


def read_checksums(export_root: Path) -> list[tuple[str, str]]:
    path = export_root / "SHA256SUMS"
    if not path.is_file():
        raise SystemExit(f"Missing checksum manifest: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise SystemExit(f"Invalid SHA256SUMS line {line_number}") from exc
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise SystemExit(f"Invalid SHA-256 on SHA256SUMS line {line_number}")
        rows.append((digest, relative))
    if not rows:
        raise SystemExit("SHA256SUMS contains no files")
    return rows


def safe_export_path(export_root: Path, relative: str) -> Path:
    path = (export_root / relative).resolve()
    try:
        path.relative_to(export_root.resolve())
    except ValueError as exc:
        raise SystemExit(f"Export path escapes bundle root: {relative}") from exc
    return path


def expected_target_root(project_root: Path) -> Path:
    return (project_root / "outputs/aaai27_experiments").resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid or missing JSON {path}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify and restore a deleted canonical AAAI result root.")
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--execute", action="store_true", help="Perform the restore after verification and planning.")
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Hard-link immutable media; copy mutable metadata so later collection cannot alter the export.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
