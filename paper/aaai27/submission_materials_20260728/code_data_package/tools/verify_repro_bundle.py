#!/usr/bin/env python3
"""Verify custom checkpoints and file hashes in a reproduction bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, size: int, expected_sha256: str) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing: {path}"]
    actual_size = path.stat().st_size
    if actual_size != size:
        errors.append(f"size mismatch: {path}: expected {size}, got {actual_size}")
        return errors
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        errors.append(
            f"SHA-256 mismatch: {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    return errors


def load_manifest(bundle_root: Path) -> dict[str, Any]:
    candidates = (
        bundle_root / "reproduction_assets.json",
        Path(__file__).resolve().parent.parent / "reproduction_assets.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError("reproduction_assets.json not found")


def verify_bundle_hashes(bundle_root: Path) -> list[str]:
    hash_list = bundle_root / "BUNDLE_SHA256SUMS"
    if not hash_list.is_file():
        return ["BUNDLE_SHA256SUMS is missing"]
    errors: list[str] = []
    for line_number, line in enumerate(
        hash_list.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError:
            errors.append(f"invalid BUNDLE_SHA256SUMS line {line_number}")
            continue
        relative = relative.removeprefix("*").removeprefix("./")
        path = bundle_root / relative
        if not path.is_file():
            errors.append(f"bundle member missing: {relative}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            errors.append(
                f"bundle member SHA-256 mismatch: {relative}: "
                f"expected {expected}, got {actual}"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--require-checkpoints", action="store_true")
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve()
    manifest = load_manifest(bundle_root)
    errors = verify_bundle_hashes(bundle_root)
    verified_checkpoints = 0
    checkpoint_root = bundle_root / "checkpoints" / "custom"
    for logical_name, spec in manifest["custom_checkpoints"].items():
        path = checkpoint_root / spec["bundle_name"]
        if path.exists() or args.require_checkpoints:
            file_errors = verify_file(path, spec["size_bytes"], spec["sha256"])
            errors.extend(f"{logical_name}: {item}" for item in file_errors)
            if not file_errors:
                verified_checkpoints += 1

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("bundle_hashes=ok")
    print(f"verified_custom_checkpoints={verified_checkpoints}")


if __name__ == "__main__":
    main()
