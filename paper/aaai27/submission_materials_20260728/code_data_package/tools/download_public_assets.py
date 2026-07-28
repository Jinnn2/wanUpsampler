#!/usr/bin/env python3
"""Download pinned public base-model snapshots and verify critical files.

Planning is the default. Pass --execute to start the large downloads.
Custom ITU/TTD checkpoints are not public assets; export them with
export_full_repro_bundle.sh on the experiment machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "reproduction_assets.json"
    return json.loads(path.read_text(encoding="utf-8"))


def verify_model(root: Path, spec: dict[str, Any]) -> None:
    failures: list[str] = []
    for relative, expected in spec["required_files"].items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        size = path.stat().st_size
        if size != expected["size_bytes"]:
            failures.append(
                f"size mismatch: {relative}: expected {expected['size_bytes']}, got {size}"
            )
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected["sha256"]:
            failures.append(
                f"SHA-256 mismatch: {relative}: expected {expected['sha256']}, "
                f"got {actual_hash}"
            )
    if failures:
        raise RuntimeError("\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--asset",
        choices=("wan13b", "distill14b", "all"),
        default="all",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform downloads. Without this flag, only print a disk-space plan.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing files without network downloads.",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    selected = {
        "wan13b": ("wan_t2v_1p3b",),
        "distill14b": ("wan_t2v_14b_stepdistill_cfgdistill",),
        "all": (
            "wan_t2v_1p3b",
            "wan_t2v_14b_stepdistill_cfgdistill",
        ),
    }[args.asset]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    minimum_bytes = sum(
        item["size_bytes"]
        for name in selected
        for item in manifest["public_models"][name]["required_files"].values()
    )
    free_bytes = shutil.disk_usage(output_root).free
    print(f"output_root={output_root}")
    print(f"critical_files_minimum_bytes={minimum_bytes}")
    print(f"free_bytes={free_bytes}")
    for name in selected:
        spec = manifest["public_models"][name]
        print(f"{name}: {spec['repo_id']}@{spec['revision']}")

    if not args.execute and not args.verify_only:
        print("plan_only=true")
        print("Re-run with --execute after confirming at least 65 GiB free space.")
        return
    if args.execute and free_bytes < 65 * 1024**3:
        raise SystemExit(
            "less than 65 GiB free; refusing to start the two-model download"
        )

    if args.execute:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit(
                "Install the downloader first: python -m pip install huggingface_hub"
            ) from exc

        for name in selected:
            spec = manifest["public_models"][name]
            destination = output_root / name
            print(f"downloading {spec['repo_id']} to {destination}")
            snapshot_download(
                repo_id=spec["repo_id"],
                revision=spec["revision"],
                local_dir=destination,
            )

    for name in selected:
        destination = output_root / name
        verify_model(destination, manifest["public_models"][name])
        print(f"verified={name}")


if __name__ == "__main__":
    main()
