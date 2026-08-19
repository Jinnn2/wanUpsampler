#!/usr/bin/env python3
"""
Inspect and locate all generated oracle dataset directories, prompt ranges,
seed distributions, and video completion status.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect oracle dataset directories.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(REPO_ROOT / "data" / "changing_resolution_uni"),
        help="Base directory containing dataset directories.",
    )
    return parser.parse_args()


def inspect_directory(dir_path: Path) -> dict:
    """Inspect an oracle dataset directory."""
    records_dir = dir_path / "records"
    parts_dir = dir_path / "_parts"
    t5_dir = dir_path / "t5_embeddings"
    manifest_file = dir_path / "dataset_manifest.json"

    # Find records either in records/ or in _parts/*/records/
    record_files = list(records_dir.glob("*.json")) if records_dir.exists() else []
    part_dirs = list(parts_dir.glob("part_*")) if parts_dir.exists() else []

    part_records = {}
    for p in part_dirs:
        p_recs = list((p / "records").glob("*.json"))
        part_records[p.name] = len(p_recs)

    # Count prompts and seeds
    all_recs = list(record_files)
    for p in part_dirs:
        all_recs.extend((p / "records").glob("*.json"))

    prompts = set()
    seeds = set()
    sample_ids = []
    for r in all_recs:
        try:
            name = r.stem  # p000042_s42
            parts = name.split("_")
            if len(parts) >= 2 and parts[0].startswith("p") and parts[1].startswith("s"):
                p_id = int(parts[0][1:])
                s_id = int(parts[1][1:])
                prompts.add(p_id)
                seeds.add(s_id)
                sample_ids.append(name)
        except Exception:
            pass

    t5_count = len(list(t5_dir.glob("*.npz"))) if t5_dir.exists() else 0

    # Check for raw videos
    mp4_count = len(list(dir_path.rglob("*.mp4")))

    return {
        "path": str(dir_path),
        "exists": dir_path.is_dir(),
        "total_json_records": len(all_recs),
        "unique_prompts": len(prompts),
        "prompt_range": f"[{min(prompts)} ~ {max(prompts)}]" if prompts else "None",
        "seeds": sorted(list(seeds)),
        "t5_embeddings": t5_count,
        "parts": part_records,
        "raw_mp4_videos": mp4_count,
        "has_manifest": manifest_file.is_file(),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    print("=" * 85)
    print(f" SCANNING ORACLE DATASETS UNDER: {data_root}")
    print("=" * 85)

    if not data_root.is_dir():
        print(f"Directory not found: {data_root}")
        # Search parent directories
        parent = data_root.parent
        if parent.is_dir():
            print(f"\nSearching available subdirectories in parent: {parent}")
            for sub in sorted(parent.iterdir()):
                if sub.is_dir():
                    print(f"  - {sub.name}")
        return

    found_dirs = []
    for entry in sorted(data_root.iterdir()):
        if entry.is_dir():
            info = inspect_directory(entry)
            if info["total_json_records"] > 0 or info["parts"] or info["t5_embeddings"] > 0:
                found_dirs.append(info)

    if not found_dirs:
        print(f"No oracle dataset subdirectories found in {data_root}.")
        print("Existing subdirectories:")
        for sub in sorted(data_root.iterdir()):
            print(f"  - {sub.name}")
        return

    for idx, info in enumerate(found_dirs, start=1):
        print(f"\n[{idx}] Directory: {info['path']}")
        print(f"    - Unique Prompts      : {info['unique_prompts']} {info['prompt_range']}")
        print(f"    - Seeds Found         : {info['seeds']}")
        print(f"    - Total Trajectory JSON: {info['total_json_records']}")
        print(f"    - Parts Distribution  : {info['parts']}")
        print(f"    - T5 Embeddings       : {info['t5_embeddings']} files")
        print(f"    - Raw MP4 Videos      : {info['raw_mp4_videos']}")
        print(f"    - Master Manifest     : {'[YES]' if info['has_manifest'] else '[NO]'}")

    print("\n" + "=" * 85)


if __name__ == "__main__":
    main()
