#!/usr/bin/env python3
"""
Find and report all directories containing .mp4 video files in data/changing_resolution_uni/
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
data_root = REPO_ROOT / "data" / "changing_resolution_uni"

print(f"Scanning for MP4 video files under {data_root}...")
found = {}
for p in data_root.rglob("*.mp4"):
    parent = p.parent
    found[str(parent)] = found.get(str(parent), 0) + 1

if not found:
    print("No .mp4 files found anywhere under data/changing_resolution_uni/")
else:
    for d, count in sorted(found.items()):
        print(f"  {count:5d} videos: {d}")
