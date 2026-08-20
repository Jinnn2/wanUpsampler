#!/usr/bin/env python3
"""
Inspect record filenames, count duplicates per (prompt_id, seed), and check vbench5 values
"""
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
rec_dir = REPO_ROOT / "data" / "changing_resolution_uni" / "oracle_dataset_1k" / "records"

files = sorted(rec_dir.glob("*.json"))
print(f"Total .json files in records/: {len(files)}")

by_key = defaultdict(list)
zero_vbench_files = []
valid_vbench_files = []

for f in files:
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        pid = int(data.get("prompt_id", -1))
        seed = int(data.get("seed", -1))
        by_key[(pid, seed)].append(f.name)
        
        # Check vbench5 in candidates
        cands = data.get("candidates", [])
        vbs = [float(c.get("vbench5", 0.0)) for c in cands]
        if vbs and max(vbs) > 0.1:
            valid_vbench_files.append(f.name)
        else:
            zero_vbench_files.append(f.name)
    except Exception as e:
        print(f"Error reading {f.name}: {e}")

print(f"Unique (prompt_id, seed) pairs: {len(by_key)}")
dup_counts = [len(v) for v in by_key.values()]
print(f"Duplicates per key: min={min(dup_counts) if dup_counts else 0}, max={max(dup_counts) if dup_counts else 0}, avg={sum(dup_counts)/len(dup_counts) if dup_counts else 0:.2f}")

print(f"Files with VALID VBench5 (>0.1): {len(valid_vbench_files)}")
print(f"Files with ZERO VBench5 (<=0.1): {len(zero_vbench_files)}")

print("\nSample filenames:")
for f in files[:10]:
    print(" ", f.name)
if len(files) > 10:
    print("  ...")
    for f in files[-10:]:
        print(" ", f.name)
