#!/usr/bin/env python3
"""
Clean up duplicate or legacy un-scored record files in records/
Keeps only genuine scored records matching p{prompt_id:06d}_s{seed}.json
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
rec_dir = REPO_ROOT / "data" / "changing_resolution_uni" / "oracle_dataset_1k" / "records"

print(f"Cleaning legacy un-scored records in {rec_dir}...")
deleted = 0
kept = 0

for f in sorted(rec_dir.glob("*.json")):
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        cands = data.get("candidates", [])
        has_scores = any(float(c.get("vbench5", 0.0)) > 0.1 for c in cands)
        
        # If it doesn't follow the unified naming convention p*_s*.json or has zero scores
        if not f.name.startswith("p") or "_s" not in f.name or not has_scores:
            f.unlink()
            deleted += 1
        else:
            kept += 1
    except Exception:
        f.unlink()
        deleted += 1

print(f"Cleanup complete! Kept: {kept} valid scored records, Removed: {deleted} legacy/duplicate files.")
