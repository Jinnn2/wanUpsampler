#!/usr/bin/env python3
"""
Inspect actual VBench dimension scores and records in oracle_dataset_1k
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
rec_dir = REPO_ROOT / "data" / "changing_resolution_uni" / "oracle_dataset_1k" / "records"

print("Inspecting records under:", rec_dir)
rec_files = sorted(rec_dir.glob("*.json"))
print(f"Total record files found: {len(rec_files)}")

if rec_files:
    sample_file = rec_files[0]
    data = json.loads(sample_file.read_text(encoding="utf-8"))
    print(f"\n--- Sample Record: {sample_file.name} ---")
    print(f"Prompt ID: {data.get('prompt_id')}, Seed: {data.get('seed')}")
    print(f"Native VBench5: {data.get('native_vbench5')}")
    print(f"Optimal Step (lambda=0.05): {data.get('optimal_step_lambda_005')}")
    print("\nCandidates:")
    for c in data.get("candidates", [])[:5]:
        print(f"  Step {c.get('step')}: vbench5={c.get('vbench5')}, dims={c.get('dimensions')}, lat={c.get('latency_seconds')}s, u_0.05={c.get('utilities', {}).get('u_0.05')}")
