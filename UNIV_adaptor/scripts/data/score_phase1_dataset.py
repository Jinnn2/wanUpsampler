"""Evaluate finalized standalone phase1 videos; never runs the generator."""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from UNIV_adaptor.data_protocol import canonical_sha256, sha256_file, write_json_atomic
from UNIV_adaptor.combined_v3 import QUALITY_DIMENSIONS
from UNIV_adaptor.low_budget_protocol import validate_plan
from UNIV_adaptor.scripts.data.run_low_budget_extension import validate_manifest

DIAGNOSTICS = ("dynamic_degree", "overall_consistency")
PAIRS = (("P1_B20_BASE", "P1_B25_SPATIAL"),
         ("P1_B25_SPATIAL", "P1_B30_SPATIAL"),
         ("P1_B30_SPATIAL", "P1_B40_SPATIAL"),
         ("P1_B30_SPATIAL", "P1_B35_HR"))

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def collect(root):
    manifest = validate_manifest(read(root / "extension_manifest.json"))
    plan = validate_plan(read(manifest["plan_path"]))
    if plan["plan_sha256"] != manifest["plan_sha256"]:
        raise ValueError("generation manifest/plan mismatch")
    if manifest["base_dataset"] is not None:
        raise ValueError("This evaluator requires standalone phase1 records")
    rows = []
    for assignment in plan["assignments"]:
        if assignment["split"] not in ("train", "validation"):
            continue
        path = root / "combined_records" / assignment["split"] / (assignment["trajectory_key"] + ".json")
        record = read(path)
        body = {k: v for k, v in record.items() if k not in ("schema", "record_sha256")}
        if canonical_sha256(body) != record["record_sha256"]:
            raise ValueError(f"record hash mismatch: {path}")
        for key in ("trajectory_key", "split", "prompt_id", "prompt", "prompt_sha256", "seed"):
            if record[key] != assignment[key]:
                raise ValueError(f"record identity mismatch: {path}: {key}")
        expected = {c["artifact_id"]: c for c in assignment["low_budget_candidates"]}
        candidates = record["budget_candidates"]
        if len(candidates) != 6 or {c["artifact_id"] for c in candidates} != set(expected):
            raise ValueError(f"incomplete candidates: {path}")
        for c in candidates:
            if c["action_key"] != expected[c["artifact_id"]]["action_key"]:
                raise ValueError("candidate action mismatch")
            for file_key, hash_key in (("video_path", "video_sha256"), ("runtime_sidecar_path", "runtime_sidecar_sha256")):
                if sha256_file(c[file_key]) != c[hash_key]:
                    raise ValueError(f"changed artifact: {c[file_key]}")
            seconds = float(c["cost"]["pipeline_seconds"])
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("pipeline_seconds must be finite and positive")
            rows.append(dict(split=record["split"], prompt_id=record["prompt_id"],
                             seed=record["seed"], prompt=record["prompt"],
                             trajectory_key=record["trajectory_key"], action_id=c["artifact_id"],
                             stem=record["trajectory_key"] + "__" + c["artifact_id"],
                             video_path=c["video_path"], video_sha256=c["video_sha256"],
                             record_sha256=record["record_sha256"],
                             pipeline_seconds=seconds, proxy=c["proxy_compute_density"]))
    if not rows:
        raise ValueError("no selected records")
    return rows

def csv_write(path, rows):
    if not rows:
        raise ValueError(f"empty report: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def report(rows, scores, out):
    if set(scores) != {r["stem"] for r in rows}:
        raise ValueError("score coverage does not exactly match input videos")
    enriched = []
    for row in rows:
        values = {d: float(scores[row["stem"]][d]) for d in (*QUALITY_DIMENSIONS, *DIAGNOSTICS)}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("nonfinite quality score")
        enriched.append({**row, **values, "vbench5": st.fmean(values[d] for d in QUALITY_DIMENSIONS)})
    csv_write(out / "quality_by_video.csv", enriched)
    # Average seeds first: each prompt has equal weight within its split.
    grouped = defaultdict(list)
    for r in enriched:
        grouped[(r["split"], r["prompt_id"], r["action_id"])].append(r)
    fields = ("pipeline_seconds", "vbench5", *QUALITY_DIMENSIONS, *DIAGNOSTICS)
    prompts = [dict(split=k[0], prompt_id=k[1], action_id=k[2], seeds=len(v),
                    **{f: st.fmean(r[f] for r in v) for f in fields}) for k, v in sorted(grouped.items())]
    csv_write(out / "quality_by_prompt.csv", prompts)
    catalog = defaultdict(list)
    for r in prompts:
        catalog[(r["split"], r["action_id"])].append(r)
    summary = [dict(split=k[0], action_id=k[1], prompts=len(v),
                    **{f: st.fmean(r[f] for r in v) for f in fields}) for k, v in sorted(catalog.items())]
    csv_write(out / "quality_cost_summary.csv", summary)
    lookup = {(r["split"], r["prompt_id"], r["action_id"]): r for r in prompts}
    paired = []
    for split, pid in sorted({(r["split"], r["prompt_id"]) for r in prompts}):
        for a, b in PAIRS:
            x, y = lookup[(split, pid, a)], lookup[(split, pid, b)]
            dt = y["pipeline_seconds"] - x["pipeline_seconds"]
            dq = y["vbench5"] - x["vbench5"]
            paired.append(dict(split=split, prompt_id=pid, before=a, after=b,
                               delta_seconds=dt, delta_vbench5=dq,
                               gain_per_extra_second=dq/dt if dt > 0 else None,
                               **{"delta_"+d: y[d]-x[d] for d in (*QUALITY_DIMENSIONS, *DIAGNOSTICS)}))
    csv_write(out / "paired_gains.csv", paired)
    lines = ["# Phase1 quality and cost", "",
             "VBench5 is an unweighted working proxy, NOT the official VBench total score.",
             "Seeds are averaged within each prompt; train and validation remain separate.",
             "Pipeline time is observed generation latency, not a dedicated warmed benchmark.",
             "No native-HR reference was generated: native fidelity and native speedup are unavailable.",
             "The current candidates have different costs; this is not a fixed-budget allocator comparison.",
             "Dynamic degree and overall consistency are diagnostics, excluded from VBench5.", "",
             "| Split | Action | Prompts | Seconds | VBench5 |", "|---|---|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['split']} | {r['action_id']} | {r['prompts']} | {r['pipeline_seconds']:.2f} | {r['vbench5']:.5f} |")
    lines += ["", "## Paired prompt-level gains", "", "| Split | Comparison | Mean quality delta | Mean seconds delta | Positive quality fraction |", "|---|---|---:|---:|---:|"]
    for split in sorted({r["split"] for r in paired}):
        for a, b in PAIRS:
            items = [r for r in paired if r["split"] == split and r["before"] == a and r["after"] == b]
            lines.append(f"| {split} | {a} -> {b} | {st.fmean(r['delta_vbench5'] for r in items):.5f} | {st.fmean(r['delta_seconds'] for r in items):.2f} | {st.fmean(r['delta_vbench5'] > 0 for r in items):.1%} |")
    (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "score", "report", "all"))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--vbench-root", default="/mnt/afs_2/houze/VBench")
    parser.add_argument("--vbench-python", default=sys.executable)
    parser.add_argument("--ngpus", type=int, default=8)
    args = parser.parse_args()
    if args.ngpus < 1:
        parser.error("ngpus must be positive")
    root = Path(args.dataset_root).resolve()
    rows = collect(root)
    digest = canonical_sha256(rows)
    out = root / "metrics" / "phase1_quality"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Verified {len(rows)} videos, {len({r['trajectory_key'] for r in rows})} trajectories", flush=True)
    if args.mode == "check":
        return
    score_path = out / "scores.json"
    if args.mode in ("score", "all"):
        from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import inspect_vbench_checkout, score_case_directory
        inputs = out / "inputs"
        inputs.mkdir(exist_ok=True)
        pmap = {}
        import os
        import shutil
        for row in rows:
            dst = inputs / (row["stem"] + ".mp4")
            src = Path(row["video_path"]).resolve()
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            if dst.is_symlink() or sha256_file(dst) != row["video_sha256"]:
                raise ValueError(f"staged content mismatch: {dst}")
            pmap[str(dst)] = row["prompt"]
        write_json_atomic(out / "prompt_map.json", pmap)
        vroot = Path(args.vbench_root).resolve()
        identity = inspect_vbench_checkout(vroot, expected_commit=None)
        bundle = score_case_directory(vroot, args.vbench_python, inputs, out / "prompt_map.json", out / "vbench",
                                      list(QUALITY_DIMENSIONS)+list(DIAGNOSTICS), list(QUALITY_DIMENSIONS), list(DIAGNOSTICS), args.ngpus, False, identity)
        write_json_atomic(score_path, dict(input_sha256=digest, scores=bundle.scores, provenance=bundle.provenance))
    if args.mode in ("report", "all"):
        payload = read(score_path)
        if payload["input_sha256"] != digest:
            raise ValueError("scores belong to changed input records")
        report(rows, payload["scores"], out)
        print(out / "report.md")

if __name__ == "__main__":
    main()
