"""Strict paired VBench-5 plus dynamic/semantic diagnostics; no test/training access."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import canonical_sha256, sha256_file
from UNIV_adaptor.hy15_protocol import immutable_json, read, record_paths, verify_plan, verify_record
from UNIV_adaptor.scripts.data.score_controlled_factor_dataset import (
    QUALITY_DIMENSIONS, DIAGNOSTIC_DIMENSIONS, stage_inputs,
)
from UNIV_adaptor.scripts.data.score_phase2_dataset import output_lock

DIMENSIONS = (*QUALITY_DIMENSIONS, *DIAGNOSTIC_DIMENSIONS)


def csv_write(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(root, *, partial=False):
    plan = read(root / "plan.json")
    verify_plan(plan)
    if partial:
        environment = canonical_sha256(read(root / "environment.json"))
        complete = defaultdict(dict)
        missing = defaultdict(list)
        for job in plan["jobs"]:
            video, record = record_paths(root, job)
            if not record.exists():
                missing[job["group"]].append(job["case"]["id"])
                continue
            # A completion record is written last. If it exists, corruption must
            # fail loudly rather than silently excluding an unfavorable action.
            verified = verify_record(root, job, plan, environment)
            complete[job["group"]][job["case"]["id"]] = verified
        required = {case["id"] for case in plan["protocol"]["cases"]}
        selected_groups = {group for group, cases in complete.items() if set(cases) == required}
        jobs = [job for job in plan["jobs"] if job["group"] in selected_groups]
        if not jobs:
            raise ValueError("No complete eight-case prompt-seed groups yet")
        manifest = {"schema": "hy15_partial_snapshot_v1", "plan_sha256": plan["plan_sha256"],
                    "environment_sha256": environment,
                    "records": [{"id": job["id"],
                                 "record_sha256": sha256_file(record_paths(root, job)[1])} for job in jobs],
                    "complete_groups": len(selected_groups),
                    "excluded_incomplete_groups": len(missing),
                    "expected_groups": len(plan["jobs"]) // len(required)}
    else:
        manifest = read(root / "dataset.json")
        jobs = plan["jobs"]
    if manifest["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Dataset/plan mismatch")
    expected = {item["id"]: item["record_sha256"] for item in manifest["records"]}
    if set(expected) != {j["id"] for j in jobs} or len(expected) != len(manifest["records"]):
        raise ValueError("Dataset has missing/duplicate/extra records")
    rows = []
    for job in jobs:
        video, record = record_paths(root, job)
        if sha256_file(record) != expected[job["id"]]:
            raise ValueError("Finalized record changed")
        r = verify_record(root, job, plan, manifest["environment_sha256"])
        prompt, case = job["prompt"], job["case"]
        rows.append({"stem": job["id"], "group": job["group"], "prompt_id": prompt["prompt_id"],
                     "family_id": prompt["family_id"], "motion": prompt["motion_level"],
                     "detail": prompt["detail_level"], "prompt": prompt["prompt"],
                     "base_seed": job["base_seed"], "seed": job["seed"],
                     "case": case["id"], "axis": case["axis"], "nominal_main_budget": case["budget"],
                     **job["density"], "seconds": r["timing_seconds"]["candidate_total"],
                     "main_seconds": r["timing_seconds"]["main"],
                     "transition_seconds": r["timing_seconds"]["transition"],
                     "refine_seconds": r["timing_seconds"]["refine"],
                     "video_path": str(video.resolve()), "video_sha256": r["video_sha256"]})
    return rows, canonical_sha256(manifest), manifest


def coverage_report(root):
    """CPU-only live progress; completed records are checked for video identity."""
    plan = read(root / "plan.json")
    verify_plan(plan)
    environment = canonical_sha256(read(root / "environment.json"))
    by_group = defaultdict(dict)
    by_case = defaultdict(list)
    for job in plan["jobs"]:
        _, path = record_paths(root, job)
        if path.exists():
            record = verify_record(root, job, plan, environment)
            by_group[job["group"]][job["case"]["id"]] = record
            by_case[job["case"]["id"]].append(record)
    case_ids = {case["id"] for case in plan["protocol"]["cases"]}
    complete = {group for group, cases in by_group.items() if set(cases) == case_ids}
    prompts = {job["group"]: job["prompt"] for job in plan["jobs"]}
    report = {"completed_videos": sum(map(len, by_case.values())),
              "expected_videos": len(plan["jobs"]),
              "complete_groups": len(complete),
              "expected_groups": len(plan["jobs"]) // len(case_ids),
              "complete_prompts": sorted({prompts[group]["prompt_id"] for group in complete}),
              "complete_factor_cells": sorted({prompts[group]["motion_level"] + "/" +
                                                 prompts[group]["detail_level"] for group in complete}),
              "cases": {case: {"n": len(items),
                                "mean_seconds": statistics.fmean(r["timing_seconds"]["candidate_total"] for r in items),
                                "mean_main_seconds": statistics.fmean(r["timing_seconds"]["main"] for r in items),
                                "mean_transition_seconds": statistics.fmean(r["timing_seconds"]["transition"] for r in items),
                                "mean_refine_seconds": statistics.fmean(r["timing_seconds"]["refine"] for r in items)}
                        for case, items in sorted(by_case.items()) if items}}
    return report


def report(rows, scores, out):
    enriched = []
    for r in rows:
        values = {d: float(scores[r["stem"]][d]) for d in DIMENSIONS}
        if any(not math.isfinite(v) for v in values.values()):
            raise ValueError("Non-finite score")
        enriched.append({**r, **values, "vbench5": statistics.fmean(values[d] for d in QUALITY_DIMENSIONS)})
    csv_write(out / "quality_by_video.csv", enriched)
    groups = defaultdict(dict)
    for r in enriched:
        groups[r["group"]][r["case"]] = r
    relative = []
    for cases in groups.values():
        for r in cases.values():
            full, repaired = cases["FULL50"], cases["FULL50_HR4"]
            relative.append({**r, "delta_q_vs_full50": r["vbench5"] - full["vbench5"],
                             "delta_q_vs_full50_hr4": r["vbench5"] - repaired["vbench5"],
                             "time_ratio_vs_full50": r["seconds"] / full["seconds"],
                             "speedup_vs_full50": full["seconds"] / r["seconds"]})
    csv_write(out / "relative_to_full.csv", relative)
    prompt_groups = defaultdict(list)
    factor_groups = defaultdict(list)
    for r in relative:
        prompt_groups[(r["prompt_id"], r["case"])].append(r)
        factor_groups[(r["motion"], r["detail"], r["case"])].append(r)
    keys = ("vbench5", "delta_q_vs_full50", "delta_q_vs_full50_hr4", "seconds", "time_ratio_vs_full50")
    prompt_rows = []
    for (pid, case), items in sorted(prompt_groups.items()):
        prompt_rows.append({"prompt_id": pid, "family_id": items[0]["family_id"], "case": case,
                            "n_seeds": len(items), **{k: statistics.fmean(r[k] for r in items) for k in keys},
                            "delta_q_seed_sd": statistics.stdev(r["delta_q_vs_full50"] for r in items) if len(items) > 1 else 0})
    csv_write(out / "prompt_mean_targets.csv", prompt_rows)
    factor_rows = [{"motion": m, "detail": d, "case": c, "n": len(items),
                    **{k: statistics.fmean(r[k] for r in items) for k in keys}}
                   for (m, d, c), items in sorted(factor_groups.items())]
    csv_write(out / "factor_summary.csv", factor_rows)
    pairs = []
    for cases in groups.values():
        for budget in ("050", "025"):
            s, t = cases[f"S_B{budget}"], cases[f"T_B{budget}"]
            pairs.append({"group": s["group"], "prompt_id": s["prompt_id"], "base_seed": s["base_seed"],
                          "family_id": s["family_id"], "motion": s["motion"], "detail": s["detail"],
                          "budget": budget, "T_minus_S_quality": t["vbench5"] - s["vbench5"],
                          "T_minus_S_seconds": t["seconds"] - s["seconds"]})
    csv_write(out / "st_pairs_by_seed.csv", pairs)
    (out / "report.md").write_text(
        "# HY15 endpoint pilot (development only)\n\n"
        "Claim under test: prompt predicts method/budget expected utility under fixed sigma0.2 HR4.\n\n"
        "Quality is the arithmetic mean of five VBench dimensions, NOT official VBench Total. "
        "Dynamic degree and overall consistency are separate diagnostics. "
        "0.5/0.25 are nominal main-stage proxy densities, NOT equal latency or measured speedups. "
        "Use relative_to_full.csv for paired outcomes and actual costs; FULL50_HR4 isolates repair effects. "
        "C uses last-CFG-velocity reuse, not TeaCache. "
        "All four families are development data; no automatic policy efficacy is established by this report. "
        "Do not treat seeds as independent prompts or select/test a policy on the same rows.\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "score", "report", "partial-check", "partial-score", "partial-report"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--vbench-python", default=sys.executable)
    parser.add_argument("--expected-vbench-commit", default="")
    parser.add_argument("--ngpus", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "partial-check":
        import json
        print(json.dumps(coverage_report(args.out), ensure_ascii=False, indent=2))
        return
    partial = args.mode.startswith("partial-")
    rows, digest, manifest = collect(args.out, partial=partial)
    print(f"Verified {len(rows)} videos; {len({r['group'] for r in rows})} matched groups")
    if args.mode == "check":
        return
    out = (args.out / "metrics" / "hy15_endpoint_vbench_partial" / digest[:16]) if partial else \
        args.out / "metrics" / "hy15_endpoint_vbench"
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        if partial:
            immutable_json(out / "snapshot.json", manifest)
        if args.mode in {"score", "partial-score"}:
            from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import inspect_vbench_checkout, score_case_directory
            identity = inspect_vbench_checkout(args.vbench_root, expected_commit=args.expected_vbench_commit or None)
            immutable_json(out / "vbench.lock.json", identity)
            inputs = stage_inputs(rows, out)
            scores = {r["stem"]: {} for r in rows}
            provenance = {}
            for dimension in DIMENSIONS:
                print(f"Scoring {dimension} on {args.ngpus} GPUs", flush=True)
                bundle = score_case_directory(args.vbench_root, args.vbench_python, inputs,
                    out / "prompt_map.json", out / "vbench" / dimension, [dimension],
                    [dimension] if dimension in QUALITY_DIMENSIONS else [],
                    [dimension] if dimension in DIAGNOSTIC_DIMENSIONS else [], args.ngpus, False, identity)
                if set(bundle.scores) != set(scores):
                    raise ValueError(f"VBench coverage mismatch: {dimension}")
                for stem in scores:
                    scores[stem][dimension] = float(bundle.scores[stem][dimension])
                provenance[dimension] = bundle.provenance
            immutable_json(out / "scores.json", {"input_sha256": digest, "scores": scores, "provenance": provenance})
        payload = read(out / "scores.json")
        if payload["input_sha256"] != digest:
            raise ValueError("Scoring inputs changed")
        report(rows, payload["scores"], out)
    print(out / "report.md")


if __name__ == "__main__":
    main()
