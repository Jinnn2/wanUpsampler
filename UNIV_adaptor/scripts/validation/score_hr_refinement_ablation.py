"""Score existing HR10/06/04/02 videos in one VBench custom-input batch."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic

CASES = {"HR10": 10, "HR06": 6, "HR04": 4, "HR02": 2}
QUALITY_DIMENSIONS = [
    "subject_consistency", "background_consistency", "motion_smoothness",
    "aesthetic_quality", "imaging_quality",
]
DIAGNOSTIC_DIMENSIONS = ["dynamic_degree"]
DIMENSIONS = [*QUALITY_DIMENSIONS, *DIAGNOSTIC_DIMENSIONS]


def load_inputs(out_dir: Path):
    """Validate the completed experiment without importing any scoring models."""
    summary_path = out_dir / "comparison_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != "univ_hr_refinement_ablation_results_v1":
        raise ValueError("expected an HR refinement ablation comparison_summary.json")
    if summary.get("complete") is not True:
        raise ValueError("comparison is incomplete; all four generation branches are required")
    if not isinstance(summary.get("prompt"), str) or not summary["prompt"].strip():
        raise ValueError("comparison summary has no prompt")
    entries = summary.get("cases", [])
    if len(entries) != 4 or {row.get("id") for row in entries} != set(CASES):
        raise ValueError("comparison must contain exactly HR10, HR06, HR04 and HR02")
    by_id = {row["id"]: row for row in entries}
    cases = []
    for case_id, steps in CASES.items():
        row = dict(by_id[case_id])
        if row.get("hr_steps") != steps:
            raise ValueError(f"wrong HR step count for {case_id}")
        seconds = float(row["hr_seconds"])
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"invalid HR timing for {case_id}")
        digest = str(row.get("boundary_sha256", ""))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"missing or invalid shared-boundary hash for {case_id}")
        # Resolve relative to the selected run, so copying a run to another host works.
        video = out_dir / f"{case_id}.mp4"
        if not video.is_file() or video.stat().st_size < 1024:
            raise FileNotFoundError(f"missing or undersized video: {video}")
        if sha256_file(video) != row.get("video_sha256"):
            raise ValueError(f"video hash differs from generation summary: {video}")
        grid = row.get("hr_schedule", {}).get("sigmas", [])
        if len(grid) != steps + 1 or grid[-1] != 0:
            raise ValueError(f"invalid HR grid for {case_id}")
        if any(not math.isfinite(float(v)) for v in grid) or any(a <= b for a, b in zip(grid, grid[1:])):
            raise ValueError(f"HR grid must be finite and strictly decreasing: {case_id}")
        row["video_path"] = str(video)
        row["hr_seconds"] = seconds
        cases.append(row)
    if len({row["boundary_sha256"] for row in cases}) != 1:
        raise ValueError("videos did not share one HR transition state")
    if len({row["hr_schedule"]["sigmas"][0] for row in cases}) != 1:
        raise ValueError("videos did not start at the same HR sigma")
    return summary, cases


def stage_inputs(cases, prompt: str, metrics_dir: Path):
    """Use hard links (copy fallback), excluding any montages in the source run."""
    inputs = metrics_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    expected = {f"{row['id']}.mp4" for row in cases}
    if {path.name for path in inputs.glob("*.mp4")} - expected:
        raise ValueError(f"unexpected videos in private VBench input directory: {inputs}")
    prompt_map = {}
    for row in cases:
        source = Path(row["video_path"])
        destination = inputs / source.name
        if destination.exists():
            if sha256_file(destination) != row["video_sha256"]:
                raise ValueError(f"staged video changed; use a new metrics directory: {destination}")
        else:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        prompt_map[str(destination.resolve())] = prompt
    mapping_path = metrics_dir / "prompt_map.json"
    write_json_atomic(mapping_path, prompt_map)
    return inputs, mapping_path


def comparison_rows(cases, scores):
    if set(scores) != set(CASES):
        raise ValueError("VBench results must cover exactly the four HR videos")
    for case_id in CASES:
        if set(scores[case_id]) != set(DIMENSIONS):
            raise ValueError(f"incomplete VBench dimensions for {case_id}")
        for dimension, value in scores[case_id].items():
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"invalid VBench score: {case_id}.{dimension}")
    base_seconds = next(row["hr_seconds"] for row in cases if row["id"] == "HR10")
    rows = []
    for case in cases:
        name = case["id"]
        row = {
            "case": name, "hr_steps": case["hr_steps"], "hr_seconds": case["hr_seconds"],
            "hr_speedup_vs_hr10": base_seconds / case["hr_seconds"],
        }
        for dimension in DIMENSIONS:
            row[dimension] = float(scores[name][dimension])
            row[f"delta_{dimension}_pp"] = 100 * (float(scores[name][dimension]) - float(scores["HR10"][dimension]))
        rows.append(row)
    return rows


def write_reports(metrics_dir: Path, payload):
    write_json_atomic(metrics_dir / "vbench_scores.json", payload)
    rows = payload["rows"]
    csv_path = metrics_dir / "comparison.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    lines = [
        "# HR refinement VBench comparison", "",
        f"Prompt: {payload['prompt']}", f"Seed: {payload['seed']}", "",
        "One prompt/seed, shared HR boundary. HR10 is the reference, not ground truth.",
        "Scores are raw [0,1]; deltas are percentage points (100 x score difference).",
        "Dynamic degree measures motion magnitude and is reported separately; higher is not necessarily better.",
        "No official VBench overall score or significance claim is computed.", "",
        "| Case | HR steps | HR seconds | HR speedup vs HR10 |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [f"| {r['case']} | {r['hr_steps']} | {r['hr_seconds']:.3f} | {r['hr_speedup_vs_hr10']:.3f}x |" for r in rows]
    lines += ["", "Times are exploratory single-pass HR timings; they exclude VBench evaluation time.",
              "The original first branch may include cold HR kernel overhead. Whole-pipeline speedup is not inferred.", "",
              "| Dimension | HR10 | HR06 (delta pp) | HR04 (delta pp) | HR02 (delta pp) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    by_case = {r["case"]: r for r in rows}
    for dimension in DIMENSIONS:
        values = [f"{by_case['HR10'][dimension]:.6f}"]
        values += [f"{by_case[name][dimension]:.6f} ({by_case[name][f'delta_{dimension}_pp']:+.3f})" for name in ("HR06", "HR04", "HR02")]
        lines.append(f"| {dimension} | " + " | ".join(values) + " |")
    report = metrics_dir / "comparison.md"
    temporary = report.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(report)
    print("\n".join(lines))
    print(f"\nReport: {report}\nCSV: {csv_path}")


def evaluate(args):
    out_dir = Path(args.out_dir).resolve()
    summary, cases = load_inputs(out_dir)
    metrics_dir = out_dir / "metrics/hr_refinement"
    from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import (
        inspect_vbench_checkout, score_case_directory,
    )
    vbench_root = Path(args.vbench_root).resolve()
    identity = inspect_vbench_checkout(vbench_root, expected_commit=args.vbench_commit or None)
    print(f"Verified four videos and shared boundary; VBench commit: {identity['git_commit']}")
    if args.mode == "check":
        return
    inputs, prompt_map = stage_inputs(cases, summary["prompt"], metrics_dir)
    # One evaluation batch loads each metric model for all four videos.
    # No separate dependency warmup: reuse the existing VBench model cache.
    bundle = score_case_directory(
        vbench_root, args.vbench_python, inputs, prompt_map, metrics_dir / "vbench",
        DIMENSIONS, QUALITY_DIMENSIONS, DIAGNOSTIC_DIMENSIONS, 1, args.force, identity,
    )
    # Reject edits to videos/summary while the evaluator was running.
    fresh_summary, fresh_cases = load_inputs(out_dir)
    if fresh_summary != summary or fresh_cases != cases:
        raise RuntimeError("experiment changed during scoring")
    payload = {
        "schema": "univ_hr_refinement_vbench_v1",
        "prompt": summary["prompt"], "seed": summary["seed"],
        "generation_summary_sha256": sha256_file(out_dir / "comparison_summary.json"),
        "boundary_sha256": cases[0]["boundary_sha256"],
        "quality_dimensions": QUALITY_DIMENSIONS,
        "diagnostic_dimensions": DIAGNOSTIC_DIMENSIONS,
        "video_sha256": {row["id"]: row["video_sha256"] for row in cases},
        "vbench_provenance": bundle.provenance,
        "rows": comparison_rows(cases, bundle.scores),
    }
    write_reports(metrics_dir, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run"), nargs="?", default="run")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs/univ_hr_refinement_ablation_v1"))
    parser.add_argument("--vbench-root", default=os.environ.get("VBENCH_ROOT", "/mnt/afs_2/houze/VBench"))
    parser.add_argument("--vbench-python", default=sys.executable)
    parser.add_argument("--vbench-commit", default="")
    parser.add_argument("--force", action="store_true", help="Rescore instead of reusing content-matched cached scores")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
