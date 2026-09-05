"""Score completed full-LR direct-sigma refinement videos with VBench."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic
from UNIV_adaptor.hr_refinement import direct_hr_sigmas
from UNIV_adaptor.scripts.validation.run_mrflow_refinement_ablation import (
    CONTROL_ID,
    build_cases,
)
from UNIV_adaptor.scripts.validation.score_hr_refinement_ablation import (
    DIAGNOSTIC_DIMENSIONS,
    DIMENSIONS,
    QUALITY_DIMENSIONS,
    stage_inputs,
)


def _valid_digest(value) -> bool:
    digest = str(value)
    return len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def load_inputs(out_dir: Path):
    summary_path = out_dir / "comparison_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != "univ_mrflow_refinement_results_v1":
        raise ValueError("expected a full-LR direct-sigma comparison summary")
    if summary.get("complete") is not True:
        raise ValueError("comparison is incomplete")
    if not isinstance(summary.get("prompt"), str) or not summary["prompt"].strip():
        raise ValueError("comparison summary has no prompt")

    sigmas = summary.get("sigmas")
    hr_steps = summary.get("hr_steps")
    if not isinstance(sigmas, list) or not sigmas or any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1
        for value in sigmas
    ) or len(set(sigmas)) != len(sigmas):
        raise ValueError("comparison summary has invalid sigmas")
    if not isinstance(hr_steps, list) or not hr_steps or any(
        type(value) is not int or value < 1 for value in hr_steps
    ) or len(set(hr_steps)) != len(hr_steps):
        raise ValueError("comparison summary has invalid HR steps")

    expected = build_cases(sigmas, hr_steps, out_dir)
    expected_by_id = {row["id"]: row for row in expected}
    entries = summary.get("cases", [])
    if len(entries) != len(expected) or {row.get("id") for row in entries} != set(expected_by_id):
        raise ValueError("comparison cases do not match the recorded sigma-step matrix")
    if summary.get("run_order") != [row["id"] for row in expected]:
        raise ValueError("comparison run order does not place the transition-only control first")

    cases = []
    for entry in entries:
        row = dict(entry)
        expected_row = expected_by_id[row["id"]]
        sigma = float(row.get("refine_sigma", -1))
        steps = int(row.get("hr_steps", -1))
        if sigma != expected_row["refine_sigma"] or steps != expected_row["hr_steps"]:
            raise ValueError(f"wrong sigma or HR steps for {row['id']}")
        if row.get("total_nfe") != 50 + steps:
            raise ValueError(f"wrong total NFE for {row['id']}")
        for key in ("clean_lr_sha256", "clean_hr_sha256", "hr_noise_sha256", "branch_start_sha256"):
            if not _valid_digest(row.get(key)):
                raise ValueError(f"missing or invalid {key} for {row['id']}")

        video = out_dir / f"{row['id']}.mp4"
        if not video.is_file() or video.stat().st_size < 1024:
            raise FileNotFoundError(f"missing or undersized video: {video}")
        if sha256_file(video) != row.get("video_sha256"):
            raise ValueError(f"video hash differs from generation summary: {video}")
        row["video_path"] = str(video)

        grid = row.get("hr_schedule", {})
        actual_sigmas = [float(value) for value in grid.get("sigmas", [])]
        expected_sigmas = (
            [0.0]
            if steps == 0
            else list(direct_hr_sigmas(start_sigma=sigma, hr_steps=steps))
        )
        if len(actual_sigmas) != len(expected_sigmas) or any(
            not math.isclose(actual, planned, rel_tol=0, abs_tol=1e-6)
            for actual, planned in zip(actual_sigmas, expected_sigmas)
        ):
            raise ValueError(f"invalid direct-sigma HR grid for {row['id']}")
        if grid.get("hr_steps") != steps or grid.get("compute_indices") != list(range(steps)):
            raise ValueError(f"invalid HR compute indices for {row['id']}")
        if grid.get("model_timesteps") != expected_row["planned_model_timesteps"]:
            raise ValueError(f"invalid direct model timesteps for {row['id']}")

        hr_seconds = float(row.get("hr_seconds", -1))
        candidate_seconds = float(row.get("candidate_denoise_seconds", -1))
        if not math.isfinite(hr_seconds) or hr_seconds < 0 or (steps > 0 and hr_seconds <= 0):
            raise ValueError(f"invalid HR timing for {row['id']}")
        if steps == 0 and hr_seconds != 0:
            raise ValueError("transition-only control must have zero HR time")
        if not math.isfinite(candidate_seconds) or candidate_seconds <= hr_seconds:
            raise ValueError(f"invalid candidate denoising timing for {row['id']}")
        row["hr_seconds"] = hr_seconds
        row["candidate_denoise_seconds"] = candidate_seconds
        cases.append(row)

    for key in ("clean_lr_sha256", "clean_hr_sha256", "hr_noise_sha256"):
        if len({row[key] for row in cases}) != 1:
            raise ValueError(f"branches did not share one {key}")
    for sigma in summary["sigmas"]:
        starts = {row["branch_start_sha256"] for row in cases if row["refine_sigma"] == sigma}
        if len(starts) != 1:
            raise ValueError(f"sigma={sigma} branches did not share one starting tensor")
    control = next(row for row in cases if row["id"] == CONTROL_ID)
    if control["branch_start_sha256"] != control["clean_hr_sha256"]:
        raise ValueError("transition-only control does not start from clean HR")
    return summary, cases


def comparison_rows(cases, scores):
    case_ids = {row["id"] for row in cases}
    if set(scores) != case_ids:
        raise ValueError("VBench results do not cover the direct-sigma cases exactly")
    for case_id in case_ids:
        if set(scores[case_id]) != set(DIMENSIONS):
            raise ValueError(f"incomplete VBench dimensions for {case_id}")
        if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1
               for value in scores[case_id].values()):
            raise ValueError(f"invalid VBench score for {case_id}")

    baseline = scores[CONTROL_ID]
    rows = []
    for case in cases:
        case_id = case["id"]
        row = {
            "case": case_id,
            "refine_sigma": case["refine_sigma"],
            "hr_steps": case["hr_steps"],
            "total_nfe": case["total_nfe"],
            "hr_seconds": case["hr_seconds"],
            "candidate_denoise_seconds": case["candidate_denoise_seconds"],
        }
        for dimension in DIMENSIONS:
            row[dimension] = float(scores[case_id][dimension])
            row[f"delta_{dimension}_pp"] = 100 * (
                row[dimension] - float(baseline[dimension])
            )
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
        "# Full-LR direct-sigma HR refinement", "",
        f"Prompt: {payload['prompt']}", f"Seed: {payload['seed']}", "",
        "All cases share one completed LR50 endpoint, DVG-anchor clean HR transition and HR noise tensor.",
        f"Quality deltas are percentage points relative to {CONTROL_ID}, the transition-only control.",
        "Dynamic degree is a motion diagnostic; higher is not necessarily better.",
        "No official VBench overall score or significance claim is computed.", "",
        "| Case | Sigma | HR steps | Total NFE | HR seconds | Candidate denoise seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {row['case']} | {row['refine_sigma']:.3f} | {row['hr_steps']} | "
        f"{row['total_nfe']} | {row['hr_seconds']:.3f} | "
        f"{row['candidate_denoise_seconds']:.3f} |"
        for row in rows
    ]
    lines += ["", "| Case | " + " | ".join(DIMENSIONS) + " |",
              "| --- | " + " | ".join("---:" for _ in DIMENSIONS) + " |"]
    for row in rows:
        values = [
            f"{row[dimension]:.6f} ({row[f'delta_{dimension}_pp']:+.3f})"
            for dimension in DIMENSIONS
        ]
        lines.append(f"| {row['case']} | " + " | ".join(values) + " |")
    report = metrics_dir / "comparison.md"
    temporary = report.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(report)
    print("\n".join(lines))
    print(f"\nReport: {report}\nCSV: {csv_path}")


def evaluate(args):
    out_dir = Path(args.out_dir).resolve()
    summary, cases = load_inputs(out_dir)
    metrics_dir = out_dir / "metrics/mrflow_refinement"
    from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import (
        inspect_vbench_checkout,
        score_case_directory,
    )

    vbench_root = Path(args.vbench_root).resolve()
    identity = inspect_vbench_checkout(
        vbench_root, expected_commit=args.vbench_commit or None
    )
    print(f"Verified {len(cases)} videos and direct-sigma schedules; VBench commit: {identity['git_commit']}")
    if args.mode == "check":
        return
    inputs, prompt_map = stage_inputs(cases, summary["prompt"], metrics_dir)
    bundle = score_case_directory(
        vbench_root,
        args.vbench_python,
        inputs,
        prompt_map,
        metrics_dir / "vbench",
        DIMENSIONS,
        QUALITY_DIMENSIONS,
        DIAGNOSTIC_DIMENSIONS,
        1,
        args.force,
        identity,
    )
    fresh_summary, fresh_cases = load_inputs(out_dir)
    if fresh_summary != summary or fresh_cases != cases:
        raise RuntimeError("experiment changed during scoring")
    payload = {
        "schema": "univ_mrflow_refinement_vbench_v1",
        "comparison": "full-lr-direct-sigma",
        "prompt": summary["prompt"],
        "seed": summary["seed"],
        "generation_summary_sha256": sha256_file(out_dir / "comparison_summary.json"),
        "clean_hr_sha256": cases[0]["clean_hr_sha256"],
        "hr_noise_sha256": cases[0]["hr_noise_sha256"],
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
    parser.add_argument(
        "--out-dir", default=str(REPO_ROOT / "outputs/univ_mrflow_refinement_v1")
    )
    parser.add_argument(
        "--vbench-root", default=os.environ.get("VBENCH_ROOT", "/mnt/afs_2/houze/VBench")
    )
    parser.add_argument("--vbench-python", default=sys.executable)
    parser.add_argument("--vbench-commit", default="")
    parser.add_argument("--force", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
