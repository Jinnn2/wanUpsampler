#!/usr/bin/env python3
"""Quickly inspect progress of UNIV prompt-budget 8-GPU data generation."""
"""Quickly inspect progress of UNIV prompt-budget multi-GPU / multi-machine data generation."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOTS = [
    "/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_full_v3",
    "/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_reserve_v1",
]


def format_duration(seconds: float) -> str:
    if seconds < 0:
        return "N/A"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}m {secs:02d}s"


def inspect_progress(out_root: Path, detail: bool = False) -> None:
def inspect_progress(out_root: Path, detail: bool = False) -> dict[str, Any] | None:
    manifest_path = out_root / "generation_manifest.json"
    if not manifest_path.is_file():
        print(f"[Error] Manifest not found at: {manifest_path}")
        print("The generation may not have started or the prepare step failed.")
        return
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Error] Failed to parse manifest {manifest_path}: {exc}")
        return
        return None

    jobs = manifest.get("jobs", [])
    if not jobs:
        print("[Warning] No jobs found in manifest.")
        return
        return None

    lock_path = out_root / ".prompt_budget_generation.lock"
    is_running = lock_path.is_dir() or lock_path.is_file()

    total_jobs = len(jobs)
    total_videos = sum(j.get("prompt_count", 0) for j in jobs)

    completed_jobs = 0
    in_progress_jobs = 0
    total_generated_videos = 0
    total_video_latency = 0.0

    worker_stats: dict[int, dict[str, Any]] = {
        i: {
            "total_jobs": 0,
            "done_jobs": 0,
            "total_videos": 0,
            "done_videos": 0,
            "current_job": None,
            "current_job_done": 0,
            "current_job_total": 0,
            "last_log_line": "",
        }
        for i in range(8)
    }

    split_case_stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "done": 0})
    )

    for job in jobs:
        slot = job.get("worker_slot", 0)
        split = job.get("split", "unknown")
        case_id = job.get("case_id", "unknown")
        prompt_count = job.get("prompt_count", 0)

        worker_stats[slot]["total_jobs"] += 1
        worker_stats[slot]["total_videos"] += prompt_count
        split_case_stats[split][case_id]["total"] += prompt_count

        timing_path = Path(job.get("timing_path", ""))
        done_in_job = 0
        if timing_path.is_file():
            try:
                for line in timing_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    if '"kind": "video"' in line or '"kind":"video"' in line:
                        done_in_job += 1
                        try:
                            rec = json.loads(line)
                            if "pipeline_elapsed_s" in rec:
                                total_video_latency += float(rec["pipeline_elapsed_s"])
                        except Exception:
                            pass
            except Exception:
                pass

        total_generated_videos += done_in_job
        worker_stats[slot]["done_videos"] += done_in_job
        split_case_stats[split][case_id]["done"] += done_in_job

        if done_in_job >= prompt_count:
            completed_jobs += 1
            worker_stats[slot]["done_jobs"] += 1
        elif done_in_job > 0:
            in_progress_jobs += 1
            worker_stats[slot]["current_job"] = job.get("job_id")
            worker_stats[slot]["current_job_done"] = done_in_job
            worker_stats[slot]["current_job_total"] = prompt_count
        elif worker_stats[slot]["current_job"] is None:
            # Check if this could be the job currently starting
            worker_stats[slot]["current_job"] = job.get("job_id")
            worker_stats[slot]["current_job_done"] = 0
            worker_stats[slot]["current_job_total"] = prompt_count

    # Fetch last log line for each worker
    log_dir = out_root / "logs" / "8gpu_data"
    for slot in range(8):
        # By default GPU index matches slot in 0..7
        gpu_log = log_dir / f"gpu_{slot}.log"
        if gpu_log.is_file():
            try:
                lines = [line.strip() for line in gpu_log.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
                lines = [
                    line.strip()
                    for line in gpu_log.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if line.strip()
                ]
                if lines:
                    worker_stats[slot]["last_log_line"] = lines[-1]
            except Exception:
                pass

    # Check finalized records if any
    records_dir = out_root / "records"
    finalized_count = 0
    if records_dir.is_dir():
        finalized_count = sum(1 for _ in records_dir.glob("*/*.json"))

    job_pct = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0.0
    vid_pct = (total_generated_videos / total_videos * 100) if total_videos > 0 else 0.0

    print("=" * 72)
    print("  UNIV 8-GPU Prompt Budget Generation Status")
    print(f"  Shard: {out_root.name}  ({'[RUNNING]' if is_running else '[IDLE/FINISHED]'})")
    print("=" * 72)
    print(f"Output Root : {out_root}")
    print(f"Run State   : {'[RUNNING] (lock active)' if is_running else '[IDLE/FINISHED] (no active lock)'}")
    print(f"Path        : {out_root}")
    print(f"Jobs Done   : {completed_jobs}/{total_jobs} ({job_pct:.1f}%) | In-progress: {in_progress_jobs}")
    print(f"Videos Done : {total_generated_videos}/{total_videos} ({vid_pct:.1f}%)")

    if finalized_count > 0:
        print(f"Final Records: {finalized_count} finalized trajectory records written")

    avg_s = 0.0
    if total_generated_videos > 0:
        avg_s = total_video_latency / total_generated_videos
        remaining_vids = total_videos - total_generated_videos
        eta_seconds = (remaining_vids * avg_s) / 8.0 if remaining_vids > 0 else 0
        print(f"Speed / Latency: ~{avg_s:.1f}s / video (wall ETA across 8 GPUs: ~{format_duration(eta_seconds)})")

    print("-" * 72)
    print(f"{'Worker/GPU':<12}{'Jobs':<12}{'Videos':<15}{'Current Activity / Status'}")
    print("-" * 72)

    for slot in range(8):
        w = worker_stats[slot]
        j_str = f"{w['done_jobs']}/{w['total_jobs']}"
        v_str = f"{w['done_videos']}/{w['total_videos']}"

        if w["done_jobs"] == w["total_jobs"] and w["total_jobs"] > 0:
            status = "ALL JOBS DONE"
        elif w["current_job_done"] > 0:
            status = f"BUSY: {w['current_job_done']}/{w['current_job_total']} vids in chunk"
        elif w["last_log_line"]:
            short_log = (w["last_log_line"][:40] + "...") if len(w["last_log_line"]) > 40 else w["last_log_line"]
            short_log = (
                (w["last_log_line"][:40] + "...")
                if len(w["last_log_line"]) > 40
                else w["last_log_line"]
            )
            status = f"RUNNING: {short_log}"
        else:
            status = "PENDING / IDLE"

        print(f"GPU {slot:<8}{j_str:<12}{v_str:<15}{status}")

    if detail:
        print("-" * 72)
        print("Breakdown by Split & Case:")
        for split, cases in sorted(split_case_stats.items()):
            case_summaries = [f"{case}: {data['done']}/{data['total']}" for case, data in sorted(cases.items())]
            case_summaries = [
                f"{case}: {data['done']}/{data['total']}"
                for case, data in sorted(cases.items())
            ]
            print(f"  [{split}]: {', '.join(case_summaries)}")

    print("=" * 72)
    return {
        "out_root": str(out_root),
        "is_running": is_running,
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "total_videos": total_videos,
        "generated_videos": total_generated_videos,
        "total_video_latency": total_video_latency,
        "finalized_count": finalized_count,
        "avg_s": avg_s,
    }


def print_multi_machine_summary(summaries: list[dict[str, Any]]) -> None:
    if len(summaries) <= 1:
        return

    print("\n" + "#" * 72)
    print("  MULTI-MACHINE / MULTI-SHARD AGGREGATE SUMMARY")
    print("#" * 72)

    total_shards = len(summaries)
    running_shards = sum(1 for s in summaries if s["is_running"])
    total_jobs = sum(s["total_jobs"] for s in summaries)
    completed_jobs = sum(s["completed_jobs"] for s in summaries)
    total_videos = sum(s["total_videos"] for s in summaries)
    generated_videos = sum(s["generated_videos"] for s in summaries)
    total_latency = sum(s["total_video_latency"] for s in summaries)
    total_finalized = sum(s["finalized_count"] for s in summaries)

    job_pct = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0.0
    vid_pct = (generated_videos / total_videos * 100) if total_videos > 0 else 0.0

    print(f"Active Shards / Machines : {running_shards}/{total_shards} running ({total_shards * 8} GPUs total)")
    print(f"Global Jobs Completed    : {completed_jobs}/{total_jobs} ({job_pct:.1f}%)")
    print(f"Global Videos Generated  : {generated_videos}/{total_videos} ({vid_pct:.1f}%)")

    if total_finalized > 0:
        print(f"Global Finalized Records : {total_finalized}")

    if generated_videos > 0:
        avg_s = total_latency / generated_videos
        remaining_vids = total_videos - generated_videos
        active_gpus = max(8, running_shards * 8)
        global_eta = (remaining_vids * avg_s) / float(active_gpus) if remaining_vids > 0 else 0
        print(f"Average Video Latency    : ~{avg_s:.1f}s")
        print(f"Global Combined ETA      : ~{format_duration(global_eta)} (across {active_gpus} GPUs)")

    print("#" * 72 + "\n")


def resolve_roots(raw_roots: list[str]) -> list[Path]:
    if raw_roots:
        return [Path(r).resolve() for r in raw_roots]
    # Check default paths
    existing = [Path(r).resolve() for r in DEFAULT_ROOTS if Path(r).exists()]
    if existing:
        return existing
    return [Path(DEFAULT_ROOTS[0]).resolve()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check progress of UNIV prompt-budget generation.")
    parser = argparse.ArgumentParser(
        description="Check progress of UNIV prompt-budget generation (single or multi-machine)."
    )
    parser.add_argument(
        "out_root",
        nargs="?",
        default="/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_full_v3",
        help="Path to OUT_ROOT (default: /mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_full_v3)",
        "out_roots",
        nargs="*",
        default=[],
        help="One or more OUT_ROOT paths (defaults to primary and reserve shards if present)",
    )
    parser.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds (e.g. --watch 10)")
    parser.add_argument(
        "--watch", type=int, default=0, help="Refresh interval in seconds (e.g. --watch 10)"
    )
    parser.add_argument("--detail", action="store_true", help="Show breakdown by split and case")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    targets = resolve_roots(args.out_roots)

    def run_check() -> None:
        summaries = []
        for target in targets:
            summary = inspect_progress(target, detail=args.detail)
            if summary:
                summaries.append(summary)
        print_multi_machine_summary(summaries)

    if args.watch > 0:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                print(f"Refreshing every {args.watch}s (Ctrl+C to quit)... Time: {dt.datetime.now().strftime('%H:%M:%S')}")
                inspect_progress(out_root, detail=args.detail)
                print(
                    f"Refreshing every {args.watch}s (Ctrl+C to quit)... Time: {dt.datetime.now().strftime('%H:%M:%S')}"
                )
                run_check()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nExiting watch mode.")
    else:
        inspect_progress(out_root, detail=args.detail)
        run_check()


if __name__ == "__main__":
    main()

