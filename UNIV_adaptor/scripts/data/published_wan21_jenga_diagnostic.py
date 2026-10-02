"""Read-only reuse of the frozen pilot plus six isolated Jenga diagnostic runs.

This does not release the failed pilot, tune thresholds, or change published arms.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from UNIV_adaptor.scripts.data import published_wan21_pilot as pilot

WORKER = Path(__file__).with_name("published_wan21_jenga_diagnostic_worker.py")
ARMS = [
    ("JENGA_ZERO_COUNTED", {"cache": "upstream_zero", "order": "gilbert"}),
    ("JENGA_HARD_OFF", {"cache": "hard_off", "order": "gilbert"}),
    ("JENGA_IDENTITY_OFF", {"cache": "hard_off", "order": "identity"}),
]


def implementation():
    return {p.name: pilot.file_hash(p) for p in (Path(__file__), WORKER, Path(pilot.__file__), pilot.WORKER)}


def separate_roots(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Diagnostic output must be a separate, non-nested directory; source assets are read-only")


def source_rows(source):
    # Require the actual old worker/driver hashes to remain identical, not a
    # blanket bypass of frozen implementation checks after an arbitrary update.
    frozen = pilot.load_plan(source)
    pilot.check_sources(frozen["config"])
    rows = pilot.collect(source, frozen, "calibration")
    expected = [j for j in frozen["jobs"] if j["phase"] == "calibration"]
    if len(rows) != len(expected):
        raise ValueError("Source calibration incomplete; diagnostic reuses complete calibration assets")
    return frozen, rows


def freeze(args):
    separate_roots(args.source, args.out)
    if args.ngpus < 1:
        raise ValueError("NGPUS must be positive")
    if (args.out / "plan.json").exists():
        result = load_plan(args.out)
        if Path(result["source_root"]) != args.source or result["ngpus"] != args.ngpus:
            raise ValueError("Frozen source/GPU configuration differs; choose a new diagnostic directory")
        return result
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Unplanned diagnostic output is nonempty; choose a fresh directory")
    frozen, rows = source_rows(args.source)
    if pilot.weight_inventory(Path(frozen["model_root"])) != frozen["weight_inventory"]:
        raise ValueError("Model weights changed since source calibration")
    off = next(a for a in frozen["config"]["disabled_arms"] if a["id"] == "JENGA_OFF")
    if off["flags"] != ["--teacache_thresh", "0", "--p_remain_rates", "1", "--sa_drop_rates", "0", "0"] or off["steps"] != 50:
        raise ValueError("Diagnostic expects the locked zero-threshold, dense 50-step Jenga control")
    jobs = []
    for arm_index, (name, settings) in enumerate(ARMS):
        arm = off | {"id": name, "role": "diagnostic_only"}
        for prompt in frozen["config"]["calibration"]["prompts"]:
            seed = frozen["config"]["calibration"]["seed"]
            group = f"{prompt['id']}_s{seed}"
            jobs.append({"id": f"calibration_{group}_{name}", "group_id": group,
                "phase": "calibration", "prompt_id": prompt["id"], "prompt": prompt["prompt"], "seed": seed,
                "gpu": arm_index % args.ngpus, "arm": arm, "diagnostic": settings})
    refs = []
    for row in rows:
        path = args.source / "records" / (row["job"]["id"] + ".json")
        refs.append({"job_id": row["job"]["id"], "path": str(path.resolve()), "sha256": pilot.file_hash(path)})
    body = {"schema": "published_wan21_jenga_diagnostic_plan_v1", "config": frozen["config"],
        "model_root": frozen["model_root"], "weight_inventory": frozen["weight_inventory"],
        "source_commits": frozen["source_commits"], "source_root": str(args.source),
        "source_plan_sha256": frozen["plan_sha256"], "source_plan_file_sha256": pilot.file_hash(args.source / "plan.json"),
        "source_records": refs, "ngpus": args.ngpus, "jobs": jobs, "implementation": implementation(),
        "claim": "Implementation diagnosis only; does not establish perceptual degradation, metric failure or prompt routing gain"}
    result = body | {"plan_sha256": pilot.digest(body)}
    pilot.immutable(args.out / "plan.json", result)
    print(f"Frozen diagnostic: {len(jobs)} new videos, {min(len(ARMS), args.ngpus)} persistent GPU/arm workers; source untouched")
    return result


def load_plan(out):
    result = pilot.read(Path(out) / "plan.json")
    if result.get("schema") != "published_wan21_jenga_diagnostic_plan_v1":
        raise ValueError("Wrong diagnostic schema")
    if pilot.digest({k: v for k, v in result.items() if k != "plan_sha256"}) != result["plan_sha256"]:
        raise ValueError("Diagnostic plan hash mismatch")
    if result["implementation"] != implementation():
        raise ValueError("Diagnostic implementation changed after freezing; use a new diagnostic directory")
    source = Path(result["source_root"])
    separate_roots(source, out)
    if pilot.file_hash(source / "plan.json") != result["source_plan_file_sha256"]:
        raise ValueError("Source plan changed")
    for ref in result["source_records"]:
        if pilot.file_hash(ref["path"]) != ref["sha256"]:
            raise ValueError(f"Source receipt changed: {ref['path']}")
        row = pilot.read(ref["path"])
        if row["plan_sha256"] != result["source_plan_sha256"] or row["job"]["id"] != ref["job_id"]:
            raise ValueError("Source receipt binding mismatch")
        if pilot.file_hash(row["video_path"]) != row["video_sha256"] or pilot.file_hash(row["sample_path"]) != row["sample_sha256"]:
            raise ValueError(f"Source video/sample changed: {ref['path']}")
    return result


def launch(args, probe=False):
    frozen = freeze(args)
    if pilot.check_sources(frozen["config"]) != frozen["source_commits"]:
        raise ValueError("Source commits changed")
    if pilot.weight_inventory(Path(frozen["model_root"])) != frozen["weight_inventory"]:
        raise ValueError("Weight inventory changed")
    queues = {g: [] for g in range(frozen["ngpus"])}
    for name, _ in ARMS:
        for g in queues:
            jobs = [j for j in frozen["jobs"] if j["gpu"] == g and j["arm"]["id"] == name]
            if jobs and (probe or any(not pilot.receipt_valid(args.out, frozen, j) for j in jobs)):
                queues[g].append(name)
    active, failures = {}, []
    (args.out / "logs").mkdir(parents=True, exist_ok=True)
    last_progress = time.monotonic()
    try:
        while any(queues.values()) or active:
            for gpu, queue in queues.items():
                if not queue or gpu in active:
                    continue
                arm = queue.pop(0)
                log = args.out / "logs" / f"{'probe' if probe else 'diagnostic'}_{arm}_gpu{gpu}_{time.time_ns()}.log"
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
                for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                    env.pop(key, None)
                command = [os.environ.get("PYTHON_JENGA", args.python), str(WORKER), "--out", str(args.out), "--gpu", str(gpu), "--arm", arm]
                if probe:
                    command.append("--probe")
                handle = log.open("w", encoding="utf-8")
                try:
                    process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                except BaseException:
                    handle.close()
                    raise
                active[gpu] = (process, handle, log)
                print(f"GPU {gpu}: {arm} -> {log}", flush=True)
            for gpu, (process, handle, log) in list(active.items()):
                if process.poll() is not None:
                    handle.close()
                    del active[gpu]
                    if process.returncode:
                        queues[gpu].clear()
                        failures.append(str(log))
                        print(pilot.log_excerpt(log), flush=True)
            if time.monotonic() - last_progress >= 30:
                done = sum((args.out / "records" / (j["id"] + ".json")).exists() for j in frozen["jobs"])
                print(f"Diagnostic: {done}/{len(frozen['jobs'])} receipts; {len(active)} active GPUs", flush=True)
                last_progress = time.monotonic()
            time.sleep(0.5)
    except BaseException:
        for process, _, _ in active.values():
            process.terminate()
        for process, handle, _ in active.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            handle.close()
        raise
    if failures:
        raise RuntimeError(f"Diagnostic worker failures: {failures}")


def compare(row, reference, tolerance):
    import numpy as np
    with np.load(row["sample_path"]) as a, np.load(reference["sample_path"]) as b:
        if a["frames"].shape != b["frames"].shape:
            raise ValueError("Sample grid shapes differ")
        diff = np.abs(a["frames"] - b["frames"])
        if not np.isfinite(diff).all():
            raise ValueError("Nonfinite diagnostic samples")
        mae, maximum = float(diff.mean()), float(diff.max())
    noise = row["noise"] == reference["noise"]
    sampling = row["sampling_identity"] == reference["sampling_identity"]
    env = {k:v for k,v in row["environment"].items() if k != "wan_module"} == {k:v for k,v in reference["environment"].items() if k != "wan_module"}
    return {"noise_match": noise, "sampling_match": sampling, "environment_match": env,
        "sample_mae": mae, "sample_max": maximum,
        "compatible": noise and sampling and env and mae <= tolerance["sample_mae_tolerance"] and maximum <= tolerance["sample_max_tolerance"]}


def report(args):
    frozen = load_plan(args.out)
    rows = pilot.collect(args.out, frozen, "calibration")
    if len(rows) != len(frozen["jobs"]):
        raise RuntimeError(f"Diagnostic incomplete: {len(rows)}/{len(frozen['jobs'])}; resume generate")
    refs = [pilot.read(r["path"]) for r in frozen["source_records"]]
    by_group = {}
    for ref in refs:
        by_group.setdefault(ref["job"]["group_id"], {})[ref["job"]["arm"]["id"]] = ref
    tolerance = frozen["config"]["calibration"]
    checks = []
    for row in rows:
        j = row["job"]
        diagnostics = row["jenga_diagnostics"]
        c = diagnostics["counters"]
        expected = 2 * j["arm"]["steps"]
        count_valid = all(type(value) is int and value >= 0 for value in c.values())
        count_valid &= c["forward_calls"] == expected and c["sparse_attention_calls"] == 0
        # The locked 1.3B model has 30 blocks, each with dense self- and
        # cross-attention. A cache hit skips both calls in all those blocks.
        count_valid &= 0 <= c["cache_hits"] <= expected
        count_valid &= c["dense_attention_calls"] == (expected - c["cache_hits"]) * 30 * 2
        if j["diagnostic"]["cache"] == "hard_off":
            count_valid &= c["cache_hits"] == 0 and c["hard_off_calls"] == expected and c["cache_enabled_calls"] == 0
        else:
            count_valid &= c["hard_off_calls"] == 0 and c["cache_enabled_calls"] == expected
        count_valid &= c["identity_order_calls"] == (expected if j["diagnostic"]["order"] == "identity" else 0)
        if diagnostics["settings"] != j["diagnostic"] or not count_valid:
            raise ValueError(f"Diagnostic intervention/count validation failed: {j['id']}")
        peers = by_group[j["group_id"]]
        checks.append({"group": j["group_id"], "arm": j["arm"]["id"], "counters": c,
            "pipeline_seconds": row["runtime"]["pipeline_seconds"],
            "vs_full50": compare(row, peers["FULL50"], tolerance),
            "vs_original_jenga_off": compare(row, peers["JENGA_OFF"], tolerance)})
    group_findings = []
    for group in by_group:
        current = {r["job"]["arm"]["id"]: r for r in rows if r["job"]["group_id"] == group}
        hard = current["JENGA_HARD_OFF"]
        identity = current["JENGA_IDENTITY_OFF"]
        group_checks = {c["arm"]: c for c in checks if c["group"] == group}
        group_findings.append({"group": group,
            "legacy_zero_repeat_vs_original_compatible": group_checks["JENGA_ZERO_COUNTED"]["vs_original_jenga_off"]["compatible"],
            "legacy_zero_cache_hits": group_checks["JENGA_ZERO_COUNTED"]["counters"]["cache_hits"],
            "hard_off_vs_full_compatible": group_checks["JENGA_HARD_OFF"]["vs_full50"]["compatible"],
            "identity_off_vs_full_compatible": group_checks["JENGA_IDENTITY_OFF"]["vs_full50"]["compatible"],
            "identity_vs_hard_off": compare(identity, hard, tolerance)})
    timing = []
    for arm in frozen["config"]["arms"]:
        values = [r for r in refs if r["job"]["arm"]["id"] == arm["id"]]
        ratios = [by_group[r["job"]["group_id"]]["FULL50"]["runtime"]["pipeline_seconds"] / r["runtime"]["pipeline_seconds"] for r in values]
        timing.append({"arm": arm["id"], "n": len(values), "mean_pipeline_seconds": statistics.mean(r["runtime"]["pipeline_seconds"] for r in values),
            "mean_paired_speedup": statistics.mean(ratios), "scope": "existing two-prompt calibration only, excludes warmup"})
    result = {"schema": "published_wan21_jenga_diagnostic_report_v1", "plan_sha256": frozen["plan_sha256"],
        "source_plan_sha256": frozen["source_plan_sha256"], "checks": checks, "group_findings": group_findings,
        "existing_calibration_timing": timing, "thresholds": tolerance,
        "pilot_released": False, "claim": frozen["claim"],
        "interpretation": "Zero hits rules out cache reuse in the counted run only. Identity-only recovery implicates ordering/numerical effects, not necessarily incorrect indexing. No automatic pilot release, exclusion or threshold relaxation."}
    pilot.write(args.out / "diagnostic_report.json", result)
    lines = ["# Jenga implementation diagnostic", "", "Diagnostic only. Original pilot remains blocked; thresholds unchanged.", "",
        "| Group | Legacy cache hits | Hard OFF vs FULL compatible | Identity OFF vs FULL compatible | Legacy repeat vs old OFF compatible |",
        "|---|---:|---|---|---|"]
    for finding in group_findings:
        lines.append(f"| {finding['group']} | {finding['legacy_zero_cache_hits']} | {finding['hard_off_vs_full_compatible']} | {finding['identity_off_vs_full_compatible']} | {finding['legacy_zero_repeat_vs_original_compatible']} |")
    (args.out / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for finding in group_findings:
        print(finding)
    print(f"Report: {args.out / 'diagnostic_report.json'}; no old assets changed and no pilot released")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "check", "generate", "report"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    args.source, args.out = args.source.resolve(), args.out.resolve()
    separate_roots(args.source, args.out)
    if args.mode == "plan":
        freeze(args)
    elif args.mode in ("check", "generate"):
        launch(args, probe=args.mode == "check")
    else:
        report(args)


if __name__ == "__main__":
    main()
