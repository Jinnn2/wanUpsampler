"""Frozen published-method Wan2.1 pilot: calibration, eight-GPU generation, scoring.

Plan/status/finalize are CPU-only. Workers import published GPU implementations
in isolated processes; scoring reuses the content-bound VBench runner. This is
NOT a reproduction of official full-suite VBench Total or a trained router.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "UNIV_adaptor/configs/published_wan21_pilot_v1.json"
WORKER = Path(__file__).with_name("published_wan21_worker.py")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def immutable(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Frozen file differs; use a new output root: {path}")
    else:
        write(path, value)


def csv_write(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_config(cfg):
    if cfg["schema"] != "published_wan21_pilot_v1":
        raise ValueError("Wrong pilot schema")
    if cfg["sampling"]["task"] != "t2v-1.3B":
        raise ValueError("This locked pilot requires Wan2.1 T2V-1.3B")
    if not isinstance(cfg["warmup"], bool) or cfg["sampling"]["frame_num"] % 4 != 1:
        raise ValueError("Invalid warmup/frame configuration")
    if len(set(cfg["seeds"])) != len(cfg["seeds"]) or any(s < 0 for s in cfg["seeds"]):
        raise ValueError("Seeds must be distinct nonnegative integers")
    arms = cfg["arms"] + cfg["disabled_arms"]
    if len({a["id"] for a in arms}) != len(arms) or cfg["arms"][0]["id"] != "FULL50":
        raise ValueError("Duplicate arms or missing FULL50 reference")
    prompts = cfg["prompts"] + cfg["calibration"]["prompts"]
    if len({p["id"] for p in prompts}) != len(prompts) or len({p["prompt"] for p in prompts}) != len(prompts):
        raise ValueError("Duplicate/overlapping calibration and study prompts")
    if {p["motion"] + "/" + p["detail"] for p in cfg["prompts"]} != {"low/low", "low/high", "high/low", "high/high"}:
        raise ValueError("All four nominal content cells must be present")
    for arm in arms:
        if arm["source"] not in ("wan21", "teacache", "scalingcache", "jenga") or arm["steps"] < 2:
            raise ValueError("Unsupported arm")
        for flag in arm["flags"]:
            if flag.startswith("--") and flag not in {"--teacache_thresh", "--use_ret_steps", "--mode", "--first_enhance", "--dynamic_cache", "--use_alpha", "--p_remain_rates", "--sa_drop_rates"}:
                raise ValueError(f"Arm flags cannot override common protocol: {flag}")
    return cfg


def source_records(cfg):
    from UNIV_adaptor.scripts.data.fetch_published_accelerators import load_manifest
    manifest, root = load_manifest()
    needed = {a["source"] for a in cfg["arms"] + cfg["disabled_arms"]} | {"wan21", cfg["prompt_source"]["repository"]}
    records = [r for r in manifest["repositories"] if r["name"] in needed]
    if {r["name"] for r in records} != needed:
        raise ValueError("Source registry does not cover pilot methods")
    return records, root


def check_sources(cfg):
    from UNIV_adaptor.scripts.data.fetch_published_accelerators import check_checkout
    records, root = source_records(cfg)
    for record in records:
        state = check_checkout(record, root / record["name"])
        if state != "ok":
            raise RuntimeError(f"{record['name']}: {state}; run this pilot's fetch mode first")
    return {r["name"]: r["commit"] for r in records}


def weight_inventory(root):
    root = Path(root).resolve()
    required = [root / "config.json", root / "Wan2.1_VAE.pth", root / "models_t5_umt5-xxl-enc-bf16.pth",
                root / "google/umt5-xxl/tokenizer_config.json"]
    weights = sorted(root.glob("diffusion_pytorch_model*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No native Wan DiT safetensors in {root}; a LightX2V-only converted checkpoint is insufficient")
    missing = [str(p) for p in required if not p.is_file()]
    tokenizer_assets = list((root / "google/umt5-xxl").glob("*.model")) + list((root / "google/umt5-xxl").glob("tokenizer.json"))
    if missing or not tokenizer_assets:
        raise FileNotFoundError(f"Incomplete native Wan weights/tokenizer: {missing}")
    config = read(root / "config.json")
    if config.get("dim") != 1536 or config.get("num_layers") != 30:
        raise ValueError("Checkpoint is not the expected Wan2.1 T2V-1.3B architecture")
    # Large weights are bound by path/size/mtime here, not claimed to be SHA256-verified.
    paths = sorted({p for p in required + weights + tokenizer_assets + list(root.glob("*.index.json"))})
    return [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
             "sha256": file_hash(p) if p.suffix == ".json" else None} for p in paths]


def build_plan(cfg, model_root, ngpus, commits, prompt_metadata, weights):
    cfg = validate_config(cfg)
    if ngpus < 1:
        raise ValueError("NGPUS must be positive")
    jobs = []
    for phase, prompts, seeds, arms in (
        ("pilot", cfg["prompts"], cfg["seeds"], cfg["arms"]),
        ("calibration", cfg["calibration"]["prompts"], [cfg["calibration"]["seed"]], cfg["arms"] + cfg["disabled_arms"]),
    ):
        for index, (prompt, seed) in enumerate((p, s) for p in prompts for s in seeds):
            group = f"{prompt['id']}_s{seed}"
            for arm_index, arm in enumerate(arms):
                jobs.append({"id": f"{phase}_{group}_{arm['id']}", "group_id": group,
                    "phase": phase, "prompt_id": prompt["id"], "prompt": prompt["prompt"], "seed": seed,
                    "gpu": (index + arm_index) % ngpus, "arm": arm})
    body = {"schema": "published_wan21_plan_v1", "config": cfg, "model_root": str(model_root), "ngpus": ngpus,
            "source_commits": commits, "prompt_metadata": prompt_metadata, "weight_inventory": weights,
            "implementation": {p.name: file_hash(p) for p in (Path(__file__), WORKER)}, "jobs": jobs}
    return body | {"plan_sha256": digest(body)}


def plan(args):
    validate_output_root(args.out)
    cfg = validate_config(read(args.config))
    commits = check_sources(cfg)
    source = ROOT / "UNIV_adaptor/external" / cfg["prompt_source"]["repository"] / cfg["prompt_source"]["path"]
    public = read(source)
    metadata = {}
    for prompt in cfg["prompts"]:
        if prompt["origin"] == "vbench":
            matches = [p for p in public if p["prompt_en"] == prompt["prompt"]]
            if len(matches) != 1:
                raise ValueError(f"Public prompt source is missing or ambiguous: {prompt['id']}")
            metadata[prompt["id"]] = matches[0]
    metadata["source_sha256"] = file_hash(source)
    result = build_plan(cfg, args.model_root.resolve(), args.ngpus, commits, metadata, weight_inventory(args.model_root))
    immutable(args.out / "plan.json", result)
    print(f"Frozen plan {result['plan_sha256'][:12]}: {sum(j['phase']=='pilot' for j in result['jobs'])} pilot videos; {sum(j['phase']=='calibration' for j in result['jobs'])} calibration videos")


def validate_output_root(out):
    out = Path(out)
    markers = [name for name in ("sparse_action_plan.json", "sparse_dataset_manifest.json", "generation_manifest.json", "collection_plan.json") if (out / name).exists()]
    if markers:
        raise ValueError(f"Output belongs to another experiment: {out}: {markers}. Set PUBLISHED_WAN21_ROOT to a new dedicated directory; old assets are untouched.")
    for name, expected in (("plan.json", "published_wan21_plan_v1"), ("dataset_manifest.json", "published_wan21_dataset_v1")):
        if (out / name).exists() and read(out / name).get("schema") != expected:
            raise ValueError(f"Foreign experiment schema in {out / name}; use a dedicated published Wan2.1 output")


def load_plan(out, verify_implementation=True):
    validate_output_root(out)
    result = read(Path(out) / "plan.json")
    if digest({k: v for k, v in result.items() if k != "plan_sha256"}) != result["plan_sha256"]:
        raise ValueError("Plan hash mismatch")
    if verify_implementation:
        if result["implementation"] != {p.name: file_hash(p) for p in (Path(__file__), WORKER)}:
            raise ValueError("Pilot implementation changed after freezing; use a new output directory")
        if check_sources(result["config"]) != result["source_commits"]:
            raise ValueError("Pinned source registry changed after freezing")
    return result


def receipt_valid(out, plan, job):
    path = Path(out) / "records" / (job["id"] + ".json")
    if not path.exists():
        return False
    row = read(path)
    if row["plan_sha256"] != plan["plan_sha256"] or row["job"] != job:
        raise ValueError(f"Receipt protocol mismatch: {path}")
    if not math.isfinite(row["runtime"]["pipeline_seconds"]) or row["runtime"]["pipeline_seconds"] <= 0:
        raise ValueError(f"Invalid runtime: {path}")
    if file_hash(row["video_path"]) != row["video_sha256"]:
        raise ValueError(f"Video changed: {path}")
    if row.get("sample_path") and file_hash(row["sample_path"]) != row["sample_sha256"]:
        raise ValueError(f"Calibration sample changed: {path}")
    return True


def collect(out, plan, phase):
    rows = []
    for job in plan["jobs"]:
        if job["phase"] == phase and receipt_valid(out, plan, job):
            rows.append(read(Path(out) / "records" / (job["id"] + ".json")))
    return rows


def fetch(args):
    from UNIV_adaptor.scripts.data.fetch_published_accelerators import fetch as fetch_one
    records, root = source_records(validate_config(read(args.config)))
    for record in records:
        fetch_one(record, root / record["name"])


def python_for(arm, fallback):
    return os.environ.get("PYTHON_" + arm["source"].upper(), fallback)


def log_excerpt(path, lines=60):
    from collections import deque
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as handle:
            return "".join(deque(handle, maxlen=lines))[-16000:]
    except OSError as exc:
        return f"Cannot read worker log: {exc}"


def diagnose(args):
    """Read-only diagnostics, usable even with a foreign/old frozen plan."""
    from UNIV_adaptor.scripts.data.fetch_published_accelerators import check_checkout, git
    cfg = validate_config(read(args.config))
    print(f"Diagnostic output: {args.out}\nDriver Python: {sys.executable}\nWorker fallback: {args.python}")
    try:
        validate_output_root(args.out)
    except ValueError as exc:
        print(f"OUTPUT WARNING: {exc}")
    records, root = source_records(cfg)
    for record in records:
        target = root / record["name"]
        print(f"SOURCE {record['name']}: {check_checkout(record, target)}")
        if target.exists():
            print(git("status", "--short", "--untracked-files=all", cwd=target) or "clean")
    # Latest representative log per implementation; STEP25 shares Wan's imports.
    seen = set()
    arms = cfg["arms"] + cfg["disabled_arms"]
    paths = sorted((args.out / "logs").glob("*.log"), key=lambda p:p.stat().st_mtime_ns, reverse=True)
    for path in paths:
        arm = next((a for a in arms if f"_{a['id']}_gpu" in path.name), None)
        if not arm or arm["source"] in seen:
            continue
        seen.add(arm["source"])
        print(f"\nWORKER {arm['source']}: {path}\n{log_excerpt(path)}")
    if not seen:
        print("No published-worker logs found here; pass --out with the failed run's exact directory.")


def launch(args, calibration=False, probe=False):
    validate_output_root(args.out)
    if not (args.out / "plan.json").exists():
        plan(args)
    frozen = load_plan(args.out)
    if weight_inventory(Path(frozen["model_root"])) != frozen["weight_inventory"]:
        raise ValueError("Weight files changed after plan was frozen")
    if args.ngpus != frozen["ngpus"]:
        raise ValueError("NGPUS differs from frozen plan")
    if not calibration and not probe:
        audit(args)
    phases = "calibration" if calibration else "pilot"
    arms = frozen["config"]["arms"] + (frozen["config"]["disabled_arms"] if calibration else [])
    processes = {}
    queues = {}
    logs = args.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # One live process per GPU, each processing one arm's assigned jobs persistently.
    # Rotate method order by GPU to avoid all FULL jobs occupying the same time window.
    for gpu in range(frozen["ngpus"]):
        rotated = arms[gpu % len(arms):] + arms[:gpu % len(arms)]
        queues[gpu] = [a for a in rotated if any(j["phase"] == phases and j["gpu"] == gpu and j["arm"]["id"] == a["id"] and (probe or not receipt_valid(args.out, frozen, j)) for j in frozen["jobs"])]
    failures = []
    last_progress = time.monotonic()
    try:
        while any(queues.values()) or processes:
            for gpu, queue in queues.items():
                if gpu in processes or not queue:
                    continue
                arm = queue.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
                for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                    env.pop(key, None)
                command = [python_for(arm, args.python), str(WORKER), "--out", str(args.out), "--gpu", str(gpu), "--arm", arm["id"]]
                if calibration:
                    command += ["--calibration"]
                if probe:
                    command += ["--probe"]
                prefix = "probe" if probe else phases
                path = logs / f"{prefix}_{arm['id']}_gpu{gpu}_{time.time_ns()}.log"
                handle = path.open("w", encoding="utf-8")
                process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                processes[gpu] = (process, handle, arm, path)
                print(f"GPU {gpu}: {arm['id']} -> {path}", flush=True)
            for gpu, (process, handle, arm, path) in list(processes.items()):
                if process.poll() is not None:
                    handle.close()
                    del processes[gpu]
                    if process.returncode:
                        failures.append(str(path))
                        queues[gpu].clear()
                        print(f"FAILED GPU {gpu} / {arm['id']}; inspect {path}", flush=True)
                        print(log_excerpt(path), flush=True)
            if time.monotonic() - last_progress >= 30:
                completed = sum((args.out / "records" / (j["id"] + ".json")).exists() for j in frozen["jobs"] if j["phase"] == phases)
                print(f"{phases}: {completed} receipts saved; {len(processes)} active GPUs. Detailed progress is in logs/.", flush=True)
                last_progress = time.monotonic()
            time.sleep(0.5)
    except BaseException:
        for process, handle, _, _ in processes.values():
            process.terminate()
        for process, handle, _, _ in processes.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            handle.close()
        raise
    if failures:
        raise RuntimeError(f"Worker failures: {failures}")


def check(args):
    cfg = validate_config(read(args.config))
    check_sources(cfg)
    weight_inventory(args.model_root)
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Missing {executable}")
    launch(args, calibration=True, probe=True)
    print("Source, native checkpoint, arguments, CUDA imports and small dense FA2 kernels verified. Full-shape model and sparse Triton generation still require calibrate.")


def audit(args):
    import numpy as np
    frozen = load_plan(args.out)
    rows = collect(args.out, frozen, "calibration")
    expected = [j for j in frozen["jobs"] if j["phase"] == "calibration"]
    if len(rows) != len(expected):
        raise RuntimeError(f"Calibration incomplete: {len(rows)}/{len(expected)}. Run calibrate; pilot generation is blocked.")
    groups = defaultdict(dict)
    for row in rows:
        groups[row["job"]["group_id"]][row["job"]["arm"]["id"]] = row
    checks = []
    tolerance = frozen["config"]["calibration"]
    for group, by_arm in groups.items():
        reference = by_arm["FULL50"]
        with np.load(reference["sample_path"]) as sample:
            baseline = sample["frames"]
        for arm, row in by_arm.items():
            noise_match = row["noise"] == reference["noise"]
            sampling_match = row["sampling_identity"] == reference["sampling_identity"]
            environment_match = {k:v for k,v in row["environment"].items() if k != "wan_module"} == {k:v for k,v in reference["environment"].items() if k != "wan_module"}
            item = {"group": group, "arm": arm, "noise_match": noise_match, "sampling_match": sampling_match,
                    "environment_match": environment_match, "pipeline_seconds": row["runtime"]["pipeline_seconds"],
                    "passed": noise_match and sampling_match and environment_match}
            if arm.endswith("_OFF"):
                with np.load(row["sample_path"]) as sample:
                    difference = np.abs(sample["frames"] - baseline)
                item.update(sample_mae=float(difference.mean()), sample_max=float(difference.max()))
                item["passed"] &= item["sample_mae"] <= tolerance["sample_mae_tolerance"] and item["sample_max"] <= tolerance["sample_max_tolerance"]
            checks.append(item)
    result = {"plan_sha256": frozen["plan_sha256"], "passed": all(r["passed"] for r in checks),
              "checks": checks, "caveat": "Raw float subsample checks implementation compatibility; this is not exhaustive pixel equality. Thresholds are preregistered, not adjusted based on pilot output."}
    write(args.out / "calibration_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise RuntimeError("Calibration failed. Diagnose source/sampler/numerical differences; do not silently relax thresholds or claim a common FULL baseline.")


def status(args):
    frozen = load_plan(args.out)
    output = {}
    for phase in ("calibration", "pilot"):
        rows = collect(args.out, frozen, phase)
        arms = len(frozen["config"]["arms"]) + (len(frozen["config"]["disabled_arms"]) if phase == "calibration" else 0)
        groups = defaultdict(list)
        for row in rows:
            groups[row["job"]["group_id"]].append(row)
        output[phase] = {"completed": len(rows), "expected": sum(j["phase"] == phase for j in frozen["jobs"]),
                         "complete_groups": sum(len(v) == arms for v in groups.values())}
    print(json.dumps(output, indent=2))


def finalized(args):
    frozen = load_plan(args.out)
    saved = read(args.out / "dataset_manifest.json")
    if saved["plan_sha256"] != frozen["plan_sha256"]:
        raise ValueError("Finalized manifest belongs to another plan")
    for row in saved["records"]:
        if not receipt_valid(args.out, frozen, row["job"]) or row != read(args.out / "records" / (row["job"]["id"] + ".json")):
            raise ValueError("Finalized receipt changed")
    return frozen, saved


def finalized_score_rows(args, frozen, dataset):
    """Bind any analysis/human selection to the verified scored video inventory."""
    score_path = args.out / "metrics/quality_by_video.csv"
    provenance = read(args.out / "metrics/score_provenance.json")
    if provenance["plan_sha256"] != frozen["plan_sha256"] or provenance["quality_csv_sha256"] != file_hash(score_path):
        raise ValueError("Scored CSV or protocol changed after strict scoring")
    with score_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {r["job"]["id"]: r for r in dataset["records"]}
    if len(rows) != len(expected) or {r["observation_id"] for r in rows} != set(expected):
        raise ValueError("Score CSV does not cover the finalized inventory")
    for row in rows:
        original = expected[row["observation_id"]]
        job = original["job"]
        if (row["video_sha256"] != original["video_sha256"] or row["prompt"] != job["prompt"]
                or row["action_id"] != job["arm"]["id"] or row["group_id"] != job["group_id"] or int(row["seed"]) != job["seed"]):
            raise ValueError("Score identity mismatch")
    return rows


def finalize(args):
    frozen = load_plan(args.out)
    audit(args)
    rows = collect(args.out, frozen, "pilot")
    groups = defaultdict(list)
    for row in rows:
        groups[row["job"]["group_id"]].append(row)
    complete = {g for g,v in groups.items() if len(v) == len(frozen["config"]["arms"])}
    expected_groups = len(frozen["config"]["prompts"]) * len(frozen["config"]["seeds"])
    if len(complete) != expected_groups and not args.allow_partial:
        raise RuntimeError(f"Only {len(complete)}/{expected_groups} complete groups; use --allow-partial for an explicitly exploratory snapshot")
    if not complete:
        raise RuntimeError("No complete groups")
    records = [r for r in rows if r["job"]["group_id"] in complete]
    for group in complete:
        reference = next(r for r in groups[group] if r["job"]["arm"]["id"] == "FULL50")
        if any(r["noise"] != reference["noise"] or r["sampling_identity"] != reference["sampling_identity"] for r in groups[group]):
            raise ValueError(f"Unmatched noise/configuration in {group}")
    if getattr(args, "snapshot_out", None):
        target = args.snapshot_out.resolve()
        if target == args.out or target.is_relative_to(args.out):
            raise ValueError("Use a separate sibling snapshot directory, not the live output or its child")
        immutable(target / "plan.json", frozen)
        for record in rows + collect(args.out, frozen, "calibration"):
            immutable(target / "records" / (record["job"]["id"] + ".json"), record)
        immutable(target / "calibration_audit.json", read(args.out / "calibration_audit.json"))
        args.out = target
    elif len(complete) != expected_groups and not getattr(args, "allow_in_place_partial", False):
        raise ValueError("Use --allow-partial --snapshot-out SIBLING_PATH so the live experiment can continue without replacing a frozen manifest")
    immutable(args.out / "dataset_manifest.json", {"schema": "published_wan21_dataset_v1", "plan_sha256": frozen["plan_sha256"],
        "partial_exploratory": len(complete) != expected_groups, "expected_groups": expected_groups, "complete_groups": sorted(complete), "records": records})
    prompts = {p["id"]: p for p in frozen["config"]["prompts"]}
    review = [{"group_id": r["job"]["group_id"], "prompt": r["job"]["prompt"],
               "nominal_motion": prompts[r["job"]["prompt_id"]]["motion"], "nominal_detail": prompts[r["job"]["prompt_id"]]["detail"],
               "video_path": r["video_path"], "actual_motion": "", "actual_detail": "", "base_failure": "", "notes": ""}
              for r in records if r["job"]["arm"]["id"] == "FULL50"]
    if not (args.out / "base_observability_review.csv").exists():
        csv_write(args.out / "base_observability_review.csv", review)
    print(f"Finalized {len(records)} videos / {len(complete)} matched groups; partial={len(complete) != expected_groups}")


def score(args):
    from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import inspect_vbench_checkout, score_case_directory, warmup_vbench_cache
    frozen, dataset = finalized(args)
    cfg = frozen["config"]
    quality, diagnostic = cfg["evaluation"]["quality_dimensions"], cfg["evaluation"]["diagnostic_dimensions"]
    identity = inspect_vbench_checkout(args.vbench_root, expected_commit=args.expected_vbench_commit)
    inputs = []
    prompts = {p["id"]: p for p in cfg["prompts"]}
    for record in dataset["records"]:
        j = record["job"]
        p = prompts[j["prompt_id"]]
        inputs.append({"observation_id": j["id"], "group_id": j["group_id"], "action_id": j["arm"]["id"],
            "prompt": j["prompt"], "prompt_key": j["prompt_id"], "seed": j["seed"],
            "motion": p["motion"], "detail": p["detail"], "family_id": p["family_id"], "origin": p["origin"],
            "video_path": record["video_path"], "video_sha256": record["video_sha256"], "pipeline_seconds": record["runtime"]["pipeline_seconds"]})
    metrics = args.out / "metrics"
    immutable(metrics / "evaluation_inputs.json", {"plan_sha256": frozen["plan_sha256"], "rows": inputs})
    warmup_vbench_cache(args.vbench_python, args.vbench_root)
    output, provenance = [], {}
    for arm in cfg["arms"]:
        rows = [r for r in inputs if r["action_id"] == arm["id"]]
        # Isolate complete groups so an incomplete file cannot leak into VBench.
        directory = metrics / "inputs" / arm["id"]
        directory.mkdir(parents=True, exist_ok=True)
        for row in rows:
            target = directory / (row["observation_id"] + ".mp4")
            if target.exists():
                if file_hash(target) != row["video_sha256"]:
                    raise ValueError("Scoring link changed")
            else:
                try:
                    os.link(row["video_path"], target)
                except OSError:
                    shutil.copy2(row["video_path"], target)
        prompt_map = directory / "prompt_map.json"
        immutable(prompt_map, {str((directory / (r["observation_id"] + ".mp4")).resolve()): r["prompt"] for r in rows})
        bundle = score_case_directory(args.vbench_root, args.vbench_python, directory, prompt_map,
            metrics / "vbench" / arm["id"], quality + diagnostic, quality, diagnostic, args.ngpus, False, identity)
        provenance[arm["id"]] = bundle.provenance
        for row in rows:
            values = bundle.scores[row["observation_id"]]
            output.append(row | values | {"vbench5": statistics.mean(values[d] for d in quality)})
    csv_path = metrics / "quality_by_video.csv"
    if (metrics / "score_provenance.json").exists():
        prior = read(metrics / "score_provenance.json")
        if not csv_path.exists() or prior["quality_csv_sha256"] != file_hash(csv_path):
            raise ValueError("Previously scored CSV changed; refusing to overwrite it")
    csv_write(csv_path, output)
    immutable(metrics / "score_provenance.json", {"plan_sha256": frozen["plan_sha256"], "vbench": provenance,
        "quality_csv_sha256": file_hash(csv_path),
        "aggregate": "vbench5 is an exploratory raw five-dimension mean, NOT normalized official VBench Total; diagnostics are not included"})
    report(args)


def report(args):
    frozen, dataset = finalized(args)
    metrics = args.out / "metrics"
    scored = finalized_score_rows(args, frozen, dataset)
    references = {r["group_id"]: r for r in scored if r["action_id"] == "FULL50"}
    paired = []
    for row in scored:
        if row["action_id"] == "FULL50":
            continue
        ref = references[row["group_id"]]
        paired.append({"group_id": row["group_id"], "prompt_key": row["prompt_key"], "action_id": row["action_id"],
            "seed": row["seed"], "origin": row["origin"], "cell": row["motion"] + "/" + row["detail"],
            "speedup": float(ref["pipeline_seconds"]) / float(row["pipeline_seconds"]),
            **{"delta_" + d: float(row[d]) - float(ref[d]) for d in ["vbench5"] + frozen["config"]["evaluation"]["quality_dimensions"] + frozen["config"]["evaluation"]["diagnostic_dimensions"]}})
    csv_write(metrics / "paired_deltas.csv", paired)
    summary = []
    buckets = defaultdict(list)
    for row in paired:
        buckets[row["action_id"], row["origin"], row["cell"]].append(row)
    for (arm, origin, cell), rows in sorted(buckets.items()):
        clusters = defaultdict(list)
        for row in rows:
            clusters[row["prompt_key"]].append(row["delta_vbench5"])
        means = [statistics.mean(v) for v in clusters.values()]
        ci = None
        if len(means) >= 2:
            rng = random.Random(20261001)
            boot = sorted(statistics.mean(rng.choices(means, k=len(means))) for _ in range(2000))
            ci = [boot[49], boot[1949]]
        summary.append({"arm": arm, "origin": origin, "nominal_cell": cell, "groups": len(rows), "prompts": len(means),
            "mean_speedup": statistics.mean(r["speedup"] for r in rows), "mean_delta_vbench5": statistics.mean(means),
            "prompt_bootstrap_ci": ci,
            "close_score_fraction": statistics.mean(abs(r["delta_vbench5"]) <= frozen["config"]["evaluation"]["metric_epsilon"] for r in rows)})
    write(metrics / "report.json", {"claim": frozen["config"]["claim"], "status": "pilot; metric inadequacy and router benefit require independent human/heldout evidence",
        "partial_exploratory": dataset["partial_exploratory"], "summary": summary,
        "caveats": ["Not official VBench Total or a full paper-table reproduction.", "Published presets have different measured speedups: no latency-matched quality ranking.",
            "Cells are nominal prompt labels; inspect base_observability_review.csv without excluding base failures after viewing scores.", "Two seeds and very few prompt families cannot establish generalizable prompt-only routing.",
            "Dynamic degree is a motion diagnostic, not universally higher-is-better.", "Bootstrap conditions on observed seeds and uses prompt clusters; tiny cell intervals may be unavailable."]})
    print(f"Saved per-video dimensions, matched deltas and cautious pilot report: {metrics}")


def blind(args):
    from UNIV_adaptor.scripts.data import acceleration_blind_audit as human
    frozen, dataset = finalized(args)
    finalized_score_rows(args, frozen, dataset)
    if dataset["partial_exploratory"]:
        raise ValueError("The preregistered 96-pair human pilot requires all groups; partial scoring remains exploratory")
    cfg = frozen["config"]
    quality, diagnostic = cfg["evaluation"]["quality_dimensions"], cfg["evaluation"]["diagnostic_dimensions"]
    epsilon = {m: cfg["evaluation"]["metric_epsilon"] for m in ["vbench5"] + quality + diagnostic}
    epsilon["dynamic_degree"] = 0.0
    source, identity = human.load_source("published_wan21", args.out / "metrics", epsilon)
    spec = {"schema": "acceleration_blind_audit_v1", "seed": 20261001, "minimum_raters": cfg["evaluation"]["minimum_raters"],
        "presentation": cfg["evaluation"]["presentation"], "metric_epsilon": epsilon,
        "real_strata": [{"source": "published_wan21", "left": "FULL50", "right": arm["id"], "count": len(dataset["complete_groups"])} for arm in cfg["arms"] if arm["id"] != "FULL50"],
        "synthetic": {"source": "published_wan21", "action": "FULL50", "bases": 0},
        "seed_controls": {"source": "published_wan21", "action": "FULL50", "count": 0}}
    pairs = human.make_plan(spec, {"published_wan21": source})
    # Six separately labelled repeats, never included in primary real-pair analysis.
    for pair in random.Random(20261001).sample(pairs, 6):
        pairs.append(pair | {"id": human.digest(["repeat", pair["id"]])[:20], "kind": "reliability_repeat"})
    body = {"schema": "acceleration_blind_plan_v1", "config": spec, "sources": {"published_wan21": identity}, "pairs": pairs}
    body["plan_sha256"] = human.digest(body)
    immutable(args.out / "blind/private/plan.json", body)
    human.package(argparse.Namespace(out=args.out / "blind", path_map=[]))
    print("Packaged 96 primary + 6 reliability pairs. Record defect, A/B, severity and timestamp in notes; use >=3 independent raters.")


def blind_report(args):
    """Add method/content breakdown and repeated-question reliability to human audit."""
    from UNIV_adaptor.scripts.data import acceleration_blind_audit as human
    study = args.out / "blind"
    human.report(argparse.Namespace(out=study, presented_scores=None))
    plan = human.load_plan(study)
    package = human.load_package(study, plan)
    pairs = {p["id"]: p for p in plan["pairs"]}
    with (study / "analysis/human_consensus.csv").open(encoding="utf-8-sig", newline="") as handle:
        consensus = list(csv.DictReader(handle))
    rows = []
    for row in consensus:
        pair = pairs[row["pair"]]
        if pair["kind"] != "real":
            continue
        rows.append(row | {"arm": pair["b"]["action"], "cell": pair["cell"],
            "prompt": pair["a"]["prompt"], "seed": pair["a"]["seed"],
            "reference_wins": int(row["consensus"] == "A"), "accelerated_wins": int(row["consensus"] == "B")})
    csv_write(study / "analysis/human_by_method_pairs.csv", rows)
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["arm"], row["cell"], row["dimension"]].append(row)
    summary = []
    for (arm, cell, dimension), group in sorted(buckets.items()):
        resolved = [r for r in group if r["consensus"] in ("A", "B", "tie")]
        item = {"arm": arm, "nominal_cell": cell, "dimension": dimension, "planned_pairs": len(group),
            "minimum_raters_pairs": sum(int(r["raters"]) >= plan["config"]["minimum_raters"] for r in group),
            "reference_wins": sum(r["consensus"] == "A" for r in group),
            "accelerated_wins": sum(r["consensus"] == "B" for r in group),
            "ties": sum(r["consensus"] == "tie" for r in group),
            "unresolved": sum(r["consensus"] == "unresolved" for r in group),
            "reference_win_fraction_resolved": statistics.mean(r["reference_wins"] for r in resolved) if resolved else None}
        summary.append(item)
    with (study / "analysis/metric_pairs.csv").open(encoding="utf-8-sig", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    metric_buckets = defaultdict(list)
    for row in metrics:
        pair = pairs[row["pair"]]
        if pair["kind"] == "real" and row["scope"] == "presented":
            metric_buckets[pair["b"]["action"], pair["cell"], row["dimension"], row["metric"]].append(row)
    metric_summary = [{"arm": key[0], "nominal_cell": key[1], "dimension": key[2], "metric": key[3],
        "directional_consensus_pairs": len(group),
        "agreement": statistics.mean(int(r["correct"]) for r in group),
        "metric_tie_rate": statistics.mean(int(r["miss"]) for r in group),
        "reversal_rate": statistics.mean(int(r["reversed"]) for r in group),
        "diagnostic_only": key[3] == "dynamic_degree"} for key,group in sorted(metric_buckets.items())]
    originals = {(p["a"]["id"], p["b"]["id"]): p["id"] for p in pairs.values() if p["kind"] == "real"}
    repeats = [(p["id"], originals[p["a"]["id"], p["b"]["id"]]) for p in pairs.values() if p["kind"] == "reliability_repeat"]
    reliability = []
    for file in sorted((study / "private/ratings").glob("*.json")):
        rating = read(file)
        mapping = {r["id"]: r for r in human.session(plan, package, rating["participant"])}
        def unblind(key, dimension):
            answer = rating["answers"].get(key)
            if not answer:
                return None
            choice = answer["votes"][dimension]
            if mapping[key]["swap"] and choice in ("A", "B"):
                choice = "B" if choice == "A" else "A"
            return choice
        for dimension in human.DIMENSIONS:
            comparisons = [(unblind(rep, dimension), unblind(orig, dimension)) for rep,orig in repeats]
            valid = [(a,b) for a,b in comparisons if a in ("A", "B", "tie") and b in ("A", "B", "tie")]
            reliability.append({"participant": rating["participant"], "dimension": dimension,
                "planned_repeats": len(repeats), "valid_repeats": len(valid),
                "within_rater_agreement": statistics.mean(a == b for a,b in valid) if valid else None})
    csv_write(study / "analysis/repeat_reliability.csv", reliability,
              ["participant", "dimension", "planned_repeats", "valid_repeats", "within_rater_agreement"])
    write(study / "analysis/method_content_report.json", {"plan_sha256": plan["plan_sha256"], "package_sha256": package["package_sha256"],
        "status": "pilot; no automatic claim of metric failure or prompt-only routing benefit",
        "human_preference_by_method_cell": summary, "presented_metric_agreement_by_method_cell": metric_summary,
        "repeat_reliability": reliability, "caveats": ["All real pairs, including ties/unresolved, remain in human tables.",
            "Metric agreement conditions on directional human consensus; diagnostics are not universal quality criteria.",
            "Repeats are excluded from primary analysis. Nominal cells require separate FULL observability review.",
            "Tiny cells and two seeds do not identify a robust method-prompt interaction or establish generalization."]})
    print(f"Saved method/content human breakdown and repeat reliability: {study / 'analysis'}")


def export(args):
    from UNIV_adaptor.scripts.data import acceleration_blind_audit as human
    study = args.out / "blind"
    plan = human.load_plan(study)
    package = human.load_package(study, plan)
    public = {"schema": "acceleration_blind_local_v1", "plan_sha256": plan["plan_sha256"],
              "package_sha256": package["package_sha256"],
              "pairs": [{k:p[k] for k in ("id", "a", "b", "prompt")} for p in package["pairs"]],
              "clips": {k:{"sha256":v["sha256"]} for k,v in package["clips"].items()}}
    public["bundle_sha256"] = human.digest(public)
    write(study / "public_study.json", public)
    export_root = args.out / "exports"
    export_root.mkdir(exist_ok=True)
    # Independent <=~64MB archives, no multipart reconstruction or main-Git videos.
    files = [study / "public_study.json"] + sorted((study / "media").glob("*.mp4"))
    for clip, receipt in public["clips"].items():
        if file_hash(study / "media" / (clip + ".mp4")) != receipt["sha256"]:
            raise ValueError("Presented media changed before export")
    chunks, current, size = [], [], 0
    for path in files:
        if current and size + path.stat().st_size > 64 * 1024**2:
            chunks.append(current)
            current, size = [], 0
        current.append(path)
        size += path.stat().st_size
    if current:
        chunks.append(current)
    manifests = []
    for i, chunk in enumerate(chunks):
        target = export_root / f"blind_media_{i:03d}.tgz"
        with tarfile.open(target, "w:gz") as archive:
            for path in chunk:
                archive.add(path, arcname="study/" + str(path.relative_to(study)).replace("\\", "/"))
            if i == 0:
                for name in ("acceleration_blind_audit.py", "acceleration_blind_audit.html"):
                    archive.add(Path(human.__file__).with_name(name), arcname="study/" + name)
        manifests.append({"file": target.name, "bytes": target.stat().st_size, "sha256": file_hash(target)})
    target = export_root / "published_wan21_analysis.tgz"
    with tarfile.open(target, "w:gz") as archive:
        for path in [args.out / "plan.json", args.out / "dataset_manifest.json", args.out / "calibration_audit.json", args.out / "base_observability_review.csv"]:
            if path.exists():
                archive.add(path, arcname=path.name)
        for directory in ("records", "metrics", "blind/private", "blind/analysis"):
            for path in sorted((args.out / directory).rglob("*")):
                if path.is_file() and path.suffix in (".csv", ".json", ".log"):
                    archive.add(path, arcname=str(path.relative_to(args.out)).replace("\\", "/"))
    write(export_root / "export_manifest.json", {"public_media": manifests, "private_analysis": {"file": target.name, "sha256": file_hash(target)}})
    print(f"Public media chunks and separate PRIVATE analysis: {export_root}. Extract all media chunks to the same folder; do not give the analysis archive to raters.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fetch", "check", "plan", "calibrate", "audit", "generate", "status", "finalize", "score", "report", "blind", "blind-report", "export", "diagnose"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/published_wan21_pilot_v1")
    parser.add_argument("--model-root", type=Path, default=Path("/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--snapshot-out", type=Path, help="Separate sibling output for a frozen partial/full snapshot; source videos are reused by reference")
    parser.add_argument("--vbench-root", type=Path, default=Path("/mnt/afs_2/houze/VBench"))
    parser.add_argument("--vbench-python", default="/opt/conda/bin/python")
    parser.add_argument("--expected-vbench-commit", default="fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490")
    args = parser.parse_args()
    args.out, args.vbench_root = args.out.resolve(), args.vbench_root.resolve()
    if args.mode in ("calibrate", "generate"):
        launch(args, calibration=args.mode == "calibrate")
    else:
        globals()[args.mode.replace("-", "_")](args)


if __name__ == "__main__":
    main()
