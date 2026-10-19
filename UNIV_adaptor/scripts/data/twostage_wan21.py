"""Two-stage spatial-only / temporal-only accelerators on the platform-v1 prompts.

Stage 1 runs the pinned official Wan2.1 entrypoint at a smaller canvas (LR81) or
half the frame count (HR41), or the official rCM distilled DiT in that same Wan
model (1-4 steps). FULL50 (HR81) is regenerated on the same machine for
same-hardware timing, or reused from another finalized same-machine run.
Stage 2 runs a published restorer (FlashVSR x4, RIFE x2) or a trivial control
(bicubic, frame hold) on the SAME stage-1 frames; a chain such as
'flashvsr+rife' runs its steps in order, each in its own Python.
Orchestration is stdlib-only; GPU work runs in twostage_wan21_worker.py. The
platform-v1 driver/worker are imported for helpers only and never modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
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

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from UNIV_adaptor.scripts.data import published_wan21_pilot as pilot  # noqa: E402
import flashvsr_asset_diagnostic as diag  # noqa: E402

DEFAULT_CONFIG = ROOT / "UNIV_adaptor/configs/published_wan21_twostage_v1.json"
WORKER = Path(__file__).with_name("twostage_wan21_worker.py")
# Every file whose code runs in this protocol; any change requires a new output root.
IMPLEMENTATION = (Path(__file__), WORKER, Path(pilot.__file__),
                  Path(__file__).with_name("published_wan21_worker.py"), Path(diag.__file__))
POSTS = ("flashvsr", "bicubic", "rife", "duplicate")
SPATIAL_POSTS, TEMPORAL_POSTS = ("flashvsr", "bicubic"), ("rife", "duplicate")
GPU_POSTS = ("flashvsr", "rife")
SAMPLERS = ("wan", "rcm")
EXTERNAL = "external"  # FULL50 reused from another finalized, same-machine run (no regeneration)
PLAN_SCHEMA, DATASET_SCHEMA = "published_wan21_twostage_plan_v1", "published_wan21_twostage_dataset_v1"

read, write, immutable, file_hash, digest, csv_write = (
    pilot.read, pilot.write, pilot.immutable, pilot.file_hash, pilot.digest, pilot.csv_write)


def canvas(size):
    width, height = map(int, size.split("*"))
    return width, height


def post_steps(arm):
    """'flashvsr+rife' -> ['flashvsr', 'rife']; each step runs as its own unit, in order."""
    return [] if arm["post"] == "none" else arm["post"].split("+")


def chain_shapes(cfg, arm):
    """(frames, width, height) entering each post step, plus the final output."""
    width, height = canvas(cfg["sampling"]["size"])
    g = next(x for x in cfg["generations"] if x["id"] == arm["generation"])
    shape = (g["frame_num"], *canvas(g["size"]))
    shapes = [shape]
    for step in post_steps(arm):
        if step not in POSTS:
            raise ValueError(f"Unsupported post stage: {step}")
        if step in SPATIAL_POSTS:
            if shape[1:] != (width // 2, height // 2):
                raise ValueError(f"{arm['id']}: spatial steps take exactly half width/height")
            shape = (shape[0], width, height)
        else:
            shape = (2 * shape[0] - 1, *shape[1:])
        shapes.append(shape)
    if shape != (cfg["sampling"]["frame_num"], width, height):
        raise ValueError(f"{arm['id']}: post chain ends at {shape}, not the final canvas and length")
    return shapes


def validate_config(cfg):
    if cfg["schema"] != "published_wan21_twostage_v1" or cfg["sampling"]["task"] != "t2v-1.3B":
        raise ValueError("Wrong two-stage schema or base model")
    gens = {g["id"]: g for g in cfg["generations"]}
    if len(gens) != len(cfg["generations"]) or EXTERNAL in gens:
        raise ValueError("Duplicate or reserved generation ids")
    for g in gens.values():
        w, h = canvas(g["size"])
        sampler = g.get("sampler", "wan")
        if w % 16 or h % 16 or g["frame_num"] % 4 != 1 or sampler not in SAMPLERS:
            raise ValueError(f"Invalid generation {g['id']}: canvas must be /16, frames 4n+1, known sampler")
        if (sampler == "wan" and g["steps"] < 2) or (sampler == "rcm" and (g["steps"] not in (1, 2, 3, 4) or "rcm" not in cfg)):
            raise ValueError(f"Invalid step count or missing rcm block for {g['id']}")
    arms = cfg["arms"]
    if len({a["id"] for a in arms}) != len(arms) or arms[0]["id"] != "FULL50" or arms[0]["post"] != "none":
        raise ValueError("Duplicate arms or missing FULL50 reference")
    for index, arm in enumerate(arms):
        if arm["generation"] == EXTERNAL:
            if index != 0 or "reference" not in cfg:
                raise ValueError("Only the FULL50 reference may be external, and it needs a reference block")
            continue
        chain_shapes(cfg, arm)
    prompts = cfg["prompts"]
    if len({p["id"] for p in prompts}) != len(prompts) or any(p["split"] == "test" for p in prompts):
        raise ValueError("Duplicate prompts, or a locked test prompt was included")
    if {p["motion"] + "/" + p["detail"] for p in prompts} != {"low/low", "low/high", "high/low", "high/high"}:
        raise ValueError("All four nominal content cells must be present")
    return cfg


def wan_source():
    from UNIV_adaptor.scripts.data.fetch_published_accelerators import check_checkout, load_manifest
    manifest, root = load_manifest()
    record = next(r for r in manifest["repositories"] if r["name"] == "wan21")
    state = check_checkout(record, root / "wan21")
    if state != "ok":
        raise RuntimeError(f"wan21: {state}; run `bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh fetch`")
    return {"path": str((root / "wan21").resolve()), "commit": record["commit"]}


def rife_binding(weights):
    path = Path(weights) / "rife_weight_binding.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run install_twostage_wan21_env.sh")
    binding = read(path)
    diag.verify_files(binding["files"])
    return binding


def flash_binding(cfg, weights):
    path = Path(weights) / "diagnostic_weight_binding.json"
    binding = read(path)
    if (binding["repository"] != cfg["flashvsr"]["weights_repository"] or
            sorted(Path(x["path"]).name for x in binding["files"]) != sorted(cfg["flashvsr"]["weights_files"])):
        raise ValueError("Wrong FlashVSR weight binding")
    return binding, path


def rcm_binding(cfg, path):
    """The rCM DiT checkpoint, bound by the Hugging Face LFS sha256 frozen in the config."""
    identity = diag.bound_file(path)
    if identity["sha256"] != cfg["rcm"]["weights_sha256"]:
        raise ValueError(f"rCM checkpoint {path} is not {cfg['rcm']['weights_file']}@{cfg['rcm']['weights_revision']}")
    return identity


def reference_binding(cfg, root):
    """FULL50 from another frozen run on the same prompts/seeds; its plan must already exist."""
    if root is None:
        raise ValueError("This config reuses FULL50 from another run; pass --reference-root")
    if Path(root).resolve().name != Path(cfg["reference"]["expected_root"]).name:
        raise ValueError(f"Reference root {root} is not the configured {cfg['reference']['expected_root']}")
    frozen = read(Path(root) / "plan.json")
    return {"root": str(Path(root).resolve()), "plan_sha256": frozen["plan_sha256"], "schema": frozen["schema"],
            "arm": cfg["reference"]["arm"]}


def reference_index(p):
    ref = p["reference"]
    root = Path(ref["root"])
    frozen, manifest = read(root / "plan.json"), read(root / "dataset_manifest.json")
    if frozen["plan_sha256"] != ref["plan_sha256"] or manifest["plan_sha256"] != ref["plan_sha256"] or manifest["partial_exploratory"]:
        raise ValueError(f"Reference run {root} changed or is not completely finalized")
    return {(r["job"]["prompt"], r["job"]["seed"]): r for r in manifest["records"] if r["job"]["arm"]["id"] == ref["arm"]}


def build_jobs(cfg, ngpus):
    gen_jobs, final_jobs = [], []
    pairs = [(p, s) for p in cfg["prompts"] for s in cfg["seeds"]]
    for index, (prompt, seed) in enumerate(pairs):
        group = f"{prompt['id']}_s{seed}"
        common = {"group_id": group, "prompt_id": prompt["id"], "prompt": prompt["prompt"], "seed": seed}
        for g_index, gen in enumerate(cfg["generations"]):
            gen_jobs.append({"id": f"gen_{group}_{gen['id']}", **common, "generation": gen,
                             "gpu": (index + g_index) % ngpus})
        for a_index, arm in enumerate(cfg["arms"]):
            external = arm["generation"] == EXTERNAL
            final_jobs.append({"id": f"pilot_{group}_{arm['id']}", "phase": "pilot", **common, "arm": arm,
                               "generation_job": None if external else f"gen_{group}_{arm['generation']}",
                               "gpu": (index + a_index) % ngpus})
    return gen_jobs, final_jobs


def implementation():
    return {p.name: file_hash(p) for p in IMPLEMENTATION}


def plan(args):
    out = args.out
    if (out / "plan.json").exists():
        load_plan(out)
        print(f"Existing frozen plan verified: {out / 'plan.json'}")
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Unplanned nonempty output: {out}; choose a NEW directory")
    cfg = validate_config(read(args.config))
    used = {s for a in cfg["arms"] for s in post_steps(a)}
    body = {"schema": PLAN_SCHEMA, "config": cfg, "model_root": str(args.model_root.resolve()), "ngpus": args.ngpus}
    if "flashvsr" in used:
        binding, binding_path = flash_binding(cfg, args.flash_weights)
        print("Freezing plan: SHA256 of FlashVSR weights (several GB on shared storage; may take minutes)...", flush=True)
        diag.verify_files(binding["files"])  # full SHA256 once at planning
        body |= {"flash_source": diag.checkout(args.flash_root, cfg["flashvsr"]["commit"]),
                 "kernel_source": diag.checkout(args.kernel_root, cfg["flashvsr"]["kernel_commit"]),
                 "weights": binding, "weight_binding": diag.bound_file(binding_path)}
    if "rife" in used:
        rife = rife_binding(args.rife_weights)
        if rife["archive_sha256"] != cfg["rife"]["weights_archive_sha256"]:
            raise ValueError("RIFE weights come from a different archive")
        body |= {"rife_source": diag.checkout(args.rife_root, cfg["rife"]["commit"]), "rife_weights": rife}
    if any(g.get("sampler") == "rcm" for g in cfg["generations"]):
        print("Freezing plan: SHA256 of the rCM checkpoint (2.8 GB)...", flush=True)
        body["rcm_weights"] = rcm_binding(cfg, args.rcm_weights)
    if cfg["arms"][0]["generation"] == EXTERNAL:
        body["reference"] = reference_binding(cfg, args.reference_root)
    print("Freezing plan: Wan weight inventory and pinned Wan checkout...", flush=True)
    gen_jobs, final_jobs = build_jobs(cfg, args.ngpus)
    body |= {"weight_inventory": pilot.weight_inventory(args.model_root), "wan_source": wan_source(),
             "implementation": implementation(), "gen_jobs": gen_jobs, "final_jobs": final_jobs}
    body["plan_sha256"] = digest(body)
    immutable(out / "plan.json", body)
    print(f"Frozen plan {body['plan_sha256'][:12]}: {len(gen_jobs)} stage-1 generations, "
          f"{sum(len(post_steps(j['arm'])) for j in final_jobs)} post-processing steps, {len(final_jobs)} final videos")


def load_plan(out, verify=True):
    p = read(Path(out) / "plan.json")
    if p.get("schema") != PLAN_SCHEMA or digest({k: v for k, v in p.items() if k != "plan_sha256"}) != p["plan_sha256"]:
        raise ValueError("Two-stage plan hash/schema mismatch")
    if verify:
        if p["implementation"] != implementation():
            raise ValueError("Two-stage implementation changed after freezing; use a new output directory")
        if wan_source() != p["wan_source"]:
            raise ValueError("Pinned Wan source changed")
        for key in ("flash_source", "kernel_source", "rife_source"):
            if key in p and diag.checkout(p[key]["path"], p[key]["commit"])["commit"] != p[key]["commit"]:
                raise ValueError(f"{key} changed")
    return p


def gen_record_path(out, job_id):
    return Path(out) / "records" / "stage1" / (job_id + ".json")


def final_record_path(out, job_id):
    return Path(out) / "records" / "post" / (job_id + ".json")


def step_record_path(out, job_id, step):
    """Receipt of an intermediate step of a multi-step post chain (all but the last step)."""
    return Path(out) / "records" / "post_steps" / f"{job_id}.step{step}.json"


def step_input_receipt(out, job, step):
    return gen_record_path(out, job["generation_job"]) if step == 0 else step_record_path(out, job["id"], step - 1)


def step_receipt_valid(out, p, job, step):
    path = step_record_path(out, job["id"], step)
    if not path.exists():
        return False
    row = read(path)
    if row["plan_sha256"] != p["plan_sha256"] or row["job"] != job or row["step"] != step:
        raise ValueError(f"Post-step receipt protocol mismatch: {path}")
    if row["input_receipt_sha256"] != file_hash(step_input_receipt(out, job, step)):
        raise ValueError(f"Post-step receipt is bound to a different input: {path}")
    if not math.isfinite(row["seconds"]) or row["seconds"] <= 0 or file_hash(row["frames"]["path"]) != row["frames"]["sha256"]:
        raise ValueError(f"Invalid post-step runtime or changed frames: {path}")
    return True


def post_step_done(out, p, job, step):
    if step == len(post_steps(job["arm"])) - 1:
        return final_receipt_valid(out, p, job)
    return step_receipt_valid(out, p, job, step)


def gen_receipt_valid(out, p, job):
    path = gen_record_path(out, job["id"])
    if not path.exists():
        return False
    row = read(path)
    if row["plan_sha256"] != p["plan_sha256"] or row["job"] != job:
        raise ValueError(f"Stage-1 receipt protocol mismatch: {path}")
    if not math.isfinite(row["runtime"]["pipeline_seconds"]) or row["runtime"]["pipeline_seconds"] <= 0:
        raise ValueError(f"Invalid stage-1 runtime: {path}")
    if file_hash(row["video_path"]) != row["video_sha256"]:
        raise ValueError(f"Stage-1 video changed: {path}")
    if row["frames"] and file_hash(row["frames"]["path"]) != row["frames"]["sha256"]:
        raise ValueError(f"Stage-1 frames changed: {path}")
    return True


def final_receipt_valid(out, p, job):
    path = final_record_path(out, job["id"])
    if not path.exists():
        return False
    row = read(path)
    if row["plan_sha256"] != p["plan_sha256"] or row["job"] != job:
        raise ValueError(f"Post receipt protocol mismatch: {path}")
    if row["stage1_receipt_sha256"] != file_hash(gen_record_path(out, job["generation_job"])):
        raise ValueError(f"Post receipt is bound to a different stage-1 receipt: {path}")
    chain = range(len(post_steps(job["arm"])) - 1)
    if (not all(step_receipt_valid(out, p, job, k) for k in chain) or
            row.get("step_receipt_sha256", []) != [file_hash(step_record_path(out, job["id"], k)) for k in chain]):
        raise ValueError(f"Post receipt is bound to different intermediate steps: {path}")
    if not math.isfinite(row["runtime"]["post_seconds"]) or row["runtime"]["post_seconds"] <= 0:
        raise ValueError(f"Invalid post runtime: {path}")
    if file_hash(row["video_path"]) != row["video_sha256"]:
        raise ValueError(f"Final video changed: {path}")
    return True


def python_for(phase, unit, step, p, args):
    if phase == "post" and post_steps(next(a for a in p["config"]["arms"] if a["id"] == unit))[step] == "flashvsr":
        return args.flash_python
    return args.python


def units(p, phase, step=None):
    if phase == "stage1":
        return [g["id"] for g in p["config"]["generations"]]
    return [a["id"] for a in p["config"]["arms"] if len(post_steps(a)) > step]


def pending_units(args, p, phase, gpu, probe, step=None):
    if probe:  # one full-shape probe per unit, spread across GPUs
        return [u for i, u in enumerate(units(p, phase, step)) if i % p["ngpus"] == gpu]
    if phase == "stage1":
        jobs = [j for j in p["gen_jobs"] if j["gpu"] == gpu and not gen_receipt_valid(args.out, p, j)]
        return [u for u in units(p, phase) if any(j["generation"]["id"] == u for j in jobs)]
    jobs = [j for j in p["final_jobs"] if j["gpu"] == gpu and len(post_steps(j["arm"])) > step
            and not post_step_done(args.out, p, j, step)]
    return [u for u in units(p, phase, step) if any(j["arm"]["id"] == u for j in jobs)]


def launch(args, phase, probe=False):
    p = load_plan(args.out)
    if args.ngpus != p["ngpus"]:
        raise ValueError("NGPUS differs from frozen plan")
    if pilot.weight_inventory(Path(p["model_root"])) != p["weight_inventory"]:
        raise ValueError("Wan weights changed after the plan was frozen")
    if phase == "stage1":
        return launch_wave(args, p, phase, None, probe)
    if not probe:
        missing = [j["id"] for j in p["gen_jobs"] if not gen_receipt_valid(args.out, p, j)]
        if missing:
            raise RuntimeError(f"Stage 1 incomplete ({len(missing)} missing, e.g. {missing[:3]}); run stage1 first")
    # Step k of every chain runs after step k-1 of every chain (they may need different Pythons).
    for step in range(max(len(post_steps(a)) for a in p["config"]["arms"])):
        launch_wave(args, p, phase, step, probe)


def launch_wave(args, p, phase, step, probe):
    queues = {gpu: pending_units(args, p, phase, gpu, probe, step) for gpu in range(p["ngpus"])}
    logs = args.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes, failures, last = {}, [], time.monotonic()
    try:
        while any(queues.values()) or processes:
            for gpu, queue in queues.items():
                if gpu in processes or not queue:
                    continue
                unit = queue.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
                for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                    env.pop(key, None)
                command = [python_for(phase, unit, step, p, args), str(WORKER), phase, "--out", str(args.out),
                           "--gpu", str(gpu), "--unit", unit] + (["--probe"] if probe else [])
                command += [] if step is None else ["--step", str(step)]
                tag = unit if step is None else f"{unit}_step{step}"
                path = logs / f"{'probe_' if probe else ''}{phase}_{tag}_gpu{gpu}_{time.time_ns()}.log"
                handle = path.open("w", encoding="utf-8")
                processes[gpu] = (subprocess.Popen(command, env=env, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT),
                                  handle, unit, path)
                print(f"GPU {gpu}: {phase} {tag} -> {path}", flush=True)
            for gpu, (process, handle, unit, path) in list(processes.items()):
                if process.poll() is None:
                    continue
                handle.close()
                del processes[gpu]
                if process.returncode:
                    failures.append(str(path))
                    queues[gpu].clear()
                    print(f"FAILED GPU {gpu} / {unit}; inspect {path}\n{pilot.log_excerpt(path)}", flush=True)
            if time.monotonic() - last >= 30:
                status(args, quiet=True)
                last = time.monotonic()
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
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Missing {executable}")
    if not (args.out / "plan.json").exists():
        plan(args)
    launch(args, "stage1", probe=True)
    launch(args, "post", probe=True)
    print("Full-shape probes passed: every Wan generation canvas, FlashVSR (sparse + color fix), RIFE and the controls.")


def status(args, quiet=False):
    p = load_plan(args.out, verify=not quiet)
    gen_done = sum(gen_record_path(args.out, j["id"]).exists() for j in p["gen_jobs"])
    post_jobs = [j for j in p["final_jobs"] if j["arm"]["post"] != "none"]
    post_done = sum(final_record_path(args.out, j["id"]).exists() for j in post_jobs)
    middle = [(j, k) for j in post_jobs for k in range(len(post_steps(j["arm"])) - 1)]
    middle_done = sum(step_record_path(args.out, j["id"], k).exists() for j, k in middle)
    print(f"stage1 {gen_done}/{len(p['gen_jobs'])}; post {post_done}/{len(post_jobs)}"
          + (f" (intermediate steps {middle_done}/{len(middle)})" if middle else ""), flush=True)


def final_records(args, p):
    records = []
    reference = reference_index(p) if "reference" in p else {}
    for job in p["final_jobs"]:
        public_job = {k: v for k, v in job.items() if k not in ("gpu", "generation_job")}
        if job["generation_job"] is None:  # reused FULL50: same prompt text and seed, unchanged video
            source = reference.get((job["prompt"], job["seed"]))
            if source is None or file_hash(source["video_path"]) != source["video_sha256"]:
                raise RuntimeError(f"Reference FULL50 missing or changed for {job['id']}")
            seconds = source["runtime"]["pipeline_seconds"]
            records.append({"job": public_job, "video_path": source["video_path"], "video_sha256": source["video_sha256"],
                            "runtime": {"pipeline_seconds": seconds, "stage1_seconds": seconds, "post_seconds": 0.0},
                            "generation_job": None, "reference_job": source["job"]["id"]})
            continue
        gen = next(j for j in p["gen_jobs"] if j["id"] == job["generation_job"])
        if not gen_receipt_valid(args.out, p, gen):
            raise RuntimeError(f"Missing stage-1 receipt: {gen['id']}")
        stage1 = read(gen_record_path(args.out, gen["id"]))
        if job["arm"]["post"] == "none":
            runtime = {"pipeline_seconds": stage1["runtime"]["pipeline_seconds"],
                       "stage1_seconds": stage1["runtime"]["pipeline_seconds"], "post_seconds": 0.0}
            records.append({"job": public_job, "video_path": stage1["video_path"], "video_sha256": stage1["video_sha256"],
                            "runtime": runtime, "generation_job": gen["id"]})
            continue
        if not final_receipt_valid(args.out, p, job):
            raise RuntimeError(f"Missing post receipt: {job['id']}")
        row = read(final_record_path(args.out, job["id"]))
        records.append({"job": public_job, "video_path": row["video_path"], "video_sha256": row["video_sha256"],
                        "runtime": row["runtime"], "generation_job": gen["id"]})
    return records


def finalize(args):
    p = load_plan(args.out)
    records = final_records(args, p)
    groups = sorted({r["job"]["group_id"] for r in records})
    immutable(args.out / "dataset_manifest.json", {"schema": DATASET_SCHEMA, "plan_sha256": p["plan_sha256"],
              "partial_exploratory": False, "expected_groups": len(groups), "complete_groups": groups, "records": records})
    print(f"Finalized {len(records)} videos / {len(groups)} groups")


def finalized(args):
    p = load_plan(args.out)
    saved = read(args.out / "dataset_manifest.json")
    if saved["plan_sha256"] != p["plan_sha256"] or saved["records"] != final_records(args, p):
        raise ValueError("Finalized manifest changed or belongs to another plan")
    return p, saved


def vbench_identity(args):
    from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import inspect_vbench_checkout
    return inspect_vbench_checkout(args.vbench_root, expected_commit=args.expected_vbench_commit)


def score(args):
    from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import score_case_directory, warmup_vbench_cache
    p, dataset = finalized(args)
    cfg = p["config"]
    quality, diagnostic = cfg["evaluation"]["quality_dimensions"], cfg["evaluation"]["diagnostic_dimensions"]
    identity = vbench_identity(args)
    prompts = {x["id"]: x for x in cfg["prompts"]}
    inputs = []
    for record in dataset["records"]:
        job, prompt = record["job"], prompts[record["job"]["prompt_id"]]
        inputs.append({"observation_id": job["id"], "group_id": job["group_id"], "action_id": job["arm"]["id"],
                       "prompt": job["prompt"], "prompt_key": job["prompt_id"], "seed": job["seed"],
                       "motion": prompt["motion"], "detail": prompt["detail"], "family_id": prompt["family_id"],
                       "split": prompt["split"], "origin": prompt["origin"], "video_path": record["video_path"],
                       "video_sha256": record["video_sha256"], "pipeline_seconds": record["runtime"]["pipeline_seconds"],
                       "stage1_seconds": record["runtime"]["stage1_seconds"], "post_seconds": record["runtime"]["post_seconds"]})
    metrics = args.out / "metrics"
    immutable(metrics / "evaluation_inputs.json", {"plan_sha256": p["plan_sha256"], "rows": inputs})
    warmup_vbench_cache(args.vbench_python, args.vbench_root)
    output, provenance = [], {}
    for arm in cfg["arms"]:
        rows = [r for r in inputs if r["action_id"] == arm["id"]]
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
                                      metrics / "vbench" / arm["id"], quality + diagnostic, quality, diagnostic,
                                      args.ngpus, False, identity)
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
    immutable(metrics / "score_provenance.json", {"plan_sha256": p["plan_sha256"], "vbench": provenance,
              "quality_csv_sha256": file_hash(csv_path),
              "aggregate": "vbench5 is an exploratory raw five-dimension mean, NOT normalized official VBench Total"})
    report(args)


def report(args):
    p, dataset = finalized(args)
    cfg = p["config"]
    metrics = args.out / "metrics"
    with (metrics / "quality_by_video.csv").open(encoding="utf-8-sig", newline="") as handle:
        scored = list(csv.DictReader(handle))
    if {r["observation_id"] for r in scored} != {r["job"]["id"] for r in dataset["records"]}:
        raise ValueError("Score CSV does not cover the finalized inventory")
    dims = ["vbench5"] + cfg["evaluation"]["quality_dimensions"] + cfg["evaluation"]["diagnostic_dimensions"]
    reference = {r["group_id"]: r for r in scored if r["action_id"] == "FULL50"}
    paired = []
    for row in scored:
        if row["action_id"] == "FULL50":
            continue
        ref = reference[row["group_id"]]
        paired.append({"group_id": row["group_id"], "prompt_key": row["prompt_key"], "action_id": row["action_id"],
                       "seed": row["seed"], "cell": row["motion"] + "/" + row["detail"],
                       "speedup": float(ref["pipeline_seconds"]) / float(row["pipeline_seconds"]),
                       "stage1_speedup": float(ref["pipeline_seconds"]) / float(row["stage1_seconds"]),
                       **{"delta_" + d: float(row[d]) - float(ref[d]) for d in dims}})
    csv_write(metrics / "paired_deltas.csv", paired)
    buckets = defaultdict(list)
    for row in paired:
        buckets[row["action_id"], "all"].append(row)
        buckets[row["action_id"], row["cell"]].append(row)
    summary = []
    for (arm, cell), rows in sorted(buckets.items()):
        clusters = defaultdict(list)
        for row in rows:
            clusters[row["prompt_key"]].append(row["delta_vbench5"])
        means = [statistics.mean(v) for v in clusters.values()]
        rng = random.Random(20261019)
        boot = sorted(statistics.mean(rng.choices(means, k=len(means))) for _ in range(2000))
        summary.append({"arm": arm, "nominal_cell": cell, "groups": len(rows), "prompts": len(means),
                        "mean_speedup": statistics.mean(r["speedup"] for r in rows),
                        "mean_stage1_speedup": statistics.mean(r["stage1_speedup"] for r in rows),
                        "mean_delta_vbench5": statistics.mean(means), "prompt_bootstrap_ci": [boot[49], boot[1949]],
                        "mean_delta_dynamic_degree": statistics.mean(r["delta_dynamic_degree"] for r in rows)})
    write(metrics / "report.json", {"claim": cfg["claim"], "summary": summary, "caveats": [
        "Not official VBench Total. Speedups are same-machine FULL50 / (stage1 + post) per group.",
        "Stage-to-stage frames pass through a lossless uint8 file because the two stages use different Python "
        "environments; file I/O and MP4 encoding are excluded from timing.",
        "F41 arms play 41 generated frames as 81 at 16 fps, i.e. 2x slow motion of what Wan drew; flow-based "
        "freeze/motion metrics must be normalized for this before comparison with FULL50.",
        "LR81 is generated at 416x240, outside Wan2.1-1.3B's 480p training canvas; LR_BICUBIC separates "
        "low-resolution generation failures from SR failures."] + ([
        "rCM arms load the official rCM 1.3B DiT into the pinned official Wan2.1 model and replicate the official "
        "rcm inference loop (TrigFlow timesteps, no CFG, no negative prompt); 81 frames as all other arms."]
        if "rcm_weights" in p else []) + ([
        f"FULL50 is reused, unregenerated, from {p['reference']['root']} (same prompts, seeds and machine)."]
        if "reference" in p else [])})
    print(f"Saved paired deltas and summary: {metrics}")


def vbench_smoke(args):
    """Score ONE existing video on one GPU for every dimension, before any long run."""
    from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import score_case_directory, warmup_vbench_cache
    cfg = validate_config(read(args.config))
    quality, diagnostic = cfg["evaluation"]["quality_dimensions"], cfg["evaluation"]["diagnostic_dimensions"]
    work = args.smoke_out / f"run_{time.time_ns()}"
    (work / "videos").mkdir(parents=True)
    target = work / "videos" / "smoke.mp4"
    shutil.copy2(args.smoke_video, target)
    prompt_map = work / "prompt_map.json"
    write(prompt_map, {str(target.resolve()): args.smoke_prompt})
    warmup_vbench_cache(args.vbench_python, args.vbench_root)
    bundle = score_case_directory(args.vbench_root, args.vbench_python, work / "videos", prompt_map, work / "vbench",
                                  quality + diagnostic, quality, diagnostic, 1, False, vbench_identity(args))
    print(json.dumps(bundle.scores, indent=2))
    print(f"VBench smoke passed for all {len(quality + diagnostic)} dimensions: {work}")


def export(args):
    export_root = args.out / "exports"
    export_root.mkdir(exist_ok=True)
    target = export_root / "twostage_wan21_analysis.tgz"
    with tarfile.open(target, "w:gz") as archive:
        for name in ("plan.json", "dataset_manifest.json"):
            if (args.out / name).exists():
                archive.add(args.out / name, arcname=name)
        for directory in ("records", "metrics", "logs"):
            for path in sorted((args.out / directory).rglob("*")):
                if path.is_file() and path.suffix in (".csv", ".json", ".log"):
                    archive.add(path, arcname=str(path.relative_to(args.out)).replace("\\", "/"))
    print(f"Analysis bundle (no videos): {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "check", "stage1", "post", "status", "finalize", "score", "report",
                                         "export", "vbench-smoke"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/published_wan21_twostage_v1")
    parser.add_argument("--model-root", type=Path, default=Path("/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B"))
    parser.add_argument("--python", default=sys.executable, help="Wan generation Python (stage 1, RIFE, controls)")
    parser.add_argument("--flash-python", default=sys.executable, help="isolated FlashVSR Python")
    parser.add_argument("--flash-root", type=Path, default=ROOT / "UNIV_adaptor/external/flashvsr")
    parser.add_argument("--kernel-root", type=Path, default=ROOT / "UNIV_adaptor/external/flashvsr_block_sparse")
    parser.add_argument("--flash-weights", type=Path, default=ROOT / "checkpoints/flashvsr_v11")
    parser.add_argument("--rife-root", type=Path, default=ROOT / "UNIV_adaptor/external/practical_rife")
    parser.add_argument("--rife-weights", type=Path, default=ROOT / "checkpoints/rife_v426")
    parser.add_argument("--rcm-weights", type=Path, default=ROOT / "checkpoints/rcm/rCM_Wan2.1_T2V_1.3B_480p.pt")
    parser.add_argument("--reference-root", type=Path, help="finalized run whose FULL50 an external reference arm reuses")
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--vbench-root", type=Path, default=Path("/mnt/afs_2/houze/VBench"))
    parser.add_argument("--vbench-python", default="/opt/conda/bin/python")
    parser.add_argument("--expected-vbench-commit", default="fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490")
    parser.add_argument("--smoke-video", type=Path)
    parser.add_argument("--smoke-prompt", default="A plain blue ceramic mug slowly rotates on a turntable.")
    parser.add_argument("--smoke-out", type=Path, default=ROOT / "outputs/vbench_smoke")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.mode in ("stage1", "post"):
        launch(args, args.mode)
    else:
        globals()[args.mode.replace("-", "_")](args)


if __name__ == "__main__":
    main()
