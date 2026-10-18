"""Official DVG on Wan2.1-T2V-1.3B: matched FULL/DVG generation and VBench scoring.

The released DVG checkout (github.com/DarrenZheng303/DVG) is imported, not
modified: FULL and DVG both call its bundled `WanT2V.generate`, differing only
in `enable_dvg`/`dvg_budget`, so the baseline shares the exact codebase. DVG
defaults are kept (832x480, 81 frames, 50 UniPC steps, shift 8, CFG 6, Wan's
default negative prompt). Its first stage starts from a smaller noise tensor,
so FULL/DVG share the seed but not the initial noise field; this is inherent to
the method and recorded.

Two job sets:
- main: controlled-factor development prompts (locked test never planned) x
  seeds x {FULL_NATIVE_50, DVG_D070, DVG_D050}; feeds VBench + requirement scoring.
- repro: a deterministic subset of VBench's own prompts, seed 42, FULL vs DVG_D050,
  to check that our run reproduces DVG's "near-lossless" VBench behaviour.
- repro_ext: the same VBench prompt indices, rewritten by VBench's line-aligned
  Wan2.1 prompt extension, to separate prompt under-specification from content.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics as st
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic  # noqa: E402

PLAN_SCHEMA = "dvg_wan21_study_plan_v1"
FULL = "FULL_NATIVE_50"
ARMS = {FULL: None, "DVG_D070": 0.7, "DVG_D050": 0.5}
REPRO_ARMS = (FULL, "DVG_D050")
SETS = ("main", "repro", "repro_ext")
SET_CSV = {"main": "quality_by_video.csv", "repro": "repro_by_video.csv", "repro_ext": "repro_ext_by_video.csv"}
SEEDS = (42, 100, 2024)
SETTINGS = {"size": "832*480", "frame_num": 81, "sampling_steps": 50, "shift": 8.0,
            "guide_scale": 6.0, "sample_solver": "unipc", "negative_prompt": "wan_default"}
QUALITY = ("subject_consistency", "background_consistency", "motion_smoothness", "aesthetic_quality", "imaging_quality")
DIAGNOSTIC = ("dynamic_degree", "overall_consistency")
ATTENTION = {"backend": "flash_attn_2_varlen", "fa3_detected": None}


# ---------------------------------------------------------------- plan

def read_lines(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def plan(args):
    sets = args.sets.split(",")
    if not set(sets) <= set(SETS):
        raise ValueError(f"unknown sets: {sets}")
    jobs = []
    with Path(args.prompts).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    dev = [r for r in rows if r["split"] != "test"]
    for r in dev if "main" in sets else ():
        for seed in SEEDS:
            for arm in ARMS:
                jobs.append({
                    "stem": f"main_p{r['prompt_id']:03d}_s{seed}__{arm}", "set": "main",
                    "group_id": f"main_p{r['prompt_id']:03d}_s{seed}", "prompt_id": r["prompt_id"],
                    "family_id": r["family_id"], "split": r["split"], "motion_level": r["motion_level"],
                    "detail_level": r["detail_level"], "base_seed": seed, "case_id": arm,
                    "budget": ARMS[arm], "prompt": r["prompt"]})
    vbench_prompts = read_lines(args.vbench_prompts)
    ext_prompts = None
    if "repro_ext" in sets:
        if not args.vbench_ext_prompts:
            raise ValueError("repro_ext needs --vbench-ext-prompts")
        ext_prompts = read_lines(args.vbench_ext_prompts)
        if len(ext_prompts) != len(vbench_prompts):  # extension must be line-aligned with the short file
            raise ValueError(f"extended prompts have {len(ext_prompts)} lines, short file has {len(vbench_prompts)}")
    picks = np.linspace(0, len(vbench_prompts) - 1, args.repro_count).round().astype(int)
    for subset, tag, source in (("repro", "repro", vbench_prompts), ("repro_ext", "reproext", ext_prompts)):
        if subset not in sets:
            continue
        for i in sorted(set(picks.tolist())):
            for arm in REPRO_ARMS:
                jobs.append({
                    "stem": f"{tag}_v{i:04d}_s42__{arm}", "set": subset, "group_id": f"{tag}_v{i:04d}_s42",
                    "prompt_id": int(i), "base_seed": 42, "case_id": arm, "budget": ARMS[arm],
                    "prompt": source[i], "short_prompt": vbench_prompts[i]})
    body = {
        "schema": PLAN_SCHEMA, "settings": SETTINGS, "arms": ARMS,
        "prompts_path": str(Path(args.prompts).resolve()), "prompts_sha256": sha256_file(args.prompts),
        "vbench_prompts_path": str(Path(args.vbench_prompts).resolve()),
        "vbench_prompts_sha256": sha256_file(args.vbench_prompts),
        "vbench_ext_prompts_path": str(Path(args.vbench_ext_prompts).resolve()) if ext_prompts else None,
        "vbench_ext_prompts_sha256": sha256_file(args.vbench_ext_prompts) if ext_prompts else None,
        "sets": sets, "excluded_test_prompts": len(rows) - len(dev), "jobs": jobs,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "plan.json", body)
    counts = {s: sum(j["set"] == s for j in jobs) for s in sets}
    print(f"planned {len(jobs)} videos {counts}; excluded {body['excluded_test_prompts']} locked test prompts")


def load_plan(out):
    value = json.loads((Path(out) / "plan.json").read_text(encoding="utf-8"))
    if value.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported plan schema")
    return value


# ---------------------------------------------------------------- generate

def load_dvg_wan(dvg_root, ckpt_dir, noise_cache_dir):
    root = Path(dvg_root).resolve()
    for path in (root, root / "Wan2.1"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import wan
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
    from wan.modules import attention
    from wan.utils.utils import cache_video
    import dvg

    # Some flash_attn_interface (FA3) builds return a Tensor where Wan's FA3 branch
    # expects a tuple and indexes [0]; lock FULL and DVG to the same FA2 path.
    if not attention.FLASH_ATTN_2_AVAILABLE:
        raise RuntimeError("flash_attn (FA2) is required for the locked dense-attention path")
    ATTENTION["fa3_detected"] = bool(attention.FLASH_ATTN_3_AVAILABLE)
    attention.FLASH_ATTN_3_AVAILABLE = False
    dvg.DVG.noise_cache_dir = str(noise_cache_dir)  # per-worker cache avoids concurrent partial reads
    model = wan.WanT2V(config=WAN_CONFIGS["t2v-1.3B"], checkpoint_dir=ckpt_dir, device_id=0, rank=0)
    return model, SIZE_CONFIGS[SETTINGS["size"]], WAN_CONFIGS["t2v-1.3B"].sample_fps, cache_video


def run_one(model, size, prompt, seed, budget, steps=SETTINGS["sampling_steps"]):
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    video = model.generate(prompt, size=size, frame_num=SETTINGS["frame_num"], shift=SETTINGS["shift"],
                           sample_solver=SETTINGS["sample_solver"], sampling_steps=steps,
                           guide_scale=SETTINGS["guide_scale"], seed=seed, offload_model=False,
                           enable_dvg=budget is not None, dvg_budget=budget if budget is not None else 0.5)
    torch.cuda.synchronize()
    return video, time.perf_counter() - start


def dvg_state(model):
    d = model.dvg
    if not d.enabled:
        return None
    return {"spatial_rates": [float(x) for x in d.spatial_rates], "temporal_rates": [float(x) for x in d.temporal_rates],
            "stage_shapes": [list(map(int, s)) for s in d.stage_shapes], "density": float(d.density),
            "stage_step_counts": [int(x) for x in d.stage_step_counts]}


def generate(args):
    value = load_plan(args.out)
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "receipts").mkdir(parents=True, exist_ok=True)
    jobs = [j for i, j in enumerate(value["jobs"]) if i % args.shard_count == args.shard_index]
    todo = [j for j in jobs if not (out / "receipts" / f"{j['stem']}.json").exists()]
    for j in todo:
        if (out / "videos" / f"{j['stem']}.mp4").exists():
            raise RuntimeError(f"unreceipted video exists; inspect and move it aside: {j['stem']}.mp4")
    if not todo:
        print(f"[gen {args.shard_index}] nothing to do")
        return
    model, size, fps, cache_video = load_dvg_wan(args.dvg_root, args.ckpt_dir, out / f"noise_cache_{args.shard_index}")
    for budget in (None, 0.5):  # excluded warmup of both code paths
        run_one(model, size, "warmup", 0, budget, steps=20)
    for n, j in enumerate(todo):
        video, seconds = run_one(model, size, j["prompt"], j["base_seed"], j["budget"])
        path = out / "videos" / f"{j['stem']}.mp4"
        cache_video(tensor=video[None], save_file=str(path), fps=fps, nrow=1, normalize=True, value_range=(-1, 1))
        write_json_atomic(out / "receipts" / f"{j['stem']}.json", {
            **j, "generate_seconds": seconds, "dvg": dvg_state(model), "video_path": str(path.resolve()),
            "video_sha256": sha256_file(path), "settings": value["settings"], "attention": dict(ATTENTION)})
        print(f"[gen {args.shard_index}] {n + 1}/{len(todo)} {j['stem']} {seconds:.1f}s", flush=True)


# ---------------------------------------------------------------- score

def receipts(out, value):
    rows = []
    for j in value["jobs"]:
        path = Path(out) / "receipts" / f"{j['stem']}.json"
        if not path.is_file():
            raise ValueError(f"missing receipt: {j['stem']}")
        r = json.loads(path.read_text(encoding="utf-8"))
        if sha256_file(r["video_path"]) != r["video_sha256"]:
            raise ValueError(f"video hash mismatch: {j['stem']}")
        rows.append(r)
    return rows


def score(args):
    from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import (
        inspect_vbench_checkout, score_case_directory)

    out = Path(args.out).resolve()  # VBench runs with its own cwd
    value = load_plan(out)
    rows = receipts(out, value)
    write_json_atomic(out / "prompt_map.json", {r["video_path"]: r["prompt"] for r in rows})
    vbench_root = Path(args.vbench_root).resolve()
    identity = inspect_vbench_checkout(vbench_root, expected_commit=None)
    scores = {r["stem"]: {} for r in rows}
    for dim in (*QUALITY, *DIAGNOSTIC):
        print(f"[VBench] {dim}: {len(rows)} videos", flush=True)
        bundle = score_case_directory(vbench_root, args.vbench_python, out / "videos", out / "prompt_map.json",
                                      out / "vbench" / dim, [dim], [dim] if dim in QUALITY else [],
                                      [dim] if dim in DIAGNOSTIC else [], args.ngpus, False, identity)
        for stem in scores:
            scores[stem][dim] = float(bundle.scores[stem][dim])
    write_json_atomic(out / "vbench_scores.json", scores)
    table = []
    for r in rows:
        s = scores[r["stem"]]
        item = {"observation_id": r["stem"], **{k: r.get(k, "") for k in (
            "set", "group_id", "prompt_id", "family_id", "split", "motion_level", "detail_level", "base_seed", "case_id")},
            "pipeline_seconds": r["generate_seconds"], "vbench5": st.fmean(s[d] for d in QUALITY), **s}
        dvg = r["dvg"] or {}
        item["dvg_density"] = dvg.get("density", "")
        item["dvg_stage1_spatial"] = (dvg.get("spatial_rates") or [""])[0]
        item["dvg_stage1_temporal"] = (dvg.get("temporal_rates") or [""])[0]
        table.append(item)
    for subset, name in SET_CSV.items():
        part = [t for t in table if t["set"] == subset]
        if not part:
            continue
        with (out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(part[0]))
            writer.writeheader()
            writer.writerows(part)
    for subset in ("repro", "repro_ext"):
        if (out / SET_CSV[subset]).is_file():
            repro_summary(out, subset)


def repro_summary(out, subset="repro"):
    with (Path(out) / SET_CSV[subset]).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by = {}
    for r in rows:
        by.setdefault(r["group_id"], {})[r["case_id"]] = r
    rng = np.random.default_rng(20261017)
    lines = ["| metric | FULL | DVG_D050 | paired delta | 95% CI |", "|---|---:|---:|---:|---|"]
    for metric in ("vbench5", *QUALITY, *DIAGNOSTIC, "pipeline_seconds"):
        a = np.array([float(g[FULL][metric]) for g in by.values()])
        b = np.array([float(g["DVG_D050"][metric]) for g in by.values()])
        d = b - a
        boot = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        lines.append(f"| {metric} | {a.mean():.4f} | {b.mean():.4f} | {d.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    speed = np.array([float(g[FULL]["pipeline_seconds"]) / float(g["DVG_D050"]["pipeline_seconds"]) for g in by.values()])
    lines.append(f"\nmean paired speedup FULL/DVG_D050: {speed.mean():.2f}x over {len(by)} VBench prompts (seed 42)")
    text = "\n".join(lines)
    name = "repro_summary.md" if subset == "repro" else f"{subset}_summary.md"
    (Path(out) / name).write_text(text + "\n", encoding="utf-8")
    print(text)


def smoke(args):
    """Two full-length 50-step clips (FULL, DVG 0.5) of one prompt; prints time, shape and DVG action."""
    out = Path(args.out) / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    model, size, fps, cache_video = load_dvg_wan(args.dvg_root, args.ckpt_dir, out / "noise_cache")
    prompt = "A single large red rubber ball bounces rapidly across a plain gray studio floor."
    for budget in (None, 0.5):
        video, seconds = run_one(model, size, prompt, 42, budget)
        tag = "full" if budget is None else "dvg050"
        cache_video(tensor=video[None], save_file=str(out / f"{tag}.mp4"), fps=fps, nrow=1, normalize=True, value_range=(-1, 1))
        print(f"{tag}: {seconds:.1f}s shape={tuple(video.shape)} dvg={dvg_state(model)}", flush=True)


def status(args):
    value = load_plan(args.out)
    done = {p.stem for p in (Path(args.out) / "receipts").glob("*.json")}
    for subset in value.get("sets", ("main", "repro")):
        jobs = [j for j in value["jobs"] if j["set"] == subset]
        print(f"{subset}: {sum(j['stem'] in done for j in jobs)}/{len(jobs)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--vbench-prompts", required=True)
    p.add_argument("--repro-count", type=int, default=48)
    p.add_argument("--vbench-ext-prompts")
    p.add_argument("--sets", default="main,repro", help="comma list of " + ",".join(SETS))
    for name in ("generate", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--dvg-root", required=True)
        p.add_argument("--ckpt-dir", required=True)
        p.add_argument("--shard-index", type=int, default=0)
        p.add_argument("--shard-count", type=int, default=1)
    p = sub.add_parser("score")
    p.add_argument("--out", required=True)
    p.add_argument("--vbench-root", required=True)
    p.add_argument("--vbench-python", required=True)
    p.add_argument("--ngpus", type=int, default=4)
    sub.add_parser("status").add_argument("--out", required=True)
    p = sub.add_parser("repro-summary")
    p.add_argument("--out", required=True)
    p.add_argument("--set", default="repro", choices=("repro", "repro_ext"))
    args = parser.parse_args()
    {"plan": plan, "generate": generate, "smoke": smoke, "score": score, "status": status,
     "repro-summary": lambda a: repro_summary(a.out, a.set)}[args.mode](args)


if __name__ == "__main__":
    main()
