"""Download, plan, validate, generate and finalize the HY15 endpoint pilot."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import canonical_sha256, sha256_file, write_json_atomic
from UNIV_adaptor.hy15_protocol import (
    DEFAULT_PROMPTS, DEFAULT_PROTOCOL, immutable_json, make_plan, read,
    record_paths, verify_plan, verify_record,
)

SR_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"


def asset_inventory(root):
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and ".cache" not in p.relative_to(root).parts
            and (p.suffix in {".json", ".safetensors", ".txt", ".model", ".jinja"})}


def download(args):
    from huggingface_hub import snapshot_download
    from UNIV_adaptor.scripts.data.score_phase2_dataset import output_lock

    p = read(args.protocol)
    args.assets.mkdir(parents=True, exist_ok=True)
    with output_lock(args.assets):
        lock_path = args.assets / "assets.lock.json"
        if lock_path.exists():
            validate_assets(args.assets, p)
            print("Existing pinned assets verified; no download required")
            return
        model = args.assets / "model"
        snapshot_download(repo_id=p["model_id"], revision=p["model_revision"], local_dir=str(model))
        sr = args.assets / "RealESRGAN_x2plus.pth"
        if not sr.exists():
            temp = sr.with_suffix(".download")
            urllib.request.urlretrieve(SR_URL, temp)
            temp.replace(sr)
        if sr.stat().st_size < 1000000:
            raise ValueError("Invalid Real-ESRGAN checkpoint download")
        immutable_json(lock_path, {"model_id": p["model_id"], "revision": p["model_revision"],
                                   "files": asset_inventory(model), "sr_sha256": sha256_file(sr),
                                   "sr_url": SR_URL})
    print(f"Pinned assets and SHA256 inventory: {lock_path}")


def validate_assets(assets, protocol):
    lock = read(assets / "assets.lock.json")
    if lock["model_id"] != protocol["model_id"] or lock["revision"] != protocol["model_revision"]:
        raise ValueError("Model identity differs from protocol")
    if asset_inventory(assets / "model") != lock["files"]:
        raise ValueError("Downloaded model files changed or are incomplete")
    if sha256_file(assets / "RealESRGAN_x2plus.pth") != lock["sr_sha256"]:
        raise ValueError("SR checkpoint changed")
    return lock


def environment(assets):
    import torch
    names = ["diffusers", "transformers", "torch", "torchvision", "accelerate", "realesrgan", "basicsr"]
    sources = ["UNIV_adaptor/hy15_runtime.py", "UNIV_adaptor/hy15_protocol.py",
               "UNIV_adaptor/scripts/data/run_hy15_endpoint_prior.py", "UNIV_adaptor/transition.py",
               "UNIV_adaptor/rgb_super_resolution.py", "changing_resolution_distill/rgb_super_resolution.py",
               "changing_resolution_distill/realesrgan_compat.py"]
    return {"assets_lock_sha256": sha256_file(assets / "assets.lock.json"),
            "versions": {n: importlib.metadata.version(n) for n in names},
            "python": platform.python_version(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "sources": {n: sha256_file(ROOT / n) for n in sources}}


def preflight(args):
    p = read(args.protocol)
    validate_assets(args.assets, p)
    import torch
    from diffusers import HunyuanVideo15Pipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for generation")
    if importlib.metadata.version("diffusers") != p["diffusers_version"]:
        raise RuntimeError("Use setup mode to install the pinned Diffusers version")
    from UNIV_adaptor.rgb_super_resolution import build_univ_rgb_super_resolver
    build_univ_rgb_super_resolver({"wan_rgb_sr_checkpoint": str(args.assets / "RealESRGAN_x2plus.pth")})
    assert HunyuanVideo15Pipeline is not None
    print(json.dumps(environment(args.assets), indent=2))
    print("Preflight passed. Model forward and VAE restoration still require smoke mode.")


def prepare(args):
    plan = make_plan(args.protocol, args.prompts)
    args.out.mkdir(parents=True, exist_ok=True)
    immutable_json(args.out / "plan.json", plan)
    print(f"Development plan: {len(plan['prompts'])} prompts; {len(plan['jobs'])} videos")
    for case in plan["protocol"]["cases"]:
        from UNIV_adaptor.hy15_protocol import density
        print(case["id"], density(case, plan["protocol"]))
    return plan


def worker(args):
    from UNIV_adaptor.hy15_runtime import HY15EndpointRunner
    from UNIV_adaptor.scripts.data.score_phase2_dataset import output_lock
    import imageio.v2 as imageio
    import torch

    plan = read(args.out / "plan.json")
    verify_plan(plan)
    env = environment(args.assets)
    # A single launcher preflight hashes all weights; each worker binds the same lock/code.
    if env != read(args.out / "environment.json"):
        raise ValueError("Worker environment differs from frozen preflight")
    env_hash = canonical_sha256(env)
    lane = args.out / "workers" / str(args.rank)
    lane.mkdir(parents=True, exist_ok=True)
    with output_lock(lane):
        jobs = [j for i, j in enumerate(plan["jobs"]) if (i // len(plan["protocol"]["cases"])) % args.world == args.rank]
        if args.limit:
            jobs = jobs[:args.limit]
        runner = None
        for j in jobs:
            video, record = record_paths(args.out, j)
            if record.exists():
                verify_record(args.out, j, plan, env_hash)
                continue
            if video.exists():
                raise RuntimeError(f"Orphan video without completion record: {video}; inspect/move it before resuming")
            if runner is None:
                runner = HY15EndpointRunner(args.assets / "model", args.assets / "RealESRGAN_x2plus.pth", plan["protocol"])
            torch.cuda.reset_peak_memory_stats()
            print(f"GPU lane {args.rank}: {j['id']}", flush=True)
            frames, runtime = runner.generate(j)
            if tuple(frames.shape) != (plan["protocol"]["frames"], plan["protocol"]["height"], plan["protocol"]["width"], 3):
                raise ValueError("Output video dimensions changed")
            if not bool(torch.isfinite(frames).all()):
                raise ValueError("Non-finite decoded video")
            video.parent.mkdir(parents=True, exist_ok=True)
            temp = video.with_name(video.stem + ".partial.mp4")
            started = time.perf_counter()
            with imageio.get_writer(temp, fps=plan["protocol"]["fps"], codec="libx264", quality=8, macro_block_size=1) as writer:
                for frame in frames:
                    writer.append_data(frame.clamp(0, 1).mul(255).round().byte().numpy())
            runtime["timing_seconds"]["video_encode_io"] = time.perf_counter() - started
            runtime["timing_seconds"]["candidate_total"] += runtime["timing_seconds"]["video_encode_io"]
            temp.replace(video)
            write_json_atomic(record, {"schema": "hy15_endpoint_record_v1", "job": j,
                "plan_sha256": plan["plan_sha256"], "environment_sha256": env_hash,
                "video_sha256": sha256_file(video), "video_bytes": video.stat().st_size, **runtime})


def finalize(args):
    plan = read(args.out / "plan.json")
    verify_plan(plan)
    env = canonical_sha256(read(args.out / "environment.json"))
    records = []
    for j in plan["jobs"]:
        verify_record(args.out, j, plan, env)
        _, path = record_paths(args.out, j)
        records.append({"id": j["id"], "record_sha256": sha256_file(path)})
    immutable_json(args.out / "dataset.json", {"plan_sha256": plan["plan_sha256"],
                                               "environment_sha256": env, "records": records})
    print(f"Finalized and hash-verified {len(records)} videos")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["download", "plan", "check", "freeze", "worker", "finalize"])
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.rank < args.world or args.limit < 0:
        parser.error("Invalid worker rank/world/limit")
    if args.mode == "download":
        download(args)
    elif args.mode == "plan":
        prepare(args)
    elif args.mode == "check":
        preflight(args)
    elif args.mode == "freeze":
        prepare(args)
        preflight(args)
        immutable_json(args.out / "environment.json", environment(args.assets))
    elif args.mode == "worker":
        worker(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
