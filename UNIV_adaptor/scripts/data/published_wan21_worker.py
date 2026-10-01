"""Run an unmodified pinned entrypoint, with receipts and process-local model reuse.

No acceleration forward is reimplemented here. One process handles one arm on
one GPU. The entrypoint initializes each prompt's method state as upstream does;
ScalingCache additionally needs cache_init after its previous cache_release.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import types

# Published snapshots may not ignore __pycache__. Do not write runtime bytecode
# into a pinned checkout; this is also propagated to all worker subprocesses.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from UNIV_adaptor.scripts.data.published_wan21_pilot import (  # noqa: E402
    check_sources, digest, file_hash, immutable, load_plan, read, receipt_valid, write,
)


SCALING_COMPATIBILITY = [
    "Expose upstream CustomWanT2V/CustomWanI2V without the missing, unused textimage2video import in adapter/wan/__init__.py",
    "Guard unused xfuser distributed imports in this strictly single-GPU worker; any invocation raises, no parallel computation is emulated",
]


def install_scaling_import_guards(external):
    """Repair import-only upstream packaging defects; never replace a forward."""
    def forbidden_parallel(*pos, **kw):
        raise RuntimeError("Parallel xDiT code was invoked in a single-GPU pilot; refusing to emulate it")
    for name in ("xfuser", "xfuser.core", "xfuser.core.distributed"):
        shim = types.ModuleType(name)
        shim.__path__ = []
        shim.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = shim
    dist = sys.modules["xfuser.core.distributed"]
    for name in ("get_sequence_parallel_rank", "get_sequence_parallel_world_size", "get_sp_group"):
        setattr(dist, name, forbidden_parallel)
    # Load the two real upstream implementations by bypassing ONLY their broken
    # package __init__, which references a file absent in the pinned git tree.
    name = "scaling_cache.adapter.wan"
    facade = types.ModuleType(name)
    facade.__path__ = [str(external / "scalingcache/scaling_cache/adapter/wan")]
    facade.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = facade
    t2v = __import__(name + ".text2video", fromlist=["CustomWanT2V"])
    i2v = __import__(name + ".image2video", fromlist=["CustomWanI2V"])
    facade.CustomWanT2V, facade.CustomWanI2V = t2v.CustomWanT2V, i2v.CustomWanI2V


def load_entrypoint(source):
    external = ROOT / "UNIV_adaptor/external"
    paths = {
        "wan21": external / "wan21/generate.py",
        "teacache": external / "teacache/TeaCache4Wan2.1/teacache_generate.py",
        "scalingcache": external / "scalingcache/Wan2.1/scalingcache_generate.py",
        "jenga": external / "jenga/jenga_wan.py",
    }
    wan_root = external / ("jenga" if source == "jenga" else "wan21")
    sys.path[:0] = [str(wan_root), str(external / "scalingcache")]
    # Upstream ScalingCache loads ../assets/alpha_dict relative to this directory.
    os.chdir(external / "scalingcache/Wan2.1" if source == "scalingcache" else wan_root)
    if source == "scalingcache":
        install_scaling_import_guards(external)
    spec = importlib.util.spec_from_file_location("published_entry", paths[source])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # ScalingCache imports CustomWanT2V but not the top-level wan module.
    import wan
    if Path(wan.__file__).resolve().parent != (wan_root / "wan").resolve():
        raise RuntimeError("Wrong Wan import; refusing a mixed implementation")
    return module


def argv_for(plan, job, video):
    arm, cfg = job["arm"], plan["config"]["sampling"]
    argv = ["published_entry", "--ckpt_dir", plan["model_root"],
            "--prompt", job["prompt"], "--base_seed", str(job["seed"]),
            "--sample_steps", str(arm["steps"]), "--save_file", str(video)]
    for key, value in cfg.items():
        argv += ["--" + key, str(value).lower() if isinstance(value, bool) else str(value)]
    return argv + arm["flags"]


def versions():
    result = {"python": sys.version}
    for name in ("torch", "torchvision", "numpy", "diffusers", "transformers", "flash_attn", "triton", "imageio", "imageio-ffmpeg", "safetensors"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def configure_attention_backend(attention=None):
    """Select upstream FA2 for ALL dense attention, including Jenga's dense path.

    Some installed flash_attn_interface versions return Tensor, while these
    pinned Wan wrappers assume a tuple and index [0] in their FA3 branch. Keep
    the upstream FA2 implementation, rather than rewriting outputs or silently
    falling back to SDPA. Jenga's published sparse Triton branch is untouched.
    """
    if attention is None:
        attention = __import__("wan.modules.attention", fromlist=["flash_attention"])
    fa2 = getattr(attention, "flash_attn", None)
    if not getattr(attention, "FLASH_ATTN_2_AVAILABLE", False) or not callable(getattr(fa2, "flash_attn_varlen_func", None)):
        raise RuntimeError("Common dense backend requires working FlashAttention-2 (flash_attn.flash_attn_varlen_func). No SDPA/FA3 fallback is allowed; check the CUDA/PyTorch-compatible flash-attn installation.")
    observed_fa3 = bool(getattr(attention, "FLASH_ATTN_3_AVAILABLE", False))
    attention.FLASH_ATTN_3_AVAILABLE = False
    metadata = {"dense_backend": "flash_attention_2", "fa3_detected_before_lock": observed_fa3,
                "flash_attn_module": getattr(fa2, "__file__", None),
                "flash_attn_interface_module": getattr(getattr(attention, "flash_attn_interface", None), "__file__", None),
                "policy": "use pinned upstream FA2 branch for all dense calls; published sparse attention remains unchanged"}
    return attention, metadata


def attention_smoke_test(attention, torch_module=None):
    """Exercise real dense CUDA kernels and output layout before loading weights."""
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    with torch.no_grad():
        q = torch.randn(1, 17, 12, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 23, 12, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 23, 12, 128, device="cuda", dtype=torch.bfloat16)
        # Test the same varlen wrapper path used by Wan's real model, including
        # explicit key lengths. Small shapes suffice to catch the API mismatch.
        lengths = torch.tensor([23], device="cuda", dtype=torch.int32)
        output = attention.flash_attention(q=q, k=k, v=v, k_lens=lengths)
        torch.cuda.synchronize()
        if tuple(output.shape) != (1, 17, 12, 128) or output.dtype != q.dtype:
            raise RuntimeError(f"Dense attention smoke test failed: expected (1,17,12,128)/{q.dtype}; got {tuple(output.shape)}/{output.dtype}")
        if not torch.isfinite(output).all().item():
            raise RuntimeError("Dense attention smoke test produced non-finite values")
    return {"passed": True, "shape": list(output.shape), "dtype": str(output.dtype), "scope": "dense FA2 CUDA kernel only; no model weights or sparse Triton validation"}


def run(args):
    plan = load_plan(args.out)
    check_sources(plan["config"])
    phase = "calibration" if args.calibration else "pilot"
    jobs = [j for j in plan["jobs"] if j["phase"] == phase and j["gpu"] == args.gpu and j["arm"]["id"] == args.arm]
    if not jobs:
        return
    pending = [j for j in jobs if not receipt_valid(args.out, plan, j)]
    if not pending and not args.probe:
        return
    module = load_entrypoint(jobs[0]["arm"]["source"])
    # Parse even during probe: catches upstream argument drift without loading weights.
    sys.argv = argv_for(plan, jobs[0], args.out / "probe.mp4")
    module._parse_args()
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this method's Python environment")
    if importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError("flash-attn is required for a common A800 attention backend")
    attention, backend = configure_attention_backend()
    env = versions() | {"gpu_name": torch.cuda.get_device_name(0),
                        "cuda": torch.version.cuda, "wan_module": str(sys.modules["wan"].__file__),
                        "dense_attention_backend": backend["dense_backend"]}
    print(f"Dense attention locked to FA2; detected flash_attn_interface={backend['flash_attn_interface_module']}", flush=True)
    if args.probe:
        smoke = attention_smoke_test(attention)
        print(json.dumps({"arm": args.arm, "environment": env, "attention_backend": backend, "dense_kernel_smoke_test": smoke}, ensure_ascii=False))
        return
    # Match attention/TF32 policy across official and vendored Wan implementations.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    pipeline = None
    process_model_load_seconds = 0.0
    excluded_warmup_seconds = 0.0
    state = {}
    source = jobs[0]["arm"]["source"]
    owner = module if source == "scalingcache" else module.wan
    attribute = "CustomWanT2V" if source == "scalingcache" else "WanT2V"
    original_constructor = getattr(owner, attribute)

    def cached_constructor(*pos, **kw):
        nonlocal pipeline, process_model_load_seconds
        start = time.perf_counter()
        if pipeline is None:
            pipeline = original_constructor(*pos, **kw)
            torch.cuda.synchronize()
            state["load_seconds"] = time.perf_counter() - start
            process_model_load_seconds = state["load_seconds"]
        elif source == "scalingcache":
            pipeline.model.cache_init(num_steps=jobs[0]["arm"]["steps"])
        return pipeline

    setattr(owner, attribute, cached_constructor)
    original_randn, original_save = torch.randn, module.cache_video

    def capture_noise(*pos, **kw):
        tensor = original_randn(*pos, **kw)
        if kw.get("generator") is not None and tensor.ndim == 4 and not state.get("noise"):
            torch.cuda.synchronize()
            start = time.perf_counter()
            array = tensor.detach().float().cpu().contiguous().numpy()
            state["noise"] = {"shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
            state["instrument_seconds"] += time.perf_counter() - start
        return tensor

    def capture_video(*pos, **kw):
        # Attribute all outstanding denoising/VAE GPU work to the pipeline,
        # never to the encoding/audit subtraction that follows.
        torch.cuda.synchronize()
        tensor = kw["tensor"][0]
        if state["warmup"]:
            return None
        if phase == "calibration":
            start = time.perf_counter()
            # A fixed raw-float grid, before lossy MP4 encoding. This is a diagnostic,
            # not proof of exact equivalence of all output pixels.
            array = tensor[:, ::4, ::4, ::4].detach().float().cpu().numpy()
            sample = args.out / "samples" / (state["job"]["id"] + ".npz")
            sample.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sample, frames=array)
            state["sample"] = str(sample.resolve())
            state["instrument_seconds"] += time.perf_counter() - start
        start = time.perf_counter()
        original_save(*pos, **kw)
        state["save_seconds"] += time.perf_counter() - start

    torch.randn, module.cache_video = capture_noise, capture_video

    def execute(job, warmup=False):
        video = args.out / "videos" / phase / job["arm"]["id"] / (job["id"] + ".mp4")
        receipt = args.out / "records" / (job["id"] + ".json")
        if not warmup and video.exists() and not receipt.exists():
            raise RuntimeError(f"Unreceipted video exists: {video}. Inspect/move it before resuming.")
        video.parent.mkdir(parents=True, exist_ok=True)
        state.clear()
        state.update(job=job, warmup=warmup, load_seconds=0.0, instrument_seconds=0.0, save_seconds=0.0)
        sys.argv = argv_for(plan, job, video)
        options = module._parse_args()
        if options.ulysses_size != 1 or options.ring_size != 1 or options.dit_fsdp or options.t5_fsdp:
            raise RuntimeError("Published pilot workers are single-GPU; parallel options are forbidden")
        module.args = options  # Jenga's upstream forward reads module-global args.
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        started = time.perf_counter()
        module.generate(options)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if warmup:
            return elapsed
        if not video.is_file() or not state.get("noise"):
            raise RuntimeError("Missing saved MP4 or actual initial-noise capture")
        # Upstream cache_video catches exceptions; independently validate output.
        from UNIV_adaptor.scripts.data.acceleration_blind_audit import probe
        info = probe(video)
        width, height = map(int, plan["config"]["sampling"]["size"].split("*"))
        expected_duration = plan["config"]["sampling"]["frame_num"] / pipeline.config.sample_fps
        if info["width"] != width or info["height"] != height or abs(info["duration"] - expected_duration) > 0.1:
            raise RuntimeError(f"Incorrect saved video geometry/duration: {info}")
        runtime = {"pipeline_seconds": elapsed - state["load_seconds"] - state["instrument_seconds"] - state["save_seconds"],
                   "model_load_seconds": state["load_seconds"], "encoding_seconds": state["save_seconds"],
                   "process_model_load_seconds": process_model_load_seconds, "excluded_warmup_seconds": excluded_warmup_seconds,
                   "instrumentation_seconds": state["instrument_seconds"], "outer_seconds": elapsed,
                   "started_utc": started_utc, "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "warmed": plan["config"]["warmup"],
                   "scope": "prompt encoding + pipeline setup + denoising + VAE; excludes model loading, MP4 encoding, audit instrumentation"}
        if runtime["pipeline_seconds"] <= 0:
            raise RuntimeError("Invalid measured pipeline time")
        immutable(receipt, {"schema": "published_wan21_record_v1", "plan_sha256": plan["plan_sha256"],
            "job": job, "video_path": str(video.resolve()), "video_sha256": file_hash(video),
            "runtime": runtime, "environment": env, "noise": state["noise"],
            "attention_backend": backend,
            "compatibility_adaptations": SCALING_COMPATIBILITY if source == "scalingcache" else [],
            "sample_path": state.get("sample"), "sample_sha256": file_hash(state["sample"]) if state.get("sample") else None,
            "sampling_identity": {"negative_prompt": pipeline.sample_neg_prompt, "param_dtype": str(pipeline.param_dtype),
                                  "text_dtype": str(pipeline.config.t5_dtype), "fps": pipeline.config.sample_fps,
                                  "dense_attention_backend": backend["dense_backend"]}})
        print(f"Completed {job['id']}: {runtime['pipeline_seconds']:.2f}s", flush=True)

    if plan["config"]["warmup"]:
        # One excluded, full-shaped run per process/arm. No repeated weight loading.
        warm = dict(pending[0], prompt=plan["config"]["calibration"]["prompts"][0]["prompt"],
                    seed=plan["config"]["calibration"]["seed"])
        print(f"Excluded full-shape warmup: GPU {args.gpu} / {args.arm}", flush=True)
        excluded_warmup_seconds = execute(warm, warmup=True)
    for job in pending:
        execute(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--probe", action="store_true")
    run(parser.parse_args())
