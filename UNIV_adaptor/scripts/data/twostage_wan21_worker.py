"""GPU workers for the two-stage platform. One process = one unit on one GPU.

stage1: the pinned, unmodified official Wan2.1 generate.py, reusing the
platform-v1 worker's entrypoint loader and FA2 lock. The only adaptation is
registering the extra 416*240 canvas in wan.configs' size tables (the sampler
and model are untouched). Frames are also kept as lossless uint8, quantized
exactly as Wan's cache_video does, for stage 2.

post: FlashVSR through the proven flashvsr_asset_diagnostic helpers, RIFE
through Practical-RIFE's model.inference, or a trivial control. Outputs are
encoded with the same imageio/libx264 settings as Wan's cache_video.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from UNIV_adaptor.scripts.data import twostage_wan21 as driver  # noqa: E402
import flashvsr_asset_diagnostic as diag  # noqa: E402

WARMUP_PROMPT = "A plain blue ceramic mug slowly rotates on a turntable against a neutral gray background. Fixed camera, soft studio light."


def wan_frames(video):
    """uint8 THWC exactly as wan.utils.utils.cache_video quantizes (make_grid normalize, then truncation)."""
    import torch
    frames = video.float().clamp(-1, 1).sub(-1).div(2)
    return (frames * 255).type(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()


def encode_like_wan(frames, path, fps):
    import imageio
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def save_frames(frames, path):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, frames=frames)
    return {"path": str(path.resolve()), "sha256": driver.file_hash(path), "shape": list(frames.shape)}


def load_frames(identity):
    import numpy as np
    if driver.file_hash(identity["path"]) != identity["sha256"]:
        raise ValueError(f"Stage-1 frames changed: {identity['path']}")
    with np.load(identity["path"], allow_pickle=False) as archive:
        frames = archive["frames"]
    if frames.dtype != np.uint8 or list(frames.shape) != identity["shape"]:
        raise ValueError("Stage-1 frame tensor mismatch")
    return frames


def check_geometry(video, width, height, frames, fps):
    from UNIV_adaptor.scripts.data.acceleration_blind_audit import probe
    info = probe(video)
    if info["width"] != width or info["height"] != height or abs(info["duration"] - frames / fps) > 0.1:
        raise RuntimeError(f"Incorrect saved video geometry/duration: {info}")


def register_size(size):
    import wan.configs as configs
    if size in configs.SUPPORTED_SIZES["t2v-1.3B"]:
        return []
    width, height = driver.canvas(size)
    # generate.py imported these dict objects by name, so mutation is visible there.
    configs.SIZE_CONFIGS[size] = (width, height)
    configs.MAX_AREA_CONFIGS[size] = width * height
    configs.SUPPORTED_SIZES["t2v-1.3B"] = configs.SUPPORTED_SIZES["t2v-1.3B"] + (size,)
    return [f"Registered canvas {size} in wan.configs size tables so generate.py accepts it; sampler and model unchanged"]


def stage1_argv(p, job, gen, video):
    sampling = dict(p["config"]["sampling"], size=gen["size"], frame_num=gen["frame_num"])
    argv = ["published_entry", "--ckpt_dir", p["model_root"], "--prompt", job["prompt"],
            "--base_seed", str(job["seed"]), "--sample_steps", str(gen["steps"]), "--save_file", str(video)]
    for key, value in sampling.items():
        argv += ["--" + key, str(value).lower() if isinstance(value, bool) else str(value)]
    return argv


def stage1(args):
    p = driver.load_plan(args.out)
    gen = next(g for g in p["config"]["generations"] if g["id"] == args.unit)
    jobs = [j for j in p["gen_jobs"] if j["gpu"] == args.gpu and j["generation"]["id"] == gen["id"]]
    pending = [j for j in jobs if not driver.gen_receipt_valid(args.out, p, j)]
    if not pending and not args.probe:
        return
    from UNIV_adaptor.scripts.data.published_wan21_worker import (
        attention_smoke_test, configure_attention_backend, load_entrypoint, versions)
    module = load_entrypoint("wan21")
    adaptations = register_size(gen["size"])
    import numpy as np
    import torch
    import random
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Wan generation Python")
    attention, backend = configure_attention_backend()
    smoke = attention_smoke_test(attention)
    env = versions() | {"gpu_name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                        "wan_module": str(sys.modules["wan"].__file__), "dense_attention_backend": backend["dense_backend"]}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    keep_frames = any(a["generation"] == gen["id"] and a["post"] != "none" for a in p["config"]["arms"])
    state, pipeline, load_seconds = {}, None, 0.0
    original_constructor, original_save = module.wan.WanT2V, module.cache_video

    def cached_constructor(*pos, **kw):
        nonlocal pipeline, load_seconds
        start = time.perf_counter()
        if pipeline is None:
            pipeline = original_constructor(*pos, **kw)
            torch.cuda.synchronize()
            state["load_seconds"] = load_seconds = time.perf_counter() - start
        return pipeline

    def capture_video(*pos, **kw):
        torch.cuda.synchronize()  # attribute all denoising/VAE GPU work to the pipeline
        if state["warmup"]:
            return None
        if keep_frames:
            start = time.perf_counter()
            state["frames"] = save_frames(wan_frames(kw["tensor"][0]), state["frames_path"])
            state["instrument_seconds"] += time.perf_counter() - start
        start = time.perf_counter()
        original_save(*pos, **kw)
        state["save_seconds"] += time.perf_counter() - start

    module.wan.WanT2V, module.cache_video = cached_constructor, capture_video

    def execute(job, warmup=False):
        video = args.out / "videos" / "stage1" / gen["id"] / (job["id"] + ".mp4")
        frames_path = args.out / "frames" / gen["id"] / (job["id"] + ".npz")
        receipt = driver.gen_record_path(args.out, job["id"])
        if not warmup and (video.exists() or frames_path.exists()) and not receipt.exists():
            raise RuntimeError(f"Unreceipted output exists: {video} / {frames_path}. Inspect/move it before resuming.")
        video.parent.mkdir(parents=True, exist_ok=True)
        state.clear()
        state.update(warmup=warmup, load_seconds=0.0, instrument_seconds=0.0, save_seconds=0.0, frames_path=frames_path)
        sys.argv = stage1_argv(p, job, gen, video)
        options = module._parse_args()
        if options.ulysses_size != 1 or options.ring_size != 1 or options.dit_fsdp or options.t5_fsdp:
            raise RuntimeError("Workers are single-GPU; parallel options are forbidden")
        module.args = options
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        started = time.perf_counter()
        module.generate(options)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if warmup:
            return elapsed
        if not video.is_file() or (keep_frames and not state.get("frames")):
            raise RuntimeError("Missing saved MP4 or lossless frames")
        width, height = driver.canvas(gen["size"])
        check_geometry(video, width, height, gen["frame_num"], pipeline.config.sample_fps)
        runtime = {"pipeline_seconds": elapsed - state["load_seconds"] - state["instrument_seconds"] - state["save_seconds"],
                   "model_load_seconds": state["load_seconds"], "encoding_seconds": state["save_seconds"],
                   "instrumentation_seconds": state["instrument_seconds"], "outer_seconds": elapsed,
                   "excluded_warmup_seconds": warmup_seconds, "process_model_load_seconds": load_seconds,
                   "started_utc": started_utc, "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                   "scope": "prompt encoding + pipeline setup + denoising + VAE; excludes model loading, MP4 encoding, lossless frame saving"}
        if runtime["pipeline_seconds"] <= 0:
            raise RuntimeError("Invalid measured pipeline time")
        driver.immutable(receipt, {"schema": "twostage_stage1_record_v1", "plan_sha256": p["plan_sha256"], "job": job,
            "video_path": str(video.resolve()), "video_sha256": driver.file_hash(video),
            "frames": state.get("frames"), "runtime": runtime, "environment": env, "attention_backend": backend,
            "compatibility_adaptations": adaptations,
            "sampling_identity": {"negative_prompt": pipeline.sample_neg_prompt, "param_dtype": str(pipeline.param_dtype),
                                  "fps": pipeline.config.sample_fps, "size": gen["size"], "frame_num": gen["frame_num"]}})
        print(f"Completed {job['id']}: {runtime['pipeline_seconds']:.2f}s", flush=True)

    warmup_seconds = 0.0
    probe_job = {"id": f"probe_{gen['id']}", "prompt": WARMUP_PROMPT, "seed": 12345}
    if p["config"]["warmup"] or args.probe:
        print(f"Excluded full-shape warmup: GPU {args.gpu} / {gen['id']} {gen['size']} x {gen['frame_num']}", flush=True)
        warmup_seconds = execute(dict(pending[0] if pending else probe_job, prompt=WARMUP_PROMPT, seed=12345), warmup=True)
    if args.probe:
        print(f"PROBE OK {gen['id']}: dense smoke {smoke['passed']}, full-shape generation {warmup_seconds:.1f}s, "
              f"adaptations={adaptations}", flush=True)
        return
    for job in pending:
        execute(job)


def flash_processor(p, size):
    pipe = diag.init_flash(p)
    cfg = p["config"]["flashvsr"]

    def run(frames):
        output, timing = diag.flash_infer(pipe, frames, cfg, size)
        return output, timing
    return run


def rife_processor(p, size):
    import torch
    import torch.nn.functional as F
    cfg = p["config"]["rife"]
    if cfg["multi"] != 2 or cfg["fp16"]:
        raise ValueError("Only the frozen multi=2, fp32 protocol is implemented")
    # Official inference_video.py settings; train_log comes from the author's archive.
    sys.path[:0] = [p["rife_weights"]["model_parent"], p["rife_source"]["path"]]
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    from train_log.RIFE_HDv3 import Model
    model = Model()
    model.load_model(str(Path(p["rife_weights"]["model_parent"]) / "train_log"), -1)
    model.eval()
    model.device()
    scale = cfg["scale"]

    def run(frames):
        count, height, width, _ = frames.shape
        tile = max(128, int(128 / scale))
        padding = (0, ((width - 1) // tile + 1) * tile - width, 0, ((height - 1) // tile + 1) * tile - height)
        torch.cuda.synchronize()
        started = time.perf_counter()
        tensor = torch.from_numpy(frames).to("cuda", non_blocking=True).permute(0, 3, 1, 2).float() / 255.
        tensor = F.pad(tensor, padding)
        output = []
        for k in range(count - 1):
            middle = model.inference(tensor[k:k + 1], tensor[k + 1:k + 2], 0.5, scale)
            output += [frames[k], (middle[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)[:height, :width]]
        output.append(frames[-1])
        torch.cuda.synchronize()
        import numpy as np
        return np.stack(output), {"rife_model_seconds": time.perf_counter() - started, "pairs": count - 1,
                                  "omitted_cli_heuristics": "ssim>0.996 static-frame skip and ssim<0.2 scene-cut hold; both change frame count"}
    return run


def bicubic_processor(p, size):
    import numpy as np
    from PIL import Image

    def run(frames):
        return np.stack([np.asarray(Image.fromarray(f).resize(size, Image.Resampling.BICUBIC), dtype=np.uint8) for f in frames]), {}
    return run


def duplicate_processor(p, size):
    import numpy as np

    def run(frames):
        return frames[np.arange(2 * len(frames) - 1) // 2], {"policy": "output frame i <- input frame i//2"}
    return run


PROCESSORS = {"flashvsr": flash_processor, "rife": rife_processor, "bicubic": bicubic_processor, "duplicate": duplicate_processor}


def post(args):
    import numpy as np
    p = driver.load_plan(args.out)
    arm = next(a for a in p["config"]["arms"] if a["id"] == args.unit)
    gen = next(g for g in p["config"]["generations"] if g["id"] == arm["generation"])
    jobs = [j for j in p["final_jobs"] if j["gpu"] == args.gpu and j["arm"]["id"] == arm["id"]]
    pending = [j for j in jobs if not driver.final_receipt_valid(args.out, p, j)]
    if not pending and not args.probe:
        return
    from UNIV_adaptor.scripts.data.published_wan21_worker import versions
    size = driver.canvas(p["config"]["sampling"]["size"])
    frames_out = p["config"]["sampling"]["frame_num"]
    gpu = arm["post"] in ("flashvsr", "rife")
    if gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f"{arm['post']} requires CUDA")
    started = time.perf_counter()
    process = PROCESSORS[arm["post"]](p, size)
    load_seconds = time.perf_counter() - started
    env = versions() | ({"gpu_name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda} if gpu else {"device": "cpu"})

    def timed(frames):
        if gpu:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        output, detail = process(frames)
        if gpu:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if output.dtype != np.uint8 or output.shape != (frames_out, size[1], size[0], 3):
            raise RuntimeError(f"{arm['post']} output {output.shape}/{output.dtype} != {(frames_out, size[1], size[0], 3)}")
        return output, detail | ({"peak_cuda_bytes": torch.cuda.max_memory_allocated()} if gpu else {}), seconds

    width, height = driver.canvas(gen["size"])
    shape = (gen["frame_num"], height, width, 3)
    if args.probe:  # stage 1 may not have run yet
        warm = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    else:
        warm = load_frames(driver.read(driver.gen_record_path(args.out, pending[0]["generation_job"]))["frames"])
    _, _, warmup_seconds = timed(warm)
    print(f"Excluded full-shape warmup {arm['id']}: {warmup_seconds:.2f}s", flush=True)
    if args.probe:
        print(f"PROBE OK {arm['id']} ({arm['post']}): input {shape} -> output {(frames_out, size[1], size[0], 3)}", flush=True)
        return
    for job in pending:
        gen_job = next(j for j in p["gen_jobs"] if j["id"] == job["generation_job"])
        if not driver.gen_receipt_valid(args.out, p, gen_job):
            raise RuntimeError(f"Missing stage-1 receipt: {gen_job['id']}")
        stage1_path = driver.gen_record_path(args.out, gen_job["id"])
        stage1 = driver.read(stage1_path)
        video = args.out / "videos" / "final" / arm["id"] / (job["id"] + ".mp4")
        receipt = driver.final_record_path(args.out, job["id"])
        if video.exists() and not receipt.exists():
            raise RuntimeError(f"Unreceipted video exists: {video}. Inspect/move it before resuming.")
        frames = load_frames(stage1["frames"])
        output, detail, seconds = timed(frames)
        video.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        encode_like_wan(output, video, stage1["sampling_identity"]["fps"])
        encoding_seconds = time.perf_counter() - start
        check_geometry(video, size[0], size[1], frames_out, stage1["sampling_identity"]["fps"])
        stage1_seconds = stage1["runtime"]["pipeline_seconds"]
        runtime = {"pipeline_seconds": stage1_seconds + seconds, "stage1_seconds": stage1_seconds, "post_seconds": seconds,
                   "post_model_load_seconds": load_seconds, "excluded_post_warmup_seconds": warmup_seconds,
                   "encoding_seconds": encoding_seconds,
                   "scope": "stage-1 pipeline + stage-2 uint8-frames-in to uint8-frames-out (incl. pre/post resize); "
                            "excludes model loading, frame file I/O and MP4 encoding"}
        driver.immutable(receipt, {"schema": "twostage_post_record_v1", "plan_sha256": p["plan_sha256"], "job": job,
            "stage1_receipt_sha256": driver.file_hash(stage1_path), "video_path": str(video.resolve()),
            "video_sha256": driver.file_hash(video), "runtime": runtime, "post_detail": detail, "environment": env})
        print(f"Completed {job['id']}: stage1 {stage1_seconds:.2f}s + post {seconds:.2f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["stage1", "post"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    (stage1 if args.phase == "stage1" else post)(args)
