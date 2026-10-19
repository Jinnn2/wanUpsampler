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
    if gen.get("sampler") == "rcm":
        return rcm_stage1(args, p, gen, pending)
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


def load_rcm_weights(model, path):
    """Official rCM DiT -> official Wan2.1 WanModel. Same module names; rCM patchifies with a Linear over
    (c kt kh kw), which is exactly the flattened Conv3d kernel. Every Wan parameter must be replaced."""
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    target = model.state_dict()
    converted, ignored = {}, []
    for key, value in state.items():
        name = key[len("net."):] if key.startswith("net.") else key
        if name not in target:
            ignored.append(key)
            continue
        if value.shape != target[name].shape:
            if name != "patch_embedding.weight" or value.numel() != target[name].numel():
                raise ValueError(f"rCM tensor {key} {tuple(value.shape)} != Wan {tuple(target[name].shape)}")
            value = value.reshape(target[name].shape)
        converted[name] = value
    if set(converted) != set(target) or any(not k.startswith("net.accum_") for k in ignored):
        raise ValueError(f"rCM checkpoint does not cover the Wan model exactly: missing "
                         f"{sorted(set(target) - set(converted))[:5]}, unexpected {ignored[:5]}")
    model.load_state_dict(converted, strict=True)
    return {"tensors": len(converted), "ignored_training_counters": ignored,
            "source_dtypes": sorted({str(v.dtype) for v in state.values()}),
            "conversion": "strip 'net.'; patch_embedding Linear(64) -> Conv3d(16,1,2,2) reshape"}


def rcm_stage1(args, p, gen, pending):
    """rCM (NVlabs/rcm wan2pt1_t2v_rcm_infer.py) sampling with the official Wan2.1 T5, VAE and DiT code."""
    from UNIV_adaptor.scripts.data.published_wan21_worker import (
        attention_smoke_test, configure_attention_backend, load_entrypoint, versions)
    module = load_entrypoint("wan21")
    import math
    import random
    import numpy as np
    import torch
    import torch.cuda.amp as amp
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Wan generation Python")
    attention, backend = configure_attention_backend()
    smoke = attention_smoke_test(attention)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    binding, rc = p["rcm_weights"], p["config"]["rcm"]
    if Path(binding["path"]).stat().st_size != binding["bytes"]:
        raise ValueError("rCM checkpoint changed after the plan was frozen")
    wan = module.wan
    config = wan.configs.WAN_CONFIGS[p["config"]["sampling"]["task"]]
    started = time.perf_counter()
    pipe = wan.WanT2V(config=config, checkpoint_dir=p["model_root"], device_id=0, rank=0,
                      t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False)
    load_report = load_rcm_weights(pipe.model, binding["path"])
    pipe.text_encoder.model.to(pipe.device)  # official generate() does this on its first call (t5_cpu=False)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    env = versions() | {"gpu_name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                        "wan_module": str(sys.modules["wan"].__file__), "dense_attention_backend": backend["dense_backend"]}
    width, height = driver.canvas(gen["size"])
    frames = gen["frame_num"]

    def sample(prompt, seed):
        target = (pipe.vae.model.z_dim, (frames - 1) // pipe.vae_stride[0] + 1,
                  height // pipe.vae_stride[1], width // pipe.vae_stride[2])
        seq_len = math.ceil(target[2] * target[3] / (pipe.patch_size[1] * pipe.patch_size[2]) * target[1])
        generator = torch.Generator(device=pipe.device)
        generator.manual_seed(seed)
        context = pipe.text_encoder([prompt], pipe.device)
        noise = torch.randn(*target, dtype=torch.float32, device=pipe.device, generator=generator)
        # rCM TrigFlow schedule converted to rectified flow, exactly as the official script.
        t = torch.tensor([math.atan(rc["sigma_max"]), *rc["mid_t"][:gen["steps"] - 1], 0.0],
                         dtype=torch.float64, device=pipe.device)
        t = torch.sin(t) / (torch.cos(t) + torch.sin(t))
        x = noise.to(torch.float64) * t[0]
        with amp.autocast(dtype=pipe.param_dtype), torch.no_grad():
            for t_cur, t_next in zip(t[:-1], t[1:]):
                # The official script feeds t*1000 in bfloat16 into the time embedding.
                timestep = (t_cur.float() * 1000).to(torch.bfloat16).float().reshape(1)
                v = pipe.model([x.to(torch.bfloat16)], t=timestep, context=context, seq_len=seq_len)[0].to(torch.float64)
                x = (1 - t_next) * (x - t_cur * v) + t_next * torch.randn(
                    *target, dtype=torch.float32, device=pipe.device, generator=generator)
            return pipe.vae.decode([x.float()])[0]

    def execute(job, warmup=False, save=None):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        start = time.perf_counter()
        video_tensor = sample(job["prompt"], job["seed"])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if not torch.isfinite(video_tensor).all() or video_tensor.float().std() < 0.02:
            raise RuntimeError(f"rCM output is non-finite or blank for {job['id']}")
        if save is None:
            return seconds, None
        save.parent.mkdir(parents=True, exist_ok=True)
        encode_start = time.perf_counter()
        module.cache_video(tensor=video_tensor[None], save_file=str(save), fps=config.sample_fps, nrow=1,
                           normalize=True, value_range=(-1, 1))
        check_geometry(save, width, height, frames, config.sample_fps)
        return seconds, {"started_utc": started_utc, "encoding_seconds": time.perf_counter() - encode_start,
                         "peak_cuda_bytes": torch.cuda.max_memory_allocated()}

    probe_job = {"id": f"probe_{gen['id']}", "prompt": WARMUP_PROMPT, "seed": 12345}
    print(f"Excluded full-shape warmup: GPU {args.gpu} / {gen['id']} rCM {gen['steps']} step(s)", flush=True)
    warmup_seconds, _ = execute(probe_job, warmup=True)
    if args.probe:  # keep the probe video so a human can confirm the converted weights draw a sane scene
        look = args.out / "probe_videos" / f"{gen['id']}_{time.time_ns()}.mp4"
        seconds, _ = execute(probe_job, save=look)
        print(f"PROBE OK {gen['id']}: dense smoke {smoke['passed']}, rCM {gen['steps']} step(s) {seconds:.1f}s, "
              f"weights {load_report['tensors']} tensors; LOOK AT {look}", flush=True)
        return
    for job in pending:
        video = args.out / "videos" / "stage1" / gen["id"] / (job["id"] + ".mp4")
        receipt = driver.gen_record_path(args.out, job["id"])
        if video.exists() and not receipt.exists():
            raise RuntimeError(f"Unreceipted output exists: {video}. Inspect/move it before resuming.")
        seconds, extra = execute(job, save=video)
        runtime = {"pipeline_seconds": seconds, "model_load_seconds": 0.0, "process_model_load_seconds": load_seconds,
                   "excluded_warmup_seconds": warmup_seconds, "encoding_seconds": extra["encoding_seconds"],
                   "started_utc": extra["started_utc"], "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "peak_cuda_bytes": extra["peak_cuda_bytes"],
                   "scope": "prompt encoding + rCM sampling + VAE; excludes model loading and MP4 encoding"}
        driver.immutable(receipt, {"schema": "twostage_stage1_record_v1", "plan_sha256": p["plan_sha256"], "job": job,
            "video_path": str(video.resolve()), "video_sha256": driver.file_hash(video), "frames": None,
            "runtime": runtime, "environment": env, "attention_backend": backend, "rcm_weight_load": load_report,
            "compatibility_adaptations": ["rCM DiT weights loaded into the official Wan2.1 WanModel (see rcm_weight_load)"],
            "sampling_identity": {"sampler": "rcm", "steps": gen["steps"], "sigma_max": rc["sigma_max"],
                                  "mid_t": rc["mid_t"][:gen["steps"] - 1], "guidance": "none", "negative_prompt": None,
                                  "param_dtype": str(pipe.param_dtype), "fps": config.sample_fps,
                                  "size": gen["size"], "frame_num": frames}})
        print(f"Completed {job['id']}: {seconds:.2f}s", flush=True)


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
    """One step (args.step) of one arm's post chain; the last step writes the final video and receipt."""
    import numpy as np
    p = driver.load_plan(args.out)
    arm = next(a for a in p["config"]["arms"] if a["id"] == args.unit)
    steps, step = driver.post_steps(arm), args.step
    kind, last = steps[step], step == len(steps) - 1
    (in_frames, in_w, in_h), (out_frames, out_w, out_h) = driver.chain_shapes(p["config"], arm)[step:step + 2]
    jobs = [j for j in p["final_jobs"] if j["gpu"] == args.gpu and j["arm"]["id"] == arm["id"]]
    pending = [j for j in jobs if not driver.post_step_done(args.out, p, j, step)]
    if not pending and not args.probe:
        return
    from UNIV_adaptor.scripts.data.published_wan21_worker import versions
    gpu = kind in driver.GPU_POSTS
    if gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f"{kind} requires CUDA")
    started = time.perf_counter()
    process = PROCESSORS[kind](p, (out_w, out_h))
    load_seconds = time.perf_counter() - started
    env = versions() | ({"gpu_name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda} if gpu else {"device": "cpu"})
    expected = (out_frames, out_h, out_w, 3)

    def timed(frames):
        if gpu:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        output, detail = process(frames)
        if gpu:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if output.dtype != np.uint8 or output.shape != expected:
            raise RuntimeError(f"{kind} output {output.shape}/{output.dtype} != {expected}")
        return output, detail | ({"peak_cuda_bytes": torch.cuda.max_memory_allocated()} if gpu else {}), seconds

    def step_input(job):
        """Frames entering this step, the receipt they come from, and the stage-1 receipt of the chain."""
        gen_job = next(j for j in p["gen_jobs"] if j["id"] == job["generation_job"])
        if not driver.gen_receipt_valid(args.out, p, gen_job):
            raise RuntimeError(f"Missing stage-1 receipt: {gen_job['id']}")
        if step and not driver.step_receipt_valid(args.out, p, job, step - 1):
            raise RuntimeError(f"Missing post step {step - 1} for {job['id']}")
        source = driver.step_input_receipt(args.out, job, step)
        return load_frames(driver.read(source)["frames"]), source, driver.gen_record_path(args.out, gen_job["id"])

    shape = (in_frames, in_h, in_w, 3)
    if args.probe:  # earlier stages may not have run yet
        warm = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    else:
        warm = step_input(pending[0])[0]
    _, _, warmup_seconds = timed(warm)
    print(f"Excluded full-shape warmup {arm['id']} step {step} ({kind}): {warmup_seconds:.2f}s", flush=True)
    if args.probe:
        print(f"PROBE OK {arm['id']} step {step} ({kind}): input {shape} -> output {expected}", flush=True)
        return
    for job in pending:
        frames, source, stage1_path = step_input(job)
        stage1 = driver.read(stage1_path)
        earlier = [driver.read(driver.step_record_path(args.out, job["id"], k)) for k in range(step)]
        fps = stage1["sampling_identity"]["fps"]
        if not last:
            receipt = driver.step_record_path(args.out, job["id"], step)
            frames_path = args.out / "frames_post" / arm["id"] / f"{job['id']}.step{step}.npz"
            if frames_path.exists() and not receipt.exists():
                raise RuntimeError(f"Unreceipted intermediate frames exist: {frames_path}. Inspect/move it before resuming.")
            output, detail, seconds = timed(frames)
            driver.immutable(receipt, {"schema": "twostage_post_step_record_v1", "plan_sha256": p["plan_sha256"],
                "job": job, "step": step, "kind": kind, "input_receipt_sha256": driver.file_hash(source),
                "frames": save_frames(output, frames_path), "seconds": seconds, "post_model_load_seconds": load_seconds,
                "excluded_post_warmup_seconds": warmup_seconds, "post_detail": detail, "environment": env})
            print(f"Completed {job['id']} step {step} ({kind}): {seconds:.2f}s", flush=True)
            continue
        video = args.out / "videos" / "final" / arm["id"] / (job["id"] + ".mp4")
        receipt = driver.final_record_path(args.out, job["id"])
        if video.exists() and not receipt.exists():
            raise RuntimeError(f"Unreceipted video exists: {video}. Inspect/move it before resuming.")
        output, detail, seconds = timed(frames)
        video.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        encode_like_wan(output, video, fps)
        encoding_seconds = time.perf_counter() - start
        check_geometry(video, out_w, out_h, out_frames, fps)
        stage1_seconds = stage1["runtime"]["pipeline_seconds"]
        post_seconds = sum(e["seconds"] for e in earlier) + seconds
        runtime = {"pipeline_seconds": stage1_seconds + post_seconds, "stage1_seconds": stage1_seconds,
                   "post_seconds": post_seconds, "post_step_seconds": [e["seconds"] for e in earlier] + [seconds],
                   "post_model_load_seconds": load_seconds, "excluded_post_warmup_seconds": warmup_seconds,
                   "encoding_seconds": encoding_seconds,
                   "scope": "stage-1 pipeline + every stage-2 step uint8-frames-in to uint8-frames-out (incl. pre/post "
                            "resize); excludes model loading, frame file I/O and MP4 encoding"}
        driver.immutable(receipt, {"schema": "twostage_post_record_v1", "plan_sha256": p["plan_sha256"], "job": job,
            "stage1_receipt_sha256": driver.file_hash(stage1_path),
            "step_receipt_sha256": [driver.file_hash(driver.step_record_path(args.out, job["id"], k)) for k in range(step)],
            "video_path": str(video.resolve()), "video_sha256": driver.file_hash(video), "runtime": runtime,
            "post_detail": detail, "earlier_step_detail": [e["post_detail"] for e in earlier], "environment": env})
        print(f"Completed {job['id']}: stage1 {stage1_seconds:.2f}s + post {post_seconds:.2f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["stage1", "post"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--step", type=int, default=0, help="post: index of the step in the arm's post chain")
    args = parser.parse_args()
    args.out = args.out.resolve()
    (stage1 if args.phase == "stage1" else post)(args)
