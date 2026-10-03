"""Native Wan adapter for the existing clean-endpoint/RGB-SR/direct-HR protocol.

Only custom arms use this adapter. Published cache entrypoints remain unchanged.
Heavy dependencies are deliberately deferred so planning/tests remain CPU-only.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math
from pathlib import Path
import time

from UNIV_adaptor.flow import wan_renoise
from UNIV_adaptor.hr_refinement import direct_hr_sigmas
from UNIV_adaptor.transition import (
    WanDVGAnchorTransition, WanRGBSRTransition, dvg_rounded_anchors,
)


def latent_shape(width, height, frames):
    if min(width, height, frames) <= 0 or width % 16 or height % 16 or frames % 4 != 1:
        raise ValueError("Wan geometry requires positive patch-aligned W/H and 4n+1 frames")
    return (16, (frames - 1) // 4 + 1, height // 8, width // 8)


def geometry(case, target):
    full = latent_shape(*target)
    small = latent_shape(case["width"], case["height"], case["frames"])
    if any(a > b for a, b in zip(small, full)):
        raise ValueError("Endpoint restoration cannot shrink the target")
    return {"main_latent_shape": list(small), "target_latent_shape": list(full),
            "realized_main_token_density": math.prod(small[1:]) / math.prod(full[1:]),
            "nominal_main_budget": case.get("budget", 1.0),
            "budget_scope": "main-pass token density only; excludes restoration and four full HR steps"}


def tensor_hash(tensor):
    array = tensor.detach().float().cpu().contiguous().numpy()
    return {"shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def coarse_noise(full, shape):
    """Nested samples of a shared iid full field, not interpolated noise.

    Each smaller field still has iid N(0,1) entries. This fixes marginal noise
    variance; it cannot make different-resolution trajectories identical.
    """
    import torch
    anchors = {}
    small = full
    for axis in (1, 2, 3):
        indices = dvg_rounded_anchors(shape[axis], full.shape[axis])
        anchors[str(axis)] = list(indices)
        small = small.index_select(axis, torch.tensor(indices, device=full.device))
    return small.contiguous(), anchors


class NativeCodec:
    def __init__(self, vae):
        self.vae = vae

    def decode(self, latent):
        return self.vae.decode([latent])[0]

    def encode(self, video):
        return self.vae.encode([video[0]])


def fresh_hr_scheduler(scheduler_class, *, device, num_train_timesteps, sigma, steps):
    import numpy as np
    scheduler = scheduler_class(num_train_timesteps=num_train_timesteps,
                                shift=1, use_dynamic_shifting=False)
    grid = direct_hr_sigmas(start_sigma=sigma, hr_steps=steps)
    # Native Wan's set_timesteps requires an ndarray, not a Python list. Shift
    # must be ONE: the explicit sigma grid must not be shifted a second time.
    scheduler.set_timesteps(device=device, sigmas=np.asarray(grid[:-1]), shift=1)
    actual = scheduler.sigmas.cpu().tolist()
    if len(scheduler.timesteps) != steps or not np.allclose(actual, grid, atol=1e-7, rtol=0):
        raise RuntimeError(f"Incorrect independent HR grid: {actual}")
    if scheduler.step_index is not None or scheduler.lower_order_nums != 0 or any(
            item is not None for item in scheduler.model_outputs):
        raise RuntimeError("HR solver history is not fresh")
    return scheduler


class EndpointRuntime:
    def __init__(self, pipeline, case, protocol, out, arm, warmup):
        self.pipeline, self.case, self.protocol = pipeline, case, protocol
        self.out, self.arm, self.warmup = Path(out), arm, warmup
        self.calls, self.record, self.instrument_seconds = 0, None, 0.0
        self.sr = None
        if case["transition"] == "rgb_sr_vae":
            from UNIV_adaptor.rgb_super_resolution import build_univ_rgb_super_resolver
            self.sr = build_univ_rgb_super_resolver(protocol["sr"])

    def audit_action(self, function):
        import torch
        # Outstanding generation kernels belong to generation, not to this
        # instrumentation subtraction.
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = function()
        torch.cuda.synchronize()
        self.instrument_seconds += time.perf_counter() - start
        return result

    def generate(self, input_prompt, size=(832, 480), frame_num=81, shift=8.0,
                 sample_solver="unipc", sampling_steps=50, guide_scale=6.0,
                 n_prompt="", seed=-1, offload_model=False):
        import numpy as np
        import torch
        from torch.cuda import amp
        from tqdm import tqdm
        from wan.text2video import FlowUniPCMultistepScheduler
        p = self.pipeline
        is_warmup = self.warmup and self.calls == 0
        self.calls += 1
        self.instrument_seconds, self.record = 0.0, None
        if sample_solver != "unipc" or sampling_steps != 50 or offload_model or p.t5_cpu or p.sp_size != 1 or p.rank != 0 or seed < 0:
            raise ValueError("Endpoint adapter requires the frozen native single-GPU/no-offload 50-step UniPC protocol")
        target = (*size, frame_num)
        info = geometry(self.case, target)
        full_shape, main_shape = tuple(info["target_latent_shape"]), tuple(info["main_latent_shape"])
        if p.vae.model.z_dim != 16 or tuple(p.vae_stride) != (4, 8, 8) or tuple(p.patch_size) != (1, 2, 2):
            raise ValueError("Unexpected native Wan latent/patch geometry")
        generator = torch.Generator(device=p.device).manual_seed(seed)
        p.text_encoder.model.to(p.device)
        context = p.text_encoder([input_prompt], p.device)
        context_null = p.text_encoder([n_prompt or p.sample_neg_prompt], p.device)
        # The existing worker captures this FULL field for all arms. Smaller
        # executed fields and their anchor mapping are recorded separately.
        full_noise = torch.randn(*full_shape, dtype=torch.float32, device=p.device, generator=generator)
        noise, anchors = coarse_noise(full_noise, main_shape)
        executed_noise = self.audit_action(lambda: tensor_hash(noise))
        del full_noise
        timing, counters = {}, {"main": 0, "hr": 0}

        def synchronize():
            torch.cuda.synchronize()
            return time.perf_counter()

        def denoise(latent, scheduler, phase):
            seq_len = math.ceil(latent.shape[2] * latent.shape[3] /
                                (p.patch_size[1] * p.patch_size[2]) * latent.shape[1])
            cond, uncond = {"context": context, "seq_len": seq_len}, {"context": context_null, "seq_len": seq_len}
            for t in tqdm(scheduler.timesteps, desc=f"{self.arm}/{phase}"):
                timestep = torch.stack([t])
                p.model.to(p.device)
                prediction_c = p.model([latent], t=timestep, **cond)[0]
                counters[phase] += 1
                prediction_u = p.model([latent], t=timestep, **uncond)[0]
                counters[phase] += 1
                prediction = prediction_u + guide_scale * (prediction_c - prediction_u)
                latent = scheduler.step(prediction.unsqueeze(0), t, latent.unsqueeze(0),
                                        return_dict=False, generator=generator)[0].squeeze(0)
            return latent

        artifacts = []

        def save_endpoint(clean):
            if is_warmup:
                return
            phase = "calibration" if seed == self.protocol["calibration_seed"] and input_prompt in self.protocol["calibration_prompts"] else "pilot"
            key = hashlib.sha256(f"{phase}|{input_prompt}|{seed}|{self.arm}".encode()).hexdigest()[:24]
            path = self.out / "endpoints" / f"{key}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise RuntimeError(f"Unreceipted endpoint exists; inspect/move before resuming: {path}")
            torch.save({"main_clean": clean.detach().cpu(), "seed": seed,
                        "prompt": input_prompt, "arm": self.arm, "geometry": info}, path)
            with path.open("rb") as handle:
                sha = hashlib.file_digest(handle, "sha256").hexdigest()
            artifacts.append({"kind": "main_clean_sigma_zero", "path": str(path.resolve()), "sha256": sha})

        no_sync = getattr(p.model, "no_sync", nullcontext)
        with amp.autocast(dtype=p.param_dtype), torch.no_grad(), no_sync():
            scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=p.num_train_timesteps,
                                                   shift=1, use_dynamic_shifting=False)
            scheduler.set_timesteps(sampling_steps, device=p.device, shift=shift)
            start = synchronize()
            clean = denoise(noise, scheduler, "main")
            timing["main_seconds"] = synchronize() - start
            if float(scheduler.sigmas[-1]) != 0 or scheduler.step_index != 50 or counters["main"] != 100:
                raise RuntimeError("Main pass did not finish all 50 intervals at sigma=0")
            clean_hash = self.audit_action(lambda: tensor_hash(clean))
            self.audit_action(lambda: save_endpoint(clean))
            transition = self.case["transition"]
            start = synchronize()
            transition_info = {"baseline": transition}
            if transition == "rgb_sr_vae":
                # Preserve the existing SR network's FP16/FP32 policy rather
                # than accidentally running its convolutions in DiT BF16.
                # Native Wan VAE methods install their own codec autocast.
                with amp.autocast(enabled=False):
                    lifted = WanRGBSRTransition(vae_codec=NativeCodec(p.vae), spatial_resolver=self.sr,
                        target_height=size[1], target_width=size[0]).lift(clean, target_latent_shape=full_shape)
                clean_hr = lifted.clean_hr
                transition_info = asdict_without_tensor(lifted)
            elif transition == "dvg_latent_anchor":
                lifted = WanDVGAnchorTransition().lift(clean, target_latent_shape=full_shape)
                clean_hr = lifted.clean_hr
                transition_info = asdict_without_tensor(lifted)
            elif transition == "vae_roundtrip":
                # WanRGBSRTransition.lift intentionally skips equal geometry.
                # This dedicated control explicitly forces deterministic VAE
                # encode(decode(z)), with no SR and no latent interpolation.
                codec = WanRGBSRTransition(vae_codec=NativeCodec(p.vae), spatial_resolver=None,
                                          target_height=size[1], target_width=size[0])
                decoded = codec._decode(clean)
                rgb = ((decoded.float().clamp(-1, 1) + 1) * .5).permute(1, 2, 3, 0).contiguous().cpu()
                clean_hr = codec._encode(rgb, device=clean.device, dtype=clean.dtype)
                del decoded, rgb
            elif transition == "identity":
                clean_hr = clean
            else:
                raise ValueError(f"Unknown transition: {transition}")
            timing["transition_seconds"] = synchronize() - start
            if tuple(clean_hr.shape) != full_shape:
                raise RuntimeError("Restoration did not produce the exact target latent shape")
            restored_hash = self.audit_action(lambda: tensor_hash(clean_hr))
            hr_meta = None
            if self.case["refine"]:
                hr = self.protocol["hr"]
                repair_seed = (seed + hr["noise_seed_offset"]) % (2**63 - 1)
                repair_generator = torch.Generator(device=p.device).manual_seed(repair_seed)
                repair_noise = torch.randn(*full_shape, dtype=torch.float32, device=p.device, generator=repair_generator)
                repair_hash = self.audit_action(lambda: tensor_hash(repair_noise))
                start = synchronize()
                latent = wan_renoise(clean_hr, repair_noise, hr["sigma"])
                hr_scheduler = fresh_hr_scheduler(FlowUniPCMultistepScheduler, device=p.device,
                    num_train_timesteps=p.num_train_timesteps, sigma=hr["sigma"], steps=hr["steps"])
                hr_meta = {"sigmas": hr_scheduler.sigmas.cpu().tolist(),
                           "timesteps": hr_scheduler.timesteps.cpu().tolist(), "fresh_solver_history": True,
                           "noise_seed": repair_seed, "noise": repair_hash,
                           "formula": "(1-sigma)*clean_hr + sigma*noise", "shift": 1}
                clean_hr = denoise(latent, hr_scheduler, "hr")
                timing["hr_seconds"] = synchronize() - start
                if hr_scheduler.step_index != 4 or counters["hr"] != 8:
                    raise RuntimeError("Expected four independent full HR steps/eight CFG forwards")
            else:
                timing["hr_seconds"] = 0.0
            start = synchronize()
            video = p.vae.decode([clean_hr])[0]
            timing["final_decode_seconds"] = synchronize() - start
        self.record = {"schema": "native_wan21_endpoint_receipt_v1", **info,
            "transition": transition_info, "main_steps": 50, "main_terminal_sigma": 0.0,
            "main_clean": clean_hash, "restored_clean": restored_hash, "hr": hr_meta,
            "model_forward_counts": counters, "executed_main_noise": executed_noise,
            "noise_alignment": {"policy": "nested_iid_full_field_anchor_subsample_v1", "anchors": anchors,
                                "caveat": "Shared field, not identical tensors/trajectories across geometries"},
            "timing": timing, "artifacts": artifacts}
        if not np.isfinite(list(timing.values())).all():
            raise RuntimeError("Non-finite stage timing")
        return video


def asdict_without_tensor(result):
    # dataclasses.asdict would deep-copy the large CUDA clean_hr tensor.
    return {key: getattr(result, key) for key in result.__dataclass_fields__ if key != "clean_hr"}
