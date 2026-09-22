"""HY1.5 native 50-step flow sampling + RGB restoration + direct sigma0.2 HR4.

The native Diffusers transformer, text encoders, VAE and Euler scheduler are
used. C is explicitly a last-CFG-velocity cache, NOT TeaCache or DeepCache.
"""
from __future__ import annotations

import time

from UNIV_adaptor.hy15_protocol import refresh_indices


def aligned_noise(full, shape):
    """Subsample a shared Gaussian field; never interpolate initial noise variance."""
    import torch
    result = full
    for axis, size in zip((2, 3, 4), shape):
        indices = torch.linspace(0, full.shape[axis] - 1, size, device=full.device).round().long()
        if len(indices.unique()) != size:
            raise ValueError("Noise mapping repeats coordinates")
        result = result.index_select(axis, indices)
    return result.clone()


class HY15EndpointRunner:
    def __init__(self, model_root, sr_checkpoint, protocol):
        import torch
        import diffusers
        from diffusers import HunyuanVideo15Pipeline
        from UNIV_adaptor.rgb_super_resolution import build_univ_rgb_super_resolver

        if diffusers.__version__ != protocol["diffusers_version"]:
            raise RuntimeError("Use the pinned HY15 environment; unexpected Diffusers API version")
        self.torch = torch
        self.p = protocol
        self.pipe = HunyuanVideo15Pipeline.from_pretrained(
            str(model_root), torch_dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        self.pipe.vae.enable_tiling()
        self.pipe.set_progress_bar_config(disable=True)
        if self.pipe.vae_scale_factor_temporal != 4 or self.pipe.vae_scale_factor_spatial != 16:
            raise RuntimeError("Unexpected VAE compression")
        if self.pipe.transformer.config.in_channels != 65:
            raise RuntimeError("Not the expected HY15 T2V checkpoint")
        self.sr = build_univ_rgb_super_resolver({
            "wan_rgb_sr_backend": "realesrgan", "wan_rgb_sr_checkpoint": str(sr_checkpoint),
            "wan_rgb_sr_tile": 256, "wan_rgb_sr_gpu_id": 0,
        })
        self.embeddings = {}
        self.native_parity = None

    def sync(self):
        self.torch.cuda.synchronize()

    def encode(self, prompt):
        if prompt not in self.embeddings:
            # Bounded cache: one positive prompt and the common negative prompt.
            if len(self.embeddings) >= 2:
                self.embeddings = {k: v for k, v in self.embeddings.items() if k == self.p["negative_prompt"]}
            self.sync()
            started = time.perf_counter()
            value = self.pipe.encode_prompt(prompt=prompt, device="cuda", dtype=self.torch.bfloat16)
            self.sync()
            self.embeddings[prompt] = (value, time.perf_counter() - started)
        return self.embeddings[prompt]

    def predict(self, x, timestep, positive, negative):
        torch = self.torch
        pipe = self.pipe
        cond, mask = pipe.prepare_cond_latents_and_mask(x, x.dtype, x.device)
        model_input = torch.cat([x, cond, mask], dim=1)
        image = torch.zeros(1, pipe.vision_num_semantic_tokens, pipe.vision_states_dim,
                            device=x.device, dtype=x.dtype)
        preds = []
        for label, embeds in (("cond", positive), ("uncond", negative)):
            with pipe.transformer.cache_context(label):
                preds.append(pipe.transformer(
                    hidden_states=model_input, image_embeds=image,
                    timestep=timestep.expand(1).to(x.dtype),
                    encoder_hidden_states=embeds[0], encoder_attention_mask=embeds[1],
                    encoder_hidden_states_2=embeds[2], encoder_attention_mask_2=embeds[3],
                    return_dict=False,
                )[0])
        # Exact pinned checkpoint guider: use_original_formulation=False, rescale=0.
        return preds[1] + self.p["cfg"] * (preds[0] - preds[1])

    def solve(self, x, positive, negative, *, refine=False, refreshes=50, steps=50):
        from diffusers import FlowMatchEulerDiscreteScheduler
        torch = self.torch
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, shift=1.0 if refine else self.p["shift"]
        )
        # shift=1 is crucial: otherwise sigma=.2 would be shifted a second time.
        inputs = self.p["refine_sigmas"][:-1] if refine else [1 - i / steps for i in range(steps)]
        scheduler.set_timesteps(sigmas=inputs, device=x.device)
        if refine and not torch.allclose(scheduler.sigmas.cpu(), torch.tensor(self.p["refine_sigmas"]), atol=1e-7):
            raise RuntimeError("Refinement sigma grid was altered by scheduler")
        refresh = set(range(4)) if refine else set(refresh_indices(steps, refreshes))
        velocity = None
        for i, timestep in enumerate(scheduler.timesteps):
            if i in refresh:
                velocity = self.predict(x, timestep, positive, negative)
            if velocity is None:
                raise RuntimeError("Cache used before first refresh")
            x = scheduler.step(velocity, timestep, x, return_dict=False)[0]
        if float(scheduler.sigmas[-1]) != 0 or not bool(torch.isfinite(x).all()):
            raise RuntimeError("Sampling did not produce a finite clean endpoint")
        return x, {"sigmas": scheduler.sigmas.cpu().tolist(), "refresh_indices": sorted(refresh),
                   "cfg_branch_forward_calls": 2 * len(refresh)}

    def check_native_parity(self, positive, negative):
        """Tiny two-step adapter check; never used as an experimental sample."""
        torch = self.torch
        x = torch.randn((1, 32, 2, 4, 4), device="cuda", dtype=torch.bfloat16,
                        generator=torch.Generator(device="cuda").manual_seed(7919))
        custom, _ = self.solve(x.clone(), positive, negative, steps=2, refreshes=2)
        names = ("prompt_embeds", "prompt_embeds_mask", "prompt_embeds_2", "prompt_embeds_mask_2")
        kwargs = dict(zip(names, positive))
        kwargs.update({"negative_" + k: v for k, v in zip(names, negative)})
        native = self.pipe(height=64, width=64, num_frames=5, num_inference_steps=2,
                           latents=x.clone(), output_type="latent", **kwargs).frames
        error = float((custom.float() - native.float()).abs().max())
        if not torch.allclose(custom.float(), native.float(), atol=0.02, rtol=0.005):
            raise RuntimeError(f"Custom/native sampling parity failed: max_abs={error}")
        self.native_parity = {"steps": 2, "shape": list(x.shape), "max_abs_error": error,
                              "atol": 0.02, "rtol": 0.005}

    def decode(self, x):
        vae = self.pipe.vae
        video = vae.decode(x.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
        return ((video[0].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).cpu()

    def restore(self, x, case):
        from UNIV_adaptor.transition import linear_resample_video
        p = self.p
        if all(case[k] == p[k] for k in ("height", "width", "frames")):
            return x, "identity_no_vae_roundtrip"
        video = self.decode(x)
        if case["height"] != p["height"] or case["width"] != p["width"]:
            video = self.sr.resize(video, target_height=p["height"], target_width=p["width"])
        if case["frames"] != p["frames"]:
            video = linear_resample_video(video, p["frames"])
        vae = self.pipe.vae
        inputs = video.permute(3, 0, 1, 2).unsqueeze(0).to("cuda", dtype=vae.dtype) * 2 - 1
        # Deterministic posterior mode, no additional seed-dependent VAE sampling.
        latent = vae.encode(inputs).latent_dist.mode() * vae.config.scaling_factor
        expected = (1, 32, (p["frames"] - 1) // 4 + 1, p["height"] // 16, p["width"] // 16)
        if tuple(latent.shape) != expected:
            raise RuntimeError(f"Restoration shape mismatch: {latent.shape}, expected {expected}")
        return latent.to(self.torch.bfloat16), "rgb_sr_or_linear_temporal_then_vae_mode"

    def generate(self, job):
        torch = self.torch
        p, case = self.p, job["case"]
        with torch.inference_mode():
            positive, pos_seconds = self.encode(job["prompt"]["prompt"])
            negative, neg_seconds = self.encode(p["negative_prompt"])
            if self.native_parity is None:
                self.check_native_parity(positive, negative)
            shape = (1, 32, (p["frames"] - 1) // 4 + 1, p["height"] // 16, p["width"] // 16)
            self.sync()
            started = time.perf_counter()
            generator = torch.Generator(device="cuda").manual_seed(job["seed"])
            field = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16)
            x = aligned_noise(field, ((case["frames"] - 1) // 4 + 1, case["height"] // 16, case["width"] // 16))
            x, main_grid = self.solve(x, positive, negative, refreshes=case["refreshes"])
            self.sync()
            main_seconds = time.perf_counter() - started
            started = time.perf_counter()
            if case["refine"]:
                x, transition = self.restore(x, case)
            else:
                transition = "native"
            self.sync()
            transition_seconds = time.perf_counter() - started
            started = time.perf_counter()
            refine_grid = None
            if case["refine"]:
                # Independent repair noise, shared across actions of the same prompt-seed.
                generator.manual_seed(job["seed"] + 1000000007)
                noise = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32)
                x = (0.8 * x.float() + 0.2 * noise).to(torch.bfloat16)
                x, refine_grid = self.solve(x, positive, negative, refine=True)
            self.sync()
            refine_seconds = time.perf_counter() - started
            started = time.perf_counter()
            video = self.decode(x)
            self.sync()
            decode_seconds = time.perf_counter() - started
        timing = {"text_encode_charged": pos_seconds + neg_seconds, "main": main_seconds,
                  "transition": transition_seconds, "refine": refine_seconds, "decode": decode_seconds}
        timing["candidate_total"] = sum(timing.values())
        return video, {"timing_seconds": timing, "main_grid": main_grid, "refine_grid": refine_grid,
                       "native_parity": self.native_parity,
                       "transition": transition, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                       "noise_protocol": "shared_full_grid_endpoint_subsample_and_independent_shared_repair_field"}
