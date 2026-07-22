from __future__ import annotations

import torch
from loguru import logger

from lightx2v.models.networks.wan.infer.module_io import GridOutput
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

from changing_resolution.ralu_nt_math import ralu_resume_parameters, ralu_stage_sigmas
from changing_resolution.ralu_wan_state import (
    RALUPackedState,
    first_ralu_handoff,
    scatter_wan_patches,
    second_ralu_handoff,
)


class WanRALUQualityScheduler(WanScheduler):
    """Wan state container with exact 368x640 Quality-stage initialization."""

    def prepare_latents(self, seed, latent_shape, dtype=torch.float32):
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)
        low_h, low_w = (int(v) for v in self.config.get("wan_ralu_low_latent_size", [46, 80]))
        expected_output = tuple(
            int(v) for v in self.config.get("wan_ralu_output_latent_size", [90, 156])
        )
        if tuple(int(v) for v in latent_shape[-2:]) != expected_output:
            raise ValueError(
                f"RALU geometry A expects output latent {expected_output}, "
                f"got {tuple(latent_shape[-2:])}"
            )
        self.latents = torch.randn(
            latent_shape[0],
            latent_shape[1],
            low_h,
            low_w,
            dtype=dtype,
            device=AI_DEVICE,
            generator=self.generator,
        )
        self.ralu_stage = 1
        self.ralu_packed_state = None


class WanRALUQualityPreInfer(WanPreInfer):
    """Wan pre-infer that embeds only the real packed mixed-resolution tokens."""

    def _cos_sin_from_coords(self, coords: torch.Tensor) -> torch.Tensor:
        c = self.head_size // 2
        split_freqs = self.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        parts = []
        for axis, axis_freqs in enumerate(split_freqs):
            unit_phase = torch.angle(axis_freqs[1]).to(device=coords.device, dtype=torch.float32)
            phase = coords[:, axis : axis + 1].to(torch.float32) * unit_phase.unsqueeze(0)
            parts.append(torch.polar(torch.ones_like(phase), phase))
        complex_rope = torch.cat(parts, dim=-1)
        if self.config.get("rope_type", "flashinfer") == "flashinfer":
            return torch.cat([complex_rope.real.contiguous(), complex_rope.imag.contiguous()], dim=-1)
        return complex_rope.reshape(complex_rope.shape[0], 1, -1)

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        scheduler = self.scheduler
        if int(getattr(scheduler, "ralu_stage", 1)) != 2:
            return super().infer(weights, inputs, kv_start=kv_start, kv_end=kv_end)

        state: RALUPackedState = scheduler.ralu_packed_state
        coarse_dense = scatter_wan_patches(
            state.coarse_values,
            state.coarse_parent_indices,
            channels=state.channels,
            frames=state.frames,
            grid=state.coarse_grid,
        )
        scheduler.latents = coarse_dense
        output = super().infer(weights, inputs, kv_start=kv_start, kv_end=kv_end)
        coarse_embeddings = output.x[state.coarse_parent_indices.long()]

        fine_dense = scatter_wan_patches(
            state.fine_values,
            state.fine_indices,
            channels=state.channels,
            frames=state.frames,
            grid=state.fine_grid,
        )
        fine_embeddings = weights.patch_embedding.apply(fine_dense.unsqueeze(0))
        fine_embeddings = fine_embeddings.flatten(2).transpose(1, 2).contiguous().squeeze(0)
        fine_embeddings = fine_embeddings[state.fine_indices.long()]

        output.x = torch.cat([coarse_embeddings, fine_embeddings], dim=0)
        output.cos_sin = self._cos_sin_from_coords(state.coords)
        output.grid_sizes = GridOutput(
            tensor=torch.tensor(
                [[state.frames, state.fine_grid[0], state.fine_grid[1]]],
                dtype=torch.int32,
                device=output.x.device,
            ),
            tuple=(state.frames, state.fine_grid[0], state.fine_grid[1]),
        )
        return output


class WanRALUQualityPostInfer(WanPostInfer):
    """Keep Wan head outputs packed during the mixed-resolution stage."""

    @torch.no_grad()
    def infer(self, x, pre_infer_out):
        if int(getattr(self.scheduler, "ralu_stage", 1)) == 2:
            return [x.float()]
        return super().infer(x, pre_infer_out)


class WanRALUQualityModel(WanModel):
    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = WanRALUQualityPreInfer
        self.post_infer_class = WanRALUQualityPostInfer


@RUNNER_REGISTER("wan2.1_ralu_quality")
class WanRALUQualityRunner(WanRunner):
    """Full three-stage RALU Quality adaptation for Wan2.1 T2V."""

    def __init__(self, config):
        super().__init__(config)
        self._validate_ralu_config()

    def _validate_ralu_config(self):
        if self.config["task"] != "t2v":
            raise NotImplementedError("wan2.1_ralu_quality currently supports t2v only")
        if self.config.get("feature_caching", "NoCaching") != "NoCaching":
            raise ValueError("RALU packed mixed tokens require feature_caching=NoCaching")
        if self.config.get("seq_parallel", False):
            raise ValueError("RALU packed mixed tokens currently require seq_parallel=false")
        if (
            self.config.get("cpu_offload", False)
            or self.config.get("lazy_load", False)
            or self.config.get("unload_modules", False)
        ):
            raise ValueError("RALU Quality requires resident DiT/VAE weights for both handoffs")
        steps = [int(v) for v in self.config.get("wan_ralu_stage_steps", [5, 6, 7])]
        ends = [float(v) for v in self.config.get("wan_ralu_end_times", [0.3, 0.45, 1.0])]
        shifts = [float(v) for v in self.config.get("wan_ralu_stage_shifts", [10.0, 8.8787, 5.3374])]
        if len(steps) != 3 or len(ends) != 3 or len(shifts) != 3:
            raise ValueError("RALU requires exactly three stage steps/end-times/shifts")
        if steps != [5, 6, 7]:
            raise ValueError(f"this runner is the fixed Quality operating point [5,6,7], got {steps}")
        if not (0.0 < ends[0] < ends[1] < ends[2] == 1.0):
            raise ValueError(f"invalid RALU end times: {ends}")
        if any(value <= 0.0 for value in shifts):
            raise ValueError(f"invalid RALU stage shifts: {shifts}")
        z_value = float(self.config.get("wan_ralu_z", 100.0))
        if z_value < 2.0:
            raise ValueError(f"wan_ralu_z must be at least 2, got {z_value}")
        configured_c = float(self.config.get("wan_ralu_covariance_c", 1.0 / z_value**2))
        if abs(configured_c - 1.0 / z_value**2) > 1e-12:
            raise ValueError("wan_ralu_covariance_c must equal 1 / wan_ralu_z^2")
        ratio = float(self.config.get("wan_ralu_up_ratio", 0.3))
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"wan_ralu_up_ratio must be in (0,1), got {ratio}")
        if int(self.config.get("infer_steps", sum(steps))) != sum(steps):
            raise ValueError(f"infer_steps must equal total RALU NFE {sum(steps)}")
        geometry = {
            "low": tuple(int(v) for v in self.config.get("wan_ralu_low_latent_size", [46, 80])),
            "coarse": tuple(int(v) for v in self.config.get("wan_ralu_coarse_token_grid", [23, 40])),
            "aligned": tuple(int(v) for v in self.config.get("wan_ralu_aligned_latent_size", [92, 160])),
            "output": tuple(int(v) for v in self.config.get("wan_ralu_output_latent_size", [90, 156])),
        }
        expected_geometry = {
            "low": (46, 80),
            "coarse": (23, 40),
            "aligned": (92, 160),
            "output": (90, 156),
        }
        if geometry != expected_geometry:
            raise ValueError(f"RALU Quality geometry A mismatch: {geometry!r}")
        target_pixels = (
            int(self.config.get("target_height", 0)),
            int(self.config.get("target_width", 0)),
        )
        if target_pixels != (720, 1248):
            raise ValueError(f"RALU Quality geometry A expects 720x1248, got {target_pixels}")

    def init_scheduler(self):
        self.scheduler = WanRALUQualityScheduler(self.config)

    def load_transformer(self):
        return WanRALUQualityModel(
            model_path=self.config["model_path"],
            config=self.config,
            device=self.init_device,
        )

    def _stage_sigmas(self) -> list[list[float]]:
        steps = [int(v) for v in self.config["wan_ralu_stage_steps"]]
        ends = [float(v) for v in self.config["wan_ralu_end_times"]]
        shifts = [float(v) for v in self.config["wan_ralu_stage_shifts"]]
        z_value = float(self.config["wan_ralu_z"])
        s2, _, _ = ralu_resume_parameters(ends[0], z_value)
        s3, _, _ = ralu_resume_parameters(ends[1], z_value)
        return [
            ralu_stage_sigmas(start_data_time=0.0, end_data_time=ends[0], num_steps=steps[0], shift=shifts[0]),
            ralu_stage_sigmas(start_data_time=s2, end_data_time=ends[1], num_steps=steps[1], shift=shifts[1]),
            ralu_stage_sigmas(start_data_time=s3, end_data_time=1.0, num_steps=steps[2], shift=shifts[2]),
        ]

    def _set_model_time(self, sigma: float, global_step: int, stage: int):
        scheduler = self.model.scheduler
        scheduler.ralu_stage = int(stage)
        scheduler.step_index = int(global_step)
        scheduler.timestep_input = torch.tensor([sigma * 1000.0], device=AI_DEVICE, dtype=torch.float32)

    @torch.no_grad()
    def _edge_mask(self, clean_lr: torch.Tensor) -> torch.Tensor:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Full RALU requires opencv-python for Canny edge selection") from exc

        decoded = self.vae_decoder.decode(clean_lr.to(GET_DTYPE()))
        if decoded.ndim == 5 and decoded.shape[0] == 1:
            decoded = decoded[0]
        if decoded.ndim != 4 or decoded.shape[0] != 3:
            raise RuntimeError(f"unexpected Wan VAE output for RALU edge selection: {tuple(decoded.shape)}")
        rgb = ((decoded.float().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
        gray = (
            0.299 * rgb[0].float()
            + 0.587 * rgb[1].float()
            + 0.114 * rgb[2].float()
        ).to(torch.uint8).cpu().numpy()
        threshold1 = int(self.config.get("wan_ralu_canny_low", 100))
        threshold2 = int(self.config.get("wan_ralu_canny_high", 200))
        edges = torch.stack(
            [torch.from_numpy(cv2.Canny(frame, threshold1=threshold1, threshold2=threshold2)) for frame in gray],
            dim=0,
        ).to(torch.float32)
        grid_h, grid_w = (int(v) for v in self.config.get("wan_ralu_coarse_token_grid", [23, 40]))
        block_h = edges.shape[-2] // grid_h
        block_w = edges.shape[-1] // grid_w
        if (grid_h * block_h, grid_w * block_w) != tuple(edges.shape[-2:]):
            raise RuntimeError("decoded LR video is not exactly divisible by the RALU coarse token grid")
        per_frame = edges.reshape(edges.shape[0], grid_h, block_h, grid_w, block_w).sum(dim=(2, 4))
        quantile = float(self.config.get("wan_ralu_edge_temporal_quantile", 0.75))
        scores = torch.quantile(per_frame, quantile, dim=0)
        # Stable index-based epsilon makes ties deterministic without random mask jitter.
        scores = scores.flatten() + torch.arange(scores.numel(), dtype=scores.dtype) * 1e-7
        count = max(1, int(scores.numel() * float(self.config.get("wan_ralu_up_ratio", 0.3))))
        selected = torch.topk(scores, count, largest=True).indices
        mask = torch.zeros(scores.numel(), dtype=torch.bool)
        mask[selected] = True
        del decoded, rgb, gray, edges, per_frame
        getattr(torch, AI_DEVICE).empty_cache()
        return mask.reshape(grid_h, grid_w).to(AI_DEVICE)

    def _run_dense_stage(
        self,
        latents: torch.Tensor,
        sigmas: list[float],
        *,
        stage: int,
        global_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        last_velocity = None
        for local_step, (sigma, sigma_next) in enumerate(zip(sigmas, sigmas[1:]), start=1):
            if self.video_segment_num == 1:
                self.check_stop()
            logger.info(
                f"RALU stage={stage} local={local_step}/{len(sigmas)-1} "
                f"global={global_step+1}/{self.model.scheduler.infer_steps} sigma={sigma:.6f}"
            )
            self.model.scheduler.latents = latents
            self._set_model_time(sigma, global_step, stage)
            self.model.infer(self.inputs)
            last_velocity = self.model.scheduler.noise_pred.to(torch.float32)
            latents = (
                latents.to(torch.float32) + last_velocity * float(sigma_next - sigma)
            ).to(latents.dtype)
            global_step += 1
            if self.progress_callback:
                self.progress_callback(global_step / self.model.scheduler.infer_steps * 100.0, 100)
        assert last_velocity is not None
        return latents, last_velocity, global_step

    def _run_mixed_stage(
        self,
        state: RALUPackedState,
        sigmas: list[float],
        *,
        global_step: int,
    ) -> tuple[RALUPackedState, torch.Tensor, int]:
        last_velocity = None
        self.model.scheduler.ralu_packed_state = state
        for local_step, (sigma, sigma_next) in enumerate(zip(sigmas, sigmas[1:]), start=1):
            if self.video_segment_num == 1:
                self.check_stop()
            logger.info(
                f"RALU stage=2 local={local_step}/{len(sigmas)-1} "
                f"global={global_step+1}/{self.model.scheduler.infer_steps} "
                f"sigma={sigma:.6f} packed_tokens={state.values.shape[0]}"
            )
            self.model.scheduler.ralu_packed_state = state
            self._set_model_time(sigma, global_step, 2)
            self.model.infer(self.inputs)
            last_velocity = self.model.scheduler.noise_pred.to(torch.float32)
            state.values = (
                state.values.to(torch.float32) + last_velocity * float(sigma_next - sigma)
            ).to(state.values.dtype)
            global_step += 1
            if self.progress_callback:
                self.progress_callback(global_step / self.model.scheduler.infer_steps * 100.0, 100)
        assert last_velocity is not None
        return state, last_velocity, global_step

    def run_segment(self, segment_idx=0):
        if self.video_segment_num != 1:
            raise NotImplementedError("RALU Quality currently supports a single 81-frame segment")
        scheduler = self.model.scheduler
        stage_sigmas = self._stage_sigmas()
        ends = [float(v) for v in self.config["wan_ralu_end_times"]]
        z_value = float(self.config["wan_ralu_z"])
        global_step = 0

        lr_latents, lr_velocity, global_step = self._run_dense_stage(
            scheduler.latents,
            stage_sigmas[0],
            stage=1,
            global_step=global_step,
        )
        endpoint_sigma = stage_sigmas[0][-1]
        clean_lr = (lr_latents.to(torch.float32) - endpoint_sigma * lr_velocity).to(lr_latents.dtype)
        edge_mask = self._edge_mask(clean_lr)
        packed = first_ralu_handoff(
            lr_latents,
            edge_mask,
            end_data_time=ends[0],
            z_value=z_value,
            generator=scheduler.generator,
        )
        packed, _, global_step = self._run_mixed_stage(
            packed,
            stage_sigmas[1],
            global_step=global_step,
        )
        hr_latents = second_ralu_handoff(
            packed,
            end_data_time=ends[1],
            z_value=z_value,
            generator=scheduler.generator,
            output_latent_size=tuple(self.config.get("wan_ralu_output_latent_size", [90, 156])),
        )
        scheduler.ralu_packed_state = None
        hr_latents, _, global_step = self._run_dense_stage(
            hr_latents,
            stage_sigmas[2],
            stage=3,
            global_step=global_step,
        )
        if global_step != scheduler.infer_steps:
            raise RuntimeError(f"RALU consumed {global_step} evaluations, expected {scheduler.infer_steps}")
        scheduler.latents = hr_latents

        if segment_idx is not None and segment_idx == self.video_segment_num - 1:
            del self.inputs
            getattr(torch, AI_DEVICE).empty_cache()
        return hr_latents
