"""Full-LR endpoint followed by independent direct-sigma HR refinement branches."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from loguru import logger
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import RUNNER_REGISTER

from .data_protocol import canonical_sha256
from .hr_ablation_runner import synchronize, tensor_sha256
from .hr_refinement import install_direct_hr_grid, install_lr_grid
from .wan_runner import WanUniversalRGBPipelineRunner


@RUNNER_REGISTER("wan2.1_univ_mrflow_ablation")
@RUNNER_REGISTER("wan2.1_univ_mrflow_budget")
class WanMrFlowRefinementAblationRunner(WanUniversalRGBPipelineRunner):
    """Run one true LR grid, then reuse its clean HR transition across branches."""

    def __init__(self, config):
        super().__init__(config)
        if config.get("cpu_offload", False) or config.get("compile", False):
            raise ValueError(
                "MrFlow ablation requires resident, uncompiled model weights"
            )
        self.refine_sigma = float(config.get("univ_mrflow_refine_sigma", 0.0))
        self.hr_steps = int(config.get("univ_mrflow_hr_steps", 0))
        self.lr_steps = int(config.get("univ_mrflow_lr_steps", 50))
        self.reuse_shared_endpoint = bool(
            config.get("univ_mrflow_reuse_endpoint", True)
        )
        self.endpoint_state_dtype = str(
            config.get("univ_mrflow_endpoint_state_dtype", "original")
        )
        if not 1 <= self.lr_steps <= 50:
            raise ValueError("univ_mrflow_lr_steps must be in [1, 50]")
        if self.hr_steps < 0:
            raise ValueError("univ_mrflow_hr_steps must be non-negative")
        if self.hr_steps == 0 and self.refine_sigma != 0.0:
            raise ValueError("HR0 requires univ_mrflow_refine_sigma=0")
        if self.hr_steps > 0 and not 0.0 < self.refine_sigma < 1.0:
            raise ValueError("MRFlow refinement sigma must be in (0, 1)")
        if self.endpoint_state_dtype not in {"original", "fp16", "bf16", "fp32"}:
            raise ValueError(
                "univ_mrflow_endpoint_state_dtype must be original, fp16, bf16, or fp32"
            )
        self.shared_clean_lr = None
        self.shared_clean_hr = None
        self.shared_hr_noise = None
        self.shared_identity = None
        self.shared_record = None
        self.shared_lr_grid = None
        self.shared_boundary_path = None
        self.shared_archive_hashes = None

    def reset_shared_endpoint(self) -> None:
        """Discard one LR-grid endpoint before starting a different LR schedule."""
        self.shared_clean_lr = None
        self.shared_clean_hr = None
        self.shared_hr_noise = None
        self.shared_identity = None
        self.shared_record = None
        self.shared_lr_grid = None
        self.shared_boundary_path = None
        self.shared_archive_hashes = None

    def _boundary_path(self) -> Path:
        configured = str(self.config.get("univ_mrflow_boundary_path", "")).strip()
        if configured:
            return Path(configured)
        output = str(getattr(self.input_info, "save_result_path", "")).strip()
        if not output:
            raise ValueError(
                "MRFlow endpoint saving requires univ_mrflow_boundary_path or save_result_path"
            )
        path = Path(output)
        return path.with_suffix(path.suffix + ".endpoint.pt")

    def _archive_tensor(self, value):
        archive_dtype = getattr(self, "endpoint_state_dtype", "original")
        dtype = {
            "original": value.dtype,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[archive_dtype]
        return value.detach().to(device="cpu", dtype=dtype).contiguous()

    def _identity(self, schedule):
        return (
            str(self.input_info.prompt),
            str(getattr(self.input_info, "negative_prompt", "")),
            int(self.input_info.seed),
            tuple(schedule.target_latent_shape),
            int(self.lr_steps),
        )

    def _complete_lr_and_transition(self, scheduler, schedule):
        if schedule.switch_step != schedule.reference_nfe or schedule.hr_compute_steps:
            raise ValueError(
                "MrFlow ablation requires switch_ratio=1.0 before LR resampling"
            )
        if tuple(schedule.lr_compute_steps) != tuple(range(schedule.reference_nfe)):
            raise ValueError(
                "disable LR prediction caching; reduced-LR grids fully compute retained positions"
            )

        lr_steps = int(self.lr_steps)
        if not 1 <= lr_steps <= schedule.reference_nfe:
            raise ValueError("MrFlow LR steps must be in [1, reference_nfe]")
        lr_grid = install_lr_grid(
            scheduler,
            reference_sigmas=scheduler.sigmas.tolist(),
            lr_steps=lr_steps,
        )

        synchronize(scheduler.latents)
        lr_started = time.perf_counter()
        for position, step_index in enumerate(lr_grid["compute_indices"]):
            self.check_stop()
            scheduler.step_pre(step_index=step_index)
            self.model.infer(self.inputs)
            # Unlike the normal handoff runner, include the final solver update to sigma=0.
            scheduler.step_post()
            logger.info(f"MrFlow ablation LR endpoint: {position + 1}/{lr_steps}")
            if self.progress_callback:
                self.progress_callback(80 * (position + 1) / lr_steps, 100)
        synchronize(scheduler.latents)
        lr_seconds = time.perf_counter() - lr_started
        if float(scheduler.sigmas[lr_steps]) != 0.0:
            raise RuntimeError(f"LR{lr_steps} did not terminate at sigma=0")

        clean_lr = scheduler.latents.detach().clone()
        transition_started = time.perf_counter()
        spatial_needed = clean_lr.shape[-2:] != schedule.target_latent_shape[-2:]
        transition = self._build_transition(spatial_needed=spatial_needed)
        result = transition.lift(
            clean_lr.to(GET_DTYPE()), target_latent_shape=schedule.target_latent_shape
        )
        synchronize(result.clean_hr)
        transition_seconds = time.perf_counter() - transition_started

        self.shared_clean_lr = clean_lr.detach().cpu().clone()
        self.shared_clean_hr = result.clean_hr.detach().cpu().clone()
        self.shared_hr_noise = scheduler.univ_hr_noise.detach().cpu().clone()
        self.clean_lr_sha256 = tensor_sha256(self.shared_clean_lr)
        self.clean_hr_sha256 = tensor_sha256(self.shared_clean_hr)
        self.hr_noise_sha256 = tensor_sha256(self.shared_hr_noise)
        self.shared_lr_seconds = lr_seconds
        self.shared_lr_grid = lr_grid
        self.shared_transition_seconds = transition_seconds
        self.shared_transition_record = {
            "baseline": result.baseline,
            "source_latent_shape": list(result.source_latent_shape),
            "target_latent_shape": list(result.target_latent_shape),
            "decoded_frames": result.decoded_frames,
            "reconstructed_frames": result.reconstructed_frames,
            "source_height": result.source_height,
            "source_width": result.source_width,
            "target_height": result.target_height,
            "target_width": result.target_width,
            "spatial_restore_applied": result.spatial_restore_applied,
            "temporal_restore_applied": result.temporal_restore_applied,
        }

        boundary = self._boundary_path()
        boundary.parent.mkdir(parents=True, exist_ok=True)
        if boundary.exists() and getattr(self, "reuse_shared_endpoint", True):
            raise FileExistsError(f"MrFlow shared boundary already exists: {boundary}")
        archive_clean_lr = self._archive_tensor(self.shared_clean_lr)
        archive_clean_hr = self._archive_tensor(self.shared_clean_hr)
        archive_hr_noise = self._archive_tensor(self.shared_hr_noise)
        archive_hashes = {
            "clean_lr_sha256": tensor_sha256(archive_clean_lr),
            "clean_hr_sha256": tensor_sha256(archive_clean_hr),
            "hr_noise_sha256": tensor_sha256(archive_hr_noise),
        }
        endpoint_payload = {
            "schema": "univ_mrflow_clean_transition_v1",
            "archive_dtype": getattr(self, "endpoint_state_dtype", "original"),
            "clean_lr": archive_clean_lr,
            "clean_hr": archive_clean_hr,
            "hr_noise": archive_hr_noise,
            **archive_hashes,
            "runtime_tensor_sha256": {
                "clean_lr": self.clean_lr_sha256,
                "clean_hr": self.clean_hr_sha256,
                "hr_noise": self.hr_noise_sha256,
            },
            "lr_steps": lr_steps,
            "reference_nfe": schedule.reference_nfe,
            "lr_schedule": lr_grid,
            "lr_endpoint_sigma": 0.0,
            "prompt": self.shared_identity[0],
            "prompt_sha256": canonical_sha256(self.shared_identity[0]),
            "negative_prompt": self.shared_identity[1],
            "seed": self.shared_identity[2],
            "artifact_id": str(self.config.get("univ_low_budget_artifact_id", "")),
            "action_key": str(self.config.get("univ_low_budget_action_key", "")),
            "action": dict(self.config["univ_action"]),
            "mrflow_refinement": {
                "renoise_sigma": float(self.refine_sigma),
                "hr_steps": int(self.hr_steps),
            },
            "transition": self.shared_transition_record,
        }
        temporary = boundary.with_name(f".{boundary.name}.tmp.{id(self)}")
        torch.save(endpoint_payload, temporary)
        temporary.replace(boundary)
        self.shared_boundary_path = boundary
        self.shared_archive_hashes = archive_hashes
        del clean_lr, result

    def _run_direct_refinement(self, scheduler):
        device = scheduler.latents.device
        clean_hr = self.shared_clean_hr.to(device=device)
        noise = scheduler.univ_hr_noise
        if tensor_sha256(noise) != self.hr_noise_sha256:
            raise RuntimeError(
                "prepared HR noise differs from the shared coordinate field"
            )
        sigma = float(self.refine_sigma)
        steps = int(self.hr_steps)
        if steps == 0:
            if sigma != 0.0:
                raise ValueError("the transition-only branch requires sigma=0")
            scheduler.latents = clean_hr.clone()
            scheduler.reset_solver_history()
            grid = {
                "grid_policy": "transition_only",
                "start_sigma": 0.0,
                "hr_steps": 0,
                "sigmas": [0.0],
                "model_timesteps": [],
                "compute_indices": [],
            }
            return grid, 0.0, tensor_sha256(scheduler.latents)
        if not 0.0 < sigma < 1.0:
            raise ValueError("a refinement branch requires sigma in (0, 1)")

        scheduler.latents = self._renoise(clean_hr, noise, sigma)
        branch_start_sha256 = tensor_sha256(scheduler.latents)
        grid = install_direct_hr_grid(scheduler, start_sigma=sigma, hr_steps=steps)
        synchronize(scheduler.latents)
        started = time.perf_counter()
        for position, step_index in enumerate(grid["compute_indices"]):
            self.check_stop()
            scheduler.step_pre(step_index=step_index)
            self.model.infer(self.inputs)
            scheduler.step_post()
            logger.info(
                f"MrFlow ablation sigma={sigma:.3f}, HR={steps}: {position + 1}/{steps}"
            )
            if self.progress_callback:
                self.progress_callback(80 + 20 * (position + 1) / steps, 100)
        synchronize(scheduler.latents)
        elapsed = time.perf_counter() - started
        if not bool(torch.isfinite(scheduler.latents).all()):
            raise RuntimeError(
                f"non-finite direct-sigma output for sigma={sigma}, HR={steps}"
            )
        return grid, elapsed, branch_start_sha256

    def run_segment(self, segment_idx=0):
        if self.video_segment_num != 1:
            raise ValueError("MrFlow ablation supports exactly one video segment")
        scheduler = self.model.scheduler
        schedule = scheduler.univ_schedule
        reference_infer_steps = scheduler.infer_steps
        identity = self._identity(schedule)
        reused = self.shared_clean_hr is not None
        try:
            if not reused:
                self.shared_identity = identity
                self._complete_lr_and_transition(scheduler, schedule)
            elif identity != self.shared_identity:
                raise ValueError(
                    "shared MrFlow transition cannot be reused for a different request"
                )

            grid, hr_seconds, branch_start_sha256 = self._run_direct_refinement(
                scheduler
            )
            self.univ_runtime_record = {
                "schema": "wan_univ_mrflow_ablation_v1",
                "prompt": identity[0],
                "seed": identity[2],
                "artifact_id": str(self.config.get("univ_low_budget_artifact_id", "")),
                "action_key": str(self.config.get("univ_low_budget_action_key", "")),
                "model_path": str(self.config.get("model_path", "")),
                "action": dict(self.config["univ_action"]),
                "reference_schedule": schedule.as_dict(),
                "lr_endpoint": {
                    "steps": int(self.lr_steps),
                    "reference_nfe": schedule.reference_nfe,
                    "sigma": 0.0,
                    "final_step_post_completed": True,
                    "clean_lr_sha256": self.clean_lr_sha256,
                    "lr_schedule": copy.deepcopy(self.shared_lr_grid),
                },
                "transition": copy.deepcopy(self.shared_transition_record),
                "endpoint_state": {
                    "schema": "univ_mrflow_clean_transition_v1",
                    "path": str(self.shared_boundary_path),
                    "archive_dtype": getattr(self, "endpoint_state_dtype", "original"),
                    "seed": identity[2],
                    "prompt_sha256": canonical_sha256(identity[0]),
                    **self.shared_archive_hashes,
                    "runtime_tensor_sha256": {
                        "clean_lr": self.clean_lr_sha256,
                        "clean_hr": self.clean_hr_sha256,
                        "hr_noise": self.hr_noise_sha256,
                    },
                },
                "shared_clean_hr": {
                    "path": str(self.shared_boundary_path),
                    "clean_hr_sha256": self.clean_hr_sha256,
                    "hr_noise_sha256": self.hr_noise_sha256,
                    "branch_start_sha256": branch_start_sha256,
                    "reused": reused,
                },
                "hr_schedule": grid,
                "timing_seconds": {
                    "lr_full_compute": 0.0 if reused else self.shared_lr_seconds,
                    "transition": 0.0 if reused else self.shared_transition_seconds,
                    "hr_full_compute": hr_seconds,
                    "candidate_denoise": self.shared_lr_seconds
                    + self.shared_transition_seconds
                    + hr_seconds,
                },
            }
            self._write_runtime_record()
            return scheduler.latents
        finally:
            scheduler.infer_steps = reference_infer_steps
            if not getattr(self, "reuse_shared_endpoint", True):
                self.reset_shared_endpoint()
