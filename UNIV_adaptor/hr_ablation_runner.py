"""One LR prefix and transition, followed by independently discretized HR branches."""
from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path

import torch
from loguru import logger
from lightx2v.utils.registry_factory import RUNNER_REGISTER

from .hr_refinement import install_hr_grid
from .wan_runner import WanUniversalRGBPipelineRunner


def tensor_sha256(tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def synchronize(tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


@RUNNER_REGISTER("wan2.1_univ_hr_ablation")
class WanHRRefinementAblationRunner(WanUniversalRGBPipelineRunner):
    """Single-prompt experiment. A new runner is required for a different request."""

    def __init__(self, config):
        super().__init__(config)
        if config.get("cpu_offload", False) or config.get("compile", False):
            raise ValueError("HR ablation requires resident, uncompiled model weights")
        self.hr_steps = 10
        self.shared_boundary = None
        self.shared_identity = None
        self.shared_record = None

    def _identity(self):
        return (
            str(self.input_info.prompt),
            str(getattr(self.input_info, "negative_prompt", "")),
            int(self.input_info.seed),
            tuple(self.model.scheduler.univ_schedule.target_latent_shape),
        )

    def run_segment(self, segment_idx=0):
        if self.video_segment_num != 1:
            raise ValueError("HR ablation supports exactly one video segment")
        scheduler = self.model.scheduler
        identity = self._identity()
        reference_infer_steps = scheduler.infer_steps
        reused = self.shared_boundary is not None
        try:
            if not reused:
                self.shared_identity = identity
                synchronize(scheduler.latents)
                self._prefix_started = time.perf_counter()
                result = super().run_segment(segment_idx)
                self.shared_record = copy.deepcopy(self.univ_runtime_record)
            else:
                if identity != self.shared_identity:
                    raise ValueError("shared HR boundary cannot be reused for a different request")
                elapsed = self._run_hr_suffix(scheduler.univ_schedule)
                self.univ_runtime_record = copy.deepcopy(self.shared_record)
                self.univ_runtime_record["timing_seconds"] = {
                    "lr_full_compute": 0.0,
                    "lr_cache_reuse": 0.0,
                    "transition": 0.0,
                    "transition_diagnostics": 0.0,
                    "hr_full_compute": elapsed,
                }
                result = scheduler.latents
                del self.inputs

            self.univ_runtime_record["schema"] = "wan_univ_hr_ablation_v1"
            self.univ_runtime_record["reference_schedule"] = self.univ_runtime_record.pop("schedule")
            self.univ_runtime_record["hr_schedule"] = self.hr_grid
            self.univ_runtime_record["shared_boundary"] = {
                "tensor_sha256": self.boundary_sha256,
                "path": str(self.boundary_path),
                "shape": list(self.shared_boundary.shape),
                "dtype": str(self.shared_boundary.dtype),
                "reused": reused,
                "prefix_and_transition_seconds": self.shared_prefix_seconds,
            }
            self._write_runtime_record()
            return result
        finally:
            # prepare_latents resolves the next run against the 50-step reference.
            scheduler.infer_steps = reference_infer_steps

    def _run_hr_suffix(self, schedule):
        scheduler = self.model.scheduler
        if self.shared_boundary is None:
            synchronize(scheduler.latents)
            self.shared_prefix_seconds = time.perf_counter() - self._prefix_started
            self.shared_boundary = scheduler.latents.detach().cpu().clone()
            self.reference_sigmas = scheduler.sigmas.detach().cpu().clone()
            self.reference_timesteps = scheduler.timesteps.detach().clone()
            self.boundary_sha256 = tensor_sha256(self.shared_boundary)
            self.boundary_path = Path(self.config["univ_hr_boundary_path"])
            self.boundary_path.parent.mkdir(parents=True, exist_ok=True)
            if self.boundary_path.exists():
                raise FileExistsError(f"boundary already exists: {self.boundary_path}")
            torch.save(
                {
                    "schema": "univ_shared_hr_boundary_v1",
                    "state": self.shared_boundary,
                    "tensor_sha256": self.boundary_sha256,
                    "boundary_step": schedule.switch_step,
                    "boundary_sigma": float(self.reference_sigmas[schedule.switch_step]),
                    "reference_sigmas": self.reference_sigmas,
                    "prompt": self.shared_identity[0],
                    "negative_prompt": self.shared_identity[1],
                    "seed": self.shared_identity[2],
                    "action": dict(self.config["univ_action"]),
                },
                self.boundary_path,
            )

        device = scheduler.latents.device
        scheduler.latents = self.shared_boundary.to(device=device).clone()
        # Also verify the actual branch input, rather than repeating a stored hash.
        if tensor_sha256(scheduler.latents) != self.boundary_sha256:
            raise RuntimeError("HR branch input differs from the shared boundary")
        scheduler.timesteps = self.reference_timesteps.clone()
        self.hr_grid = install_hr_grid(
            scheduler,
            reference_sigmas=self.reference_sigmas.tolist(),
            boundary_step=schedule.switch_step,
            hr_steps=self.hr_steps,
        )
        synchronize(scheduler.latents)
        started = time.perf_counter()
        for position, step_index in enumerate(self.hr_grid["compute_indices"]):
            self.check_stop()
            scheduler.step_pre(step_index=step_index)
            self.model.infer(self.inputs)
            scheduler.step_post()
            logger.info(f"HR ablation {self.hr_steps} steps: {position + 1}/{self.hr_steps}")
            if self.progress_callback:
                self.progress_callback(100 * (position + 1) / self.hr_steps, 100)
        synchronize(scheduler.latents)
        elapsed = time.perf_counter() - started
        if not bool(torch.isfinite(scheduler.latents).all()):
            raise RuntimeError(f"non-finite output from HR{self.hr_steps}")
        return elapsed
