from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from loguru import logger

from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

from .diagnostics import transition_state_diagnostics
from .flow import wan_clean_from_velocity, wan_renoise
from .noise import coordinate_gaussian_tensor
from .schedule import action_from_config, resolve_schedule
from .transition import (
    DVG_LATENT_ANCHOR,
    RGB_SR_VAE,
    TRANSITION_BASELINES,
    WanDVGAnchorTransition,
    WanRGBSRTransition,
)


class WanUniversalScheduler(WanScheduler):
    """Wan scheduler with a low-grid prefix and coordinate-aligned HR noise."""

    def prepare_latents(self, seed, latent_shape, dtype=torch.float32):
        target_shape = tuple(int(value) for value in latent_shape)
        action = action_from_config(self.config)
        self.univ_schedule = resolve_schedule(
            action,
            reference_nfe=int(self.infer_steps),
            target_latent_shape=target_shape,
        )
        self.univ_seed = int(seed)
        self.univ_hr_noise = coordinate_gaussian_tensor(
            target_shape,
            seed=self.univ_seed,
            device=torch.device(AI_DEVICE),
            dtype=dtype,
        )
        self.latents = coordinate_gaussian_tensor(
            self.univ_schedule.low_latent_shape,
            seed=self.univ_seed,
            reference_shape=target_shape,
            device=torch.device(AI_DEVICE),
            dtype=dtype,
        )
        logger.info(
            "UNIV coordinate noise prepared: "
            f"LR={self.univ_schedule.low_latent_shape}, HR={target_shape}"
        )

    def reset_solver_history(self) -> None:
        """Drop LR-shaped UniPC history before the HR suffix."""

        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.last_sample = None
        self.noise_pred = None
        self.this_order = None
        self.lower_order_nums = 0


class WanCoordinateNativeScheduler(WanScheduler):
    """Native Wan scheduler initialized with the UNIV coordinate noise field."""

    def prepare_latents(self, seed, latent_shape, dtype=torch.float32):
        target_shape = tuple(int(value) for value in latent_shape)
        self.latents = coordinate_gaussian_tensor(
            target_shape,
            seed=int(seed),
            device=torch.device(AI_DEVICE),
            dtype=dtype,
        )
        logger.info(f"UNIV validation native coordinate noise: shape={target_shape}")


@RUNNER_REGISTER("wan2.1_univ_native")
class WanCoordinateNativeRunner(WanRunner):
    """Unmodified 50-step HR Wan execution with coordinate-aligned noise."""

    def init_scheduler(self):
        self.scheduler = WanCoordinateNativeScheduler(self.config)

    def load_transformer(self):
        return WanModel(
            model_path=self.config["model_path"],
            config=self.config,
            device=self.init_device,
            model_type="wan2.1",
        )


@RUNNER_REGISTER("wan2.1_univ_pipeline")
@RUNNER_REGISTER("wan2.1_univ_rgb_pipeline")
class WanUniversalRGBPipelineRunner(WanRunner):
    """Complete LR/cache -> selectable transition -> HR suffix runner."""

    def __init__(self, config):
        super().__init__(config)
        self._validate_univ_config()
        self._univ_transition = None
        self.univ_runtime_record: dict[str, object] = {}

    def _validate_univ_config(self) -> None:
        if self.config["task"] != "t2v":
            raise NotImplementedError("wan2.1_univ_pipeline currently supports T2V only")
        if int(self.config["infer_steps"]) != 50:
            raise ValueError(
                "the first UNIV implementation uses the Wan 50-step reference schedule"
            )
        if self.config.get("feature_caching", "NoCaching") != "NoCaching":
            raise ValueError(
                "set feature_caching=NoCaching; UNIV owns the exact LR cache policy"
            )
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            raise ValueError("the first UNIV runner does not support lazy/unload modules")
        if self.config.get("parallel", False):
            raise ValueError("the first UNIV transition runner is single-device only")
        cache_mode = str(self.config.get("univ_cache_mode", "residual"))
        if cache_mode not in {"residual", "velocity"}:
            raise ValueError("univ_cache_mode must be 'residual' or 'velocity'")
        baseline = str(self.config.get("univ_transition_baseline", RGB_SR_VAE))
        if baseline not in TRANSITION_BASELINES:
            raise ValueError(
                "univ_transition_baseline must be one of "
                f"{sorted(TRANSITION_BASELINES)}, got {baseline!r}"
            )
        if baseline == RGB_SR_VAE and self.config.get("use_tae", False):
            raise ValueError("rgb_sr_vae requires the full Wan VAE codec, not TAE")
        action_from_config(self.config)

    def init_scheduler(self):
        self.scheduler = WanUniversalScheduler(self.config)

    def load_transformer(self):
        return WanModel(
            model_path=self.config["model_path"],
            config=self.config,
            device=self.init_device,
            model_type="wan2.1",
        )

    def _cached_prediction(self, current_latents, cached_value):
        mode = str(self.config.get("univ_cache_mode", "residual"))
        if mode == "velocity":
            return cached_value.to(device=current_latents.device, dtype=current_latents.dtype)
        return (current_latents.to(torch.float32) + cached_value).to(
            dtype=current_latents.dtype
        )

    def _new_cache_value(self, current_latents, prediction):
        mode = str(self.config.get("univ_cache_mode", "residual"))
        if mode == "velocity":
            return prediction.detach().to(torch.float32)
        return (
            prediction.detach().to(torch.float32)
            - current_latents.detach().to(torch.float32)
        )

    def _build_transition(self, *, spatial_needed: bool):
        if self._univ_transition is not None:
            return self._univ_transition
        baseline = str(self.config.get("univ_transition_baseline", RGB_SR_VAE))
        if baseline == DVG_LATENT_ANCHOR:
            self._univ_transition = WanDVGAnchorTransition()
            return self._univ_transition

        resolver = None
        if spatial_needed:
            from .rgb_super_resolution import (
                build_univ_rgb_super_resolver,
            )

            resolver = build_univ_rgb_super_resolver(self.config)
        self._univ_transition = WanRGBSRTransition(
            vae_codec=self.vae_decoder,
            spatial_resolver=resolver,
            target_height=int(self.config["target_height"]),
            target_width=int(self.config["target_width"]),
        )
        return self._univ_transition

    @staticmethod
    def _renoise(clean_hr, hr_noise, sigma):
        sigma = torch.as_tensor(sigma, device=clean_hr.device, dtype=torch.float32)
        return wan_renoise(
            clean_hr.to(torch.float32),
            hr_noise.to(device=clean_hr.device, dtype=torch.float32),
            sigma,
        ).to(dtype=clean_hr.dtype)

    def _load_native_hr_state(
        self,
        *,
        expected_shape,
        device,
        boundary_step: int,
        boundary_sigma: float,
    ):
        path_value = str(self.config.get("univ_native_hr_state_path", "")).strip()
        if not path_value:
            return None, None
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"native HR state does not exist: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        reference_record = {
            "path": str(path),
            "tensor_key": None,
            "validated_boundary_fields": [],
        }
        if isinstance(payload, dict):
            if "boundary_step" in payload:
                recorded_step = int(payload["boundary_step"])
                if recorded_step != int(boundary_step):
                    raise ValueError(
                        "native HR boundary_step mismatch: "
                        f"checkpoint={recorded_step}, runtime={boundary_step}"
                    )
                reference_record["validated_boundary_fields"].append("boundary_step")
            if "boundary_sigma" in payload:
                recorded_sigma = float(payload["boundary_sigma"])
                if abs(recorded_sigma - float(boundary_sigma)) > 1e-6:
                    raise ValueError(
                        "native HR boundary_sigma mismatch: "
                        f"checkpoint={recorded_sigma}, runtime={boundary_sigma}"
                    )
                reference_record["validated_boundary_fields"].append("boundary_sigma")
            requested_key = str(self.config.get("univ_native_hr_state_key", "state"))
            candidate_keys = (requested_key, "state", "latents", "sample")
            for key in candidate_keys:
                if key in payload:
                    payload = payload[key]
                    reference_record["tensor_key"] = key
                    break
            else:
                raise KeyError(
                    f"native HR checkpoint {path} has none of keys {candidate_keys}"
                )
        if not torch.is_tensor(payload):
            raise TypeError(f"native HR state must be a tensor, got {type(payload)!r}")
        if payload.ndim == 5 and int(payload.shape[0]) == 1:
            payload = payload[0]
        if tuple(payload.shape) != tuple(expected_shape):
            raise ValueError(
                "native HR state shape mismatch: "
                f"got {tuple(payload.shape)}, expected {tuple(expected_shape)}"
            )
        logger.info(f"UNIV native HR diagnostic state loaded: {path}")
        return payload.to(device=device), reference_record

    def _write_runtime_record(self) -> None:
        output = getattr(self.input_info, "save_result_path", None)
        if not output:
            return
        path = Path(str(output))
        sidecar = path.with_suffix(path.suffix + ".univ.json")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(self.univ_runtime_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"UNIV runtime record: {sidecar}")

    def run_segment(self, segment_idx=0):
        scheduler: WanUniversalScheduler = self.model.scheduler
        schedule = scheduler.univ_schedule
        compute_steps = set(schedule.lr_compute_steps)
        cached_value = None
        lr_compute_seconds = 0.0
        lr_cache_seconds = 0.0
        hr_compute_seconds = 0.0
        transition_seconds = 0.0
        diagnostics_seconds = 0.0

        logger.info(f"UNIV resolved schedule: {json.dumps(schedule.as_dict())}")
        for step_index in schedule.lr_solver_steps:
            if self.video_segment_num == 1:
                self.check_stop()
            scheduler.step_pre(step_index=step_index)
            current_latents = scheduler.latents
            start = time.perf_counter()
            if step_index in compute_steps:
                self.model.infer(self.inputs)
                cached_value = self._new_cache_value(
                    current_latents, scheduler.noise_pred
                )
                lr_compute_seconds += time.perf_counter() - start
                mode = "compute"
            else:
                if cached_value is None:
                    raise RuntimeError("cache reuse requested before cache initialization")
                scheduler.noise_pred = self._cached_prediction(
                    current_latents, cached_value
                )
                lr_cache_seconds += time.perf_counter() - start
                mode = "cache"
            logger.info(
                f"==> UNIV LR step {step_index + 1}/{schedule.switch_step}: {mode}"
            )

            if step_index + 1 == schedule.switch_step:
                # The final LR position is forced to compute by the planner.
                sigma_current = scheduler.sigmas[step_index].to(
                    device=current_latents.device, dtype=torch.float32
                )
                clean_lr = wan_clean_from_velocity(
                    current_latents.to(torch.float32),
                    scheduler.noise_pred.to(torch.float32),
                    sigma_current,
                ).to(dtype=current_latents.dtype)
                break
            scheduler.step_post()
            if self.progress_callback:
                self.progress_callback(((step_index + 1) / schedule.reference_nfe) * 100, 100)
        else:
            raise RuntimeError("UNIV LR loop ended without producing a transition state")

        cached_value = None
        del current_latents
        transition_start = time.perf_counter()
        spatial_needed = clean_lr.shape[-2:] != schedule.target_latent_shape[-2:]
        transition = self._build_transition(spatial_needed=spatial_needed)
        transition_result = transition.lift(
            clean_lr.to(GET_DTYPE()),
            target_latent_shape=schedule.target_latent_shape,
        )
        boundary_sigma = scheduler.sigmas[schedule.switch_step]
        scheduler.latents = self._renoise(
            transition_result.clean_hr,
            scheduler.univ_hr_noise,
            boundary_sigma,
        )
        scheduler.reset_solver_history()
        transition_seconds = time.perf_counter() - transition_start

        diagnostics_enabled = bool(
            self.config.get("univ_enable_transition_diagnostics", True)
        )
        native_hr_reference = None
        if diagnostics_enabled:
            diagnostics_start = time.perf_counter()
            native_hr_state, native_hr_reference = self._load_native_hr_state(
                expected_shape=schedule.target_latent_shape,
                device=scheduler.latents.device,
                boundary_step=schedule.switch_step,
                boundary_sigma=float(boundary_sigma),
            )
            diagnostics = transition_state_diagnostics(
                clean_lr=clean_lr,
                clean_hr=transition_result.clean_hr,
                renoised_hr=scheduler.latents,
                native_hr_state=native_hr_state,
            )
            del native_hr_state
            diagnostics_seconds = time.perf_counter() - diagnostics_start
        else:
            diagnostics = {
                "schema": "univ_transition_diagnostics_v1",
                "enabled": False,
                "reason": "disabled_for_unbiased_timing",
            }
        transition_record = {
            "baseline": transition_result.baseline,
            "source_latent_shape": list(transition_result.source_latent_shape),
            "target_latent_shape": list(transition_result.target_latent_shape),
            "decoded_frames": transition_result.decoded_frames,
            "reconstructed_frames": transition_result.reconstructed_frames,
            "source_height": transition_result.source_height,
            "source_width": transition_result.source_width,
            "target_height": transition_result.target_height,
            "target_width": transition_result.target_width,
            "spatial_restore_applied": transition_result.spatial_restore_applied,
            "temporal_restore_applied": transition_result.temporal_restore_applied,
            "boundary_sigma": float(boundary_sigma),
            "native_hr_reference": native_hr_reference,
        }
        del clean_lr, transition_result
        if torch.cuda.is_available() and scheduler.latents.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info(
            "==> UNIV transition complete: "
            f"boundary={schedule.switch_step}/{schedule.reference_nfe}, "
            f"sigma={float(boundary_sigma)}, shape={tuple(scheduler.latents.shape)}"
        )

        hr_compute_seconds = self._run_hr_suffix(schedule)

        self.univ_runtime_record = {
            "schema": "wan_univ_pipeline_v2",
            "prompt": str(getattr(self.input_info, "prompt", "")),
            "seed": int(getattr(self.input_info, "seed", scheduler.univ_seed)),
            "model_path": str(self.config.get("model_path", "")),
            "rgb_sr_checkpoint": str(self.config.get("wan_rgb_sr_checkpoint", "")),
            "action": dict(self.config["univ_action"]),
            "schedule": schedule.as_dict(),
            "cache_mode": str(self.config.get("univ_cache_mode", "residual")),
            "cfg_enabled": bool(self.config.get("enable_cfg", False)),
            "physical_dit_pass_multiplier": 2
            if self.config.get("enable_cfg", False)
            else 1,
            "transition": transition_record,
            "transition_diagnostics": diagnostics,
            "timing_seconds": {
                "lr_full_compute": lr_compute_seconds,
                "lr_cache_reuse": lr_cache_seconds,
                "transition": transition_seconds,
                "transition_diagnostics": diagnostics_seconds,
                "hr_full_compute": hr_compute_seconds,
            },
        }
        self._write_runtime_record()

        if segment_idx is not None and segment_idx == self.video_segment_num - 1:
            del self.inputs
            getattr(torch, AI_DEVICE).empty_cache()
        return scheduler.latents

    def _run_hr_suffix(self, schedule):
        """Run the reference suffix; experiments may override this boundary hook."""
        scheduler = self.model.scheduler
        elapsed = 0.0
        for step_index in schedule.hr_compute_steps:
            if self.video_segment_num == 1:
                self.check_stop()
            start = time.perf_counter()
            scheduler.step_pre(step_index=step_index)
            self.model.infer(self.inputs)
            scheduler.step_post()
            elapsed += time.perf_counter() - start
            logger.info(
                f"==> UNIV HR full compute {step_index + 1}/{schedule.reference_nfe}"
            )
            if self.progress_callback:
                self.progress_callback(((step_index + 1) / schedule.reference_nfe) * 100, 100)
        if not schedule.hr_compute_steps and self.progress_callback:
            self.progress_callback(100, 100)
        return elapsed
