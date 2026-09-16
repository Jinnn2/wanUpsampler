"""Wan runner for the LR-policy/restart/HR-budget existence experiment."""

from __future__ import annotations

import copy
import time

import torch
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from loguru import logger

from .data_protocol import canonical_sha256
from .flow import wan_clean_from_velocity
from .hr_ablation_runner import synchronize, tensor_sha256
from .mrflow_ablation_runner import WanMrFlowRefinementAblationRunner
from .online_policy import full_compute_steps


def _tensor_features(value: torch.Tensor) -> dict[str, object]:
    x = value.detach().to(torch.float32)
    result: dict[str, object] = {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "rms": float(x.square().mean().sqrt()),
        "abs_max": float(x.abs().max()),
    }
    if x.ndim == 4:
        reduce_dims = (1, 2, 3)
        result["channel_mean"] = [float(v) for v in x.mean(dim=reduce_dims).cpu()]
        result["channel_std"] = [
            float(v) for v in x.std(dim=reduce_dims, unbiased=False).cpu()
        ]
        if x.shape[1] > 1:
            temporal = x[:, 1:] - x[:, :-1]
            result["temporal_diff_mae"] = float(temporal.abs().mean())
            result["temporal_diff_rms"] = float(temporal.square().mean().sqrt())
        else:
            result["temporal_diff_mae"] = 0.0
            result["temporal_diff_rms"] = 0.0
        spatial_terms = []
        if x.shape[2] > 1:
            spatial_terms.append((x[:, :, 1:] - x[:, :, :-1]).flatten())
        if x.shape[3] > 1:
            spatial_terms.append((x[:, :, :, 1:] - x[:, :, :, :-1]).flatten())
        if spatial_terms:
            spatial = torch.cat(spatial_terms)
            result["spatial_diff_mae"] = float(spatial.abs().mean())
            result["spatial_diff_rms"] = float(spatial.square().mean().sqrt())
        else:
            result["spatial_diff_mae"] = 0.0
            result["spatial_diff_rms"] = 0.0
    return result


def _prompt_context_features(inputs: object) -> dict[str, object]:
    if not isinstance(inputs, dict):
        return {"available": False}
    text = inputs.get("text_encoder_output")
    if not isinstance(text, dict) or not torch.is_tensor(text.get("context")):
        return {"available": False}
    context = text["context"].detach().to(torch.float32)
    tokens = context.reshape(-1, context.shape[-1])
    norms = tokens.square().mean(dim=-1).sqrt()
    valid = norms > 0
    selected = tokens[valid]
    if selected.numel() == 0:
        return {"available": True, "valid_tokens": 0}
    return {
        "available": True,
        "valid_tokens": int(valid.sum()),
        "mean": float(selected.mean()),
        "std": float(selected.std(unbiased=False)),
        "rms": float(selected.square().mean().sqrt()),
    }


@RUNNER_REGISTER("wan2.1_univ_online_policy_existence")
class WanOnlinePolicyExistenceRunner(WanMrFlowRefinementAblationRunner):
    """Share a full-compute prefix, then alter only remaining LR recomputation.

    Separate invocations with the same prompt, seed, geometry and decision step
    deterministically reconstruct the same decision state. Runtime hashes make
    this counterfactual contract auditable. Alternative geometries represent a
    real restart from step zero and are accounted for as such by the analyzer.
    """

    def __init__(self, config):
        super().__init__(config)
        self.decision_step = int(config.get("univ_online_decision_step", 0))
        self.remaining_lr_full_compute = int(
            config.get("univ_online_remaining_lr_full_compute", 0)
        )
        self.online_case_id = str(config.get("univ_online_case_id", "")).strip()
        self.initial_group = str(config.get("univ_online_initial_group", "")).strip()
        self.case_role = str(config.get("univ_online_case_role", "")).strip()
        if not self.online_case_id or not self.initial_group:
            raise ValueError("online policy case id and initial group are required")
        if self.case_role not in {"continue", "restart"}:
            raise ValueError("online policy case role must be continue or restart")
        if self.lr_steps != 50:
            raise ValueError(
                "online policy existence runner requires a 50-position LR grid"
            )
        full_compute_steps(
            decision_step=self.decision_step,
            reference_nfe=50,
            remaining_full_compute=self.remaining_lr_full_compute,
        )
        self.online_decision_record: dict[str, object] = {}
        self.online_timing_record: dict[str, float] = {}

    def _complete_lr_and_transition(self, scheduler, schedule):
        if schedule.switch_step != schedule.reference_nfe or schedule.hr_compute_steps:
            raise ValueError("online policy pilot requires switch_ratio=1.0")
        if tuple(schedule.lr_compute_steps) != tuple(range(schedule.reference_nfe)):
            raise ValueError(
                "set lr_nfe_ratio=1.0; this runner owns the online cache policy"
            )

        compute_steps = full_compute_steps(
            decision_step=self.decision_step,
            reference_nfe=schedule.reference_nfe,
            remaining_full_compute=self.remaining_lr_full_compute,
        )
        compute_set = set(compute_steps)
        prefix_steps = tuple(range(self.decision_step))
        suffix_steps = tuple(range(self.decision_step, schedule.reference_nfe))
        cache_steps = tuple(step for step in suffix_steps if step not in compute_set)
        cached_value = None
        predicted_clean = None

        synchronize(scheduler.latents)
        lr_started = time.perf_counter()
        prefix_seconds = None
        decision_state = None
        decision_state_features = None
        decision_clean_features = None
        prompt_context_features = None
        observation_seconds = None
        pre_step_observation_seconds = None
        for step_index in range(schedule.reference_nfe):
            self.check_stop()
            scheduler.step_pre(step_index=step_index)
            current_latents = scheduler.latents
            if step_index in compute_set:
                self.model.infer(self.inputs)
                cached_value = self._new_cache_value(
                    current_latents, scheduler.noise_pred
                )
                mode = "compute"
            else:
                if cached_value is None:
                    raise RuntimeError(
                        "online cache reuse requested before initialization"
                    )
                scheduler.noise_pred = self._cached_prediction(
                    current_latents, cached_value
                )
                mode = "cache"

            if step_index + 1 == self.decision_step:
                synchronize(current_latents)
                observation_started = time.perf_counter()
                sigma = scheduler.sigmas[step_index].to(
                    device=current_latents.device, dtype=torch.float32
                )
                predicted_clean = wan_clean_from_velocity(
                    current_latents.to(torch.float32),
                    scheduler.noise_pred.to(torch.float32),
                    sigma,
                ).to(dtype=current_latents.dtype)
                decision_clean_features = _tensor_features(predicted_clean)
                prompt_context_features = _prompt_context_features(self.inputs)
                synchronize(predicted_clean)
                pre_step_observation_seconds = time.perf_counter() - observation_started

            scheduler.step_post()
            if step_index + 1 == self.decision_step:
                synchronize(scheduler.latents)
                if pre_step_observation_seconds is None:
                    raise RuntimeError("pre-step online observation was not captured")
                prefix_seconds = (
                    time.perf_counter() - lr_started - pre_step_observation_seconds
                )
                observation_started = time.perf_counter()
                decision_state = scheduler.latents.detach().clone()
                decision_state_features = _tensor_features(decision_state)
                synchronize(scheduler.latents)
                observation_seconds = (
                    pre_step_observation_seconds
                    + time.perf_counter()
                    - observation_started
                )
            logger.info(
                f"==> UNIV online LR step {step_index + 1}/{schedule.reference_nfe}: {mode}"
            )
            if self.progress_callback:
                self.progress_callback(
                    80 * (step_index + 1) / schedule.reference_nfe, 100
                )

        synchronize(scheduler.latents)
        lr_seconds = time.perf_counter() - lr_started
        if (
            prefix_seconds is None
            or decision_state is None
            or predicted_clean is None
            or decision_state_features is None
            or decision_clean_features is None
            or prompt_context_features is None
            or observation_seconds is None
        ):
            raise RuntimeError("online decision state was not captured")
        if float(scheduler.sigmas[schedule.reference_nfe]) != 0.0:
            raise RuntimeError("online LR trajectory did not terminate at sigma zero")

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
        self.shared_lr_grid = {
            "grid_policy": "full_reference_grid_with_online_residual_reuse",
            "reference_nfe": schedule.reference_nfe,
            "decision_step": self.decision_step,
            "prefix_full_compute_steps": list(prefix_steps),
            "remaining_full_compute_steps": [
                step for step in compute_steps if step >= self.decision_step
            ],
            "remaining_cache_steps": list(cache_steps),
            "total_full_compute": len(compute_steps),
            "terminal_sigma": 0.0,
        }
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
        self.online_decision_record = {
            "schema": "univ_online_decision_observation_v1",
            "case_id": self.online_case_id,
            "case_role": self.case_role,
            "initial_group": self.initial_group,
            "decision_step": self.decision_step,
            "decision_sigma": float(scheduler.sigmas[self.decision_step]),
            "preview_source_step": self.decision_step - 1,
            "preview_source_sigma": float(scheduler.sigmas[self.decision_step - 1]),
            "state_sha256": tensor_sha256(decision_state),
            "predicted_clean_sha256": tensor_sha256(predicted_clean),
            "state_features": decision_state_features,
            "predicted_clean_features": decision_clean_features,
            "prompt_context_features": prompt_context_features,
            "remaining_lr_full_compute": self.remaining_lr_full_compute,
            "remaining_lr_compute_steps": self.shared_lr_grid[
                "remaining_full_compute_steps"
            ],
            "remaining_lr_cache_steps": list(cache_steps),
        }
        self.online_timing_record = {
            "decision_prefix_lr": prefix_seconds,
            "online_observation": observation_seconds,
            "remaining_lr": lr_seconds - prefix_seconds - observation_seconds,
        }

        boundary = self._boundary_path()
        boundary.parent.mkdir(parents=True, exist_ok=True)
        if boundary.exists() and getattr(self, "reuse_shared_endpoint", True):
            raise FileExistsError(f"online policy endpoint already exists: {boundary}")
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
            "lr_steps": schedule.reference_nfe,
            "reference_nfe": schedule.reference_nfe,
            "lr_schedule": copy.deepcopy(self.shared_lr_grid),
            "lr_endpoint_sigma": 0.0,
            "prompt": self.shared_identity[0],
            "prompt_sha256": canonical_sha256(self.shared_identity[0]),
            "negative_prompt": self.shared_identity[1],
            "seed": self.shared_identity[2],
            "artifact_id": self.online_case_id,
            "action_key": str(self.config.get("univ_low_budget_action_key", "")),
            "action": dict(self.config["univ_action"]),
            "online_decision": copy.deepcopy(self.online_decision_record),
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
        del clean_lr, result, decision_state, predicted_clean

    def run_segment(self, segment_idx=0):
        result = super().run_segment(segment_idx)
        self.univ_runtime_record["schema"] = "wan_univ_online_policy_existence_v1"
        self.univ_runtime_record["online_decision"] = copy.deepcopy(
            self.online_decision_record
        )
        self.univ_runtime_record["timing_seconds"].update(self.online_timing_record)
        self.univ_runtime_record["timing_seconds"]["candidate_denoise"] = (
            self.shared_lr_seconds
            - self.online_timing_record["online_observation"]
            + self.shared_transition_seconds
            + float(self.univ_runtime_record["timing_seconds"]["hr_full_compute"])
        )
        self._write_runtime_record()
        return result
