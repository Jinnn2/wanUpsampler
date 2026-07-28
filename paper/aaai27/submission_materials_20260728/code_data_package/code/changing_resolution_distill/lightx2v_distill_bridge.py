"""Distill4 scheduler reconstruction and InTraScale path, paper Secs. 3.1--3.3."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

from changing_resolution.lightx2v_clean_bridge import WanV2CleanLatentResizerBridge
from changing_resolution_distill.runtime_weights import (
    checkpoint_model_state,
    iter_lora_branches,
    set_registered_lora_strength,
)
from lightx2v.models.networks.wan.distill_model import WanDistillModel
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner
from lightx2v.models.schedulers.wan.step_distill.scheduler import (
    WanStepDistillScheduler,
)
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


class WanStepDistillScheduler4CleanResizerBridge(WanStepDistillScheduler):
    """4-step distill scheduler with one LR->HR clean-latent handoff."""

    def __init__(self, config):
        super().__init__(config)
        if "resolution_rate" not in config:
            config["resolution_rate"] = [2.0 / 3.0]
        if "changing_resolution_steps" not in config:
            config["changing_resolution_steps"] = [2]
        if len(config["resolution_rate"]) != len(config["changing_resolution_steps"]):
            raise ValueError(
                "resolution_rate and changing_resolution_steps must have the same length"
            )
        self.clean_latent_resizer = None

    def set_clean_latent_resizer(self, resizer):
        self.clean_latent_resizer = resizer

    def prepare_latents(self, seed, latent_shape, dtype=torch.float32):
        """Prepare an exact LR latent when height and width use different rates."""

        lowres_size = self.config.get("wan_lowres_latent_size")
        if lowres_size is not None:
            if len(self.config.get("resolution_rate", [])) != 1:
                raise ValueError(
                    "wan_lowres_latent_size currently supports exactly one resolution stage"
                )
            if not isinstance(lowres_size, (list, tuple)) or len(lowres_size) != 2:
                raise ValueError(
                    "wan_lowres_latent_size must be [latent_height, latent_width]"
                )

            low_h, low_w = (int(lowres_size[0]), int(lowres_size[1]))
            if low_h <= 0 or low_w <= 0 or low_h % 2 != 0 or low_w % 2 != 0:
                raise ValueError(
                    "wan_lowres_latent_size values must be positive even integers, "
                    f"got {(low_h, low_w)}"
                )
            if low_h > latent_shape[2] or low_w > latent_shape[3]:
                raise ValueError(
                    "wan_lowres_latent_size cannot exceed target latent size: "
                    f"lowres={(low_h, low_w)}, target={tuple(latent_shape[-2:])}"
                )

            self.generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)
            self.latents_list = [
                torch.randn(
                    latent_shape[0],
                    latent_shape[1],
                    low_h,
                    low_w,
                    dtype=dtype,
                    device=AI_DEVICE,
                    generator=self.generator,
                ),
                torch.randn(
                    *latent_shape,
                    dtype=dtype,
                    device=AI_DEVICE,
                    generator=self.generator,
                ),
            ]
            self.latents = self.latents_list[0]
            self.changing_resolution_index = 0
            logger.info(
                "Prepared explicit distill changing-resolution latents: "
                f"{tuple(self.latents_list[0].shape)} -> {tuple(self.latents_list[1].shape)}"
            )
            return

        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)
        self.latents_list = []
        for rate in self.config["resolution_rate"]:
            self.latents_list.append(
                torch.randn(
                    latent_shape[0],
                    latent_shape[1],
                    int(latent_shape[2] * rate) // 2 * 2,
                    int(latent_shape[3] * rate) // 2 * 2,
                    dtype=dtype,
                    device=AI_DEVICE,
                    generator=self.generator,
                )
            )
        self.latents_list.append(
            torch.randn(
                latent_shape[0],
                latent_shape[1],
                latent_shape[2],
                latent_shape[3],
                dtype=dtype,
                device=AI_DEVICE,
                generator=self.generator,
            )
        )
        self.latents = self.latents_list[0]
        self.changing_resolution_index = 0

    def step_post(self):
        if self.step_index + 1 in self.config["changing_resolution_steps"]:
            self.step_post_upsample()
            self.changing_resolution_index += 1
        else:
            super().step_post()

    def step_post_upsample(self):
        flow_pred = self.noise_pred.to(torch.float32)
        sample = self.latents.to(torch.float32)
        sigma = self.sigmas[self.step_index].to(
            device=sample.device, dtype=torch.float32
        )
        x0_pred = sample - sigma * flow_pred
        clean_sample = self._resize_clean_latent_to_next_stage(x0_pred.to(sample.dtype))

        if self.step_index + 1 >= self.infer_steps:
            logger.info(
                "Distill changing-resolution at final step; keep resized clean latent."
            )
            self.latents = clean_sample
            return

        sigma_next = self.sigmas[self.step_index + 1].to(
            device=sample.device, dtype=torch.float32
        )
        renoise_mode = self.config.get("wan_distill_bridge_renoise_mode", "random")
        if renoise_mode == "resize_flow":
            flow_hr = torch.nn.functional.interpolate(
                flow_pred.unsqueeze(0),
                size=clean_sample.shape[1:],
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
            noisy_sample = clean_sample.to(torch.float32) + sigma_next * flow_hr
        elif renoise_mode == "random":
            target_noise = self.latents_list[self.changing_resolution_index + 1].to(
                torch.float32
            )
            noisy_sample = self.add_noise(
                clean_sample.to(torch.float32), target_noise, sigma_next
            )
        else:
            raise ValueError(
                "wan_distill_bridge_renoise_mode must be 'resize_flow' or 'random', "
                f"got {renoise_mode!r}"
            )
        self.latents = noisy_sample.to(dtype=self.latents.dtype)

    def _resize_clean_latent_to_next_stage(self, denoised_sample):
        target_latent_shape = self.latents_list[
            self.changing_resolution_index + 1
        ].shape
        can_use_bridge = (
            self.clean_latent_resizer is not None
            and target_latent_shape[0] == denoised_sample.shape[0]
            and target_latent_shape[1] == denoised_sample.shape[1]
        )
        if can_use_bridge:
            logger.info(
                "Use Wan distill clean bridge to resize clean latent: "
                f"{tuple(denoised_sample.shape)} -> {tuple(target_latent_shape)}"
            )
            return self.clean_latent_resizer.resize(
                latent=denoised_sample,
                target_latent_shape=target_latent_shape,
                step_index=self.step_index,
                changing_resolution_index=self.changing_resolution_index,
            )

        logger.warning(
            "Wan distill clean bridge unavailable for this shape, fallback to trilinear: "
            f"{tuple(denoised_sample.shape)} -> {tuple(target_latent_shape)}"
        )
        return torch.nn.functional.interpolate(
            denoised_sample.unsqueeze(0),
            size=target_latent_shape[1:],
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)


class WanDistillModelLastStepLoRA(WanDistillModel):
    """Wan distill model with LoRA key normalization for DiffSynth/PEFT checkpoints."""

    def _load_lora_file(self, file_path):
        if self.device.type != "cpu" and torch.distributed.is_initialized():
            device = f"{AI_DEVICE}:{torch.distributed.get_rank()}"
        else:
            device = str(self.device)

        def normalize_key(key: str) -> str:
            for prefix in (
                "pipe.dit.",
                "model.diffusion_model.",
                "diffusion_model.",
                "transformer.",
            ):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            key = key.replace(".lora_A.default.weight", ".lora_down.weight")
            key = key.replace(".lora_B.default.weight", ".lora_up.weight")
            key = key.replace(".lora_A.weight", ".lora_down.weight")
            key = key.replace(".lora_B.weight", ".lora_up.weight")
            key = key.replace(".lora_A", ".lora_down")
            key = key.replace(".lora_B", ".lora_up")
            return key

        with safe_open(file_path, framework="pt", device=device) as handle:
            tensor_dict = {
                normalize_key(key): handle.get_tensor(key).to(GET_DTYPE())
                for key in handle.keys()
            }
        self._last_lora_file_key_count = len(tensor_dict)
        sample_keys = list(tensor_dict)[:8]
        logger.info(
            f"Loaded LoRA tensors: {len(tensor_dict)} from {file_path}; sample_keys={sample_keys}"
        )
        return tensor_dict

    def _update_lora(self, lora_path, strength):
        if Path(lora_path).resolve() != Path(self.lora_path).resolve():
            raise ValueError(
                "WanDistillModelLastStepLoRA supports one registered LoRA path; "
                f"got update for {lora_path!r}, registered={self.lora_path!r}"
            )
        branch_count = set_registered_lora_strength(self, strength)
        if branch_count <= 0:
            raise RuntimeError("Cannot update LoRA strength: no registered branches")
        logger.debug(
            f"Updated {branch_count} cached LoRA branch strengths to {float(strength)}"
        )


@RUNNER_REGISTER("wan2.1_distill_last_step_lora")
class WanDistillLastStepLoRARunner(WanDistillRunner):
    """WAN 4-step distill runner with LoRA enabled only on configured denoise steps."""

    def __init__(self, config):
        super().__init__(config)
        self.lora_configs = list(config.get("lora_configs") or [])
        if len(self.lora_configs) != 1:
            raise ValueError(
                "wan2.1_distill_last_step_lora expects exactly one LoRA config."
            )
        self.lora_path = self.lora_configs[0]["path"]
        self.lora_strength = float(self.lora_configs[0].get("strength", 1.0))
        self.lora_active_steps = {
            int(step)
            for step in config.get(
                "lora_active_steps", [int(config.get("infer_steps", 3))]
            )
        }
        self.return_clean_pred_steps = {
            int(step) for step in config.get("return_clean_pred_steps", [])
        }

    def load_transformer(self):
        wan_model_kwargs = {
            "model_path": self.config["model_path"],
            "config": self.config,
            "device": self.init_device,
            "lora_path": self.lora_path,
            "lora_strength": 0.0,
        }
        model = WanDistillModelLastStepLoRA(**wan_model_kwargs)
        branch_count = count_lora_branches(model)
        if branch_count <= 0:
            raise RuntimeError(
                "LoRA checkpoint loaded but matched zero LightX2V LoRA branches. "
                f"Check key format in: {self.lora_path}"
            )
        self._current_lora_strength = 0.0
        logger.info(
            f"Registered {branch_count} LightX2V LoRA branches from {self.lora_path}"
        )
        return model

    def run_segment(self, segment_idx=0):
        infer_steps = self.model.scheduler.infer_steps
        device_module = getattr(torch, AI_DEVICE)
        current_lora_strength = float(getattr(self, "_current_lora_strength", 0.0))

        for step_index in range(infer_steps):
            if self.video_segment_num == 1:
                self.check_stop()
            step_number = step_index + 1
            strength = (
                self.lora_strength if step_number in self.lora_active_steps else 0.0
            )
            logger.info(
                f"==> step_index: {step_number} / {infer_steps}, lora_strength={strength}"
            )
            if strength != current_lora_strength:
                self.model._update_lora(self.lora_path, strength)
                current_lora_strength = strength
                self._current_lora_strength = strength
            self.model.scheduler.step_pre(step_index=step_index)
            self.model.infer(self.inputs)
            if step_number in self.return_clean_pred_steps:
                flow_pred = self.model.scheduler.noise_pred.to(torch.float32)
                sample = self.model.scheduler.latents.to(torch.float32)
                sigma = self.model.scheduler.sigmas[step_index].to(
                    device=sample.device, dtype=torch.float32
                )
                self.model.scheduler.latents = (sample - sigma * flow_pred).to(
                    dtype=self.model.scheduler.latents.dtype
                )
                logger.info(
                    f"Return clean prediction after step {step_number}; skip scheduler.step_post()."
                )
                if self.progress_callback:
                    current_step = segment_idx * infer_steps + step_number
                    total_all_steps = self.video_segment_num * infer_steps
                    self.progress_callback((current_step / total_all_steps) * 100, 100)
                break
            self.model.scheduler.step_post()

            if self.progress_callback:
                current_step = segment_idx * infer_steps + step_number
                total_all_steps = self.video_segment_num * infer_steps
                self.progress_callback((current_step / total_all_steps) * 100, 100)

        if segment_idx is not None and segment_idx == self.video_segment_num - 1:
            del self.inputs
            device_module.empty_cache()

        return self.model.scheduler.latents


def count_lora_branches(obj) -> int:
    return sum(1 for _ in iter_lora_branches(obj))


@RUNNER_REGISTER("wan2.1_distill_clean_resizer_bridge")
class WanDistillCleanResizerBridgeRunner(WanDistillRunner):
    """WAN 4-step distill changing-resolution runner backed by the Stage 3 resizer."""

    def __init__(self, config):
        super().__init__(config)
        self._validate_bridge_config()

    def _validate_bridge_config(self):
        if self.config["task"] != "t2v":
            raise NotImplementedError(
                "wan2.1_distill_clean_resizer_bridge currently only supports t2v."
            )
        if self.config.get("use_tae", False):
            raise ValueError(
                "wan2.1_distill_clean_resizer_bridge requires the full WAN VAE decoder, not TAE."
            )
        if self.config.get("lazy_load", False) or self.config.get(
            "unload_modules", False
        ):
            raise ValueError(
                "wan2.1_distill_clean_resizer_bridge does not support lazy_load/unload_modules yet."
            )
        if len(self.config.get("resolution_rate", [])) != 1:
            raise ValueError(
                "wan2.1_distill_clean_resizer_bridge expects exactly one lowres->highres stage."
            )
        if len(self.config.get("changing_resolution_steps", [])) != 1:
            raise ValueError(
                "wan2.1_distill_clean_resizer_bridge expects exactly one changing_resolution step."
            )
        if self.config.get("wan_clean_resizer_ckpt") is None:
            raise ValueError(
                "wan2.1_distill_clean_resizer_bridge requires wan_clean_resizer_ckpt in config."
            )
        if self.config.get("wan_clean_resizer_repo") is None:
            raise ValueError(
                "wan2.1_distill_clean_resizer_bridge requires wan_clean_resizer_repo in config."
            )

    def init_scheduler(self):
        if self.config["feature_caching"] != "NoCaching":
            raise NotImplementedError(
                "wan2.1_distill_clean_resizer_bridge currently supports only NoCaching."
            )
        self.scheduler = WanStepDistillScheduler4CleanResizerBridge(self.config)

    def load_clean_resizer(self):
        repo_path = Path(self.config["wan_clean_resizer_repo"])
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        from wan_sr.models import (
            build_clean_latent_resizer,
            infer_clean_resizer_model_type,
        )
        from wan_sr.training.config import load_yaml

        ckpt_path = self.config["wan_clean_resizer_ckpt"]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model_config = checkpoint.get("config", {}).get("model", {})
        if not model_config and self.config.get("wan_clean_resizer_train_config"):
            train_cfg = load_yaml(self.config["wan_clean_resizer_train_config"])
            model_config = train_cfg.get("model", {})
        if not model_config:
            raise ValueError(
                "Failed to infer clean resizer config from checkpoint or train config."
            )

        model_config = self._apply_clean_resizer_overrides(model_config)
        model_type = infer_clean_resizer_model_type(model_config)
        model = build_clean_latent_resizer(model_config)
        model.load_state_dict(checkpoint_model_state(checkpoint))
        if self.config.get("wan_clean_resizer_use_ema", True) and "ema" in checkpoint:
            from wan_sr.training.ema import EMA

            ema = EMA(model)
            ema.load_state_dict(checkpoint["ema"])
            ema.copy_to(model)
            del ema

        del checkpoint

        model = model.to(device=torch.device(AI_DEVICE), dtype=GET_DTYPE())
        model.eval()
        logger.info(
            f"Initialized Wan distill clean resizer model_type={model_type} from {ckpt_path}"
        )
        return model

    def _apply_clean_resizer_overrides(self, model_config):
        from wan_sr.models import infer_clean_resizer_model_type

        model_config = dict(model_config)
        configured = self.config.get("wan_clean_resizer_model_class")
        if configured:
            model_config["model_type"] = str(configured)
        if infer_clean_resizer_model_type(model_config) == "stage2":
            if "wan_clean_resizer_residual_skip" in self.config:
                residual_skip = self.config["wan_clean_resizer_residual_skip"]
                model_config["residual_skip"] = bool(residual_skip)
        return model_config

    def load_model(self):
        super().load_model()
        self.wan_clean_resizer = self.load_clean_resizer()
        self.clean_latent_resizer = WanV2CleanLatentResizerBridge(
            resizer=self.wan_clean_resizer,
            config=self.config,
        )
        self.scheduler.set_clean_latent_resizer(self.clean_latent_resizer)
        logger.info("Initialized WAN distill + clean-latent bridge resizer.")

    def init_run(self):
        super().init_run()
        if hasattr(self.scheduler, "set_clean_latent_resizer"):
            self.scheduler.set_clean_latent_resizer(self.clean_latent_resizer)


@RUNNER_REGISTER("wan2.1_distill_last_step_lora_clean_resizer_bridge")
class WanDistillLastStepLoRACleanResizerBridgeRunner(
    WanDistillCleanResizerBridgeRunner
):
    """WAN 4-step distill runner: LR LoRA handoff -> Stage2 resize -> HR final denoise."""

    def __init__(self, config):
        super().__init__(config)
        self.lora_configs = list(config.get("lora_configs") or [])
        if len(self.lora_configs) != 1:
            raise ValueError(
                "wan2.1_distill_last_step_lora_clean_resizer_bridge expects exactly one LoRA config."
            )
        self.lora_path = self.lora_configs[0]["path"]
        self.lora_strength = float(self.lora_configs[0].get("strength", 1.0))
        self.lora_active_steps = {
            int(step)
            for step in config.get(
                "lora_active_steps", [int(config.get("infer_steps", 3))]
            )
        }

    def load_transformer(self):
        wan_model_kwargs = {
            "model_path": self.config["model_path"],
            "config": self.config,
            "device": self.init_device,
            "lora_path": self.lora_path,
            "lora_strength": 0.0,
        }
        model = WanDistillModelLastStepLoRA(**wan_model_kwargs)
        branch_count = count_lora_branches(model)
        if branch_count <= 0:
            raise RuntimeError(
                "LoRA checkpoint loaded but matched zero LightX2V LoRA branches. "
                f"Check key format in: {self.lora_path}"
            )
        self._current_lora_strength = 0.0
        logger.info(
            f"Registered {branch_count} LightX2V LoRA branches from {self.lora_path}"
        )
        return model

    def run_segment(self, segment_idx=0):
        infer_steps = self.model.scheduler.infer_steps
        device_module = getattr(torch, AI_DEVICE)
        current_lora_strength = float(getattr(self, "_current_lora_strength", 0.0))

        for step_index in range(infer_steps):
            if self.video_segment_num == 1:
                self.check_stop()
            step_number = step_index + 1
            strength = (
                self.lora_strength if step_number in self.lora_active_steps else 0.0
            )
            logger.info(
                f"==> step_index: {step_number} / {infer_steps}, lora_strength={strength}"
            )
            if strength != current_lora_strength:
                self.model._update_lora(self.lora_path, strength)
                current_lora_strength = strength
                self._current_lora_strength = strength
            self.model.scheduler.step_pre(step_index=step_index)
            self.model.infer(self.inputs)
            self.model.scheduler.step_post()

            if self.progress_callback:
                current_step = segment_idx * infer_steps + step_number
                total_all_steps = self.video_segment_num * infer_steps
                self.progress_callback((current_step / total_all_steps) * 100, 100)

        if segment_idx is not None and segment_idx == self.video_segment_num - 1:
            del self.inputs
            device_module.empty_cache()

        return self.model.scheduler.latents


@RUNNER_REGISTER("wan2.1_distill_interp_bridge")
class WanDistillInterpBridgeRunner(WanDistillRunner):
    """WAN 4-step distill changing-resolution runner with trilinear clean-latent resize."""

    def __init__(self, config):
        super().__init__(config)
        self._validate_interp_config()

    def _validate_interp_config(self):
        if self.config["task"] != "t2v":
            raise NotImplementedError(
                "wan2.1_distill_interp_bridge currently only supports t2v."
            )
        if len(self.config.get("resolution_rate", [])) != 1:
            raise ValueError(
                "wan2.1_distill_interp_bridge expects exactly one lowres->highres stage."
            )
        if len(self.config.get("changing_resolution_steps", [])) != 1:
            raise ValueError(
                "wan2.1_distill_interp_bridge expects exactly one changing_resolution step."
            )

    def init_scheduler(self):
        if self.config["feature_caching"] != "NoCaching":
            raise NotImplementedError(
                "wan2.1_distill_interp_bridge currently supports only NoCaching."
            )
        self.scheduler = WanStepDistillScheduler4CleanResizerBridge(self.config)


@RUNNER_REGISTER("wan2.1_distill_last_step_lora_interp_bridge")
class WanDistillLastStepLoRAInterpBridgeRunner(WanDistillInterpBridgeRunner):
    """WAN four-step runner: step-local LoRA handoff -> interpolation -> HR suffix."""

    def __init__(self, config):
        super().__init__(config)
        self.lora_configs = list(config.get("lora_configs") or [])
        if len(self.lora_configs) != 1:
            raise ValueError(
                "wan2.1_distill_last_step_lora_interp_bridge expects exactly one LoRA config."
            )
        self.lora_path = self.lora_configs[0]["path"]
        self.lora_strength = float(self.lora_configs[0].get("strength", 1.0))
        self.lora_active_steps = {
            int(step)
            for step in config.get(
                "lora_active_steps", [int(config.get("infer_steps", 3))]
            )
        }

    def load_transformer(self):
        model = WanDistillModelLastStepLoRA(
            model_path=self.config["model_path"],
            config=self.config,
            device=self.init_device,
            lora_path=self.lora_path,
            lora_strength=0.0,
        )
        branch_count = count_lora_branches(model)
        if branch_count <= 0:
            raise RuntimeError(
                "LoRA checkpoint loaded but matched zero LightX2V LoRA branches. "
                f"Check key format in: {self.lora_path}"
            )
        self._current_lora_strength = 0.0
        logger.info(
            f"Registered {branch_count} LightX2V LoRA branches from {self.lora_path}"
        )
        return model

    def run_segment(self, segment_idx=0):
        infer_steps = self.model.scheduler.infer_steps
        device_module = getattr(torch, AI_DEVICE)
        current_lora_strength = float(getattr(self, "_current_lora_strength", 0.0))

        for step_index in range(infer_steps):
            if self.video_segment_num == 1:
                self.check_stop()
            step_number = step_index + 1
            strength = (
                self.lora_strength if step_number in self.lora_active_steps else 0.0
            )
            logger.info(
                f"==> step_index: {step_number} / {infer_steps}, lora_strength={strength}"
            )
            if strength != current_lora_strength:
                self.model._update_lora(self.lora_path, strength)
                current_lora_strength = strength
                self._current_lora_strength = strength
            self.model.scheduler.step_pre(step_index=step_index)
            self.model.infer(self.inputs)
            self.model.scheduler.step_post()

            if self.progress_callback:
                current_step = segment_idx * infer_steps + step_number
                total_all_steps = self.video_segment_num * infer_steps
                self.progress_callback((current_step / total_all_steps) * 100, 100)

        if segment_idx is not None and segment_idx == self.video_segment_num - 1:
            del self.inputs
            device_module.empty_cache()

        return self.model.scheduler.latents


class WanDistillFullLREndpointMixin:
    """Finish the canonical four-step LR path, lift once, then replay K HR suffix steps."""

    endpoint_resizer = "abstract"

    def __init__(self, config):
        super().__init__(config)
        infer_steps = int(config["infer_steps"])
        if infer_steps != 4:
            raise ValueError(
                f"Distill endpoint runners require infer_steps=4, got {infer_steps}"
            )
        if [int(step) for step in config.get("changing_resolution_steps", [])] != [
            infer_steps
        ]:
            raise ValueError(
                "Distill endpoint runners require changing_resolution_steps=[infer_steps]"
            )
        refinement_steps = int(config.get("wan_final_refine_steps", 1))
        if refinement_steps < 0 or refinement_steps > infer_steps:
            raise ValueError(
                f"wan_final_refine_steps must be in [0, {infer_steps}], got {refinement_steps}"
            )
        self.final_refine_steps = refinement_steps
        direct_sigma = config.get("wan_final_refine_sigma")
        if direct_sigma is not None:
            direct_sigma = float(direct_sigma)
            if refinement_steps != 1:
                raise ValueError(
                    "wan_final_refine_sigma is only defined for exactly one HR refinement step"
                )
            if not 0.0 < direct_sigma < 1.0:
                raise ValueError(
                    f"wan_final_refine_sigma must be in (0, 1), got {direct_sigma}"
                )
        self.final_refine_sigma = direct_sigma

    def _lift_endpoint(self, clean_lr, target_latent_shape):
        raise NotImplementedError

    @staticmethod
    def _set_schedule_scalar(sequence, index, value):
        current = sequence[index]
        if torch.is_tensor(current):
            replacement = current.new_tensor(value)
        elif torch.is_tensor(sequence):
            replacement = sequence.new_tensor(value)
        else:
            replacement = value
        sequence[index] = replacement

    def run_segment(self, segment_idx=0):
        infer_steps = self.model.scheduler.infer_steps
        scheduler = self.model.scheduler
        device_module = getattr(torch, AI_DEVICE)
        total_evaluations = infer_steps + self.final_refine_steps

        for step_index in range(infer_steps):
            if self.video_segment_num == 1:
                self.check_stop()
            step_number = step_index + 1
            logger.info(f"==> Distill LR step_index: {step_number} / {infer_steps}")
            scheduler.step_pre(step_index=step_index)
            self.model.infer(self.inputs)
            # The endpoint runner must complete the LR solver before lifting.
            # Bypass the configured step-4 resolution switch on the final update.
            if step_number == infer_steps:
                super(WanStepDistillScheduler4CleanResizerBridge, scheduler).step_post()
            else:
                scheduler.step_post()
            if self.progress_callback:
                self.progress_callback((step_number / total_evaluations) * 100, 100)

        clean_lr = scheduler.latents
        target_latent_shape = scheduler.latents_list[-1].shape
        clean_hr = self._lift_endpoint(clean_lr, target_latent_shape)
        if tuple(clean_hr.shape) != tuple(target_latent_shape):
            raise RuntimeError(
                f"{self.endpoint_resizer} endpoint produced {tuple(clean_hr.shape)}, "
                f"expected {tuple(target_latent_shape)}"
            )
        clean_hr = clean_hr.to(device=clean_lr.device, dtype=clean_lr.dtype)

        if self.final_refine_steps == 0:
            logger.info(
                f"==> Distill endpoint {self.endpoint_resizer}: decode lifted LR endpoint without HR refinement"
            )
            scheduler.latents = clean_hr
            if self.progress_callback:
                self.progress_callback(100, 100)
            if segment_idx is not None and segment_idx == self.video_segment_num - 1:
                del self.inputs
                device_module.empty_cache()
            return scheduler.latents

        first_refine_index = infer_steps - self.final_refine_steps
        if self.final_refine_sigma is not None:
            # MrFlow's 12+1 protocol performs the single HR correction directly
            # at sigma=0.12. Keep the injected noise and the DiT conditioning
            # timestep consistent instead of replaying the distilled t=250 step.
            self._set_schedule_scalar(
                scheduler.sigmas, first_refine_index, self.final_refine_sigma
            )
            self._set_schedule_scalar(
                scheduler.timesteps,
                first_refine_index,
                self.final_refine_sigma * 1000.0,
            )
        refine_sigma = scheduler.sigmas[first_refine_index].to(
            device=clean_hr.device, dtype=torch.float32
        )
        hr_noise = scheduler.latents_list[-1].to(
            device=clean_hr.device, dtype=torch.float32
        )
        scheduler.latents = scheduler.add_noise(
            clean_hr.to(torch.float32), hr_noise, refine_sigma
        ).to(dtype=clean_hr.dtype)
        logger.info(
            f"==> Distill endpoint {self.endpoint_resizer}: "
            f"{self.final_refine_steps} HR suffix step(s), "
            f"start_step={first_refine_index + 1}, sigma={float(refine_sigma)}, "
            f"protocol={'direct_sigma' if self.final_refine_sigma is not None else 'distilled_suffix'}"
        )
        for refinement_index, step_index in enumerate(
            range(first_refine_index, infer_steps), start=1
        ):
            scheduler.step_pre(step_index=step_index)
            self.model.infer(self.inputs)
            # HR replay must not trigger the configured LR->HR switch again.
            super(WanStepDistillScheduler4CleanResizerBridge, scheduler).step_post()
            if self.progress_callback:
                completed = infer_steps + refinement_index
                self.progress_callback((completed / total_evaluations) * 100, 100)

        if segment_idx is not None and segment_idx == self.video_segment_num - 1:
            del self.inputs
            device_module.empty_cache()
        return scheduler.latents


@RUNNER_REGISTER("wan2.1_distill_full_lr_stage2_k_hr")
class WanDistillFullLRStage2KHRRunner(
    WanDistillFullLREndpointMixin, WanDistillCleanResizerBridgeRunner
):
    endpoint_resizer = "stage2"

    def _lift_endpoint(self, clean_lr, target_latent_shape):
        return self.model.scheduler._resize_clean_latent_to_next_stage(clean_lr)


@RUNNER_REGISTER("wan2.1_distill_full_lr_interp_k_hr")
class WanDistillFullLRInterpKHRRunner(
    WanDistillFullLREndpointMixin, WanDistillInterpBridgeRunner
):
    endpoint_resizer = "interp"

    def _lift_endpoint(self, clean_lr, target_latent_shape):
        return torch.nn.functional.interpolate(
            clean_lr.unsqueeze(0),
            size=target_latent_shape[1:],
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)


@RUNNER_REGISTER("wan2.1_distill_full_lr_rgb_k_hr")
class WanDistillFullLRRGBKHRRunner(
    WanDistillFullLREndpointMixin, WanDistillInterpBridgeRunner
):
    endpoint_resizer = "rgb"

    def __init__(self, config):
        super().__init__(config)
        if config.get("use_tae", False):
            raise ValueError("RGB endpoint requires the full Wan VAE codec, not TAE")
        self.rgb_super_resolver = None

    def _load_rgb_super_resolver(self):
        if self.rgb_super_resolver is None:
            from changing_resolution_distill.rgb_super_resolution import (
                build_rgb_super_resolver,
            )

            self.rgb_super_resolver = build_rgb_super_resolver(self.config)
        return self.rgb_super_resolver

    @torch.no_grad()
    def _lift_endpoint(self, clean_lr, target_latent_shape):
        if not hasattr(self.vae_decoder, "encode"):
            raise RuntimeError(
                "RGB endpoint requires the loaded LightX2V Wan VAE object to expose encode()"
            )
        decoded = self.vae_decoder.decode(clean_lr.to(GET_DTYPE()))
        if decoded.ndim == 5 and decoded.shape[0] == 1:
            decoded = decoded[0]
        if decoded.ndim != 4 or decoded.shape[0] != 3:
            raise RuntimeError(
                f"unexpected Wan VAE decode shape: {tuple(decoded.shape)}"
            )
        rgb_video = (
            ((decoded.float().clamp(-1, 1) + 1.0) * 0.5)
            .permute(1, 2, 3, 0)
            .contiguous()
            .cpu()
        )
        del decoded
        device_module = getattr(torch, AI_DEVICE)
        device_module.empty_cache()
        resolver = self._load_rgb_super_resolver()
        target_height = int(self.config["target_height"])
        target_width = int(self.config["target_width"])
        upscaled = resolver.resize(
            rgb_video, target_height=target_height, target_width=target_width
        )
        del rgb_video
        device_module.empty_cache()

        vae_input = (
            upscaled.permute(3, 0, 1, 2)
            .unsqueeze(0)
            .to(device=AI_DEVICE, dtype=GET_DTYPE())
        )
        vae_input = vae_input.mul(2.0).sub(1.0)
        encoded = self.vae_decoder.encode(vae_input)
        if isinstance(encoded, (list, tuple)):
            if len(encoded) != 1:
                raise RuntimeError(
                    f"unexpected Wan VAE encode list length: {len(encoded)}"
                )
            encoded = encoded[0]
        if encoded.ndim == 5 and encoded.shape[0] == 1:
            encoded = encoded[0]
        if encoded.ndim != 4:
            raise RuntimeError(
                f"unexpected Wan VAE encode shape: {tuple(encoded.shape)}"
            )
        del upscaled, vae_input
        device_module.empty_cache()
        return encoded.to(device=clean_lr.device, dtype=clean_lr.dtype)
