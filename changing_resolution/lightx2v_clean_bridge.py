from __future__ import annotations

import sys
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.changing_resolution.scheduler import (
    WanScheduler4ChangingResolution,
)
from lightx2v.models.schedulers.wan.feature_caching.scheduler import (
    WanSchedulerCaching,
    WanSchedulerTaylorCaching,
)
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


class WanTailSkipLoRAModel(WanModel):
    """Wan LoRA update compatible with LightX2V models that have no post_weight.

    LightX2V's generic ``BaseTransformerModel._update_lora`` currently updates
    ``pre_weight``, ``transformer_weights``, and ``post_weight`` unconditionally.
    WanModel has no post-weight container, so a dynamic strength change fails at
    the handoff step.  The initial dynamic LoRA registration already handles the
    first two containers correctly; this override mirrors that behavior when
    toggling strength during sampling.
    """

    def _load_lora_file(self, file_path):
        if self.device.type != "cpu" and torch.distributed.is_initialized():
            device = f"{AI_DEVICE}:{torch.distributed.get_rank()}"
        else:
            device = str(self.device)

        def normalize_key(key: str) -> str:
            for prefix in ("pipe.dit.", "model.diffusion_model.", "diffusion_model.", "transformer."):
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
        logger.info(f"Loaded {len(tensor_dict)} normalized tail-skip LoRA tensors from {file_path}")
        return tensor_dict

    def _update_lora(self, lora_path, strength):
        lora_weight = lora_path if isinstance(lora_path, dict) else self._load_lora_file(lora_path)
        self.pre_weight.update_lora(lora_weight, strength)
        self.transformer_weights.update_lora(lora_weight, strength)
        post_weight = getattr(self, "post_weight", None)
        if post_weight is not None:
            post_weight.update_lora(lora_weight, strength)


def count_lora_branches(obj, seen=None) -> int:
    """Count registered LightX2V LoRA branches without double-counting modules."""

    if obj is None:
        return 0
    if seen is None:
        seen = set()
    object_id = id(obj)
    if object_id in seen:
        return 0
    seen.add(object_id)

    count = int(bool(getattr(obj, "has_lora_branch", False)))
    for child in getattr(obj, "_modules", {}).values():
        count += count_lora_branches(child, seen)
    for child in getattr(obj, "_parameters", {}).values():
        count += count_lora_branches(child, seen)
    for name in ("pre_weight", "transformer_weights", "post_weight"):
        count += count_lora_branches(getattr(obj, name, None), seen)
    return count


class WanScheduler4CleanResizerBridgeInterface:
    """Changing-resolution scheduler using a trained clean-latent resizer."""

    def __new__(cls, father_scheduler, config):
        class NewClass(WanScheduler4CleanResizerBridge, father_scheduler):
            def __init__(self, config):
                father_scheduler.__init__(self, config)
                WanScheduler4CleanResizerBridge.__init__(self, config)

        return NewClass(config)


class WanScheduler4CleanResizerBridge(WanScheduler4ChangingResolution):
    def __init__(self, config):
        super().__init__(config)
        self.clean_latent_resizer = None

    def set_clean_latent_resizer(self, resizer):
        self.clean_latent_resizer = resizer

    def prepare_latents(self, seed, latent_shape, dtype=torch.float32):
        """Allow an exact LR latent size when pixel aspect ratios do not share one rate."""

        lowres_size = self.config.get("wan_lowres_latent_size")
        if lowres_size is None:
            return super().prepare_latents(seed, latent_shape, dtype=dtype)
        if len(self.config.get("resolution_rate", [])) != 1:
            raise ValueError("wan_lowres_latent_size currently supports exactly one resolution stage")
        if not isinstance(lowres_size, (list, tuple)) or len(lowres_size) != 2:
            raise ValueError("wan_lowres_latent_size must be [latent_height, latent_width]")

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
            "Prepared explicit changing-resolution latents: "
            f"{tuple(self.latents_list[0].shape)} -> {tuple(self.latents_list[1].shape)}"
        )

    def _resize_clean_latent_to_next_stage(self, denoised_sample, target_latent_shape):
        can_use_bridge = (
            self.clean_latent_resizer is not None
            and target_latent_shape[0] == denoised_sample.shape[0]
            and target_latent_shape[1] == denoised_sample.shape[1]
        )
        if can_use_bridge:
            logger.info(
                "Use Wan V2 clean bridge to resize clean latent: "
                f"{tuple(denoised_sample.shape)} -> {tuple(target_latent_shape)}"
            )
            return self.clean_latent_resizer.resize(
                latent=denoised_sample,
                target_latent_shape=target_latent_shape,
                step_index=self.step_index,
                changing_resolution_index=self.changing_resolution_index,
            )

        logger.warning(
            "Wan V2 clean bridge unavailable for this shape, fallback to trilinear: "
            f"{tuple(denoised_sample.shape)} -> {tuple(target_latent_shape)}"
        )
        return torch.nn.functional.interpolate(
            denoised_sample.unsqueeze(0),
            size=target_latent_shape[1:],
            mode="trilinear",
        ).squeeze(0)

    def step_post_upsample(self):
        model_output = self.noise_pred.to(torch.float32)
        sample = self.latents.to(torch.float32)
        sigma_t = self.sigmas[self.step_index]
        x0_pred = sample - sigma_t * model_output
        denoised_sample = x0_pred.to(sample.dtype)

        target_latent_shape = self.latents_list[self.changing_resolution_index + 1].shape
        clean_sample = self._resize_clean_latent_to_next_stage(denoised_sample, target_latent_shape)

        if self.step_index + 1 >= len(self.timesteps):
            logger.info("Changing resolution at final step; decode resized clean latent without re-noise.")
            self.latents = clean_sample
            return

        noisy_sample = self.add_noise(
            clean_sample,
            self.latents_list[self.changing_resolution_index + 1],
            self.timesteps[self.step_index + 1],
        )

        self.latents = noisy_sample
        self.set_timesteps(
            self.infer_steps,
            device=AI_DEVICE,
            shift=self.sample_shift + self.changing_resolution_index + 1,
        )


class WanV2CleanLatentResizerBridge:
    def __init__(self, resizer, config):
        self.resizer = resizer
        self.config = config
        self.dtype = GET_DTYPE()
        self.device = torch.device(AI_DEVICE)

    @torch.no_grad()
    def resize(self, latent, target_latent_shape, step_index=None, changing_resolution_index=None):
        if latent.dim() != 4:
            raise ValueError(f"Expected WAN latent shape [C, T, H, W], got {tuple(latent.shape)}")

        target_latent_shape = tuple(target_latent_shape)
        current_shape = tuple(latent.shape)
        if target_latent_shape[0] != current_shape[0] or target_latent_shape[1] != current_shape[1]:
            raise ValueError(
                "Wan V2 clean bridge expects channel/time to stay unchanged. "
                f"Current={current_shape}, target={target_latent_shape}"
            )

        logger.info(
            "Wan V2 clean latent bridge resize: "
            f"step={step_index}, stage={changing_resolution_index}, "
            f"wan_latent={current_shape} -> {target_latent_shape}"
        )

        batch = latent.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        pred = self.resizer(batch, output_size=target_latent_shape[-2:]).squeeze(0)

        if tuple(pred.shape) != target_latent_shape:
            logger.warning(
                "Wan V2 bridge shape mismatch, fallback to trilinear resize: "
                f"{tuple(pred.shape)} -> {target_latent_shape}"
            )
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(0),
                size=target_latent_shape[1:],
                mode="trilinear",
            ).squeeze(0)

        return pred.to(dtype=latent.dtype, device=latent.device)


@RUNNER_REGISTER("wan2.1_clean_resizer_bridge")
class WanCleanResizerBridgeRunner(WanRunner):
    """WAN changing-resolution runner backed by the Stage 2 clean-latent resizer."""

    def __init__(self, config):
        super().__init__(config)
        self._validate_bridge_config()

    def _validate_bridge_config(self):
        if self.config["task"] != "t2v":
            raise NotImplementedError("wan2.1_clean_resizer_bridge currently only supports t2v.")
        if self.config.get("use_tae", False):
            raise ValueError("wan2.1_clean_resizer_bridge requires the full WAN VAE decoder, not TAE.")
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            raise ValueError("wan2.1_clean_resizer_bridge does not support lazy_load/unload_modules yet.")
        if len(self.config.get("resolution_rate", [])) != 1:
            raise ValueError("wan2.1_clean_resizer_bridge expects exactly one lowres->highres stage.")
        if len(self.config.get("changing_resolution_steps", [])) != 1:
            raise ValueError("wan2.1_clean_resizer_bridge expects exactly one changing_resolution step.")
        if self.config.get("wan_clean_resizer_ckpt") is None:
            raise ValueError("wan2.1_clean_resizer_bridge requires wan_clean_resizer_ckpt in config.")
        if self.config.get("wan_clean_resizer_repo") is None:
            raise ValueError("wan2.1_clean_resizer_bridge requires wan_clean_resizer_repo in config.")

    def init_scheduler(self):
        if self.config["feature_caching"] == "NoCaching":
            scheduler_class = WanScheduler
        elif self.config["feature_caching"] == "TaylorSeer":
            scheduler_class = WanSchedulerTaylorCaching
        elif self.config.feature_caching in ["Tea", "Ada", "Custom", "FirstBlock", "DualBlock", "DynamicBlock", "Mag"]:
            scheduler_class = WanSchedulerCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config.feature_caching}")

        self.scheduler = WanScheduler4CleanResizerBridgeInterface(scheduler_class, self.config)

    def load_clean_resizer(self):
        repo_path = Path(self.config["wan_clean_resizer_repo"])
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        from wan_sr.models import build_clean_latent_resizer, infer_clean_resizer_model_type
        from wan_sr.training.checkpoint import load_checkpoint
        from wan_sr.training.config import load_yaml

        ckpt_path = self.config["wan_clean_resizer_ckpt"]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model_config = checkpoint.get("config", {}).get("model", {})
        if not model_config and self.config.get("wan_clean_resizer_train_config"):
            train_cfg = load_yaml(self.config["wan_clean_resizer_train_config"])
            model_config = train_cfg.get("model", {})
        if not model_config:
            raise ValueError("Failed to infer clean resizer config from checkpoint or train config.")

        model_config = self._apply_clean_resizer_overrides(model_config)
        model_type = infer_clean_resizer_model_type(model_config)
        model = build_clean_latent_resizer(model_config)
        load_checkpoint(ckpt_path, model, map_location="cpu")
        if self.config.get("wan_clean_resizer_use_ema", True) and "ema" in checkpoint:
            from wan_sr.training.ema import EMA

            ema = EMA(model)
            ema.load_state_dict(checkpoint["ema"])
            ema.copy_to(model)

        model = model.to(device=torch.device(AI_DEVICE), dtype=GET_DTYPE())
        model.eval()
        logger.info(f"Initialized Wan V2 clean resizer model_type={model_type} from {ckpt_path}")
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
        logger.info("Initialized WAN + Wan V2 clean-latent bridge resizer.")

    def init_run(self):
        super().init_run()
        if hasattr(self.scheduler, "set_clean_latent_resizer"):
            self.scheduler.set_clean_latent_resizer(self.clean_latent_resizer)


@RUNNER_REGISTER("wan2.1_tail_skip_lora")
class WanTailSkipLoRARunner(WanRunner):
    """WAN 50-step runner with LoRA enabled only on configured handoff steps."""

    def __init__(self, config):
        super().__init__(config)
        self.lora_configs = list(config.get("lora_configs") or [])
        if len(self.lora_configs) != 1:
            raise ValueError("wan2.1_tail_skip_lora expects exactly one LoRA config.")
        self.lora_path = self.lora_configs[0]["path"]
        self.lora_strength = float(self.lora_configs[0].get("strength", 1.0))
        self.lora_active_steps = {
            int(step)
            for step in config.get("lora_active_steps", [int(config.get("infer_steps", 45))])
        }
        self.return_clean_pred_steps = {
            int(step)
            for step in config.get("return_clean_pred_steps", [])
        }

    def load_transformer(self):
        wan_model_kwargs = {
            "model_path": self.config["model_path"],
            "config": self.config,
            "device": self.init_device,
            "lora_path": self.lora_path,
            "lora_strength": 0.0,
        }
        model = WanTailSkipLoRAModel(**wan_model_kwargs)
        branch_count = count_lora_branches(model)
        if branch_count <= 0:
            raise RuntimeError(
                "LoRA checkpoint loaded but matched zero LightX2V LoRA branches. "
                f"Check key format in: {self.lora_path}"
            )
        self._current_lora_strength = 0.0
        logger.info(f"Registered {branch_count} WAN tail-skip LoRA branches from {self.lora_path}")
        return model

    def run_segment(self, segment_idx=0):
        infer_steps = self.model.scheduler.infer_steps
        device_module = getattr(torch, AI_DEVICE)
        current_lora_strength = float(getattr(self, "_current_lora_strength", 0.0))

        for step_index in range(infer_steps):
            if self.video_segment_num == 1:
                self.check_stop()
            step_number = step_index + 1
            strength = self.lora_strength if step_number in self.lora_active_steps else 0.0
            logger.info(f"==> step_index: {step_number} / {infer_steps}, lora_strength={strength}")
            if strength != current_lora_strength:
                self.model._update_lora(self.lora_path, strength)
                current_lora_strength = strength
                self._current_lora_strength = strength
            self.model.scheduler.step_pre(step_index=step_index)
            self.model.infer(self.inputs)
            if step_number in self.return_clean_pred_steps:
                flow_pred = self.model.scheduler.noise_pred.to(torch.float32)
                sample = self.model.scheduler.latents.to(torch.float32)
                sigma = self.model.scheduler.sigmas[step_index].to(device=sample.device, dtype=torch.float32)
                self.model.scheduler.latents = (sample - sigma * flow_pred).to(dtype=self.model.scheduler.latents.dtype)
                logger.info(f"Return clean prediction after step {step_number}; skip scheduler.step_post().")
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


@RUNNER_REGISTER("wan2.1_tail_skip_lora_clean_resizer_bridge")
class WanTailSkipLoRACleanResizerBridgeRunner(WanCleanResizerBridgeRunner):
    """WAN 50-step runner: LR LoRA handoff -> Stage2 resize -> HR remaining steps."""

    def __init__(self, config):
        super().__init__(config)
        self.lora_configs = list(config.get("lora_configs") or [])
        if len(self.lora_configs) != 1:
            raise ValueError("wan2.1_tail_skip_lora_clean_resizer_bridge expects exactly one LoRA config.")
        self.lora_path = self.lora_configs[0]["path"]
        self.lora_strength = float(self.lora_configs[0].get("strength", 1.0))
        self.lora_active_steps = {
            int(step)
            for step in config.get("lora_active_steps", [int(config.get("changing_resolution_steps", [45])[0])])
        }

    def load_transformer(self):
        wan_model_kwargs = {
            "model_path": self.config["model_path"],
            "config": self.config,
            "device": self.init_device,
            "lora_path": self.lora_path,
            "lora_strength": 0.0,
        }
        model = WanTailSkipLoRAModel(**wan_model_kwargs)
        branch_count = count_lora_branches(model)
        if branch_count <= 0:
            raise RuntimeError(
                "LoRA checkpoint loaded but matched zero LightX2V LoRA branches. "
                f"Check key format in: {self.lora_path}"
            )
        self._current_lora_strength = 0.0
        logger.info(f"Registered {branch_count} WAN tail-skip LoRA branches from {self.lora_path}")
        return model

    def run_segment(self, segment_idx=0):
        infer_steps = self.model.scheduler.infer_steps
        device_module = getattr(torch, AI_DEVICE)
        current_lora_strength = float(getattr(self, "_current_lora_strength", 0.0))

        for step_index in range(infer_steps):
            if self.video_segment_num == 1:
                self.check_stop()
            step_number = step_index + 1
            strength = self.lora_strength if step_number in self.lora_active_steps else 0.0
            logger.info(f"==> step_index: {step_number} / {infer_steps}, lora_strength={strength}")
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


@RUNNER_REGISTER("wan2.1_clean_interp_bridge")
class WanCleanInterpBridgeRunner(WanRunner):
    """WAN changing-resolution runner that uses the same clean-latent switch path with trilinear resize."""

    def __init__(self, config):
        super().__init__(config)
        self._validate_interp_config()

    def _validate_interp_config(self):
        if self.config["task"] != "t2v":
            raise NotImplementedError("wan2.1_clean_interp_bridge currently only supports t2v.")
        if len(self.config.get("resolution_rate", [])) != 1:
            raise ValueError("wan2.1_clean_interp_bridge expects exactly one lowres->highres stage.")
        if len(self.config.get("changing_resolution_steps", [])) != 1:
            raise ValueError("wan2.1_clean_interp_bridge expects exactly one changing_resolution step.")

    def init_scheduler(self):
        if self.config["feature_caching"] == "NoCaching":
            scheduler_class = WanScheduler
        elif self.config["feature_caching"] == "TaylorSeer":
            scheduler_class = WanSchedulerTaylorCaching
        elif self.config.feature_caching in ["Tea", "Ada", "Custom", "FirstBlock", "DualBlock", "DynamicBlock", "Mag"]:
            scheduler_class = WanSchedulerCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config.feature_caching}")

        self.scheduler = WanScheduler4CleanResizerBridgeInterface(scheduler_class, self.config)


@RUNNER_REGISTER("wan2.1_partial_denoise_decode")
class WanPartialDenoiseDecodeRunner(WanRunner):
    """Decode the clean x0 estimate after N steps from a full-step WAN schedule."""

    def __init__(self, config):
        super().__init__(config)
        stop_after_steps = int(self.config.get("stop_after_steps", 0))
        infer_steps = int(self.config["infer_steps"])
        if stop_after_steps < 1 or stop_after_steps > infer_steps:
            raise ValueError(f"stop_after_steps must be in [1, {infer_steps}], got {stop_after_steps}")
        if self.config.get("changing_resolution", False):
            raise ValueError("wan2.1_partial_denoise_decode expects a single-resolution config.")

    def run_segment(self, segment_idx=0):
        stop_after_steps = int(self.config["stop_after_steps"])

        for step_index in range(stop_after_steps):
            if self.video_segment_num == 1:
                self.check_stop()
            logger.info(f"==> partial step_index: {step_index + 1} / {self.model.scheduler.infer_steps}")

            self.model.scheduler.step_pre(step_index=step_index)
            self.model.infer(self.inputs)

            if step_index + 1 == stop_after_steps:
                latents = self._current_denoised_latent()
            else:
                self.model.scheduler.step_post()

        if segment_idx is not None and segment_idx == self.video_segment_num - 1:
            del self.inputs
            getattr(torch, AI_DEVICE).empty_cache()

        return latents

    def _current_denoised_latent(self):
        scheduler = self.model.scheduler
        model_output = scheduler.noise_pred.to(torch.float32)
        sample = scheduler.latents.to(torch.float32)
        sigma_t = scheduler.sigmas[scheduler.step_index]
        x0_pred = sample - sigma_t * model_output
        return x0_pred.to(dtype=scheduler.latents.dtype, device=scheduler.latents.device)
