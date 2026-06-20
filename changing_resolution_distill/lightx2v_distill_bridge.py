from __future__ import annotations

import sys
from pathlib import Path

import torch
from loguru import logger

from changing_resolution.lightx2v_clean_bridge import WanV2CleanLatentResizerBridge
from lightx2v.models.networks.wan.distill_model import WanDistillModel
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner
from lightx2v.models.schedulers.wan.step_distill.scheduler import WanStepDistillScheduler
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
            raise ValueError("resolution_rate and changing_resolution_steps must have the same length")
        self.clean_latent_resizer = None

    def set_clean_latent_resizer(self, resizer):
        self.clean_latent_resizer = resizer

    def prepare_latents(self, seed, latent_shape, dtype=torch.float32):
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
        sigma = self.sigmas[self.step_index].to(device=sample.device, dtype=torch.float32)
        x0_pred = sample - sigma * flow_pred
        clean_sample = self._resize_clean_latent_to_next_stage(x0_pred.to(sample.dtype))

        if self.step_index + 1 >= self.infer_steps:
            logger.info("Distill changing-resolution at final step; keep resized clean latent.")
            self.latents = clean_sample
            return

        sigma_next = self.sigmas[self.step_index + 1].to(device=sample.device, dtype=torch.float32)
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
            target_noise = self.latents_list[self.changing_resolution_index + 1].to(torch.float32)
            noisy_sample = self.add_noise(clean_sample.to(torch.float32), target_noise, sigma_next)
        else:
            raise ValueError(
                "wan_distill_bridge_renoise_mode must be 'resize_flow' or 'random', "
                f"got {renoise_mode!r}"
            )
        self.latents = noisy_sample.to(dtype=self.latents.dtype)

    def _resize_clean_latent_to_next_stage(self, denoised_sample):
        target_latent_shape = self.latents_list[self.changing_resolution_index + 1].shape
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


@RUNNER_REGISTER("wan2.1_distill_last_step_lora")
class WanDistillLastStepLoRARunner(WanDistillRunner):
    """WAN 4-step distill runner with LoRA enabled only on configured denoise steps."""

    def __init__(self, config):
        super().__init__(config)
        self.lora_configs = list(config.get("lora_configs") or [])
        if len(self.lora_configs) != 1:
            raise ValueError("wan2.1_distill_last_step_lora expects exactly one LoRA config.")
        self.lora_path = self.lora_configs[0]["path"]
        self.lora_strength = float(self.lora_configs[0].get("strength", 1.0))
        self.lora_active_steps = {
            int(step)
            for step in config.get("lora_active_steps", [int(config.get("infer_steps", 3))])
        }

    def load_transformer(self):
        wan_model_kwargs = {
            "model_path": self.config["model_path"],
            "config": self.config,
            "device": self.init_device,
            "lora_path": self.lora_path,
            "lora_strength": 0.0,
        }
        return WanDistillModel(**wan_model_kwargs)

    def run_segment(self, segment_idx=0):
        infer_steps = self.model.scheduler.infer_steps
        device_module = getattr(torch, AI_DEVICE)

        for step_index in range(infer_steps):
            if self.video_segment_num == 1:
                self.check_stop()
            step_number = step_index + 1
            strength = self.lora_strength if step_number in self.lora_active_steps else 0.0
            logger.info(f"==> step_index: {step_number} / {infer_steps}, lora_strength={strength}")
            self.model._update_lora(self.lora_path, strength)
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


@RUNNER_REGISTER("wan2.1_distill_clean_resizer_bridge")
class WanDistillCleanResizerBridgeRunner(WanDistillRunner):
    """WAN 4-step distill changing-resolution runner backed by the Stage 3 resizer."""

    def __init__(self, config):
        super().__init__(config)
        self._validate_bridge_config()

    def _validate_bridge_config(self):
        if self.config["task"] != "t2v":
            raise NotImplementedError("wan2.1_distill_clean_resizer_bridge currently only supports t2v.")
        if self.config.get("use_tae", False):
            raise ValueError("wan2.1_distill_clean_resizer_bridge requires the full WAN VAE decoder, not TAE.")
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            raise ValueError("wan2.1_distill_clean_resizer_bridge does not support lazy_load/unload_modules yet.")
        if len(self.config.get("resolution_rate", [])) != 1:
            raise ValueError("wan2.1_distill_clean_resizer_bridge expects exactly one lowres->highres stage.")
        if len(self.config.get("changing_resolution_steps", [])) != 1:
            raise ValueError("wan2.1_distill_clean_resizer_bridge expects exactly one changing_resolution step.")
        if self.config.get("wan_clean_resizer_ckpt") is None:
            raise ValueError("wan2.1_distill_clean_resizer_bridge requires wan_clean_resizer_ckpt in config.")
        if self.config.get("wan_clean_resizer_repo") is None:
            raise ValueError("wan2.1_distill_clean_resizer_bridge requires wan_clean_resizer_repo in config.")

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
        logger.info(f"Initialized Wan distill clean resizer model_type={model_type} from {ckpt_path}")
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


@RUNNER_REGISTER("wan2.1_distill_interp_bridge")
class WanDistillInterpBridgeRunner(WanDistillRunner):
    """WAN 4-step distill changing-resolution runner with trilinear clean-latent resize."""

    def __init__(self, config):
        super().__init__(config)
        self._validate_interp_config()

    def _validate_interp_config(self):
        if self.config["task"] != "t2v":
            raise NotImplementedError("wan2.1_distill_interp_bridge currently only supports t2v.")
        if len(self.config.get("resolution_rate", [])) != 1:
            raise ValueError("wan2.1_distill_interp_bridge expects exactly one lowres->highres stage.")
        if len(self.config.get("changing_resolution_steps", [])) != 1:
            raise ValueError("wan2.1_distill_interp_bridge expects exactly one changing_resolution step.")

    def init_scheduler(self):
        if self.config["feature_caching"] != "NoCaching":
            raise NotImplementedError("wan2.1_distill_interp_bridge currently supports only NoCaching.")
        self.scheduler = WanStepDistillScheduler4CleanResizerBridge(self.config)
