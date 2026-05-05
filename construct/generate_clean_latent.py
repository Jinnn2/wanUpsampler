# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Generate clean latents for Wan 2.2 T2V model using TextDataset.
#
# Usage (multi-GPU with FSDP + Ulysses):
#   CUDA_VISIBLE_DEVICES=1,3,4,5 torchrun --nproc_per_node=4 generate_clean_latent.py \
#       --task t2v-A14B \
#       --size 832*480 \
#       --ckpt_dir /workspace/models/Wan-AI/Wan2.2-T2V-A14B \
#       --dit_fsdp \
#       --t5_fsdp \
#       --ulysses_size 4 \
#       --output_folder ./clean_latents \
#       --rawdata_path /path/to/prompts.txt

import argparse
import gc
import logging
import math
import os
import random
import sys
import warnings
from contextlib import contextmanager

warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
from tqdm import tqdm

import wan
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import str2bool

import lmdb
import numpy as np
from torch.utils.data import Dataset


# ---- LMDB utility functions ----
def get_array_shape_from_lmdb(env, array_name):
    with env.begin() as txn:
        image_shape = txn.get(f"{array_name}_shape".encode()).decode()
        image_shape = tuple(map(int, image_shape.split()))
    return image_shape


def retrieve_row_from_lmdb(lmdb_env, array_name, dtype, row_index, shape=None):
    """
    Retrieve a specific row from a specific array in the LMDB.
    """
    data_key = f'{array_name}_{row_index}_data'.encode()

    with lmdb_env.begin() as txn:
        row_bytes = txn.get(data_key)

    if dtype == str:
        array = row_bytes.decode()
    else:
        array = np.frombuffer(row_bytes, dtype=dtype)

    if shape is not None and len(shape) > 0:
        array = array.reshape(shape)
    return array


# ---- Dataset class ----
class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class LatentLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1]
        }


def generate_clean_latent(wan_t2v, prompt, size, frame_num, shift,
                          sample_solver, sampling_steps, guide_scale,
                          n_prompt, seed, offload_model):
    """
    Run the full ODE denoising process and return only the final clean latent.

    Args:
        wan_t2v: WanT2V pipeline instance.
        prompt: Text prompt string.
        size: Tuple (width, height).
        frame_num: Number of video frames.
        shift: Noise schedule shift parameter.
        sample_solver: 'unipc' or 'dpm++'.
        sampling_steps: Number of denoising steps.
        guide_scale: CFG scale, float or tuple (low_noise_scale, high_noise_scale).
        n_prompt: Negative prompt string.
        seed: Random seed.
        offload_model: Whether to offload models to CPU.
    Returns:
        torch.Tensor of shape [C, F, H, W] on rank 0, None on other ranks.
    """
    device = wan_t2v.device

    # Handle guide_scale: can be float, int, or tuple
    if isinstance(guide_scale, (float, int)):
        guide_scale = (guide_scale, guide_scale)

    F = frame_num
    target_shape = (
        wan_t2v.vae.model.z_dim,
        (F - 1) // wan_t2v.vae_stride[0] + 1,
        size[1] // wan_t2v.vae_stride[1],
        size[0] // wan_t2v.vae_stride[2],
    )

    seq_len = math.ceil(
        (target_shape[2] * target_shape[3]) /
        (wan_t2v.patch_size[1] * wan_t2v.patch_size[2]) *
        target_shape[1] / wan_t2v.sp_size
    ) * wan_t2v.sp_size

    if n_prompt == "":
        n_prompt = wan_t2v.sample_neg_prompt

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    # Encode text
    if not wan_t2v.t5_cpu:
        wan_t2v.text_encoder.model.to(device)
        context = wan_t2v.text_encoder([prompt], device)
        context_null = wan_t2v.text_encoder([n_prompt], device)
        if offload_model:
            wan_t2v.text_encoder.model.cpu()
    else:
        context = wan_t2v.text_encoder([prompt], torch.device('cpu'))
        context_null = wan_t2v.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(device) for t in context]
        context_null = [t.to(device) for t in context_null]

    # Generate initial noise
    noise = [
        torch.randn(
            target_shape[0], target_shape[1],
            target_shape[2], target_shape[3],
            dtype=torch.float32, device=device, generator=seed_g)
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_low = getattr(wan_t2v.low_noise_model, 'no_sync', noop_no_sync)
    no_sync_high = getattr(wan_t2v.high_noise_model, 'no_sync', noop_no_sync)

    with (
        torch.amp.autocast('cuda', dtype=wan_t2v.param_dtype),
        torch.no_grad(),
        no_sync_low(),
        no_sync_high(),
    ):
        boundary = wan_t2v.boundary * wan_t2v.num_train_timesteps

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=wan_t2v.num_train_timesteps,
                shift=1, use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=wan_t2v.num_train_timesteps,
                shift=1, use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError(f"Unsupported solver: {sample_solver}")

        latents = noise
        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        for t in tqdm(timesteps, disable=wan_t2v.rank != 0, desc="ODE steps"):
            latent_model_input = latents
            timestep = torch.stack([t])

            model = wan_t2v._prepare_model_for_timestep(
                t, boundary, offload_model)
            cur_guide_scale = (
                guide_scale[1] if t.item() >= boundary else guide_scale[0]
            )

            noise_pred_cond = model(
                latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = model(
                latent_model_input, t=timestep, **arg_null)[0]

            noise_pred = noise_pred_uncond + cur_guide_scale * (
                noise_pred_cond - noise_pred_uncond)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                return_dict=False, generator=seed_g)[0]
            latents = [temp_x0.squeeze(0)]

        clean_latent = latents[0].float().cpu().clone()

        if offload_model:
            wan_t2v.low_noise_model.cpu()
            wan_t2v.high_noise_model.cpu()
            torch.cuda.empty_cache()

    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return clean_latent if wan_t2v.rank == 0 else None


def main():
    parser = argparse.ArgumentParser(
        description="Generate final clean latents for Wan 2.2 T2V using TextDataset")
    parser.add_argument(
        "--task", type=str, default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size", type=str, default="832*480",
        choices=list(SIZE_CONFIGS.keys()),
        help="Video resolution (width*height).")
    parser.add_argument(
        "--frame_num", type=int, default=None,
        help="Number of video frames (must be 4n+1). Default from config.")
    parser.add_argument(
        "--ckpt_dir", type=str, required=True,
        help="Path to model checkpoint directory.")
    parser.add_argument(
        "--output_folder", type=str, default="./clean_latents",
        help="Output directory for clean latent .pt files.")
    parser.add_argument(
        "--rawdata_path", type=str, required=True,
        help="Path to prompt text file.")
    parser.add_argument(
        "--extended_prompt_path", type=str, default=None,
        help="Optional path to extended prompt text file.")
    parser.add_argument(
        "--offload_model", type=str2bool, default=None,
        help="Whether to offload model to CPU after forward.")
    parser.add_argument(
        "--ulysses_size", type=int, default=1,
        help="DeepSpeed Ulysses sequence parallel size.")
    parser.add_argument(
        "--t5_fsdp", action="store_true", default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu", action="store_true", default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp", action="store_true", default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--convert_model_dtype", action="store_true", default=False,
        help="Whether to convert model parameters dtype.")
    parser.add_argument(
        "--sample_solver", type=str, default='unipc',
        choices=['unipc', 'dpm++'],
        help="ODE solver.")
    parser.add_argument(
        "--sample_steps", type=int, default=None,
        help="Number of denoising steps. Default from config.")
    parser.add_argument(
        "--sample_shift", type=float, default=None,
        help="Noise schedule shift. Default from config.")
    parser.add_argument(
        "--sample_guide_scale", type=float, default=None,
        help="CFG scale. If not set, uses config default (may be a tuple).")
    parser.add_argument(
        "--base_seed", type=int, default=42,
        help="Base random seed.")
    args = parser.parse_args()

    # ---- Distributed setup ----
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model not specified, set to {args.offload_model}.")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://",
            rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), \
            "t5_fsdp and dit_fsdp require distributed environment."
        assert args.ulysses_size <= 1, \
            "Sequence parallelism requires distributed environment."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, \
            "ulysses_size must equal world_size."
        init_distributed_group()

    # ---- Config defaults ----
    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    if args.frame_num is None:
        args.frame_num = cfg.frame_num
    
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, \
            f"{cfg.num_heads=} not divisible by {args.ulysses_size=}."

    logging.info(f"Clean latent generation args: {args}")
    logging.info(f"Model config: {cfg}")

    # Broadcast seed
    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # ---- Create model ----
    logging.info("Creating WanT2V pipeline.")
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    # ---- Load dataset ----
    logging.info(f"Loading TextDataset from {args.rawdata_path}")
    dataset = TextDataset(
        prompt_path=args.rawdata_path,
        extended_prompt_path=args.extended_prompt_path,
    )
    logging.info(f"Total samples in dataset: {len(dataset)}")

    if rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    # ---- Generate clean latents ----
    # When using Ulysses sequence parallelism, all ranks work on the same
    # sample together, so iterate over every sample sequentially.
    # Without Ulysses (data parallelism), distribute samples across ranks.
    use_sp = args.ulysses_size > 1
    if use_sp:
        total_steps = len(dataset)
    else:
        total_steps = int(math.ceil(len(dataset) / world_size))
    
    for index in tqdm(
        range(total_steps), 
        disable=(rank != 0),
        desc="Generating clean latents"
    ):
        if use_sp:
            sample_index = index
        else:
            sample_index = index * world_size + rank
        if sample_index >= len(dataset):
            continue
        
        # Load sample from dataset
        sample = dataset[sample_index]
        prompt = sample.get("extended_prompts", sample["prompts"])
        
        logging.info(
            f"[{sample_index + 1}/{len(dataset)}] {prompt[:100]}...")

        clean_latent = generate_clean_latent(
            wan_t2v=wan_t2v,
            prompt=prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            n_prompt="",
            seed=args.base_seed + sample_index,
            offload_model=args.offload_model,
        )

        if clean_latent is not None:
            save_path = os.path.join(
                args.output_folder, f"{sample_index:05d}.pt")
            torch.save(
                {prompt: clean_latent.detach()},
                save_path
            )
            logging.info(
                f"Saved: {save_path}, shape: {list(clean_latent.shape)}")
            del clean_latent

        if dist.is_initialized():
            dist.barrier()

    # ---- Cleanup ----
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("All done.")


if __name__ == "__main__":
    main()
