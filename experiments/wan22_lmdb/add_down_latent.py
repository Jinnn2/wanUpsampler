import argparse
import torch
import torch.distributed as dist
import os
import sys
import numpy as np
import lmdb
from tqdm import tqdm
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.wan_wrapper import WanVAEWrapper
from utils.lmdb import (
    get_array_shape_from_lmdb,
    retrieve_row_from_lmdb,
    store_arrays_to_lmdb,
)

parser = argparse.ArgumentParser()
parser.add_argument("--src_lmdb", type=str,
                    default="/data/nvme0/lvchengtao/dataset/wanx_14B_shift-3.0_cfg-5.0_lmdb_70K",
                    help="Path to source sharded LMDB directory")
parser.add_argument("--dst_lmdb", type=str,
                    default="/data/nvme0/lvchengtao/dataset/wanx_14B_shift-3.0_cfg-5.0_lmdb_70K_with_down",
                    help="Path to destination sharded LMDB directory")
parser.add_argument("--down_h_factor", type=int, default=5,
                    help="Height downsampling factor in pixel space")
parser.add_argument("--down_w_factor", type=int, default=4,
                    help="Width downsampling factor in pixel space")
args = parser.parse_args()

# ---------- Initialize distributed training ----------
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
elif 'SLURM_PROCID' in os.environ:
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = rank % torch.cuda.device_count()
else:
    rank = 0
    world_size = 1
    local_rank = 0

dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
torch.set_grad_enabled(False)

# ---------- Load VAE ----------
if rank == 0:
    print("Loading VAE...")
vae = WanVAEWrapper()
vae = vae.to(device=device, dtype=torch.bfloat16)
vae.eval()

# ---------- Get shard list ----------
shard_names = sorted(os.listdir(args.src_lmdb))
num_shards = len(shard_names)

if rank == 0:
    os.makedirs(args.dst_lmdb, exist_ok=True)
    print(f"Source: {args.src_lmdb}, {num_shards} shards")
    print(f"Destination: {args.dst_lmdb}")
    print(f"Downsampling factors: h={args.down_h_factor}, w={args.down_w_factor}")
    print(f"World size: {world_size}")

dist.barrier()

# ---------- Assign shards to ranks (no concurrent writes) ----------
map_size = 5_000_000_000_000  # 5 TB

for shard_id in range(num_shards):
    if shard_id % world_size != rank:
        continue

    shard_name = shard_names[shard_id]
    src_path = os.path.join(args.src_lmdb, shard_name)
    dst_path = os.path.join(args.dst_lmdb, shard_name)

    # Open source LMDB (read-only)
    src_env = lmdb.open(src_path, readonly=True, lock=False,
                        readahead=False, meminit=False)
    latents_shape = get_array_shape_from_lmdb(src_env, 'latents')
    num_samples = latents_shape[0]

    print(f"Rank {rank}: processing {shard_name}, "
          f"{num_samples} samples, latents shape per row: {latents_shape[1:]}")

    # Open destination LMDB
    dst_env = lmdb.open(dst_path, map_size=map_size)

    down_latent_shape_per_row = None

    for idx in tqdm(range(num_samples),
                    desc=f"Rank {rank} {shard_name}",
                    disable=(rank != 0)):
        # 1. Read original data
        latents = retrieve_row_from_lmdb(
            src_env, "latents", np.float16, idx, shape=latents_shape[1:]
        )
        prompt = retrieve_row_from_lmdb(
            src_env, "prompts", str, idx
        )

        # 2. Get clean latent
        #    4D (T, C, H, W): single clean video latent
        #    5D (steps, T, C, H, W): take last step (clean)
        if len(latents.shape) == 4:
            clean_latent = torch.tensor(latents, dtype=torch.bfloat16).to(device)
            clean_latent = clean_latent.unsqueeze(0)  # [1, T, C, H, W]
        else:
            clean_latent = torch.tensor(latents[-1], dtype=torch.bfloat16).to(device)
            clean_latent = clean_latent.unsqueeze(0)  # [1, T, C, H, W]

        # 3. Decode to pixel → [B, T, C, pH, pW]
        video = vae.decode_to_pixel(clean_latent)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        vae.model.clear_cache()

        # 4. Downsample in pixel space
        b, t, c, h, w = video.shape
        video = rearrange(video, 'b t c h w -> (b t) c h w')
        video = torch.nn.functional.interpolate(
            video,
            size=(h // args.down_h_factor, w // args.down_w_factor),
            mode='area'
        )
        video_downsampled = rearrange(video, '(b t) c h w -> b t c h w', b=b, t=t)

        # 5. Normalize back to [-1, 1] and encode
        video_normalized = (video_downsampled * 2.0 - 1.0).to(device=device, dtype=torch.bfloat16)
        # Permute to [B, C, T, H, W] for VAE encode
        video_normalized = video_normalized.permute(0, 2, 1, 3, 4)
        down_latent = vae.encode_to_latent(video_normalized)
        vae.model.clear_cache()

        # down_latent: [B, T, C, nH, nW] → squeeze batch → [T, C, nH, nW]
        down_latent_np = down_latent[0].cpu().half().numpy()

        if down_latent_shape_per_row is None:
            down_latent_shape_per_row = down_latent_np.shape
            print(f"Rank {rank} {shard_name}: down_latents shape per row: "
                  f"{down_latent_shape_per_row}")

        # 6. Write all fields to destination LMDB
        store_arrays_to_lmdb(dst_env, {
            "latents":      np.array([latents]),        # [1, ...]
            "prompts":      np.array([prompt]),          # [1,]
            "down_latents": np.array([down_latent_np]),  # [1, T, C, nH, nW]
        }, start_index=idx)

    # ---------- Write shape metadata for this shard ----------
    with dst_env.begin(write=True) as txn:
        # latents shape (keep original)
        shape_str = " ".join(map(str, list(latents_shape)))
        txn.put("latents_shape".encode(), shape_str.encode())

        # prompts shape
        txn.put("prompts_shape".encode(), str(num_samples).encode())

        # down_latents shape
        if down_latent_shape_per_row is not None:
            down_shape = [num_samples] + list(down_latent_shape_per_row)
            shape_str = " ".join(map(str, down_shape))
            txn.put("down_latents_shape".encode(), shape_str.encode())

    print(f"Rank {rank}: finished {shard_name}, "
          f"{num_samples} samples written to {dst_path}")

    src_env.close()
    dst_env.close()

# ---------- Wait for all ranks to finish ----------
dist.barrier()

if rank == 0:
    print(f"Done. All {num_shards} shards processed and written to {args.dst_lmdb}")

# ---------- Cleanup ----------
dist.destroy_process_group()


'''
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 add_down_latent.py \
    --src_lmdb /data/nvme0/lvchengtao/dataset/tiny_latent \
    --dst_lmdb /data/nvme0/lvchengtao/dataset/tiny_latent_with_down_2x \
    --down_h_factor 2 \
    --down_w_factor 2

'''
