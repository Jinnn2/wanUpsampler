from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data import LastStepSkipLoRALMDBDataset  # noqa: E402


def main() -> None:
    args = parse_args()
    dataset = LastStepSkipLoRALMDBDataset(args.data_dir, dtype=torch.float32)
    print(f"data_dir: {args.data_dir}")
    print(f"samples : {len(dataset)}")
    print(f"shards  : {len(dataset.shards)}")

    if args.expect_samples is not None and len(dataset) != args.expect_samples:
        raise SystemExit(f"Expected {args.expect_samples} samples, got {len(dataset)}")

    indices = sorted(set([0, len(dataset) - 1, *range(min(args.limit, len(dataset)))]))
    for index in indices:
        row = dataset[index]
        meta = json.loads(row["meta_json"])
        recipe = meta.get("last_step_skip_recipe", {})
        print(
            "sample"
            f" index={index}"
            f" id={row['sample_id']}"
            f" seed={row['seed']}"
            f" x_pre_step3={tuple(row['x_pre_step3_lr'].shape)}"
            f" z4={tuple(row['z4_lr_teacher'].shape)}"
            f" z0_hr={tuple(row['z0_hr'].shape)}"
            f" train_step={recipe.get('train_step_name')}"
            f" input_semantic={recipe.get('semantic_input_name', meta.get('semantic_input_name'))}"
            f" input_step={recipe.get('actual_input_step', meta.get('actual_input_step'))}"
            f" mode={recipe.get('mode')}"
        )
        print_stats("  x_pre_step3", row["x_pre_step3_lr"])
        print_stats("  z4_teacher ", row["z4_lr_teacher"])
        print(f"  l1(x_pre_step3,z4_teacher)={torch.nn.functional.l1_loss(row['x_pre_step3_lr'], row['z4_lr_teacher']).item():.6f}")

    first = dataset[0]
    if first["x_pre_step3_lr"].shape != first["z4_lr_teacher"].shape:
        raise SystemExit("x_pre_step3_lr and z4_lr_teacher shapes do not match")
    if first["z0_hr"].shape[-2] <= first["z4_lr_teacher"].shape[-2]:
        raise SystemExit("z0_hr height is not larger than LR target")
    first_meta = json.loads(first["meta_json"])
    recipe = first_meta.get("last_step_skip_recipe", {})
    semantic = recipe.get("semantic_input_name", first_meta.get("semantic_input_name"))
    if semantic and semantic != "x_pre_step3_lr":
        raise SystemExit(f"Unexpected semantic input name: {semantic}")
    print("Last-step-skip LoRA LMDB check passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3",
    )
    parser.add_argument("--expect_samples", type=int)
    parser.add_argument("--limit", type=int, default=3)
    return parser.parse_args()


def print_stats(label: str, tensor: torch.Tensor) -> None:
    tensor_f = tensor.float()
    print(
        f"{label}:"
        f" mean={tensor_f.mean().item():.6f}"
        f" std={tensor_f.std().item():.6f}"
        f" min={tensor_f.min().item():.6f}"
        f" max={tensor_f.max().item():.6f}"
    )


if __name__ == "__main__":
    main()
