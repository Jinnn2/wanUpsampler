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
            f" x3={tuple(row['x3_lr'].shape)}"
            f" z4={tuple(row['z4_lr_teacher'].shape)}"
            f" z0_hr={tuple(row['z0_hr'].shape)}"
            f" train_step={recipe.get('train_step_name')}"
            f" mode={recipe.get('mode')}"
        )

    first = dataset[0]
    if first["x3_lr"].shape != first["z4_lr_teacher"].shape:
        raise SystemExit("x3_lr and z4_lr_teacher shapes do not match")
    if first["z0_hr"].shape[-2] <= first["z4_lr_teacher"].shape[-2]:
        raise SystemExit("z0_hr height is not larger than LR target")
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


if __name__ == "__main__":
    main()
