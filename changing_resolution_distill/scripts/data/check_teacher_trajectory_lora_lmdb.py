from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data import TeacherTrajectoryLoRALMDBDataset  # noqa: E402


def main() -> None:
    args = parse_args()
    dataset = TeacherTrajectoryLoRALMDBDataset(args.data_dir, strict_channels=True)
    print(f"samples={len(dataset)} shards={len(dataset.shards)}")
    count = min(int(args.samples), len(dataset))
    for index in range(count):
        row = dataset[index]
        meta = json.loads(row["meta_json"])
        recipe = meta.get("teacher_trajectory_recipe", {})
        print(
            f"[{index}] id={row['sample_id']}"
            f" x={tuple(row['x_pre_train_step'].shape)}"
            f" z={tuple(row['z_teacher_final'].shape)}"
            f" seed={row['seed']}"
            f" train_step={recipe.get('train_step_index')}"
            f" sigma={recipe.get('train_sigma')}"
            f" prompt={row['prompt'][:80]!r}"
        )
        if row["x_pre_train_step"].shape != row["z_teacher_final"].shape:
            raise SystemExit("x_pre_train_step and z_teacher_final shapes do not match")
    print("Teacher trajectory LMDB check passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data/changing_resolution_distill/lmdb_teacher_trajectory_lora_14b_cfgdistill_5k_step3",
    )
    parser.add_argument("--samples", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
