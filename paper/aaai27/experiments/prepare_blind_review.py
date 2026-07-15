from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path


DIMENSIONS = ["detail", "artifact_cleanliness", "temporal_stability", "structure_identity", "overall"]


def main() -> None:
    args = parse_args()
    root = Path(args.factorial_root)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    review_dir = root / "review"
    private_dir = root / "_private"
    video_dir = review_dir / "blinded"
    review_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.review_seed)
    ballots: list[dict[str, str]] = []
    keys: list[dict[str, str]] = []
    for sample_index, _prompt in enumerate(manifest["prompts"], start=int(manifest["prompt_offset"])):
        seed = int(manifest["seed_base"]) + sample_index
        label = f"{sample_index:02d}"
        for step in sorted({int(case["step"]) for case in manifest["cases"]}):
            pairs = comparison_pairs(step)
            for comparison, left_case, right_case in pairs:
                source_left = case_video(root, left_case, label, seed)
                source_right = case_video(root, right_case, label, seed)
                if not source_left.is_file() or not source_right.is_file():
                    continue
                if rng.random() < 0.5:
                    source_left, source_right = source_right, source_left
                    left_case, right_case = right_case, left_case
                blind_id = hashlib.sha256(
                    f"{args.review_seed}:{manifest['family']}:{sample_index}:{seed}:{comparison}".encode()
                ).hexdigest()[:12]
                blind_left = video_dir / f"{blind_id}_A.mp4"
                blind_right = video_dir / f"{blind_id}_B.mp4"
                link(source_left, blind_left)
                link(source_right, blind_right)
                ballot = {
                    "blind_id": blind_id,
                    "video_A": str(blind_left),
                    "video_B": str(blind_right),
                    **{f"{dimension}_winner_A_B_tie": "" for dimension in DIMENSIONS},
                    "confidence_1_to_5": "",
                    "severe_failure_A_B_neither": "",
                    "notes": "",
                }
                ballots.append(ballot)
                keys.append(
                    {
                        "blind_id": blind_id,
                        "family": manifest["family"],
                        "comparison": comparison,
                        "sample_index": str(sample_index),
                        "seed": str(seed),
                        "case_A": left_case,
                        "case_B": right_case,
                    }
                )

    write_csv(review_dir / "human_ratings.csv", ballots)
    write_csv(private_dir / "human_review_key.csv", keys)
    print(f"Blinded pairs: {len(ballots)}")
    print(f"Rater ballot : {review_dir / 'human_ratings.csv'}")
    print(f"Private key  : {private_dir / 'human_review_key.csv'}")


def comparison_pairs(step: int) -> list[tuple[str, str, str]]:
    prefix = f"step{step}"
    return [
        ("stage2_at_base", f"{prefix}_base_interp", f"{prefix}_base_stage2"),
        ("lora_with_interp", f"{prefix}_base_interp", f"{prefix}_lora_interp"),
        ("lora_with_stage2", f"{prefix}_base_stage2", f"{prefix}_lora_stage2"),
        ("talh_vs_plain_handoff", f"{prefix}_base_interp", f"{prefix}_lora_stage2"),
    ]


def case_video(root: Path, case: str, label: str, seed: int) -> Path:
    return root / "videos" / case / f"{case}_{label}_seed{seed}.mp4"


def link(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        destination.symlink_to(source.resolve())


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise SystemExit("No complete factorial pairs were found")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create method-hidden, randomized A/B review pairs.")
    parser.add_argument("--factorial-root", required=True)
    parser.add_argument("--review-seed", type=int, default=202707)
    return parser.parse_args()


if __name__ == "__main__":
    main()
