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
    review_dir, private_dir = review_paths(root, args.review_name)
    video_dir = review_dir / "blinded"
    review_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.review_seed)
    ballots: list[dict[str, str]] = []
    keys: list[dict[str, str]] = []
    available_steps = sorted({int(case["step"]) for case in manifest["cases"]})
    selected_steps = set(args.step or available_steps)
    unknown_steps = sorted(selected_steps - set(available_steps))
    if unknown_steps:
        raise SystemExit(f"Requested steps are absent from the factorial: {unknown_steps}")
    selected_comparisons = set(args.comparison or [])
    for sample_index, _prompt in enumerate(manifest["prompts"], start=int(manifest["prompt_offset"])):
        seed = int(manifest["seed_base"]) + sample_index
        label = f"{sample_index:02d}"
        for step in available_steps:
            if step not in selected_steps:
                continue
            pairs = comparison_pairs(step, manifest)
            for comparison, left_case, right_case in pairs:
                if selected_comparisons and comparison not in selected_comparisons:
                    continue
                source_left = case_video(root, left_case, label, seed)
                source_right = case_video(root, right_case, label, seed)
                if not source_left.is_file() or not source_right.is_file():
                    continue
                if rng.random() < 0.5:
                    source_left, source_right = source_right, source_left
                    left_case, right_case = right_case, left_case
                blind_id = make_blind_id(
                    args.review_seed,
                    str(manifest["family"]),
                    sample_index,
                    seed,
                    comparison,
                    step,
                    multi_step=len(available_steps) > 1,
                    namespace=args.review_name,
                )
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


def comparison_pairs(step: int, manifest: dict | None = None) -> list[tuple[str, str, str]]:
    configured = (manifest or {}).get("review_pairs")
    if configured is not None:
        pairs = []
        for item in configured:
            pair_step = int(item.get("step", step))
            if pair_step == step:
                pairs.append((str(item["comparison"]), str(item["left_case"]), str(item["right_case"])))
        return pairs
    prefix = f"step{step}"
    return [
        ("stage2_at_base", f"{prefix}_base_interp", f"{prefix}_base_stage2"),
        ("lora_with_interp", f"{prefix}_base_interp", f"{prefix}_lora_interp"),
        ("lora_with_stage2", f"{prefix}_base_stage2", f"{prefix}_lora_stage2"),
        ("talh_vs_plain_handoff", f"{prefix}_base_interp", f"{prefix}_lora_stage2"),
    ]


def make_blind_id(
    review_seed: int,
    family: str,
    sample_index: int,
    seed: int,
    comparison: str,
    step: int,
    *,
    multi_step: bool,
    namespace: str = "default",
) -> str:
    identity = f"{review_seed}:{family}:{sample_index}:{seed}:{comparison}"
    # Preserve IDs for single-step families such as distill4, while
    # disambiguating Wan50 packages that contain both step40 and step45.
    if multi_step:
        identity += f":step{step}"
    if namespace != "default":
        identity += f":review={namespace}"
    return hashlib.sha256(identity.encode()).hexdigest()[:12]


def review_paths(root: Path, review_name: str) -> tuple[Path, Path]:
    if review_name == "default":
        return root / "review", root / "_private"
    return root / "review" / review_name, root / "_private" / review_name


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
    parser.add_argument(
        "--review-name",
        default="default",
        help="Named review package; non-default packages are isolated under review/NAME and _private/NAME.",
    )
    parser.add_argument("--step", type=int, action="append", default=[])
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
    )
    args = parser.parse_args()
    if Path(args.review_name).name != args.review_name or args.review_name in {"", ".", ".."}:
        parser.error("--review-name must be one path-safe directory name")
    return args


if __name__ == "__main__":
    main()
