#!/usr/bin/env python3
"""Token attribution for the nonlinear B4 router via leave-one-token-out effects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("token_attribution_b4")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_uni.scripts.router.model_router import (  # noqa: E402
    SoftDistillationMLPRouter,
)
from changing_resolution_uni.scripts.router.token_word_utils import (  # noqa: E402
    ENGLISH_STOPWORDS,
    clean_token,
    merge_subtokens_to_words,
    summarize_attributions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(
            REPO_ROOT / "outputs" / "router_benchmarks_1k" / "mlp_distill_router.pt"
        ),
    )
    parser.add_argument(
        "--dataset_dir",
        default=str(
            REPO_ROOT / "data" / "changing_resolution_uni" / "oracle_dataset_1k"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=str(
            REPO_ROOT / "outputs" / "router_benchmarks_1k" / "token_attribution_b4"
        ),
    )
    parser.add_argument(
        "--t5_dir",
        default=None,
        help="Optional seq_embedding/token metadata directory.",
    )
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--min_word_count", type=int, default=3)
    parser.add_argument("--include_stopwords", action="store_true")
    parser.add_argument("--attribution_batch_size", type=int, default=64)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.top_k < 1 or args.min_word_count < 1 or args.attribution_batch_size < 1:
        parser.error(
            "top_k, min_word_count, and attribution_batch_size must be positive"
        )
    return args


def safe_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        logger.warning(
            "This PyTorch version lacks weights_only=True; using legacy loader"
        )
        return torch.load(path, map_location="cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def expected_step(
    model: SoftDistillationMLPRouter,
    pooled: torch.Tensor,
    candidate_steps: torch.Tensor,
) -> torch.Tensor:
    probabilities = model(pooled)["discrete_probs"]
    return (probabilities * candidate_steps.unsqueeze(0)).sum(dim=-1)


@torch.no_grad()
def leave_one_out_expected_step_deltas(
    model: SoftDistillationMLPRouter,
    seq_embedding: np.ndarray,
    pooled_embedding: np.ndarray,
    candidate_steps: list[int],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float, int]:
    """Return full-minus-token-removed expected-step deltas for every token."""
    sequence = torch.from_numpy(np.asarray(seq_embedding, dtype=np.float32)).to(device)
    if sequence.ndim != 2 or sequence.shape[1] != 4096:
        raise ValueError(
            f"expected seq_embedding [L,4096], got {tuple(sequence.shape)}"
        )
    pooled = torch.from_numpy(np.asarray(pooled_embedding, dtype=np.float32)).to(device)
    if pooled.shape != (4096,):
        raise ValueError(f"expected pooled_embedding [4096], got {tuple(pooled.shape)}")
    steps = torch.tensor(candidate_steps, dtype=torch.float32, device=device)
    full_output = model(pooled.unsqueeze(0))
    full_expected = float(
        (full_output["discrete_probs"] * steps.unsqueeze(0)).sum().item()
    )
    predicted_index = int(full_output["pred_step_idx"].item())

    token_count = sequence.shape[0]
    if token_count <= 1:
        return np.zeros(token_count, dtype=np.float32), full_expected, predicted_index
    removed_pooled = (sequence.sum(dim=0, keepdim=True) - sequence) / (token_count - 1)
    removed_scores: list[torch.Tensor] = []
    for start in range(0, token_count, batch_size):
        removed_scores.append(
            expected_step(
                model,
                removed_pooled[start : start + batch_size],
                steps,
            ).cpu()
        )
    removed = torch.cat(removed_scores)
    return (full_expected - removed.numpy()), full_expected, predicted_index


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = safe_load_checkpoint(checkpoint_path)
    model_type = checkpoint.get("model_type")
    if model_type != "mlp_distill":
        raise ValueError(
            f"B4 attribution requires model_type='mlp_distill', got {model_type!r}"
        )
    candidate_steps = list(checkpoint.get("candidate_steps", [30, 35, *range(40, 51)]))
    model = SoftDistillationMLPRouter(
        in_dim=4096,
        hidden_dims=[256, 128],
        num_classes=len(candidate_steps),
        dropout=0.1,
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()
    logger.info("Loaded B4 SoftDistillationMLPRouter on %s", device)

    dataset_root = Path(args.dataset_dir).resolve()
    t5_dir = (
        Path(args.t5_dir).resolve() if args.t5_dir else dataset_root / "t5_embeddings"
    )
    npz_files = sorted(t5_dir.glob("prompt_*.npz"))
    manifest_path = dataset_root / "dataset_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record_names = manifest.get("record_files", [])
        if isinstance(record_names, list) and record_names:
            selected_prompt_ids = {
                int(str(name).split("_s", 1)[0].removeprefix("p"))
                for name in record_names
            }
            npz_files = [
                path
                for path in npz_files
                if int(path.stem.split("_")[1]) in selected_prompt_ids
            ]
    if not npz_files:
        raise RuntimeError(f"No selected prompt embeddings found under {t5_dir}")

    token_scores: dict[str, list[float]] = defaultdict(list)
    natural_word_scores: dict[str, list[dict[str, float]]] = defaultdict(list)
    sample_attributions: list[dict[str, Any]] = []
    processing_errors: list[str] = []
    for npz_path in npz_files:
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                if "seq_embedding" not in data or "pooled_embedding" not in data:
                    raise ValueError("missing seq_embedding or pooled_embedding")
                sequence = np.asarray(data["seq_embedding"], dtype=np.float32)
                pooled = np.asarray(data["pooled_embedding"], dtype=np.float32)
            prompt_id = int(npz_path.stem.split("_")[1])
            metadata_path = npz_path.with_suffix(".json")
            if not metadata_path.is_file():
                raise ValueError(f"missing token metadata {metadata_path.name}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            tokens = [str(token) for token in metadata.get("tokens", [])]
            prompt_text = str(metadata.get("prompt_text", ""))
            if len(tokens) != len(sequence):
                raise ValueError(
                    f"token/sequence mismatch: tokens={len(tokens)} sequence={len(sequence)}"
                )

            deltas, full_expected, predicted_index = leave_one_out_expected_step_deltas(
                model,
                sequence,
                pooled,
                candidate_steps,
                device,
                args.attribution_batch_size,
            )
            token_list = []
            for token, delta in zip(tokens, deltas):
                cleaned = clean_token(token)
                if len(cleaned) >= 2 and not cleaned.startswith("<"):
                    token_scores[cleaned.casefold()].append(float(delta))
                    token_list.append(
                        {
                            "token": cleaned,
                            "expected_step_delta": round(float(delta), 6),
                        }
                    )
            natural_words = merge_subtokens_to_words(tokens, deltas)
            for occurrence in natural_words:
                natural_word_scores[occurrence["word"]].append(
                    {
                        "mean_piece_attribution": float(
                            occurrence["mean_piece_attribution"]
                        ),
                        "additive_contribution": float(
                            occurrence["additive_contribution"]
                        ),
                        "subtoken_count": float(occurrence["subtoken_count"]),
                    }
                )
            sample_attributions.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "expected_step": round(full_expected, 6),
                    "predicted_step": candidate_steps[predicted_index],
                    "tokens": token_list,
                    "natural_words": natural_words,
                }
            )
        except Exception as exc:
            processing_errors.append(f"{npz_path.name}: {exc}")

    if processing_errors:
        preview = "\n".join(f"  - {item}" for item in processing_errors[:20])
        suffix = (
            ""
            if len(processing_errors) <= 20
            else f"\n  ... and {len(processing_errors) - 20} more"
        )
        raise RuntimeError(
            "B4 token attribution failed closed on incomplete inputs:\n"
            f"{preview}{suffix}"
        )

    token_summary = []
    for token, scores in token_scores.items():
        if len(scores) >= args.min_word_count:
            token_summary.append(
                {
                    "word": token,
                    "count": len(scores),
                    "mean_attribution": float(np.mean(scores)),
                    "std_attribution": float(np.std(scores)),
                }
            )
    token_summary.sort(key=lambda row: row["mean_attribution"], reverse=True)
    natural_summary = summarize_attributions(
        natural_word_scores, minimum_count=args.min_word_count
    )
    if not natural_summary:
        raise RuntimeError("No natural words survived attribution filtering")
    natural_summary.sort(key=lambda row: row["mean_attribution"], reverse=True)
    ranking_candidates = [
        row
        for row in natural_summary
        if args.include_stopwords or row["word"] not in ENGLISH_STOPWORDS
    ]
    top_late = [row for row in ranking_candidates if row["mean_attribution"] > 0.0][
        : args.top_k
    ]
    top_early = sorted(
        [row for row in ranking_candidates if row["mean_attribution"] < 0.0],
        key=lambda row: row["mean_attribution"],
    )[: args.top_k]

    word_fields = [
        "rank",
        "word",
        "mean_expected_step_delta",
        "std_expected_step_delta",
        "mean_subtokens",
        "count",
    ]

    def write_word_ranking(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=word_fields)
            writer.writeheader()
            for rank, row in enumerate(rows, 1):
                writer.writerow(
                    {
                        "rank": rank,
                        "word": row["word"],
                        "mean_expected_step_delta": f"{row['mean_attribution']:+.8f}",
                        "std_expected_step_delta": f"{row['std_attribution']:.8f}",
                        "mean_subtokens": f"{row['mean_subtokens']:.3f}",
                        "count": row["count"],
                    }
                )

    write_word_ranking(out_dir / "top_late_switch_words.csv", top_late)
    write_word_ranking(out_dir / "top_early_switch_words.csv", top_early)
    write_word_ranking(out_dir / "natural_word_attributions.csv", natural_summary)

    token_fields = [
        "rank",
        "token",
        "mean_expected_step_delta",
        "std_expected_step_delta",
        "count",
    ]
    for filename, rows in (
        (
            "top_late_switch_tokens.csv",
            [row for row in token_summary if row["mean_attribution"] > 0.0][
                : args.top_k
            ],
        ),
        (
            "top_early_switch_tokens.csv",
            sorted(
                [row for row in token_summary if row["mean_attribution"] < 0.0],
                key=lambda row: row["mean_attribution"],
            )[: args.top_k],
        ),
    ):
        with (out_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=token_fields)
            writer.writeheader()
            for rank, row in enumerate(rows, 1):
                writer.writerow(
                    {
                        "rank": rank,
                        "token": row["word"],
                        "mean_expected_step_delta": f"{row['mean_attribution']:+.8f}",
                        "std_expected_step_delta": f"{row['std_attribution']:.8f}",
                        "count": row["count"],
                    }
                )

    (out_dir / "sample_token_attributions.json").write_text(
        json.dumps(sample_attributions[:100], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    attribution_metadata = {
        "schema": "b4_leave_one_out_token_attribution_v1",
        "model_type": "mlp_distill",
        "model_label": "Soft Distillation MLP (B4)",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "primary_lambda": checkpoint.get("primary_lambda"),
        "prompt_count": len(sample_attributions),
        "sample_prompt_count": min(100, len(sample_attributions)),
        "natural_vocabulary_size": len(natural_summary),
        "subtoken_vocabulary_size": len(token_summary),
        "minimum_occurrence_count": args.min_word_count,
        "stopwords_in_top_rankings": args.include_stopwords,
        "attribution_method": "leave_one_token_out_masked_mean_pooling",
        "attribution_target": "expected_candidate_timestep",
        "attribution_definition": (
            "full prompt expected timestep minus expected timestep after removing "
            "one token from masked-mean pooling"
        ),
        "positive_direction": "later_switch_stay_lr",
        "negative_direction": "earlier_switch_go_hr",
        "ranking_sign_filter": "late rankings are positive; early rankings are negative",
        "additive_completeness": False,
        "natural_word_aggregation": (
            "mean of constituent token-piece leave-one-out effects; not joint word removal"
        ),
    }
    (out_dir / "attribution_metadata.json").write_text(
        json.dumps(attribution_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 96)
    print(" B4 LEAVE-ONE-TOKEN-OUT ATTRIBUTION OF EXPECTED TIMESTEP")
    print("=" * 96)
    print(
        f"{'Words pushing LATER switch (stay LR)':<46} | "
        f"{'Words pushing EARLIER switch (go HR)':<46}"
    )
    print("-" * 96)
    for index in range(min(15, len(top_late), len(top_early))):
        late = f"{top_late[index]['word']} ({top_late[index]['mean_attribution']:+.4f})"
        early = (
            f"{top_early[index]['word']} ({top_early[index]['mean_attribution']:+.4f})"
        )
        print(f"{late:<46} | {early:<46}")
    print("=" * 96)
    logger.info("B4 token attribution saved to %s", out_dir)


if __name__ == "__main__":
    main()
