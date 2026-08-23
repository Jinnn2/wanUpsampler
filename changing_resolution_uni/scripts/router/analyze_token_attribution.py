#!/usr/bin/env python3
"""
Reverse Token Attribution & Semantic Analysis for Linear Ordinal Switch Router.
Extracts r_i = w^T h_i for every token in prompts, discovers top early-switch
and late-switch semantic keywords, and performs counterfactual prompt interventions.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("token_attribution")

import sys

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_uni.scripts.router.model_router import LinearOrdinalRouter
from changing_resolution_uni.scripts.router.token_word_utils import (
    ENGLISH_STOPWORDS,
    clean_token,
    merge_subtokens_to_words,
    summarize_attributions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze token attributions of linear ordinal router.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "outputs" / "router_benchmarks_1k" / "linear_ordinal_router.pt"),
        help="Path to trained linear ordinal router checkpoint.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(REPO_ROOT / "data" / "changing_resolution_uni" / "oracle_dataset_1k"),
        help="Path to dataset directory containing t5_embeddings/ and records/.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "router_benchmarks_1k" / "token_attribution"),
        help="Output directory for attribution reports and tables.",
    )
    parser.add_argument("--top_k", type=int, default=30, help="Top K words to export.")
    parser.add_argument(
        "--min_word_count",
        type=int,
        default=3,
        help="Minimum number of natural-word occurrences used in Top Words.",
    )
    parser.add_argument(
        "--include_stopwords",
        action="store_true",
        help="Include common English function words in natural-language Top Words.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Train the router first!")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cand_steps = ckpt.get("candidate_steps", [30, 35, *range(40, 51)])
    K = len(cand_steps)

    model = LinearOrdinalRouter(in_dim=4096, num_classes=K)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # w: [4096]
    w = model.linear.weight.squeeze(0).detach().numpy()
    thresholds = model.get_monotonic_thresholds().detach().numpy()

    logger.info(f"Loaded LinearOrdinalRouter (Weight norm: {np.linalg.norm(w):.4f}, Thresholds: {np.round(thresholds, 2)})")

    dataset_root = Path(args.dataset_dir).resolve()
    t5_dir = dataset_root / "t5_embeddings"
    records_dir = dataset_root / "records"

    # Gather token statistics across dataset
    token_scores: dict[str, list[float]] = defaultdict(list)
    natural_word_scores: dict[str, list[dict[str, float]]] = defaultdict(list)
    sample_attributions: list[dict[str, Any]] = []

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
    logger.info(f"Analyzing {len(npz_files)} prompt T5 token sequences...")

    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            seq_emb = data["seq_embedding"].astype(np.float32)  # [L, 4096]
            pid = int(npz_file.stem.split("_")[1])

            # Load tokens and prompt_text from corresponding JSON metadata file
            meta_json = t5_dir / f"{npz_file.stem}.json"
            tokens = []
            prompt_text = ""
            if meta_json.is_file():
                try:
                    m_data = json.loads(meta_json.read_text(encoding="utf-8"))
                    tokens = m_data.get("tokens", [])
                    prompt_text = m_data.get("prompt_text", "")
                except Exception:
                    pass
            if not tokens and "tokens" in data:
                tokens = list(data["tokens"])
            if not prompt_text and "prompt_text" in data:
                prompt_text = str(data["prompt_text"])

            # If token count doesn't match seq_len, pad or slice
            if len(tokens) < len(seq_emb):
                tokens = tokens + [f"tok_{i}" for i in range(len(tokens), len(seq_emb))]
            elif len(tokens) > len(seq_emb):
                tokens = tokens[: len(seq_emb)]

            # Exact token attribution r_i = w^T h_i
            r_scores = np.dot(seq_emb, w.astype(np.float32))  # [L]

            # Model prediction on this prompt
            pooled = data["pooled_embedding"].astype(np.float32)
            with torch.no_grad():
                out = model(torch.from_numpy(pooled).float().unsqueeze(0))
                pred_idx = out["pred_step_idx"].item()
                pred_step = cand_steps[pred_idx]
                switch_score = out["switch_score"].item()

            token_list = []
            for tok, score in zip(tokens, r_scores):
                c_tok = clean_token(str(tok))
                if len(c_tok) >= 2 and not c_tok.startswith("<") and not c_tok.endswith(">"):
                    token_scores[c_tok.lower()].append(float(score))
                    token_list.append({"token": c_tok, "attribution": round(float(score), 4)})
            natural_words = merge_subtokens_to_words(
                [str(token) for token in tokens], r_scores
            )
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

            sample_attributions.append({
                "prompt_id": pid,
                "prompt_text": prompt_text,
                "switch_score": round(switch_score, 4),
                "predicted_step": pred_step,
                "tokens": token_list,
                "natural_words": natural_words,
            })
        except Exception as e:
            logger.warning(f"Error processing {npz_file}: {e}")

    # Preserve subtoken-level results for traceability.
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

    # Natural words merge all SentencePiece pieces belonging to one occurrence.
    natural_summary = summarize_attributions(
        natural_word_scores, minimum_count=args.min_word_count
    )
    natural_summary.sort(key=lambda row: row["mean_attribution"], reverse=True)
    ranking_candidates = [
        row
        for row in natural_summary
        if args.include_stopwords or row["word"] not in ENGLISH_STOPWORDS
    ]
    top_late = ranking_candidates[: args.top_k]
    top_early = sorted(
        ranking_candidates, key=lambda row: row["mean_attribution"]
    )[: args.top_k]

    word_fields = [
        "rank",
        "word",
        "mean_attribution",
        "std_attribution",
        "mean_additive_contribution",
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
                        "mean_attribution": f"{row['mean_attribution']:+.6f}",
                        "std_attribution": f"{row['std_attribution']:.6f}",
                        "mean_additive_contribution": (
                            f"{row['mean_additive_contribution']:+.8f}"
                        ),
                        "mean_subtokens": f"{row['mean_subtokens']:.3f}",
                        "count": row["count"],
                    }
                )

    write_word_ranking(out_dir / "top_late_switch_words.csv", top_late)
    write_word_ranking(out_dir / "top_early_switch_words.csv", top_early)
    write_word_ranking(out_dir / "natural_word_attributions.csv", natural_summary)

    token_fields = ["rank", "token", "mean_attribution", "std_attribution", "count"]
    for filename, rows in (
        ("top_late_switch_tokens.csv", token_summary[: args.top_k]),
        (
            "top_early_switch_tokens.csv",
            sorted(token_summary, key=lambda row: row["mean_attribution"])[
                : args.top_k
            ],
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
                        "mean_attribution": f"{row['mean_attribution']:+.6f}",
                        "std_attribution": f"{row['std_attribution']:.6f}",
                        "count": row["count"],
                    }
                )

    # Export Sample Attributions JSON
    sample_json = out_dir / "sample_token_attributions.json"
    sample_json.write_text(json.dumps(sample_attributions[:100], indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "attribution_metadata.json").write_text(
        json.dumps(
            {
                "schema": "natural_word_token_attribution_v1",
                "prompt_count": len(sample_attributions),
                "natural_vocabulary_size": len(natural_summary),
                "subtoken_vocabulary_size": len(token_summary),
                "minimum_occurrence_count": args.min_word_count,
                "stopwords_in_top_rankings": args.include_stopwords,
                "word_aggregation": (
                    "SentencePiece or WordPiece pieces are merged per occurrence; "
                    "mean_attribution averages piece scores and "
                    "mean_additive_contribution sums piece scores divided by prompt "
                    "token count."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Print Summary Markdown Table
    print("\n" + "=" * 90)
    print(" NATURAL-WORD ATTRIBUTION: SEMANTIC DISCOVERY OF TIMESTEP SWITCHING")
    print("=" * 90)
    print(f"{'Top Words Pushing LATER Switch (Stay LR)':<42} | {'Top Words Pushing EARLIER Switch (Go HR)':<42}")
    print("-" * 90)
    for i in range(min(15, len(top_late), len(top_early))):
        lw = f"{i+1:2d}. {top_late[i]['word']} ({top_late[i]['mean_attribution']:+.3f})"
        ew = f"{i+1:2d}. {top_early[i]['word']} ({top_early[i]['mean_attribution']:+.3f})"
        print(f"{lw:<42} | {ew:<42}")
    print("=" * 90 + "\n")
    logger.info(f"Token attribution tables saved to {out_dir}")


if __name__ == "__main__":
    main()
