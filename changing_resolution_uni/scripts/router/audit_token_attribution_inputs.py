#!/usr/bin/env python3
"""Read-only audit of T5 files required for natural-word attribution."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_uni.scripts.router.token_word_utils import (
    merge_subtokens_to_words,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-word-count", type=int, default=3)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def selected_prompt_ids(dataset_dir: Path) -> list[int]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = manifest.get("record_files")
    if not isinstance(names, list) or not names:
        raise ValueError("dataset manifest must contain record_files")
    return sorted(
        {
            int(str(name).split("_s", 1)[0].removeprefix("p"))
            for name in names
        }
    )


def main() -> None:
    args = parse_args()
    if args.min_word_count < 1:
        raise ValueError("min-word-count must be positive")
    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    t5_dir = dataset_dir / "t5_embeddings"
    if not t5_dir.is_dir():
        raise FileNotFoundError(f"T5 directory not found: {t5_dir}")
    prompt_ids = selected_prompt_ids(dataset_dir)
    rows: list[dict[str, Any]] = []
    word_counts: Counter[str] = Counter()

    for prompt_id in prompt_ids:
        npz_path = t5_dir / f"prompt_{prompt_id:06d}.npz"
        meta_path = t5_dir / f"prompt_{prompt_id:06d}.json"
        issues = []
        npz_keys: list[str] = []
        pooled_shape = None
        sequence_shape = None
        token_count = 0
        natural_word_count = 0
        if not npz_path.is_file():
            issues.append("missing_npz")
        else:
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    npz_keys = list(data.files)
                    if "pooled_embedding" not in data.files:
                        issues.append("missing_pooled_embedding")
                    else:
                        pooled_shape = list(data["pooled_embedding"].shape)
                    if "seq_embedding" not in data.files:
                        issues.append("missing_seq_embedding")
                    else:
                        sequence_shape = list(data["seq_embedding"].shape)
            except Exception as exc:
                issues.append(f"invalid_npz:{type(exc).__name__}:{exc}")

        tokens = []
        if not meta_path.is_file():
            issues.append("missing_token_metadata_json")
        else:
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                tokens = metadata.get("tokens", [])
                if not isinstance(tokens, list) or not tokens:
                    issues.append("missing_tokens_list")
                    tokens = []
            except Exception as exc:
                issues.append(f"invalid_metadata:{type(exc).__name__}:{exc}")

        token_count = len(tokens)
        if sequence_shape is not None:
            if len(sequence_shape) != 2 or sequence_shape[1] != 4096:
                issues.append(f"unexpected_seq_shape:{sequence_shape}")
            elif token_count != sequence_shape[0]:
                issues.append(
                    f"token_sequence_length_mismatch:{token_count}:{sequence_shape[0]}"
                )
        if tokens and sequence_shape is not None and token_count == sequence_shape[0]:
            words = merge_subtokens_to_words(
                [str(token) for token in tokens],
                np.zeros(token_count, dtype=np.float32),
            )
            natural_word_count = len(words)
            word_counts.update(word["word"] for word in words)
            if not words:
                issues.append("no_natural_words_after_merge")

        rows.append(
            {
                "prompt_id": prompt_id,
                "npz_path": str(npz_path),
                "meta_path": str(meta_path),
                "npz_keys": json.dumps(npz_keys),
                "pooled_shape": json.dumps(pooled_shape),
                "sequence_shape": json.dumps(sequence_shape),
                "token_count": token_count,
                "natural_word_count": natural_word_count,
                "complete_for_attribution": not issues,
                "issues": " | ".join(issues),
            }
        )

    issue_counts: Counter[str] = Counter()
    for row in rows:
        for issue in str(row["issues"]).split(" | "):
            if issue:
                issue_counts[issue.split(":", 1)[0]] += 1
    summary = {
        "schema": "token_attribution_input_audit_v1",
        "dataset_dir": str(dataset_dir),
        "t5_dir": str(t5_dir.resolve()),
        "selected_prompt_count": len(prompt_ids),
        "complete_prompt_count": sum(
            bool(row["complete_for_attribution"]) for row in rows
        ),
        "incomplete_prompt_count": sum(
            not bool(row["complete_for_attribution"]) for row in rows
        ),
        "issue_counts": dict(sorted(issue_counts.items())),
        "natural_word_occurrence_count": sum(word_counts.values()),
        "natural_vocabulary_size": len(word_counts),
        "natural_vocabulary_at_min_count": sum(
            count >= args.min_word_count for count in word_counts.values()
        ),
        "prompt_id_min": min(prompt_ids) if prompt_ids else None,
        "prompt_id_max": max(prompt_ids) if prompt_ids else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "t5_attribution_input_audit.json"
    report_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "t5_attribution_input_details.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Audit report: {report_path}")
    if args.strict and summary["incomplete_prompt_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
