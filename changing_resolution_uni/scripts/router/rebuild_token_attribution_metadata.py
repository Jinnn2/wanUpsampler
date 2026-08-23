#!/usr/bin/env python3
"""Rebuild missing token strings from existing T5 input_ids without re-encoding."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
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
    parser.add_argument(
        "--model-root",
        default=os.environ.get(
            "MODEL_ROOT", "/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B"
        ),
    )
    parser.add_argument("--tokenizer-path", default=None)
    return parser.parse_args()


def selected_prompts(dataset_dir: Path) -> dict[int, str]:
    manifest = json.loads(
        (dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    names = manifest.get("record_files")
    if not isinstance(names, list) or not names:
        raise ValueError("dataset manifest must contain record_files")
    records_dir = (dataset_dir / "records").resolve()
    prompts: dict[int, str] = {}
    for name in names:
        path = (records_dir / str(name)).resolve()
        if path.parent != records_dir:
            raise ValueError("record path escapes records directory")
        record = json.loads(path.read_text(encoding="utf-8"))
        prompt_id = int(record["prompt_id"])
        prompt_text = str(record["prompt_text"])
        previous = prompts.setdefault(prompt_id, prompt_text)
        if previous != prompt_text:
            raise ValueError(f"prompt {prompt_id} text differs across seeds")
    return prompts


def resolve_tokenizer_path(model_root: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = [model_root / "google" / "umt5-xxl", model_root]
    for candidate in candidates:
        if (candidate / "tokenizer_config.json").is_file():
            return str(candidate)
    return "google/umt5-xxl"


def link_or_copy(source: Path, destination: Path) -> str:
    if destination.exists() or destination.is_symlink():
        if not destination.samefile(source):
            raise FileExistsError(
                f"Existing attribution NPZ points elsewhere: {destination}"
            )
        return "existing_link"
    try:
        destination.symlink_to(source)
        return "absolute_file_symlink"
    except OSError:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copied"


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    source_t5_dir = (dataset_dir / "t5_embeddings").resolve()
    out_dir = Path(args.out_dir).resolve()
    if not source_t5_dir.is_dir():
        raise FileNotFoundError(source_t5_dir)
    prompts = selected_prompts(dataset_dir)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to reconstruct token strings") from exc
    tokenizer_path = resolve_tokenizer_path(Path(args.model_root).resolve(), args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    storage_counts: dict[str, int] = {}
    entries: list[dict[str, Any]] = []
    for prompt_id, prompt_text in sorted(prompts.items()):
        source_npz = source_t5_dir / f"prompt_{prompt_id:06d}.npz"
        if not source_npz.is_file():
            raise FileNotFoundError(source_npz)
        with np.load(source_npz, allow_pickle=False) as data:
            if "seq_embedding" not in data.files:
                raise ValueError(f"{source_npz.name} is missing seq_embedding")
            if "input_ids" not in data.files:
                raise ValueError(
                    f"{source_npz.name} is missing input_ids; token strings cannot "
                    "be reconstructed without re-encoding"
                )
            sequence_length = int(data["seq_embedding"].shape[0])
            input_ids = np.asarray(data["input_ids"]).reshape(-1)[:sequence_length]
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        if len(tokens) != sequence_length:
            raise ValueError(
                f"prompt {prompt_id}: tokenizer returned {len(tokens)} tokens for "
                f"sequence length {sequence_length}"
            )
        words = merge_subtokens_to_words(
            [str(token) for token in tokens],
            np.zeros(sequence_length, dtype=np.float32),
        )
        if not words:
            raise ValueError(f"prompt {prompt_id}: reconstructed tokens contain no words")

        destination_npz = out_dir / source_npz.name
        storage = link_or_copy(source_npz, destination_npz)
        storage_counts[storage] = storage_counts.get(storage, 0) + 1
        metadata = {
            "schema": "reconstructed_t5_token_metadata_v1",
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "num_tokens": len(tokens),
            "natural_word_count": len(words),
            "tokens": [str(token) for token in tokens],
            "source_npz": str(source_npz),
            "tokenizer_path": tokenizer_path,
            "npz_file": destination_npz.name,
        }
        metadata_path = out_dir / f"prompt_{prompt_id:06d}.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        entries.append(
            {
                "prompt_id": prompt_id,
                "npz_file": destination_npz.name,
                "metadata_file": metadata_path.name,
                "token_count": len(tokens),
                "natural_word_count": len(words),
                "storage": storage,
            }
        )

    manifest = {
        "schema": "token_attribution_embeddings_manifest_v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "source_t5_dir": str(source_t5_dir),
        "tokenizer_path": tokenizer_path,
        "prompt_count": len(entries),
        "storage_counts": storage_counts,
        "prompts": entries,
    }
    manifest_path = out_dir / "token_attribution_embeddings_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "prompts"}, indent=2, ensure_ascii=False))
    print(f"Reconstructed token metadata: {manifest_path}")


if __name__ == "__main__":
    main()
