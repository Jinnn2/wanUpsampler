#!/usr/bin/env python3
"""
Extract frozen T5 prompt embeddings, attention-masked mean pooled features,
and token strings for downstream optimal-stopping router training and token attribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import logging
import numpy as np
try:
    import torch
except ImportError:
    torch = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("extract_t5")

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LIGHTX2V_REPO = os.environ.get("LIGHTX2V_REPO")
if LIGHTX2V_REPO and LIGHTX2V_REPO not in sys.path:
    sys.path.insert(0, LIGHTX2V_REPO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract T5 text embeddings for prompts.")
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Path to prompts file (txt with one prompt per line, or json list).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory to store extracted embeddings (.npz per prompt).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B",
        help="Wan model root path containing T5 encoder or tokenizer weights.",
    )
    parser.add_argument(
        "--text_encoder_ckpt",
        type=str,
        default=None,
        help="Optional explicit path to models_t5_umt5-xxl-enc-bf16.pth.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional tokenizer directory or HF hub identifier (default: google/umt5-xxl or model_path).",
    )
    parser.add_argument(
        "--prompt_offset",
        type=int,
        default=0,
        help="Start index in the prompt list.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of prompts to process.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Max sequence length for T5 tokenization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
        help="Device to run text encoder (cuda / cpu).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Inference precision for text encoder.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip extraction if output embedding file already exists.",
    )
    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
    )
    return parser.parse_args()


def load_prompts(file_path: Path, offset: int = 0, limit: int | None = None) -> list[tuple[int, str]]:
    """Load prompts from .txt or .json, returning list of (global_index, prompt_text)."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Prompts file not found: {file_path}")

    all_prompts: list[str] = []
    if file_path.suffix.lower() == ".json":
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    all_prompts.append(item.strip())
                elif isinstance(item, dict) and "prompt" in item:
                    all_prompts.append(str(item["prompt"]).strip())
        elif isinstance(raw, dict) and "prompts" in raw:
            all_prompts = [str(p).strip() for p in raw["prompts"]]
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    all_prompts.append(line)

    indexed = list(enumerate(all_prompts))
    selected = indexed[offset:]
    if limit is not None and limit > 0:
        selected = selected[:limit]
    return selected


def init_tokenizer_and_encoder(
    model_path: str,
    text_encoder_ckpt: str | None,
    tokenizer_path: str | None,
    device: torch.device,
    torch_dtype: torch.dtype,
):
    """
    Initialize Wan T5 tokenizer and encoder.
    Falls back gracefully to transformers or native LightX2V text encoder.
    """
    try:
        from transformers import AutoTokenizer, T5EncoderModel

        tok_id = tokenizer_path or model_path
        if not (Path(tok_id).exists() and (Path(tok_id) / "tokenizer_config.json").exists()):
            tok_id = "google/umt5-xxl"
        logger.info(f"Loading tokenizer from: {tok_id}")
        tokenizer = AutoTokenizer.from_pretrained(tok_id)

        # Check for standard T5 encoder weights
        enc_path = text_encoder_ckpt or os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth")
        if os.path.isfile(enc_path):
            logger.info(f"Loading Wan native T5 weights from: {enc_path}")
            # Native Wan T5 encoder structure or transformers T5
            try:
                from lightx2v.models.text_encoders.wan.text_encoder import WanT5Encoder
                encoder = WanT5Encoder(enc_path, dtype=torch_dtype, device=str(device))
                return tokenizer, encoder, "wan_native"
            except Exception as e:
                logger.warning(f"Failed to load WanT5Encoder ({e}), trying standard T5EncoderModel...")

        logger.info(f"Loading HuggingFace T5EncoderModel from: {tok_id}")
        encoder = T5EncoderModel.from_pretrained(tok_id, torch_dtype=torch_dtype)
        encoder.to(device)
        encoder.eval()
        return tokenizer, encoder, "hf_transformers"
    except Exception as exc:
        raise RuntimeError(f"Could not initialize T5 encoder/tokenizer: {exc}") from exc


def no_grad(fn):
    if torch is not None:
        return torch.no_grad()(fn)
    return fn


@no_grad
def encode_prompt_tokens(
    tokenizer: Any,
    encoder: Any,
    backend: str,
    prompt: str,
    max_seq_len: int,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> dict[str, Any]:
    """Tokenize and extract embeddings + masked mean pooling."""
    if backend == "wan_native":
        # Native LightX2V Wan encoder
        ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        emb = encoder(ids, mask)  # [1, L, 4096]
        
        # Token strings
        tokens = tokenizer.convert_ids_to_tokens(ids[0].cpu().tolist())
        seq_len = int(mask[0].gt(0).sum().item())
        emb_valid = emb[0, :seq_len]  # [valid_L, 4096]
        mask_valid = mask[0, :seq_len]  # [valid_L]
        
        # Attention-masked mean pooling: sum(m_i * h_i) / sum(m_i)
        pooled = (emb_valid * mask_valid.unsqueeze(-1)).sum(dim=0) / mask_valid.sum().clamp(min=1)
        
        return {
            "tokens": tokens[:seq_len],
            "input_ids": ids[0, :seq_len].cpu().numpy().astype(np.int64),
            "attention_mask": mask[0, :seq_len].cpu().numpy().astype(np.int64),
            "seq_embedding": emb_valid.cpu().to(torch.float16).numpy(),
            "pooled_embedding": pooled.cpu().to(torch.float16).numpy(),
        }
    else:
        # HuggingFace T5EncoderModel
        inputs = tokenizer(
            prompt,
            max_length=max_seq_len,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = outputs.last_hidden_state  # [1, L, 4096]
        
        seq_len = int(attention_mask[0].sum().item())
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())[:seq_len]
        emb_valid = emb[0, :seq_len]
        mask_valid = attention_mask[0, :seq_len]
        
        pooled = (emb_valid * mask_valid.unsqueeze(-1)).sum(dim=0) / mask_valid.sum().clamp(min=1)
        
        return {
            "tokens": tokens,
            "input_ids": input_ids[0, :seq_len].cpu().numpy().astype(np.int64),
            "attention_mask": mask_valid.cpu().numpy().astype(np.int64),
            "seq_embedding": emb_valid.cpu().to(torch.float16).numpy(),
            "pooled_embedding": pooled.cpu().to(torch.float16).numpy(),
        }


def main() -> None:
    args = parse_args()
    prompts_file = Path(args.prompts_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompts_file, offset=args.prompt_offset, limit=args.limit)
    logger.info(f"Loaded {len(prompts)} prompts from {prompts_file} (offset={args.prompt_offset})")

    device = torch.device(args.device)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.precision]

    tokenizer, encoder, backend = init_tokenizer_and_encoder(
        model_path=args.model_path,
        text_encoder_ckpt=args.text_encoder_ckpt,
        tokenizer_path=args.tokenizer_path,
        device=device,
        torch_dtype=torch_dtype,
    )

    manifest_entries: list[dict[str, Any]] = []
    num_skipped = 0
    num_processed = 0

    for global_idx, prompt_text in prompts:
        save_path = out_dir / f"prompt_{global_idx:06d}.npz"
        meta_path = out_dir / f"prompt_{global_idx:06d}.json"

        if args.skip_existing and save_path.is_file() and meta_path.is_file():
            num_skipped += 1
            manifest_entries.append({
                "prompt_id": global_idx,
                "prompt_text": prompt_text,
                "npz_file": str(save_path),
                "json_file": str(meta_path),
            })
            continue

        try:
            encoded = encode_prompt_tokens(
                tokenizer=tokenizer,
                encoder=encoder,
                backend=backend,
                prompt=prompt_text,
                max_seq_len=args.max_seq_len,
                device=device,
                torch_dtype=torch_dtype,
            )

            # Save arrays in compressed .npz
            np.savez_compressed(
                save_path,
                pooled_embedding=encoded["pooled_embedding"],
                seq_embedding=encoded["seq_embedding"],
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )

            # Save human-readable tokens and metadata
            meta = {
                "prompt_id": global_idx,
                "prompt_text": prompt_text,
                "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                "num_tokens": len(encoded["tokens"]),
                "tokens": encoded["tokens"],
                "npz_file": save_path.name,
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            manifest_entries.append({
                "prompt_id": global_idx,
                "prompt_text": prompt_text,
                "npz_file": str(save_path),
                "json_file": str(meta_path),
            })
            num_processed += 1
            if num_processed % 50 == 0 or num_processed == len(prompts):
                logger.info(f"Progress: [{num_processed}/{len(prompts)}] prompts embedded.")
        except Exception as err:
            logger.error(f"Error encoding prompt {global_idx}: {err}")
            raise

    # Write summary manifest
    manifest_path = out_dir / "t5_manifest.json"
    manifest_payload = {
        "schema": "prompt_t5_embeddings_manifest_v1",
        "total_prompts": len(manifest_entries),
        "processed": num_processed,
        "skipped": num_skipped,
        "prompts": manifest_entries,
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Done! {num_processed} processed, {num_skipped} skipped. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
