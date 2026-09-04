from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.data_protocol import (  # noqa: E402
    load_prompts,
    sha256_file,
    write_json_atomic,
)


SELECTION_SCHEMA = "univ_prompt_shard_selection_v1"


def prompt_rank(prompt: str, *, namespace: str, seed: str) -> bytes:
    payload = f"{namespace}:{seed}\0{prompt}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def select_prompt_shard(
    source: str | Path,
    *,
    exclusions: list[str | Path],
    count: int,
    seed: str,
) -> tuple[list[str], dict]:
    source_path = Path(source).resolve()
    raw = load_prompts(source_path)
    unique = list(dict.fromkeys(raw))
    excluded_prompts: set[str] = set()
    exclusion_records = []
    for value in exclusions:
        path = Path(value).resolve()
        prompts = load_prompts(path)
        excluded_prompts.update(prompts)
        exclusion_records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "valid_lines": len(prompts),
                "unique_lines": len(set(prompts)),
            }
        )
    eligible = [prompt for prompt in unique if prompt not in excluded_prompts]
    if count < 1:
        raise ValueError("count must be positive")
    if len(eligible) < count:
        raise ValueError(
            f"source has only {len(eligible)} eligible unique prompts for count={count}"
        )
    selected = sorted(
        eligible,
        key=lambda prompt: prompt_rank(prompt, namespace="select", seed=seed),
    )[:count]
    selected = sorted(
        selected,
        key=lambda prompt: prompt_rank(prompt, namespace="split", seed=seed),
    )
    metadata = {
        "schema": SELECTION_SCHEMA,
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "source_valid_lines": len(raw),
        "source_unique_lines": len(unique),
        "source_duplicates": len(raw) - len(unique),
        "exclusions": exclusion_records,
        "excluded_unique_prompts": len(excluded_prompts),
        "eligible_unique_prompts": len(eligible),
        "selection_method": "sha256_bottom_k_then_hash_shuffle",
        "selection_seed": seed,
        "selected_count": len(selected),
        "selected_unique_count": len(set(selected)),
        "selected_sha256": hashlib.sha256(
            ("\n".join(selected) + "\n").encode("utf-8")
        ).hexdigest(),
    }
    return selected, metadata


def write_selection(
    output: str | Path,
    selected: list[str],
    metadata: dict,
) -> None:
    destination = Path(output).resolve()
    text = "\n".join(selected) + "\n"
    if destination.is_file() and destination.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"refusing to replace a different prompt shard: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)
    metadata_path = destination.with_suffix(destination.suffix + ".meta.json")
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous != metadata:
            raise RuntimeError(
                f"refusing to replace different prompt metadata: {metadata_path}"
            )
    else:
        write_json_atomic(metadata_path, metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a deterministic non-overlapping UNIV prompt shard."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected, metadata = select_prompt_shard(
        args.source,
        exclusions=args.exclude,
        count=args.count,
        seed=args.seed,
    )
    write_selection(args.output, selected, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Prompt shard written: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
