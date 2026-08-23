#!/usr/bin/env python3
"""Extract a canonical prompt-offset, scalar-VBench router dataset."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_uni.scripts.data.oracle_record_schema import (
    FORMAL_STEPS,
    OracleRecordError,
    validate_scored_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select exactly three prompt-offset scored trajectories per prompt."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-prompts", type=int, default=1000)
    parser.add_argument("--base-seeds", type=int, nargs="+", default=[42, 100, 2024])
    parser.add_argument("--primary-lambda", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source_dir}")
    if output_dir == source_dir:
        raise ValueError("output-dir must differ from source-dir")

    paths_by_key: dict[tuple[int, int], list[Path]] = defaultdict(list)
    parse_errors = []
    for path in sorted(source_dir.rglob("records/p*_s*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            key = (int(record["prompt_id"]), int(record["seed"]))
            paths_by_key[key].append(path)
        except Exception as exc:
            parse_errors.append(f"{path}: {exc}")
    if parse_errors:
        preview = "\n".join(f"  - {item}" for item in parse_errors[:20])
        raise RuntimeError(f"Failed to parse source records:\n{preview}")

    selected: dict[tuple[int, int], tuple[Path, dict]] = {}
    errors = []
    base_seeds = sorted({int(seed) for seed in args.base_seeds})
    for prompt_id in range(args.expected_prompts):
        for base_seed in base_seeds:
            actual_seed = base_seed + prompt_id
            key = (prompt_id, actual_seed)
            paths = paths_by_key.get(key, [])
            if not paths:
                errors.append(f"missing prompt={prompt_id} seed={actual_seed}")
                continue
            hashes = {sha256(path) for path in paths}
            if len(hashes) > 1:
                errors.append(
                    f"conflicting duplicates for prompt={prompt_id} seed={actual_seed}: "
                    f"{[str(path) for path in paths]}"
                )
                continue
            path = sorted(
                paths,
                key=lambda item: (
                    item.parent != source_dir / "records",
                    str(item),
                ),
            )[0]
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                normalized = validate_scored_record(
                    record,
                    candidate_steps=FORMAL_STEPS,
                    require_dimensions=False,
                )
                if (normalized["prompt_id"], normalized["seed"]) != key:
                    raise OracleRecordError(
                        f"record key {(normalized['prompt_id'], normalized['seed'])} "
                        f"does not match expected {key}"
                    )
                selected[key] = (path, record)
            except (json.JSONDecodeError, OracleRecordError) as exc:
                errors.append(f"{path}: {exc}")

    t5_source = source_dir / "t5_embeddings"
    missing_t5 = [
        prompt_id
        for prompt_id in range(args.expected_prompts)
        if not (t5_source / f"prompt_{prompt_id:06d}.npz").is_file()
    ]
    if missing_t5:
        errors.append(
            f"missing T5 embeddings for {len(missing_t5)} prompts, examples={missing_t5[:20]}"
        )

    expected_records = args.expected_prompts * len(base_seeds)
    if len(selected) != expected_records:
        errors.append(f"selected {len(selected)} records, expected {expected_records}")
    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:50])
        suffix = "" if len(errors) <= 50 else f"\n  ... and {len(errors) - 50} more"
        raise RuntimeError(f"Router dataset preparation failed:\n{preview}{suffix}")

    print(
        f"Validated {args.expected_prompts} prompts x {len(base_seeds)} trajectories "
        f"from {source_dir}"
    )
    if args.dry_run:
        print("Dry run complete; no files written.")
        return

    records_dir = output_dir / "records"
    t5_output = output_dir / "t5_embeddings"
    records_dir.mkdir(parents=True, exist_ok=True)
    t5_output.mkdir(parents=True, exist_ok=True)

    record_files = []
    for (prompt_id, seed), (source_path, _) in sorted(selected.items()):
        destination = records_dir / f"p{prompt_id:06d}_s{seed}.json"
        shutil.copy2(source_path, destination)
        record_files.append(destination.name)
    for prompt_id in range(args.expected_prompts):
        source_path = t5_source / f"prompt_{prompt_id:06d}.npz"
        shutil.copy2(source_path, t5_output / source_path.name)
        metadata_path = source_path.with_suffix(".json")
        if metadata_path.is_file():
            shutil.copy2(metadata_path, t5_output / metadata_path.name)

    manifest = {
        "schema": "prompt_conditioned_router_dataset_v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "quality_profile": "vbench5_scalar",
        "total_prompts_found": args.expected_prompts,
        "expected_prompts": args.expected_prompts,
        "total_trajectories": expected_records,
        "expected_trajectories": expected_records,
        "expected_base_seeds": base_seeds,
        "seed_policy": "prompt_offset",
        "candidate_steps": FORMAL_STEPS,
        "primary_lambda": args.primary_lambda,
        "record_files": record_files,
        "is_complete": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Router dataset ready: {output_dir}")


if __name__ == "__main__":
    main()
