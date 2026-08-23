#!/usr/bin/env python3
"""Read-only audit for Native-HR score reuse and per-step quality/time behavior."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORMAL_STEPS = [30, 35, *range(40, 51)]


@dataclass(frozen=True)
class Candidate:
    step: int
    quality: float | None
    latency: float | None
    latency_source: str
    dimensions: dict[str, float]


@dataclass(frozen=True)
class Record:
    path: Path
    prompt_id: int
    seed: int
    prompt_text: str
    native_quality: float | None
    native_latency: float | None
    native_latency_source: str
    native_dimensions: dict[str, float]
    candidates: tuple[Candidate, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect legacy oracle records for the signature produced when missing "
            "candidate VBench scores fall back to Native-HR. No files are modified."
        )
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--lambda-value", type=float, default=0.01)
    parser.add_argument("--equality-tolerance", type=float, default=1e-12)
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def numeric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, raw in value.items():
        number = finite_float(raw)
        if number is not None:
            result[str(key)] = number
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_number(*values: Any) -> float | None:
    for value in values:
        number = finite_float(value)
        if number is not None:
            return number
    return None


def load_record(path: Path) -> Record:
    raw = json.loads(path.read_text(encoding="utf-8"))
    prompt_id = int(raw["prompt_id"])
    seed = int(raw["seed"])
    prompt_text = str(raw.get("prompt_text", raw.get("prompt", "")))
    manifest = raw.get("manifest") if isinstance(raw.get("manifest"), dict) else {}
    native = raw.get("native") if isinstance(raw.get("native"), dict) else {}
    manifest_native = (
        manifest.get("native_hr")
        if isinstance(manifest.get("native_hr"), dict)
        else {}
    )
    native_quality = first_number(
        raw.get("native_vbench5"),
        native.get("vbench5"),
        native.get("quality"),
    )
    native_latency = first_number(
        raw.get("native_latency_seconds"),
        native.get("latency_seconds"),
        manifest_native.get("warm_pipeline_seconds"),
        manifest_native.get("estimated_warm_pipeline_seconds"),
    )
    native_latency_source = str(raw.get("native_latency_source", "unknown"))
    native_dimensions = numeric_mapping(raw.get("native_dimensions"))

    raw_candidates = raw.get("candidates")
    if not isinstance(raw_candidates, list):
        raw_candidates = manifest.get("branches", [])
    branch_by_step = {
        int(branch["candidate_step"]): branch
        for branch in manifest.get("branches", [])
        if isinstance(branch, dict) and "candidate_step" in branch
    }
    candidates = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        step_raw = candidate.get("step", candidate.get("candidate_step"))
        try:
            step = int(step_raw)
        except (TypeError, ValueError):
            continue
        branch = branch_by_step.get(step, {})
        quality = first_number(candidate.get("vbench5"), candidate.get("quality"))
        latency = first_number(
            candidate.get("latency_seconds"),
            candidate.get("warm_pipeline_seconds"),
            candidate.get("estimated_warm_pipeline_seconds"),
            branch.get("warm_pipeline_seconds"),
            branch.get("estimated_warm_pipeline_seconds"),
        )
        explicit_source = candidate.get("latency_source")
        if explicit_source:
            latency_source = str(explicit_source)
        elif "latency_seconds" in candidate:
            latency_source = "legacy_latency_seconds"
        elif "warm_pipeline_seconds" in candidate or "warm_pipeline_seconds" in branch:
            latency_source = "warm_pipeline_seconds"
        elif (
            "estimated_warm_pipeline_seconds" in candidate
            or "estimated_warm_pipeline_seconds" in branch
        ):
            latency_source = "estimated_warm_pipeline_seconds"
        else:
            latency_source = "unknown"
        dimensions = numeric_mapping(
            candidate.get("dimensions", candidate.get("vbench_details"))
        )
        candidates.append(
            Candidate(
                step=step,
                quality=quality,
                latency=latency,
                latency_source=latency_source,
                dimensions=dimensions,
            )
        )
    return Record(
        path=path,
        prompt_id=prompt_id,
        seed=seed,
        prompt_text=prompt_text,
        native_quality=native_quality,
        native_latency=native_latency,
        native_latency_source=native_latency_source,
        native_dimensions=native_dimensions,
        candidates=tuple(sorted(candidates, key=lambda item: item.step)),
    )


def discover_records(
    dataset_dir: Path,
) -> tuple[list[Record], dict[str, Any], list[str]]:
    paths_by_key: dict[tuple[int, int], list[Path]] = defaultdict(list)
    parse_errors = []
    for path in sorted(dataset_dir.rglob("records/p*_s*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            paths_by_key[(int(raw["prompt_id"]), int(raw["seed"]))].append(path)
        except Exception as exc:
            parse_errors.append(f"{path}: {exc}")

    duplicate_keys = 0
    conflicting_duplicate_keys = 0
    selected_paths = []
    for paths in paths_by_key.values():
        if len(paths) > 1:
            duplicate_keys += 1
            hashes = {sha256_file(path) for path in paths}
            if len(hashes) > 1:
                conflicting_duplicate_keys += 1
        selected_paths.append(
            min(
                paths,
                key=lambda path: (
                    path.parent != dataset_dir / "records",
                    len(path.parts),
                    str(path),
                ),
            )
        )

    records = []
    for path in sorted(selected_paths):
        try:
            records.append(load_record(path))
        except Exception as exc:
            parse_errors.append(f"{path}: {exc}")
    duplicate_summary = {
        "discovered_record_paths": sum(len(paths) for paths in paths_by_key.values()),
        "unique_prompt_seed_keys": len(paths_by_key),
        "duplicate_keys": duplicate_keys,
        "conflicting_duplicate_keys": conflicting_duplicate_keys,
    }
    return records, duplicate_summary, parse_errors


def equal(left: float | None, right: float | None, tolerance: float) -> bool:
    return (
        left is not None
        and right is not None
        and abs(left - right) <= tolerance
    )


def record_signature(record: Record, tolerance: float, lambda_value: float) -> dict[str, Any]:
    by_step = {candidate.step: candidate for candidate in record.candidates}
    complete = set(by_step) == set(FORMAL_STEPS)
    qualities = [
        by_step[step].quality
        for step in FORMAL_STEPS
        if step in by_step and by_step[step].quality is not None
    ]
    latencies = [
        by_step[step].latency
        for step in FORMAL_STEPS
        if step in by_step and by_step[step].latency is not None
    ]
    exact_native = [
        equal(candidate.quality, record.native_quality, tolerance)
        for candidate in record.candidates
        if candidate.quality is not None and record.native_quality is not None
    ]
    quality_range = max(qualities) - min(qualities) if qualities else None
    latency_range = max(latencies) - min(latencies) if latencies else None
    quality_winner = None
    utility_winner = None
    if complete and len(qualities) == len(FORMAL_STEPS):
        quality_winner = max(
            FORMAL_STEPS,
            key=lambda step: (float(by_step[step].quality), -step),
        )
    if (
        complete
        and record.native_latency is not None
        and record.native_latency > 0
        and all(
            by_step[step].quality is not None and by_step[step].latency is not None
            for step in FORMAL_STEPS
        )
    ):
        utility_winner = max(
            FORMAL_STEPS,
            key=lambda step: (
                float(by_step[step].quality)
                - lambda_value
                * float(by_step[step].latency)
                / float(record.native_latency),
                -step,
            ),
        )
    latency_monotonic_nonincreasing = None
    if len(latencies) == len(FORMAL_STEPS):
        latency_monotonic_nonincreasing = all(
            latencies[index + 1] <= latencies[index] + tolerance
            for index in range(len(latencies) - 1)
        )
    return {
        "complete_steps": complete,
        "candidate_count": len(record.candidates),
        "scored_candidate_count": len(qualities),
        "timed_candidate_count": len(latencies),
        "quality_range": quality_range,
        "latency_range": latency_range,
        "all_candidates_same_quality": (
            complete
            and len(qualities) == len(FORMAL_STEPS)
            and quality_range is not None
            and quality_range <= tolerance
        ),
        "all_scored_candidates_equal_native": (
            complete
            and len(qualities) == len(FORMAL_STEPS)
            and len(exact_native) == len(FORMAL_STEPS)
            and all(exact_native)
        ),
        "candidate_equal_native_count": sum(exact_native),
        "candidate_native_comparison_count": len(exact_native),
        "missing_candidate_dimensions": sum(
            not bool(candidate.dimensions) for candidate in record.candidates
        ),
        "quality_winner": quality_winner,
        "utility_winner": utility_winner,
        "all_candidates_same_latency": (
            complete
            and len(latencies) == len(FORMAL_STEPS)
            and latency_range is not None
            and latency_range <= tolerance
        ),
        "latency_monotonic_nonincreasing": latency_monotonic_nonincreasing,
    }


def select_prompt_groups(
    records_by_prompt: dict[int, list[Record]],
    signatures: dict[tuple[int, int], dict[str, Any]],
    sample_count: int,
) -> list[tuple[int, list[str]]]:
    prompt_features = []
    for prompt_id, records in sorted(records_by_prompt.items()):
        record_signatures = [signatures[(record.prompt_id, record.seed)] for record in records]
        quality_ranges = [
            float(item["quality_range"])
            for item in record_signatures
            if item["quality_range"] is not None
        ]
        prompt_features.append(
            {
                "prompt_id": prompt_id,
                "all_native_reuse": bool(record_signatures)
                and all(item["all_scored_candidates_equal_native"] for item in record_signatures),
                "all_flat": bool(record_signatures)
                and all(item["all_candidates_same_quality"] for item in record_signatures),
                "step50_quality": any(item["quality_winner"] == 50 for item in record_signatures),
                "interior_quality": any(
                    item["quality_winner"] not in {None, 30, 50}
                    for item in record_signatures
                ),
                "step50_utility": any(item["utility_winner"] == 50 for item in record_signatures),
                "interior_utility": any(
                    item["utility_winner"] not in {None, 30, 50}
                    for item in record_signatures
                ),
                "mean_quality_range": (
                    sum(quality_ranges) / len(quality_ranges) if quality_ranges else -1.0
                ),
            }
        )

    selected: dict[int, list[str]] = {}

    def add(feature: dict[str, Any], category: str) -> None:
        selected.setdefault(int(feature["prompt_id"]), []).append(category)

    categories = [
        ("native_reuse", lambda item: item["all_native_reuse"]),
        ("flat", lambda item: item["all_flat"]),
        ("step50_quality", lambda item: item["step50_quality"]),
        ("interior_quality", lambda item: item["interior_quality"]),
        ("step50_utility", lambda item: item["step50_utility"]),
        ("interior_utility", lambda item: item["interior_utility"]),
    ]
    for name, predicate in categories:
        match = next((item for item in prompt_features if predicate(item)), None)
        if match is not None:
            add(match, name)

    for feature in sorted(
        prompt_features,
        key=lambda item: (-float(item["mean_quality_range"]), int(item["prompt_id"])),
    ):
        if len(selected) >= sample_count:
            break
        add(feature, "high_quality_range")

    if len(selected) < sample_count and prompt_features:
        stride = max(1, len(prompt_features) // max(1, sample_count - len(selected)))
        for feature in prompt_features[::stride]:
            if len(selected) >= sample_count:
                break
            add(feature, "coverage")
    return [(prompt_id, selected[prompt_id]) for prompt_id in sorted(selected)[:sample_count]]


def prompt_mean_rows(
    prompt_id: int,
    records: list[Record],
    lambda_value: float,
    tolerance: float,
) -> list[dict[str, Any]]:
    rows = []
    for step in FORMAL_STEPS:
        candidates = [
            next((candidate for candidate in record.candidates if candidate.step == step), None)
            for record in records
        ]
        qualities = [candidate.quality for candidate in candidates if candidate and candidate.quality is not None]
        latencies = [candidate.latency for candidate in candidates if candidate and candidate.latency is not None]
        natives = [record.native_quality for record in records if record.native_quality is not None]
        native_latencies = [record.native_latency for record in records if record.native_latency is not None]
        mean_quality = sum(qualities) / len(qualities) if qualities else None
        mean_latency = sum(latencies) / len(latencies) if latencies else None
        mean_native = sum(natives) / len(natives) if natives else None
        mean_native_latency = (
            sum(native_latencies) / len(native_latencies) if native_latencies else None
        )
        utility = None
        if (
            mean_quality is not None
            and mean_latency is not None
            and mean_native_latency is not None
            and mean_native_latency > 0
        ):
            utility = mean_quality - lambda_value * mean_latency / mean_native_latency
        rows.append(
            {
                "prompt_id": prompt_id,
                "view": "prompt_mean",
                "seed": "mean",
                "step": step,
                "quality": mean_quality,
                "native_quality": mean_native,
                "quality_minus_native": (
                    mean_quality - mean_native
                    if mean_quality is not None and mean_native is not None
                    else None
                ),
                "latency_seconds": mean_latency,
                "native_latency_seconds": mean_native_latency,
                "utility": utility,
                "equals_native": equal(mean_quality, mean_native, tolerance),
                "latency_source": "prompt_mean",
                "dimensions_json": "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.sample_count < 1:
        raise ValueError("sample-count must be positive")
    if args.equality_tolerance < 0 or not math.isfinite(args.equality_tolerance):
        raise ValueError("equality-tolerance must be finite and non-negative")
    if not math.isfinite(args.lambda_value) or args.lambda_value < 0:
        raise ValueError("lambda-value must be finite and non-negative")

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, duplicate_summary, parse_errors = discover_records(dataset_dir)
    if not records:
        raise RuntimeError(f"No readable oracle records found under {dataset_dir}")
    signatures = {
        (record.prompt_id, record.seed): record_signature(
            record, args.equality_tolerance, args.lambda_value
        )
        for record in records
    }
    records_by_prompt: dict[int, list[Record]] = defaultdict(list)
    for record in records:
        records_by_prompt[record.prompt_id].append(record)

    all_candidates = [candidate for record in records for candidate in record.candidates]
    all_signatures = list(signatures.values())
    native_comparison_count = sum(
        int(item["candidate_native_comparison_count"]) for item in all_signatures
    )
    native_equal_count = sum(
        int(item["candidate_equal_native_count"]) for item in all_signatures
    )
    latency_sources = Counter(
        candidate.latency_source for candidate in all_candidates
    )
    per_step_rows = []
    for step in FORMAL_STEPS:
        step_candidates = [candidate for candidate in all_candidates if candidate.step == step]
        qualities = [candidate.quality for candidate in step_candidates if candidate.quality is not None]
        latencies = [candidate.latency for candidate in step_candidates if candidate.latency is not None]
        native_pairs = [
            (candidate.quality, record.native_quality)
            for record in records
            for candidate in record.candidates
            if candidate.step == step
            and candidate.quality is not None
            and record.native_quality is not None
        ]
        per_step_rows.append(
            {
                "step": step,
                "candidate_count": len(step_candidates),
                "scored_count": len(qualities),
                "quality_mean": sum(qualities) / len(qualities) if qualities else None,
                "quality_min": min(qualities) if qualities else None,
                "quality_max": max(qualities) if qualities else None,
                "exact_native_reuse_count": sum(
                    equal(left, right, args.equality_tolerance)
                    for left, right in native_pairs
                ),
                "exact_native_reuse_fraction": (
                    sum(
                        equal(left, right, args.equality_tolerance)
                        for left, right in native_pairs
                    )
                    / len(native_pairs)
                    if native_pairs
                    else None
                ),
                "latency_mean": sum(latencies) / len(latencies) if latencies else None,
                "latency_min": min(latencies) if latencies else None,
                "latency_max": max(latencies) if latencies else None,
                "latency_exact_50_count": sum(
                    equal(value, 50.0, args.equality_tolerance) for value in latencies
                ),
                "unique_latency_count": len(set(latencies)),
            }
        )

    artifact_counts = {
        "sample_manifests": sum(1 for _ in dataset_dir.rglob("manifests/*.json")),
        "candidate_step_videos": sum(
            1 for _ in dataset_dir.rglob("videos/step*/*.mp4")
        ),
        "native_hr_videos": sum(
            1 for _ in dataset_dir.rglob("videos/native_hr/*.mp4")
        ),
        "vbench_eval_results": sum(
            1 for _ in dataset_dir.rglob("*eval_results*.json")
        ),
        "oracle_metrics_files": sum(
            1 for _ in dataset_dir.rglob("oracle_metrics.json")
        ),
        "strict_score_run_manifests": sum(
            1 for _ in dataset_dir.rglob("score_run_manifest.json")
        ),
    }
    summary = {
        "dataset_dir": str(dataset_dir),
        "record_count": len(records),
        "prompt_count": len(records_by_prompt),
        "candidate_row_count": len(all_candidates),
        **duplicate_summary,
        "parse_error_count": len(parse_errors),
        "complete_step_record_count": sum(
            bool(item["complete_steps"]) for item in all_signatures
        ),
        "all_candidates_same_quality_record_count": sum(
            bool(item["all_candidates_same_quality"]) for item in all_signatures
        ),
        "all_candidates_equal_native_record_count": sum(
            bool(item["all_scored_candidates_equal_native"]) for item in all_signatures
        ),
        "candidate_native_comparison_count": native_comparison_count,
        "candidate_exact_native_reuse_count": native_equal_count,
        "candidate_exact_native_reuse_fraction": (
            native_equal_count / native_comparison_count
            if native_comparison_count
            else None
        ),
        "missing_candidate_dimension_row_count": sum(
            int(item["missing_candidate_dimensions"]) for item in all_signatures
        ),
        "all_candidates_same_latency_record_count": sum(
            bool(item["all_candidates_same_latency"]) for item in all_signatures
        ),
        "latency_monotonic_nonincreasing_record_count": sum(
            item["latency_monotonic_nonincreasing"] is True
            for item in all_signatures
        ),
        "native_latency_exact_189_count": sum(
            equal(record.native_latency, 189.0, args.equality_tolerance)
            for record in records
        ),
        "latency_source_counts": dict(sorted(latency_sources.items())),
        "artifact_counts": artifact_counts,
    }

    selected = select_prompt_groups(
        records_by_prompt, signatures, args.sample_count
    )
    selected_prompt_ids = {prompt_id for prompt_id, _ in selected}
    selected_categories = dict(selected)
    sample_rows = []
    sampled_prompt_payloads = []
    for prompt_id in sorted(selected_prompt_ids):
        prompt_records = sorted(records_by_prompt[prompt_id], key=lambda item: item.seed)
        sampled_prompt_payloads.append(
            {
                "prompt_id": prompt_id,
                "categories": selected_categories[prompt_id],
                "prompt_text": prompt_records[0].prompt_text if prompt_records else "",
                "seeds": [record.seed for record in prompt_records],
            }
        )
        for record in prompt_records:
            for candidate in record.candidates:
                utility = None
                if (
                    candidate.quality is not None
                    and candidate.latency is not None
                    and record.native_latency is not None
                    and record.native_latency > 0
                ):
                    utility = (
                        candidate.quality
                        - args.lambda_value
                        * candidate.latency
                        / record.native_latency
                    )
                sample_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "view": "seed",
                        "seed": record.seed,
                        "step": candidate.step,
                        "quality": candidate.quality,
                        "native_quality": record.native_quality,
                        "quality_minus_native": (
                            candidate.quality - record.native_quality
                            if candidate.quality is not None
                            and record.native_quality is not None
                            else None
                        ),
                        "latency_seconds": candidate.latency,
                        "native_latency_seconds": record.native_latency,
                        "utility": utility,
                        "equals_native": equal(
                            candidate.quality,
                            record.native_quality,
                            args.equality_tolerance,
                        ),
                        "latency_source": candidate.latency_source,
                        "dimensions_json": json.dumps(
                            candidate.dimensions,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )
        sample_rows.extend(
            prompt_mean_rows(
                prompt_id,
                prompt_records,
                args.lambda_value,
                args.equality_tolerance,
            )
        )

    write_csv(out_dir / "per_step_summary.csv", per_step_rows)
    write_csv(out_dir / "sampled_prompt_step_table.csv", sample_rows)
    report = {
        "schema": "oracle_native_fallback_audit_v1",
        "lambda_value": args.lambda_value,
        "equality_tolerance": args.equality_tolerance,
        "summary": summary,
        "per_step": per_step_rows,
        "sampled_prompts": sampled_prompt_payloads,
        "parse_errors": parse_errors[:100],
    }
    report_path = out_dir / "oracle_native_fallback_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print("Sampled prompt groups:")
    for item in sampled_prompt_payloads:
        print(
            f"  prompt={item['prompt_id']} seeds={item['seeds']} "
            f"categories={item['categories']}"
        )
    print(f"Report: {report_path}")
    print(f"Step summary: {out_dir / 'per_step_summary.csv'}")
    print(f"Sample details: {out_dir / 'sampled_prompt_step_table.csv'}")


if __name__ == "__main__":
    main()
