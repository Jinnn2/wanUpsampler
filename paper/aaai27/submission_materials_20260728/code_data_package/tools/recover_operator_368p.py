#!/usr/bin/env python3
"""Validate, anonymize, and summarize the 368p-to-720p ITU operator evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


EXPECTED_INPUT_SHA256 = (
    "e9dccf84dc386b91616e3151d43d5ef19c29f5e4bf8dcb33eb4b862eceaf2c85"
)
METRICS = (
    ("latent_l1", "lower", "interp_latent_l1", "trained_latent_l1"),
    ("psnr", "higher", "interp_psnr", "trained_psnr"),
    ("ssim", "higher", "interp_ssim", "trained_ssim"),
    ("lpips", "lower", "interp_lpips", "trained_lpips"),
    ("temporal_l1", "lower", "interp_temporal_l1", "trained_temporal_l1"),
    (
        "hf_energy_error",
        "lower",
        "interp_hf_energy_error",
        "trained_hf_energy_error",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
            rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 50:
        raise ValueError(f"expected 50 samples, found {len(rows)}")
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if len(set(sample_ids)) != 50 or any(not item for item in sample_ids):
        raise ValueError("sample_id values must be non-empty and unique")
    required = {"sample_index", "sample_id"}
    for _, _, interp_key, trained_key in METRICS:
        required.update((interp_key, trained_key))
    for index, row in enumerate(rows):
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"sample {index} is missing fields: {missing}")
        for key in required.difference({"sample_id"}):
            value = row[key]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"sample {index} has invalid numeric field {key}: {value}")


def anonymize_row(row: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(row)
    paths = sanitized.get("paths")
    if isinstance(paths, dict):
        sanitized["paths"] = {
            key: f"<GENERATED_OUTPUT_ROOT>/{Path(str(value)).name}"
            for key, value in paths.items()
        }
    return sanitized


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric, better, interp_key, trained_key in METRICS:
        interp = [float(row[interp_key]) for row in rows]
        trained = [float(row[trained_key]) for row in rows]
        deltas = [right - left for left, right in zip(interp, trained)]
        if better == "lower":
            wins = sum(right < left for left, right in zip(interp, trained))
            relative_improvement = (
                statistics.fmean(interp) - statistics.fmean(trained)
            ) / statistics.fmean(interp)
        else:
            wins = sum(right > left for left, right in zip(interp, trained))
            relative_improvement = (
                statistics.fmean(trained) - statistics.fmean(interp)
            ) / statistics.fmean(interp)
        output.append(
            {
                "metric": metric,
                "better": better,
                "samples": len(rows),
                "interp_mean": statistics.fmean(interp),
                "trained_mean": statistics.fmean(trained),
                "delta_mean": statistics.fmean(deltas),
                "delta_std_population": statistics.pstdev(deltas),
                "relative_improvement_percent": 100.0 * relative_improvement,
                "win_rate": wins / len(rows),
                "wins": f"{wins}/{len(rows)}",
            }
        )
    return output


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    source_sha256: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "operator_368p_raw_sanitized.jsonl"
    with raw_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(anonymize_row(row), sort_keys=True) + "\n")

    samples_path = output_dir / "operator_368p_samples.csv"
    sample_fields = ["sample_index", "sample_id"]
    for metric, _, interp_key, trained_key in METRICS:
        sample_fields.extend((interp_key, trained_key, f"{metric}_delta_trained_minus_interp"))
    with samples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader()
        for row in rows:
            record: dict[str, Any] = {
                "sample_index": row["sample_index"],
                "sample_id": row["sample_id"],
            }
            for metric, _, interp_key, trained_key in METRICS:
                interp_value = float(row[interp_key])
                trained_value = float(row[trained_key])
                record[interp_key] = f"{interp_value:.12g}"
                record[trained_key] = f"{trained_value:.12g}"
                record[f"{metric}_delta_trained_minus_interp"] = (
                    f"{trained_value - interp_value:.12g}"
                )
            writer.writerow(record)

    summary_path = output_dir / "operator_368p_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    provenance = {
        "schema_version": 1,
        "source_role": "canonical per-sample operator evaluation",
        "input_sha256": source_sha256,
        "expected_input_sha256": EXPECTED_INPUT_SHA256,
        "input_sha256_verified": source_sha256 == EXPECTED_INPUT_SHA256,
        "samples": len(rows),
        "unique_sample_ids": len({row["sample_id"] for row in rows}),
        "aggregation": {
            "means": "arithmetic mean over 50 samples",
            "delta": "trained minus interpolation",
            "delta_std": "population standard deviation (ddof=0)",
            "wins": "strict per-sample comparison in the metric-preferred direction",
        },
        "paths": "machine-specific paths replaced by <GENERATED_OUTPUT_ROOT>/basename",
        "outputs": {
            "raw_jsonl": raw_path.name,
            "samples_csv": samples_path.name,
            "summary_csv": summary_path.name,
        },
    }
    (output_dir / "operator_368p_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-sha256",
        default=EXPECTED_INPUT_SHA256,
        help="Set to an empty string only when importing a new, separately audited run.",
    )
    args = parser.parse_args()

    source_hash = sha256_file(args.input_jsonl)
    if args.expected_sha256 and source_hash != args.expected_sha256.lower():
        raise SystemExit(
            f"input SHA-256 mismatch: expected {args.expected_sha256}, got {source_hash}"
        )
    rows = load_rows(args.input_jsonl)
    validate_rows(rows)
    summary = summarize(rows)
    write_outputs(args.output_dir, rows, summary, source_hash)

    print(f"validated_samples={len(rows)}")
    print(f"input_sha256={source_hash}")
    for row in summary:
        print(
            f"{row['metric']}: interp={row['interp_mean']:.9f}, "
            f"trained={row['trained_mean']:.9f}, "
            f"relative={row['relative_improvement_percent']:.4f}%, "
            f"wins={row['wins']}"
        )


if __name__ == "__main__":
    main()
