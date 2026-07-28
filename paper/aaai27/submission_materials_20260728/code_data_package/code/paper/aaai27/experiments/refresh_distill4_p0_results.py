from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper.aaai27.experiments.benchmark_warm_quality_efficiency import (  # noqa: E402
    fingerprint,
    protocol_signature,
    summarize_pairs,
    write_csv_atomic,
    write_json_atomic,
)
from paper.aaai27.experiments.prepare_quality_efficiency import (  # noqa: E402
    QUALITY_DIMENSIONS,
    vbench_case_scores,
)


P0_CASE = "endpoint_rgb_1hr"


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root).resolve()
    if args.action == "merge-vbench":
        merge_vbench(suite_root, Path(args.partial_json).resolve())
    else:
        merge_warm(suite_root, Path(args.partial_root).resolve())


def merge_vbench(suite_root: Path, partial_path: Path) -> None:
    canonical_path = suite_root / "metrics/vbench_v1_custom.json"
    canonical = load_json(canonical_path)
    partial = load_json(partial_path)
    canonical_cases = canonical.get("cases", {})
    partial_cases = partial.get("cases", {})
    if P0_CASE not in canonical_cases:
        raise SystemExit(f"{P0_CASE} is missing from {canonical_path}")
    if set(partial_cases) != {P0_CASE}:
        raise SystemExit(
            f"Partial VBench JSON must contain only {P0_CASE}, got {sorted(partial_cases)}"
        )
    if set(partial.get("dimensions", [])) != set(canonical.get("dimensions", [])):
        raise SystemExit("Partial and canonical VBench dimensions differ")
    timestamp = utc_timestamp()
    history = suite_root / "metrics/history" / f"p0_vbench_{timestamp}"
    backup(canonical_path, history / canonical_path.name)
    backup(partial_path, history / partial_path.name)

    canonical_cases[P0_CASE] = partial_cases[P0_CASE]
    canonical["generated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    canonical["p0_refresh"] = {
        "case": P0_CASE,
        "protocol": "MrFlow-style direct sigma=0.12",
        "partial_source": str(partial_path),
        "previous_canonical": str(history / canonical_path.name),
    }
    write_json_atomic(canonical_path, canonical)
    refresh_benchmark_spec(suite_root, canonical_path)
    write_json_atomic(
        suite_root / "metrics/p0_vbench_refresh_manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "case": P0_CASE,
            "partial": fingerprint(partial_path),
            "canonical": fingerprint(canonical_path),
            "history": str(history),
        },
    )
    print(f"Merged P0 VBench case into {canonical_path}")


def refresh_benchmark_spec(suite_root: Path, vbench_path: Path) -> None:
    spec_path = suite_root / "benchmark_spec.json"
    spec = load_json(spec_path)
    scores = vbench_case_scores(vbench_path, P0_CASE)
    cases = spec.get("cases", [])
    target = next((case for case in cases if case.get("name") == P0_CASE), None)
    if target is None:
        raise SystemExit(f"{P0_CASE} is missing from {spec_path}")
    target.update(
        {
            "quality_metric": "vbench_custom_quality5_mean",
            "quality_value": sum(scores.values()) / len(QUALITY_DIMENSIONS),
            "quality_components": scores,
            "vbench_source": str(vbench_path),
        }
    )
    write_json_atomic(spec_path, spec)


def merge_warm(suite_root: Path, partial_root: Path) -> None:
    canonical_root = suite_root / "warm_quality_efficiency"
    canonical_summary = canonical_root / "quality_efficiency_warm.csv"
    canonical_raw = canonical_root / "quality_efficiency_warm_raw.csv"
    canonical_pairs = canonical_root / "quality_efficiency_warm_pairs.csv"
    partial_summary = partial_root / "quality_efficiency_warm.csv"
    partial_raw = partial_root / "quality_efficiency_warm_raw.csv"
    partial_manifest = partial_root / "warm_timing_manifest.json"
    for path in (
        canonical_summary,
        canonical_raw,
        canonical_pairs,
        partial_summary,
        partial_raw,
        partial_manifest,
    ):
        if not path.is_file():
            raise SystemExit(f"Missing warm timing artifact: {path}")

    summary_rows, summary_fields = read_csv(canonical_summary)
    raw_rows, raw_fields = read_csv(canonical_raw)
    new_summary, _ = read_csv(partial_summary)
    new_raw, _ = read_csv(partial_raw)
    if {row.get("case") for row in new_summary} != {P0_CASE}:
        raise SystemExit(f"Partial warm summary must contain only {P0_CASE}")
    if {row.get("case") for row in new_raw} != {P0_CASE}:
        raise SystemExit(f"Partial warm raw table must contain only {P0_CASE}")
    if P0_CASE not in {row.get("case") for row in summary_rows}:
        raise SystemExit(f"Canonical warm summary is missing {P0_CASE}")

    timestamp = utc_timestamp()
    history = canonical_root / "history" / f"p0_warm_{timestamp}"
    for path in (
        canonical_summary,
        canonical_raw,
        canonical_pairs,
        canonical_root / "warm_timing_manifest.json",
        canonical_root / "protocol.json",
        canonical_root / "raw" / f"{P0_CASE}.jsonl",
        canonical_root / "resources" / f"{P0_CASE}.json",
        canonical_root / "configs" / f"{P0_CASE}.json",
    ):
        if path.is_file():
            backup(path, history / path.relative_to(canonical_root))

    summary_rows = [
        row for row in summary_rows if row.get("case") != P0_CASE
    ] + new_summary
    raw_rows = [row for row in raw_rows if row.get("case") != P0_CASE] + new_raw
    order = {
        case["name"]: index
        for index, case in enumerate(load_json(suite_root / "run_manifest.json")["cases"])
    }
    summary_rows.sort(key=lambda row: order.get(str(row.get("case")), 10_000))
    raw_rows.sort(
        key=lambda row: (
            order.get(str(row.get("case")), 10_000),
            0 if row.get("phase") == "warmup" else 1,
            int(row.get("repeat", 0)),
        )
    )
    refresh_speedups(summary_rows)
    pairs = summarize_pairs(
        load_json(suite_root / "run_manifest.json").get("analysis_pairs", []),
        raw_rows,
        summary_rows,
    )
    write_csv_atomic(canonical_summary, summary_rows, fieldnames=summary_fields)
    write_csv_atomic(canonical_raw, raw_rows, fieldnames=raw_fields)
    write_csv_atomic(
        canonical_pairs,
        pairs,
        fieldnames=list(pairs[0]) if pairs else None,
    )
    for subdir, suffix in (
        ("raw", ".jsonl"),
        ("resources", ".json"),
        ("configs", ".json"),
    ):
        source = partial_root / subdir / f"{P0_CASE}{suffix}"
        destination = canonical_root / subdir / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    old_manifest_path = canonical_root / "warm_timing_manifest.json"
    old_manifest = (
        load_json(history / "warm_timing_manifest.json")
        if (history / "warm_timing_manifest.json").is_file()
        else {}
    )
    merged_protocol = load_json(history / "protocol.json")
    partial_protocol = load_json(partial_root / "protocol.json")
    for key in (
        "source_manifest_sha256",
        "source_spec_sha256",
        "implementation_sha256",
    ):
        merged_protocol[key] = partial_protocol[key]
    merged_protocol.setdefault("config_sha256", {})[P0_CASE] = partial_protocol[
        "config_sha256"
    ][P0_CASE]
    merged_protocol["case_refreshes"] = {
        P0_CASE: {
            "partial_root": str(partial_root),
            "run_signature": partial_protocol["run_signature"],
        }
    }
    merged_protocol["run_signature"] = protocol_signature(merged_protocol)
    write_json_atomic(canonical_root / "protocol.json", merged_protocol)
    old_manifest.update(
        {
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "settings": merged_protocol,
            "p0_refresh": {
                "case": P0_CASE,
                "protocol": "MrFlow-style direct sigma=0.12",
                "partial_root": str(partial_root),
                "partial_manifest": fingerprint(partial_manifest),
                "history": str(history),
            },
            "outputs": {
                "summary": fingerprint(canonical_summary),
                "raw": fingerprint(canonical_raw),
                "pairs": fingerprint(canonical_pairs),
            },
        }
    )
    write_json_atomic(old_manifest_path, old_manifest)
    write_json_atomic(
        canonical_root / "p0_warm_refresh_manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "case": P0_CASE,
            "partial_root": str(partial_root),
            "canonical_summary": fingerprint(canonical_summary),
            "history": str(history),
        },
    )
    print(f"Merged P0 warm timing into {canonical_root}")


def refresh_speedups(rows: list[dict[str, Any]]) -> None:
    native = next((row for row in rows if row.get("case") == "native_hr4"), None)
    if native is None:
        raise SystemExit("Canonical warm summary is missing native_hr4")
    native_time = float(native["pipeline_mean_s"])
    for row in rows:
        elapsed = float(row["pipeline_mean_s"])
        row["speedup_vs_native"] = native_time / elapsed
        row["latency_reduction_vs_native_pct"] = 100.0 * (
            1.0 - elapsed / native_time
        )


def backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid JSON {path}: {exc}") from exc


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge targeted P0 VBench and warm timing refreshes into Distill4."
    )
    parser.add_argument("action", choices=["merge-vbench", "merge-warm"])
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--partial-json", default="")
    parser.add_argument("--partial-root", default="")
    args = parser.parse_args()
    if args.action == "merge-vbench" and not args.partial_json:
        parser.error("merge-vbench requires --partial-json")
    if args.action == "merge-warm" and not args.partial_root:
        parser.error("merge-warm requires --partial-root")
    return args


if __name__ == "__main__":
    main()
