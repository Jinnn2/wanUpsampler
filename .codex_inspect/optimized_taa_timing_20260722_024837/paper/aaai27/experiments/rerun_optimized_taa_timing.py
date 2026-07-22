from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_SCRIPT = REPO_ROOT / "paper/aaai27/experiments/benchmark_quality_efficiency.py"
OPTIMIZATION_FILES = (
    REPO_ROOT / "changing_resolution/dynamic_lora.py",
    REPO_ROOT / "changing_resolution/lightx2v_clean_bridge.py",
)
TARGET_CASES = ("talh40", "talh45")
EXPECTED_CASE_COUNT = 13


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else suite_root / "optimized_taa_timing"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    source_summary = (
        Path(args.source_summary).resolve()
        if args.source_summary
        else suite_root / "quality_efficiency.csv"
    )
    source_raw = (
        Path(args.source_raw).resolve()
        if args.source_raw
        else source_summary.with_name(f"{source_summary.stem}_raw{source_summary.suffix}")
    )
    old_summary_rows = read_csv(source_summary)
    old_raw_rows = read_csv(source_raw)
    validate_source_rows(old_summary_rows, old_raw_rows)
    source_gpu = infer_source_gpu(old_summary_rows)
    for row in old_raw_rows:
        if not row.get("gpu"):
            row["gpu"] = str(source_gpu)
    gpu = source_gpu if args.gpu is None else args.gpu
    if gpu != source_gpu and not args.allow_cross_gpu_merge:
        raise SystemExit(
            f"Existing official timing uses physical GPU {source_gpu}, but rerun requested GPU {gpu}. "
            "Use the same GPU or pass --allow-cross-gpu-merge explicitly for a non-official diagnostic run."
        )

    source_spec = suite_root / "benchmark_spec.json"
    spec = load_json(source_spec)
    filtered_spec = filter_spec(spec)
    validate_talh_configs(suite_root)
    optimization_fingerprints = [fingerprint(path) for path in OPTIMIZATION_FILES]
    filtered_spec["rerun"] = {
        "reason": "resident step-local LoRA with zero-strength compute bypass",
        "source_spec": str(source_spec),
        "source_spec_sha256": sha256_file(source_spec),
        "optimization_files": optimization_fingerprints,
        "cases": list(TARGET_CASES),
        "physical_gpu": gpu,
        "source_physical_gpu": source_gpu,
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    spec_path = output_root / "benchmark_spec_talh_optimized.json"
    spec_path.write_text(
        json.dumps(filtered_spec, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    partial_summary = output_root / "quality_efficiency_talh_optimized.csv"
    partial_raw = partial_summary.with_name(
        f"{partial_summary.stem}_raw{partial_summary.suffix}"
    )
    command = [
        args.python,
        str(BENCHMARK_SCRIPT),
        "--spec",
        str(spec_path),
        "--output",
        str(partial_summary),
        "--gpu",
        str(gpu),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--workdir",
        str(REPO_ROOT),
    ]
    if args.resume:
        command.append("--resume")

    print(f"Existing official timing GPU: {source_gpu}")
    print(f"Selected rerun GPU          : {gpu}")
    print(f"Prepared optimized TALH spec: {spec_path}")
    print("Command:")
    print("  " + shell_join(command))
    print(
        f"Planned fresh processes: {len(TARGET_CASES)} × "
        f"({args.warmup} warm-up + {args.repeats} measured) = "
        f"{len(TARGET_CASES) * (args.warmup + args.repeats)}"
    )
    if args.prepare_only:
        return

    if not args.allow_busy_gpu:
        ensure_gpu_idle(gpu)
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    new_summary_rows = read_csv(partial_summary)
    new_raw_rows = read_csv(partial_raw)
    validate_rerun_rows(new_summary_rows, new_raw_rows, gpu, args)

    merged_summary_rows = replace_cases(old_summary_rows, new_summary_rows)
    merged_raw_rows = replace_cases(old_raw_rows, new_raw_rows)
    merged_summary = output_root / "quality_efficiency_optimized_merged.csv"
    merged_raw = output_root / "quality_efficiency_optimized_merged_raw.csv"
    write_csv(merged_summary, merged_summary_rows)
    write_csv(merged_raw, merged_raw_rows)

    comparison = output_root / "talh_timing_before_after.csv"
    write_comparison(comparison, old_summary_rows, new_summary_rows)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "optimized zero-strength LoRA bypass timing rerun",
        "quality_outputs_reused": True,
        "vbench_reused": True,
        "settings": {
            "physical_gpu": gpu,
            "source_physical_gpu": source_gpu,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "cases": list(TARGET_CASES),
            "resume": args.resume,
            "cross_gpu_merge": gpu != source_gpu,
        },
        "inputs": {
            "source_spec": fingerprint(source_spec),
            "source_summary": fingerprint(source_summary),
            "source_raw": fingerprint(source_raw),
            "optimization_files": optimization_fingerprints,
        },
        "outputs": {
            "filtered_spec": fingerprint(spec_path),
            "partial_summary": fingerprint(partial_summary),
            "partial_raw": fingerprint(partial_raw),
            "merged_summary": fingerprint(merged_summary),
            "merged_raw": fingerprint(merged_raw),
            "comparison": fingerprint(comparison),
        },
    }
    manifest_path = output_root / "rerun_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print_summary(new_summary_rows)
    print(f"Partial timing : {partial_summary}")
    print(f"Partial raw    : {partial_raw}")
    print(f"Merged 13-case: {merged_summary}")
    print(f"Merged raw     : {merged_raw}")
    print(f"Before/after   : {comparison}")
    print(f"Audit manifest : {manifest_path}")


def filter_spec(spec: dict[str, Any]) -> dict[str, Any]:
    selected = [case for case in spec.get("cases", []) if case.get("name") in TARGET_CASES]
    names = [case.get("name") for case in selected]
    if set(names) != set(TARGET_CASES) or len(names) != len(TARGET_CASES):
        raise SystemExit(
            f"benchmark_spec.json must contain exactly one row for each of {TARGET_CASES}, got {names}"
        )
    payload = dict(spec)
    payload["cases"] = selected
    return payload


def validate_talh_configs(suite_root: Path) -> None:
    for case_name, expected_step in (("talh40", 40), ("talh45", 45)):
        path = suite_root / "configs" / f"{case_name}.json"
        config = load_json(path)
        lora_configs = list(config.get("lora_configs") or [])
        if len(lora_configs) != 1:
            raise SystemExit(f"{case_name}: expected exactly one LoRA config in {path}")
        strength = float(lora_configs[0].get("strength", -1.0))
        if strength != 0.75:
            raise SystemExit(f"{case_name}: expected LoRA strength=0.75, got {strength}")
        active_steps = [int(value) for value in config.get("lora_active_steps", [])]
        if active_steps != [expected_step]:
            raise SystemExit(
                f"{case_name}: expected lora_active_steps=[{expected_step}], got {active_steps}"
            )


def validate_source_rows(
    summary_rows: list[dict[str, str]],
    raw_rows: list[dict[str, str]],
) -> None:
    summary_names = [row["case"] for row in summary_rows]
    if len(summary_names) != EXPECTED_CASE_COUNT or len(set(summary_names)) != EXPECTED_CASE_COUNT:
        raise SystemExit(
            f"Source summary must contain exactly {EXPECTED_CASE_COUNT} unique cases; "
            f"got {len(summary_names)} rows and {len(set(summary_names))} unique cases"
        )
    raw_names = {row["case"] for row in raw_rows}
    missing_raw = set(summary_names) - raw_names
    if missing_raw:
        raise SystemExit(f"Source raw CSV is missing cases: {sorted(missing_raw)}")


def infer_source_gpu(summary_rows: list[dict[str, str]]) -> int:
    gpu_values = {int(row["gpu"]) for row in summary_rows}
    if len(gpu_values) != 1:
        raise SystemExit(f"Source summary mixes GPU identifiers: {sorted(gpu_values)}")
    return next(iter(gpu_values))


def validate_rerun_rows(
    summary_rows: list[dict[str, str]],
    raw_rows: list[dict[str, str]],
    gpu: int,
    args: argparse.Namespace,
) -> None:
    names = [row["case"] for row in summary_rows]
    if set(names) != set(TARGET_CASES) or len(names) != len(TARGET_CASES):
        raise RuntimeError(f"Rerun summary case mismatch: {names}")
    if any(int(row["gpu"]) != gpu for row in summary_rows):
        raise RuntimeError("Rerun summary recorded the wrong GPU")
    expected_raw = len(TARGET_CASES) * (args.warmup + args.repeats)
    if len(raw_rows) != expected_raw:
        raise RuntimeError(f"Expected {expected_raw} rerun raw rows, got {len(raw_rows)}")
    if any(int(row["gpu"]) != gpu for row in raw_rows):
        raise RuntimeError("Rerun raw CSV recorded the wrong GPU")


def replace_cases(
    old_rows: list[dict[str, str]],
    new_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    targets = set(TARGET_CASES)
    new_by_case: dict[str, list[dict[str, str]]] = {}
    for row in new_rows:
        new_by_case.setdefault(row["case"], []).append(row)
    missing = targets - set(new_by_case)
    if missing:
        raise RuntimeError(f"Replacement rows missing cases: {sorted(missing)}")

    merged: list[dict[str, str]] = []
    inserted: set[str] = set()
    for row in old_rows:
        case_name = row["case"]
        if case_name not in targets:
            merged.append(row)
        elif case_name not in inserted:
            merged.extend(new_by_case[case_name])
            inserted.add(case_name)
    return merged


def write_comparison(
    path: Path,
    old_rows: list[dict[str, str]],
    new_rows: list[dict[str, str]],
) -> None:
    old_by_case = {row["case"]: row for row in old_rows}
    new_by_case = {row["case"]: row for row in new_rows}
    rows = []
    for case_name in TARGET_CASES:
        old_time = float(old_by_case[case_name]["elapsed_mean_s"])
        new_time = float(new_by_case[case_name]["elapsed_mean_s"])
        rows.append(
            {
                "case": case_name,
                "old_gpu": old_by_case[case_name].get("gpu", ""),
                "new_gpu": new_by_case[case_name].get("gpu", ""),
                "old_elapsed_mean_s": old_time,
                "new_elapsed_mean_s": new_time,
                "delta_s": new_time - old_time,
                "delta_pct": 100.0 * (new_time / old_time - 1.0),
                "quality_value_unchanged": (
                    old_by_case[case_name].get("quality_value")
                    == new_by_case[case_name].get("quality_value")
                ),
            }
        )
    write_csv(path, rows)


def ensure_gpu_idle(gpu: int) -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    processes = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "No running processes" not in line
    ]
    if processes:
        raise SystemExit(
            f"GPU {gpu} has active compute processes:\n  "
            + "\n  ".join(processes)
            + "\nWait for it to become idle or pass --allow-busy-gpu explicitly."
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Missing CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"CSV contains no rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def print_summary(rows: list[dict[str, str]]) -> None:
    by_case = {row["case"]: row for row in rows}
    print("Optimized cold-start timing:")
    for case_name in TARGET_CASES:
        row = by_case[case_name]
        print(
            f"  {case_name}: mean={float(row['elapsed_mean_s']):.3f}s "
            f"std={float(row['elapsed_std_s']):.3f}s "
            f"median={float(row['elapsed_median_s']):.3f}s "
            f"peak={float(row['peak_memory_gib']):.3f}GiB"
        )


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def shell_join(command: list[str]) -> str:
    import shlex

    return shlex.join(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun optimized TALH-40/45 timings and merge them into the matching-GPU 13-case table."
    )
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Physical GPU. Defaults to the GPU recorded in the existing 13-case summary.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--source-summary", default="")
    parser.add_argument("--source-raw", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--allow-cross-gpu-merge", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be >= 0 and --repeats must be >= 1")
    return args


if __name__ == "__main__":
    main()
