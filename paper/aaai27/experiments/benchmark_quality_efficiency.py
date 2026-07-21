from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    args = parse_args()
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cases = list(spec.get("cases", []))
    if not cases:
        raise SystemExit(f"Benchmark spec contains no cases: {spec_path}")
    for case in cases:
        validate_case(case)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_name(f"{output.stem}_raw{output.suffix}")
    signature = run_signature(spec_path, args)
    raw_rows = load_resumable_rows(raw_output, signature) if args.resume else []
    rows: list[dict[str, Any]] = []
    for case in cases:
        for phase, repeat in desired_repeats(args.warmup, args.repeats):
            existing = find_raw_row(raw_rows, case["name"], phase, repeat)
            if existing is not None:
                print(f"[resume] {case['name']} {phase} repeat={repeat}", flush=True)
                continue
            elapsed, peak = measure(
                case["command"],
                args.gpu,
                Path(args.workdir),
                case.get("environment", {}),
            )
            raw_rows.append(
                {
                    "run_signature": signature,
                    "gpu": args.gpu,
                    "family": case["family"],
                    "case": case["name"],
                    "method": case.get("protocol", {}).get("method", "NA"),
                    "phase": phase,
                    "repeat": repeat,
                    "elapsed_s": elapsed,
                    "peak_memory_gib": peak,
                }
            )
            write_csv_atomic(raw_output, raw_rows)

        measured = [
            row
            for row in raw_rows
            if row["case"] == case["name"] and row["phase"] == "measured"
        ]
        measured.sort(key=lambda row: int(row["repeat"]))
        if len(measured) != args.repeats:
            raise RuntimeError(
                f"{case['name']}: expected {args.repeats} measured rows, got {len(measured)}"
            )
        durations = [float(row["elapsed_s"]) for row in measured]
        peaks = [float(row["peak_memory_gib"]) for row in measured]
        rows.append(summary_row(case, args, durations, peaks))
        write_csv_atomic(output, rows)

    print(f"Quality-efficiency CSV: {output}")
    print(f"Raw measurements      : {raw_output}")


def desired_repeats(warmup: int, repeats: int) -> list[tuple[str, int]]:
    return [("warmup", index) for index in range(warmup)] + [
        ("measured", index) for index in range(repeats)
    ]


def find_raw_row(
    rows: list[dict[str, Any]],
    case_name: str,
    phase: str,
    repeat: int,
) -> dict[str, Any] | None:
    matches = [
        row
        for row in rows
        if row.get("case") == case_name
        and row.get("phase") == phase
        and int(row.get("repeat", -1)) == repeat
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Duplicate raw timing row: {case_name} {phase} repeat={repeat}")
    return matches[0] if matches else None


def load_resumable_rows(path: Path, signature: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    incompatible = [row for row in rows if row.get("run_signature") != signature]
    if incompatible:
        raise SystemExit(
            f"Cannot resume {path}: run signature differs. Use a new output path or omit --resume."
        )
    return rows


def run_signature(spec_path: Path, args: argparse.Namespace) -> str:
    digest = hashlib.sha256()
    digest.update(spec_path.read_bytes())
    digest.update(
        json.dumps(
            {
                "gpu": args.gpu,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "workdir": str(Path(args.workdir).resolve()),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def summary_row(
    case: dict[str, Any],
    args: argparse.Namespace,
    durations: list[float],
    peaks: list[float],
) -> dict[str, Any]:
    return {
        "family": case["family"],
        "case": case["name"],
        "method": case.get("protocol", {}).get("method", "NA"),
        "measurement": case.get("measurement", "end_to_end_generation"),
        "lr_evaluations": case.get("protocol", {}).get("lr_evaluations", "NA"),
        "hr_evaluations": case.get("protocol", {}).get("hr_evaluations", "NA"),
        "total_evaluations": case.get("protocol", {}).get("total_evaluations", "NA"),
        "handoff_step": case.get("protocol", {}).get("handoff_step", "NA"),
        "refinement_steps": case.get("protocol", {}).get("refinement_steps", "NA"),
        "reschedule_mode": case.get("protocol", {}).get("reschedule_mode", "NA"),
        "gpu": args.gpu,
        "repeats": args.repeats,
        "elapsed_mean_s": statistics.mean(durations),
        "elapsed_std_s": statistics.stdev(durations) if len(durations) > 1 else 0.0,
        "elapsed_median_s": statistics.median(durations),
        "inference_mean_s": statistics.mean(durations),
        "inference_std_s": statistics.stdev(durations) if len(durations) > 1 else 0.0,
        "peak_memory_gib": max(peaks),
        "peak_memory_mean_gib": statistics.mean(peaks),
        "quality_metric": case["quality_metric"],
        "quality_value": case.get("quality_value", "NA"),
        "quality_components": json.dumps(case.get("quality_components", {}), sort_keys=True),
        "vbench_source": case.get("vbench_source", ""),
        "command": case["command"],
    }


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_case(case: dict[str, Any]) -> None:
    missing = [key for key in ("family", "name", "command", "quality_metric") if key not in case]
    if missing:
        raise SystemExit("Benchmark case missing keys: " + ", ".join(missing))


def measure(command: str, gpu: int, cwd: Path, extra_environment: dict[str, str]) -> tuple[float, float]:
    environment = dict(os.environ)
    environment.update({str(key): str(value) for key, value in extra_environment.items()})
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    baseline = gpu_memory_mib(gpu)
    started = time.perf_counter()
    process = subprocess.Popen(["bash", "-c", command], cwd=cwd, env=environment)
    peak = baseline
    while process.poll() is None:
        peak = max(peak, gpu_memory_mib(gpu))
        time.sleep(0.1)
    elapsed = time.perf_counter() - started
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return elapsed, max(0.0, peak - baseline) / 1024.0


def gpu_memory_mib(gpu: int) -> float:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return float(result.stdout.strip().splitlines()[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark isolated experiment commands with wall time and peak GPU memory.")
    parser.add_argument("--spec", required=True, help="JSON containing a cases list")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workdir", default=str(REPO_ROOT))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Resume compatible raw repeats from the output raw CSV.")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be >= 0 and --repeats must be >= 1")
    return args


if __name__ == "__main__":
    main()
