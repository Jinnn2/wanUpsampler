from __future__ import annotations

import argparse
import csv
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
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for case in spec.get("cases", []):
        validate_case(case)
        durations: list[float] = []
        peaks: list[float] = []
        for repeat in range(args.warmup + args.repeats):
            elapsed, peak = measure(case["command"], args.gpu, Path(args.workdir), case.get("environment", {}))
            raw_rows.append(
                {
                    "family": case["family"],
                    "case": case["name"],
                    "phase": "warmup" if repeat < args.warmup else "measured",
                    "repeat": repeat if repeat < args.warmup else repeat - args.warmup,
                    "elapsed_s": elapsed,
                    "peak_memory_gib": peak,
                }
            )
            if repeat >= args.warmup:
                durations.append(elapsed)
                peaks.append(peak)
        rows.append(
            {
                "family": case["family"],
                "case": case["name"],
                "measurement": case.get("measurement", "end_to_end_generation"),
                "lr_evaluations": case.get("protocol", {}).get("lr_evaluations", "NA"),
                "hr_evaluations": case.get("protocol", {}).get("hr_evaluations", "NA"),
                "total_evaluations": case.get("protocol", {}).get("total_evaluations", "NA"),
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
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    raw_output = output.with_name(f"{output.stem}_raw{output.suffix}")
    with raw_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"Quality-efficiency CSV: {output}")
    print(f"Raw measurements      : {raw_output}")


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
    return parser.parse_args()


if __name__ == "__main__":
    main()
