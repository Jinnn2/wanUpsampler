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
    for case in spec.get("cases", []):
        validate_case(case)
        durations: list[float] = []
        peaks: list[float] = []
        for repeat in range(args.warmup + args.repeats):
            elapsed, peak = measure(case["command"], args.gpu, Path(args.workdir), case.get("environment", {}))
            if repeat >= args.warmup:
                durations.append(elapsed)
                peaks.append(peak)
        rows.append(
            {
                "family": case["family"],
                "case": case["name"],
                "measurement": case.get("measurement", "end_to_end_generation"),
                "gpu": args.gpu,
                "repeats": args.repeats,
                "elapsed_mean_s": statistics.mean(durations),
                "elapsed_std_s": statistics.stdev(durations) if len(durations) > 1 else 0.0,
                "inference_mean_s": statistics.mean(durations),
                "inference_std_s": statistics.stdev(durations) if len(durations) > 1 else 0.0,
                "peak_memory_gib": max(peaks),
                "quality_metric": case["quality_metric"],
                "quality_value": case.get("quality_value", "NA"),
                "command": case["command"],
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Quality-efficiency CSV: {output}")


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
    process = subprocess.Popen(["bash", "-lc", command], cwd=cwd, env=environment)
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
