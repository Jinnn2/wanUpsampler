from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def main() -> None:
    args = parse_args()
    root = Path(args.factorial_root).resolve()
    manifest = load_json(root / "run_manifest.json")
    inputs = prepare_inputs(root, manifest)
    if args.action == "prepare":
        print(f"Prepared {len(inputs)} VBench cases under {root / 'metrics/vbench_inputs'}")
        return
    if args.action == "run":
        if not args.vbench_root:
            raise SystemExit("--vbench-root (or VBENCH_ROOT) is required for action=run")
        run_all(root, inputs, Path(args.vbench_root).resolve(), args.dimensions, args.ngpus, args.python)
    output = collect_results(root, manifest, args.dimensions, Path(args.vbench_root).resolve() if args.vbench_root else None)
    print(f"Canonical VBench JSON: {output}")


def prepare_inputs(root: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    prompts = manifest.get("prompts")
    cases = manifest.get("cases")
    if not isinstance(prompts, list) or not prompts or not isinstance(cases, list) or not cases:
        raise SystemExit("run_manifest.json must contain non-empty prompts and cases")
    offset = int(manifest.get("prompt_offset", 0))
    seed_base = int(manifest["seed_base"])
    input_root = root / "metrics/vbench_inputs"
    result: dict[str, Path] = {}
    for case in cases:
        name = str(case["name"])
        mapping: dict[str, str] = {}
        for position, prompt in enumerate(prompts):
            sample_index = offset + position
            seed = seed_base + sample_index
            video = root / "videos" / name / f"{name}_{sample_index:02d}_seed{seed}.mp4"
            if not video.is_file() or video.stat().st_size < 1024:
                raise SystemExit(f"Missing or undersized factorial video: {video}")
            mapping[str(video.resolve())] = str(prompt)
        case_dir = input_root / name
        case_dir.mkdir(parents=True, exist_ok=True)
        prompt_map = case_dir / "prompt_map.json"
        prompt_map.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result[name] = prompt_map
    return result


def run_all(
    root: Path,
    inputs: dict[str, Path],
    vbench_root: Path,
    dimensions: list[str],
    ngpus: int,
    python: str,
) -> None:
    evaluate = vbench_root / "evaluate.py"
    if not evaluate.is_file():
        raise SystemExit(f"Official VBench evaluate.py not found: {evaluate}")
    raw_root = root / "metrics/vbench_raw"
    for case, prompt_map in inputs.items():
        output = raw_root / case
        output.mkdir(parents=True, exist_ok=True)
        videos = root / "videos" / case
        base = [
            str(evaluate),
            "--videos_path",
            str(videos),
            "--dimension",
            *dimensions,
            "--mode",
            "custom_input",
            "--prompt_file",
            str(prompt_map),
            "--output_path",
            str(output),
        ]
        command = (
            [python, "-m", "torch.distributed.run", f"--nproc_per_node={ngpus}", "--standalone", *base]
            if ngpus > 1
            else [python, *base]
        )
        print(f"[VBench] {case}", flush=True)
        subprocess.run(command, cwd=vbench_root, check=True)


def collect_results(
    root: Path, manifest: dict[str, Any], dimensions: list[str], vbench_root: Path | None
) -> Path:
    raw_root = root / "metrics/vbench_raw"
    cases: dict[str, Any] = {}
    missing: list[str] = []
    for case in manifest["cases"]:
        name = str(case["name"])
        files = sorted((raw_root / name).rglob("*.json")) if (raw_root / name).is_dir() else []
        numeric: dict[str, float] = {}
        sources: list[str] = []
        for path in files:
            payload = load_json(path)
            flattened = flatten_numeric(payload)
            if flattened:
                sources.append(str(path.resolve()))
                for key, value in flattened.items():
                    numeric[f"{path.stem}.{key}"] = value
        if not numeric:
            missing.append(name)
        cases[name] = {"source_files": sources, "numeric_metrics": numeric}
    if missing:
        raise SystemExit("No numeric VBench JSON found for cases: " + ", ".join(missing))
    revision = None
    if vbench_root and (vbench_root / ".git").exists():
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=vbench_root, check=True, text=True, capture_output=True
        ).stdout.strip()
    payload = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmark": "VBench",
        "mode": "custom_input",
        "vbench_revision": revision,
        "family": manifest["family"],
        "dimensions": dimensions,
        "prompt_count": len(manifest["prompts"]),
        "cases": cases,
    }
    metrics = root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    output = metrics / "vbench_v1_custom.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            result.update(flatten_numeric(item, child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        result[prefix or "value"] = float(value)
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid JSON {path}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, run, and canonically collect VBench factorial evaluation.")
    parser.add_argument("action", choices=["prepare", "run", "collect"])
    parser.add_argument("--factorial-root", required=True)
    parser.add_argument("--vbench-root", default=None)
    parser.add_argument(
        "--python",
        default=os.environ.get("VBENCH_PYTHON", sys.executable),
        help="Python executable from the isolated VBench environment",
    )
    parser.add_argument("--dimension", dest="dimensions", action="append", default=[])
    parser.add_argument("--ngpus", type=int, default=1)
    args = parser.parse_args()
    if not args.dimensions:
        args.dimensions = list(DEFAULT_DIMENSIONS)
    if args.ngpus < 1:
        parser.error("--ngpus must be >= 1")
    return args


if __name__ == "__main__":
    main()
