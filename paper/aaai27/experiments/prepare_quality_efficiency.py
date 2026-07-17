from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
QUALITY_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
]


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    results = root / "outputs/aaai27_experiments"
    prompts = root / "changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt"
    wan_stage2 = root / "outputs/changing_resolution_clean_368x640_720x1248_stage2_lmdb/latest.pt"
    wan_stage2_config = root / "changing_resolution/configs/train_clean_368x640_to_720x1248_lmdb_stage2.yaml"
    wan_lora = root / "outputs/changing_resolution_tail_skip_lora_step45_to_step50/latest.safetensors"
    distill_stage2 = (
        root
        / "outputs/changing_resolution_distill_clean_368x640_720x1248_stage2_14b_cfgdistill_5k_lmdb/latest.pt"
    )
    distill_stage2_config = (
        root / "changing_resolution_distill/configs/train_clean_368x640_to_720x1248_lmdb_stage2_distill.yaml"
    )
    distill_lora = (
        root
        / "outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3/step_0010000.safetensors"
    )
    required = [
        prompts,
        wan_stage2,
        wan_stage2_config,
        wan_lora,
        distill_stage2,
        distill_stage2_config,
        distill_lora,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing final experiment inputs:\n" + "\n".join(missing))

    wan_vbench = results / "factorial_wan50/metrics/vbench_v1_custom.json"
    distill_vbench = results / "factorial_distill4/metrics/vbench_v1_custom.json"
    cases = [
        make_case(
            family="wan50",
            name="step45_base_interp",
            command=factorial_command(
                root,
                family="wan50",
                step=45,
                handoff="base",
                resizer="interp",
                prompts=prompts,
                out_root=results / "efficiency/runs/wan50_base_interp",
                stage2=wan_stage2,
                stage2_config=wan_stage2_config,
                lora=None,
                seed=args.seed,
                python=args.python,
            ),
            vbench=wan_vbench,
        ),
        make_case(
            family="wan50",
            name="step45_lora_stage2",
            command=factorial_command(
                root,
                family="wan50",
                step=45,
                handoff="lora",
                resizer="stage2",
                prompts=prompts,
                out_root=results / "efficiency/runs/wan50_lora_stage2",
                stage2=wan_stage2,
                stage2_config=wan_stage2_config,
                lora=wan_lora,
                seed=args.seed,
                python=args.python,
            ),
            vbench=wan_vbench,
        ),
        make_case(
            family="distill4",
            name="step3_base_interp",
            command=factorial_command(
                root,
                family="distill4",
                step=3,
                handoff="base",
                resizer="interp",
                prompts=prompts,
                out_root=results / "efficiency/runs/distill4_base_interp",
                stage2=distill_stage2,
                stage2_config=distill_stage2_config,
                lora=None,
                seed=args.seed,
                python=args.python,
                stage2_use_ema=True,
            ),
            vbench=distill_vbench,
        ),
        make_case(
            family="distill4",
            name="step3_lora_stage2",
            command=factorial_command(
                root,
                family="distill4",
                step=3,
                handoff="lora",
                resizer="stage2",
                prompts=prompts,
                out_root=results / "efficiency/runs/distill4_lora_stage2",
                stage2=distill_stage2,
                stage2_config=distill_stage2_config,
                lora=distill_lora,
                seed=args.seed,
                python=args.python,
                lora_strength=0.75,
                stage2_use_ema=True,
            ),
            vbench=distill_vbench,
        ),
    ]
    lightx2v_repo = Path(os.environ.get("LIGHTX2V_REPO", "/mnt/afs_2/houze/LightX2V")).resolve()
    diffsynth_repo = Path(os.environ.get("DIFFSYNTH_REPO", "/mnt/afs_2/houze/DiffSynth-Studio")).resolve()
    pythonpath = ":".join([str(lightx2v_repo), str(diffsynth_repo), str(root)])
    for case in cases:
        case["environment"] = {
            "LIGHTX2V_REPO": str(lightx2v_repo),
            "DIFFSYNTH_REPO": str(diffsynth_repo),
            "PYTHONPATH": pythonpath,
        }
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol": {
            "measurement": "cold_start_single_video_end_to_end",
            "prompt_count_per_process": 1,
            "fixed_seed": args.seed,
            "quality_dimensions": QUALITY_DIMENSIONS,
            "dynamic_degree_reported_separately": True,
        },
        "cases": cases,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Efficiency benchmark spec: {output}")
    for case in cases:
        print(f"  {case['family']}/{case['name']}: quality5={case['quality_value']:.6f}")


def make_case(family: str, name: str, command: str, vbench: Path) -> dict[str, Any]:
    components = vbench_case_scores(vbench, name)
    return {
        "family": family,
        "name": name,
        "measurement": "cold_start_single_video_end_to_end",
        "command": command,
        "quality_metric": "vbench_custom_quality5_mean",
        "quality_value": sum(components.values()) / len(components),
        "quality_components": components,
        "vbench_source": str(vbench),
    }


def factorial_command(
    root: Path,
    *,
    family: str,
    step: int,
    handoff: str,
    resizer: str,
    prompts: Path,
    out_root: Path,
    stage2: Path,
    stage2_config: Path,
    lora: Path | None,
    seed: int,
    lora_strength: float = 0.75,
    stage2_use_ema: bool = False,
    python: str = "python",
) -> str:
    command = [
        python,
        str(root / "paper/aaai27/experiments/run_factorial.py"),
        "run",
        "--family",
        family,
        "--steps",
        str(step),
        "--handoffs",
        handoff,
        "--resizers",
        resizer,
        "--prompts",
        str(prompts),
        "--out-root",
        str(out_root),
        "--stage2-checkpoint",
        str(stage2),
        "--stage2-train-config",
        str(stage2_config),
        "--limit",
        "1",
        "--seed",
        str(seed),
        "--lora-strength",
        str(lora_strength),
        "--no-skip-existing",
    ]
    if lora:
        command.extend(["--lora-checkpoint", f"{step}={lora}"])
    if stage2_use_ema:
        command.append("--stage2-use-ema")
    return shlex.join(command)


def vbench_case_scores(path: Path, case: str) -> dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Missing or invalid VBench JSON {path}: {exc}") from exc
    numeric = payload.get("cases", {}).get(case, {}).get("numeric_metrics", {})
    if not numeric:
        raise SystemExit(f"VBench case {case!r} is missing from {path}")
    scores: dict[str, float] = {}
    for dimension in QUALITY_DIMENSIONS:
        suffix = f".{dimension}.0"
        candidates = [(key, value) for key, value in numeric.items() if key.endswith(suffix)]
        if not candidates:
            raise SystemExit(f"Aggregate VBench dimension {dimension!r} not found for {case!r} in {path}")
        scores[dimension] = float(sorted(candidates)[-1][1])
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the four-case final quality-efficiency benchmark spec.")
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--output",
        default="outputs/aaai27_experiments/efficiency/benchmark_spec.json",
    )
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument("--python", default="/opt/conda/bin/python", help="Wan/LightX2V environment Python")
    return parser.parse_args()


if __name__ == "__main__":
    main()
