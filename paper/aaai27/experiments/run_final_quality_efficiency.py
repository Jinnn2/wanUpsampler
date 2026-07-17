from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper.aaai27.experiments.prepare_quality_efficiency import QUALITY_DIMENSIONS, vbench_case_scores


@dataclass(frozen=True)
class Case:
    name: str
    model_cls: str
    lr_evaluations: int
    hr_evaluations: int
    total_evaluations: int
    handoff_step: int | None = None
    lora_strength: float | None = None


def main() -> None:
    args = parse_args()
    root = Path(args.out_root).resolve()
    cases = build_cases(args)
    validate_inputs(args)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    for case in cases:
        write_config(root / "configs" / f"{case.name}.json", args, case)
    prompts = load_prompts(Path(args.prompts), args.prompt_offset, args.limit)
    write_manifest(root, args, prompts, cases)

    if args.mode == "check":
        print(f"Validated final quality-efficiency suite: {root}")
        return
    if args.mode == "benchmark-spec":
        write_benchmark_spec(root, args, cases)
        return
    for case in cases:
        run_case(root, args, case, seed=args.seed, limit=args.limit, output_root=root / "videos" / case.name)


def build_cases(args: argparse.Namespace) -> list[Case]:
    return [
        Case("full_hr50", "wan2.1", 0, 50, 50),
        Case("talh40", "wan2.1_tail_skip_lora_clean_resizer_bridge", 40, 10, 50, 40, args.step40_strength),
        Case("talh45", "wan2.1_tail_skip_lora_clean_resizer_bridge", 45, 5, 50, 45, args.step45_strength),
        Case("full_lr50_stage2_1hr", "wan2.1_full_lr_stage2_one_hr", 50, 1, 51, 50),
    ]


def base_config(args: argparse.Namespace, case: Case) -> dict[str, Any]:
    return {
        "infer_steps": 50,
        "target_video_length": args.num_frames,
        "text_len": 512,
        "target_height": 720,
        "target_width": 1248,
        "self_attn_1_type": "flash_attn3",
        "cross_attn_1_type": "flash_attn3",
        "cross_attn_2_type": "flash_attn3",
        "sample_guide_scale": args.guide_scale,
        "sample_shift": 8,
        "enable_cfg": True,
        "cpu_offload": False,
        "feature_caching": "NoCaching",
        "compare_name": case.name,
    }


def write_config(path: Path, args: argparse.Namespace, case: Case) -> None:
    config = base_config(args, case)
    if case.name != "full_hr50":
        config.update(
            {
                "changing_resolution": True,
                "resolution_rate": [368 / 720],
                "wan_lowres_latent_size": [46, 80],
                "changing_resolution_steps": [case.handoff_step],
                "wan_clean_resizer_repo": str(REPO_ROOT),
                "wan_clean_resizer_ckpt": str(Path(args.stage2_checkpoint).resolve()),
                "wan_clean_resizer_train_config": str(Path(args.stage2_train_config).resolve()),
                "wan_clean_resizer_model_class": "stage2",
                "wan_clean_resizer_use_ema": args.stage2_use_ema,
            }
        )
    if case.name == "full_lr50_stage2_1hr":
        config["wan_final_refine_shift_increment"] = args.final_refine_shift_increment
    if case.name in {"talh40", "talh45"}:
        checkpoint = args.lora40_checkpoint if case.handoff_step == 40 else args.lora45_checkpoint
        config.update(
            {
                "lora_dynamic_apply": True,
                "lora_active_steps": [case.handoff_step],
                "lora_configs": [
                    {
                        "name": "wan2.1",
                        "path": str(Path(checkpoint).resolve()),
                        "strength": case.lora_strength,
                    }
                ],
            }
        )
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_case(
    root: Path,
    args: argparse.Namespace,
    case: Case,
    *,
    seed: int,
    limit: int,
    output_root: Path,
    skip_existing: bool | None = None,
    execute: bool = True,
) -> list[str]:
    batch = REPO_ROOT / "changing_resolution/scripts/bridge/run_lightx2v_clean_bridge_batch_infer.py"
    command = [
        args.python,
        str(batch),
        "--seed",
        str(seed),
        "--increment_seed",
        "--model_cls",
        case.model_cls,
        "--task",
        "t2v",
        "--model_path",
        str(Path(args.model_root).resolve()),
        "--config_json",
        str(root / "configs" / f"{case.name}.json"),
        "--prompts_file",
        str(Path(args.prompts).resolve()),
        "--out_dir",
        str(output_root),
        "--name_prefix",
        case.name,
        "--prompt-offset",
        str(args.prompt_offset),
        "--limit",
        str(limit),
        "--target_video_length",
        str(args.num_frames),
        "--negative_prompt",
        args.negative_prompt,
    ]
    use_skip = args.skip_existing if skip_existing is None else skip_existing
    if use_skip:
        command.append("--skip-existing")
    if execute:
        print(f"[batch] {case.name}: {limit} prompt(s), one model load", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True, env=inference_environment())
    return command


def write_manifest(root: Path, args: argparse.Namespace, prompts: list[str], cases: list[Case]) -> None:
    payload = {
        "schema_version": 1,
        "family": "wan50_quality_efficiency",
        "seed_base": args.seed,
        "prompt_offset": args.prompt_offset,
        "prompts": prompts,
        "cases": [asdict(case) for case in cases],
        "analysis_pairs": [
            {
                "comparison": f"full_hr50_vs_{case.name}",
                "left_case": "full_hr50",
                "right_case": case.name,
            }
            for case in cases
            if case.name != "full_hr50"
        ]
        + [{"comparison": "talh45_vs_talh40", "left_case": "talh45", "right_case": "talh40"}],
        "artifacts": {
            "stage2": artifact_fingerprint(Path(args.stage2_checkpoint)),
            "lora40": artifact_fingerprint(Path(args.lora40_checkpoint)),
            "lora45": artifact_fingerprint(Path(args.lora45_checkpoint)),
        },
        "settings": {
            "model_root": str(Path(args.model_root).resolve()),
            "prompts_file": str(Path(args.prompts).resolve()),
            "num_frames": args.num_frames,
            "guide_scale": args.guide_scale,
            "step40_strength": args.step40_strength,
            "step45_strength": args.step45_strength,
            "final_refine_shift_increment": args.final_refine_shift_increment,
            "stage2_use_ema": args.stage2_use_ema,
        },
    }
    (root / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_benchmark_spec(root: Path, args: argparse.Namespace, cases: list[Case]) -> None:
    vbench = root / "metrics/vbench_v1_custom.json"
    if not vbench.is_file():
        raise SystemExit(f"Run VBench before preparing the benchmark spec: {vbench}")
    spec_cases = []
    for case in cases:
        components = vbench_case_scores(vbench, case.name)
        command = run_case(
            root,
            args,
            case,
            seed=args.benchmark_seed,
            limit=1,
            output_root=root / "benchmark_runs" / case.name,
            skip_existing=False,
            execute=False,
        )
        spec_cases.append(
            {
                "family": "wan50",
                "name": case.name,
                "measurement": "cold_start_single_video_end_to_end",
                "command": shlex.join(command),
                "quality_metric": "vbench_custom_quality5_mean",
                "quality_value": sum(components.values()) / len(components),
                "quality_components": components,
                "vbench_source": str(vbench),
                "protocol": {
                    "lr_evaluations": case.lr_evaluations,
                    "hr_evaluations": case.hr_evaluations,
                    "total_evaluations": case.total_evaluations,
                },
                "environment": benchmark_environment(),
            }
        )
    output = Path(args.benchmark_output) if args.benchmark_output else root / "benchmark_spec.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "protocol": {
            "measurement": "cold_start_single_video_end_to_end",
            "prompt_count_per_process": 1,
            "fixed_seed": args.benchmark_seed,
            "quality_dimensions": QUALITY_DIMENSIONS,
            "dynamic_degree_reported_separately": True,
            "same_model_prompt_seed_frames_resolution": True,
        },
        "cases": spec_cases,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Final quality-efficiency benchmark spec: {output}")


def benchmark_environment() -> dict[str, str]:
    environment = inference_environment()
    return {
        key: environment[key]
        for key in ("LIGHTX2V_REPO", "DIFFSYNTH_REPO", "PYTHONPATH")
        if key in environment
    }


def inference_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("LIGHTX2V_REPO", "/mnt/afs_2/houze/LightX2V")
    environment.setdefault("DIFFSYNTH_REPO", "/mnt/afs_2/houze/DiffSynth-Studio")
    roots = [environment["LIGHTX2V_REPO"], environment["DIFFSYNTH_REPO"], str(REPO_ROOT)]
    if environment.get("PYTHONPATH"):
        roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def validate_inputs(args: argparse.Namespace) -> None:
    required = [
        Path(args.prompts),
        Path(args.model_root),
        Path(args.stage2_checkpoint),
        Path(args.stage2_train_config),
        Path(args.lora40_checkpoint),
        Path(args.lora45_checkpoint),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing final quality-efficiency inputs:\n  " + "\n  ".join(missing))


def load_prompts(path: Path, offset: int, limit: int) -> list[str]:
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    selected = prompts[offset : offset + limit]
    if not selected:
        raise SystemExit(f"No prompts selected from {path}")
    return selected


def artifact_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final four-way Wan50 quality-efficiency comparison.")
    parser.add_argument("mode", choices=["check", "run", "benchmark-spec"])
    parser.add_argument(
        "--out-root", default=str(REPO_ROOT / "outputs/aaai27_experiments/quality_efficiency_final")
    )
    parser.add_argument(
        "--prompts",
        default=str(REPO_ROOT / "changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt"),
    )
    parser.add_argument("--model-root", default="/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument(
        "--stage2-checkpoint",
        default=str(REPO_ROOT / "outputs/changing_resolution_clean_368x640_720x1248_stage2_lmdb/latest.pt"),
    )
    parser.add_argument(
        "--stage2-train-config",
        default=str(REPO_ROOT / "changing_resolution/configs/train_clean_368x640_to_720x1248_lmdb_stage2.yaml"),
    )
    parser.add_argument(
        "--lora40-checkpoint",
        default=str(
            REPO_ROOT / "outputs/changing_resolution_tail_skip_lora_step40_to_step50_temporal/latest.safetensors"
        ),
    )
    parser.add_argument(
        "--lora45-checkpoint",
        default=str(REPO_ROOT / "outputs/changing_resolution_tail_skip_lora_step45_to_step50/latest.safetensors"),
    )
    parser.add_argument("--step40-strength", type=float, default=1.0)
    parser.add_argument("--step45-strength", type=float, default=0.75)
    parser.add_argument("--final-refine-shift-increment", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=9700)
    parser.add_argument("--benchmark-seed", type=int, default=15000)
    parser.add_argument("--benchmark-output")
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--guide-scale", type=float, default=6.0)
    parser.add_argument("--stage2-use-ema", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


if __name__ == "__main__":
    main()
