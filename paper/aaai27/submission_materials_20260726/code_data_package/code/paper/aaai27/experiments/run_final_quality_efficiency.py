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

from paper.aaai27.experiments.prepare_quality_efficiency import (  # noqa: E402
    QUALITY_DIMENSIONS,
    vbench_case_scores,
)


@dataclass(frozen=True)
class Case:
    name: str
    method: str
    model_cls: str
    lr_evaluations: int
    hr_evaluations: int
    total_evaluations: int
    handoff_step: int | None = None
    lora_strength: float | None = None
    refinement_steps: int | None = None
    reschedule_mode: str = "canonical"
    mixed_evaluations: int = 0


def main() -> None:
    args = parse_args()
    root = Path(args.out_root).resolve()
    cases = build_cases(args)
    validate_inputs(args, cases)
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
    methods = set(getattr(args, "methods", ["native", "lightx2v", "talh", "endpoint", "ralu"]))
    lightx2v_steps = tuple(getattr(args, "lightx2v_handoff_steps", [40, 45, 48]))
    endpoint_steps = tuple(getattr(args, "endpoint_refinement_steps", [0, 1, 2, 5]))
    for label, values in (("LightX2V", lightx2v_steps),):
        invalid = [step for step in values if step < 1 or step >= 50]
        if invalid:
            raise SystemExit(f"Invalid {label} handoff step(s) {invalid}; expected [1, 49]")
    invalid_refinements = [step for step in endpoint_steps if step < 0 or step > 50]
    if invalid_refinements:
        raise SystemExit(f"Invalid Endpoint refinement step(s) {invalid_refinements}; expected [0, 50]")

    cases: list[Case] = []
    if "native" in methods:
        cases.append(Case("full_hr50", "native", "wan2.1", 0, 50, 50))
    if "lightx2v" in methods:
        cases.extend(
            Case(
                f"lightx2v_cr{step}",
                "lightx2v",
                "wan2.1_clean_interp_bridge",
                step,
                50 - step,
                50,
                handoff_step=step,
            )
            for step in lightx2v_steps
        )
    if "talh" in methods:
        cases.extend(
            [
                Case(
                    "talh40",
                    "talh",
                    "wan2.1_tail_skip_lora_clean_resizer_bridge",
                    40,
                    10,
                    50,
                    handoff_step=40,
                    lora_strength=args.step40_strength,
                ),
                Case(
                    "talh45",
                    "talh",
                    "wan2.1_tail_skip_lora_clean_resizer_bridge",
                    45,
                    5,
                    50,
                    handoff_step=45,
                    lora_strength=args.step45_strength,
                ),
            ]
        )
    if "endpoint" in methods:
        cases.extend(
            Case(
                f"full_lr50_stage2_{steps}hr",
                "endpoint",
                "wan2.1_full_lr_stage2_k_hr",
                50,
                steps,
                50 + steps,
                handoff_step=50,
                refinement_steps=steps,
                reschedule_mode="lowest_noise_hr_suffix",
            )
            for steps in endpoint_steps
        )
    if "ralu" in methods:
        stage_steps = tuple(int(v) for v in getattr(args, "ralu_stage_steps", [5, 6, 7]))
        cases.append(
            Case(
                "ralu_quality",
                "ralu",
                "wan2.1_ralu_quality",
                stage_steps[0],
                stage_steps[2],
                sum(stage_steps),
                reschedule_mode="ralu_three_stage_ntdm_region_adaptive",
                mixed_evaluations=stage_steps[1],
            )
        )
    if not cases:
        raise SystemExit("No quality-efficiency cases selected")
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise SystemExit("Quality-efficiency case names are not unique")
    return cases


def base_config(args: argparse.Namespace, case: Case) -> dict[str, Any]:
    return {
        "infer_steps": case.total_evaluations if case.method == "ralu" else 50,
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
    if case.method not in {"native", "ralu"}:
        config.update(
            {
                "changing_resolution": True,
                "resolution_rate": [368 / 720],
                "wan_lowres_latent_size": [46, 80],
                "changing_resolution_steps": [case.handoff_step],
            }
        )
    if case.method in {"talh", "endpoint"}:
        config.update(
            {
                "wan_clean_resizer_repo": str(REPO_ROOT),
                "wan_clean_resizer_ckpt": str(Path(args.stage2_checkpoint).resolve()),
                "wan_clean_resizer_train_config": str(Path(args.stage2_train_config).resolve()),
                "wan_clean_resizer_model_class": "stage2",
                "wan_clean_resizer_use_ema": args.stage2_use_ema,
            }
        )
    if case.method == "endpoint":
        config["wan_final_refine_steps"] = case.refinement_steps
        config["wan_final_refine_shift_increment"] = args.final_refine_shift_increment
    if case.method == "ralu":
        config.update(
            {
                "wan_ralu_stage_steps": list(getattr(args, "ralu_stage_steps", [5, 6, 7])),
                "wan_ralu_end_times": list(getattr(args, "ralu_end_times", [0.30, 0.45, 1.0])),
                "wan_ralu_stage_shifts": list(
                    getattr(args, "ralu_stage_shifts", [10.0, 8.8787, 5.3374])
                ),
                "wan_ralu_z": float(getattr(args, "ralu_z", 100.0)),
                "wan_ralu_covariance_c": 1.0 / float(getattr(args, "ralu_z", 100.0)) ** 2,
                "wan_ralu_up_ratio": float(getattr(args, "ralu_up_ratio", 0.30)),
                "wan_ralu_low_latent_size": [46, 80],
                "wan_ralu_coarse_token_grid": [23, 40],
                "wan_ralu_aligned_latent_size": [92, 160],
                "wan_ralu_output_latent_size": [90, 156],
                "wan_ralu_canny_low": int(getattr(args, "ralu_canny_low", 100)),
                "wan_ralu_canny_high": int(getattr(args, "ralu_canny_high", 200)),
                "wan_ralu_edge_temporal_quantile": float(
                    getattr(args, "ralu_edge_temporal_quantile", 0.75)
                ),
                "wan_ralu_adaptation": "full_three_stage_region_adaptive_ntdm",
                "wan_ralu_geometry": (
                    "aligned_368x640_to_736x1280_then_patch_crop_720x1248_at_handoff2"
                ),
                "wan_ralu_ntdm_source": "official_objective_hori8_bounded_quality_5_6_7",
                "wan_ralu_patch_domain": "wan_packed_1x2x2_raw_latent_tokens",
                "wan_ralu_mixed_position_ids": "official_coarse_integer_children_half_offset",
                "wan_ralu_transition_noise": (
                    "official_unit_for_unchanged_and_I_minus_c11T_for_expanded"
                ),
            }
        )
    if case.method == "talh":
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
    case_names = {case.name for case in cases}
    artifacts: dict[str, Any] = {}
    if any(case.method in {"talh", "endpoint"} for case in cases):
        artifacts["stage2"] = artifact_fingerprint(Path(args.stage2_checkpoint))
    if "talh40" in case_names:
        artifacts["lora40"] = artifact_fingerprint(Path(args.lora40_checkpoint))
    if "talh45" in case_names:
        artifacts["lora45"] = artifact_fingerprint(Path(args.lora45_checkpoint))
    payload = {
        "schema_version": 2,
        "family": "wan50_quality_efficiency",
        "seed_base": args.seed,
        "prompt_offset": args.prompt_offset,
        "prompts": prompts,
        "cases": [asdict(case) for case in cases],
        "analysis_pairs": build_analysis_pairs(cases),
        "artifacts": artifacts,
        "settings": {
            "model_root": str(Path(args.model_root).resolve()),
            "prompts_file": str(Path(args.prompts).resolve()),
            "num_frames": args.num_frames,
            "guide_scale": args.guide_scale,
            "step40_strength": args.step40_strength,
            "step45_strength": args.step45_strength,
            "final_refine_shift_increment": args.final_refine_shift_increment,
            "stage2_use_ema": args.stage2_use_ema,
            "lightx2v_handoff_steps": list(getattr(args, "lightx2v_handoff_steps", [40, 45, 48])),
            "endpoint_refinement_steps": list(getattr(args, "endpoint_refinement_steps", [0, 1, 2, 5])),
            "ralu_stage_steps": list(getattr(args, "ralu_stage_steps", [5, 6, 7])),
            "ralu_end_times": list(getattr(args, "ralu_end_times", [0.30, 0.45, 1.0])),
            "ralu_stage_shifts": list(getattr(args, "ralu_stage_shifts", [10.0, 8.8787, 5.3374])),
            "ralu_z": getattr(args, "ralu_z", 100.0),
            "ralu_covariance_c": 1.0 / float(getattr(args, "ralu_z", 100.0)) ** 2,
            "ralu_up_ratio": getattr(args, "ralu_up_ratio", 0.30),
            "ralu_scope": "full three-stage region-adaptive RALU Quality adaptation",
        },
    }
    (root / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_analysis_pairs(cases: list[Case]) -> list[dict[str, str]]:
    by_name = {case.name: case for case in cases}
    pairs: list[dict[str, str]] = []
    if "full_hr50" in by_name:
        pairs.extend(
            {
                "comparison": f"full_hr50_vs_{case.name}",
                "left_case": "full_hr50",
                "right_case": case.name,
            }
            for case in cases
            if case.name != "full_hr50"
        )
    for step in sorted(
        {
            int(case.handoff_step)
            for case in cases
            if case.handoff_step is not None and case.handoff_step < 50
        }
    ):
        candidates = [f"lightx2v_cr{step}", f"talh{step}"]
        available = [name for name in candidates if name in by_name]
        for left, right in zip(available, available[1:]):
            pairs.append(
                {
                    "comparison": f"{left}_vs_{right}",
                    "left_case": left,
                    "right_case": right,
                }
            )
    if "ralu_quality" in by_name:
        for peer in ("lightx2v_cr45", "talh45"):
            if peer in by_name:
                pairs.append(
                    {
                        "comparison": f"ralu_quality_vs_{peer}",
                        "left_case": "ralu_quality",
                        "right_case": peer,
                    }
                )
    endpoint_cases = sorted(
        (case for case in cases if case.method == "endpoint"),
        key=lambda case: int(case.refinement_steps or 0),
    )
    for left, right in zip(endpoint_cases, endpoint_cases[1:]):
        pairs.append(
            {
                "comparison": f"{left.name}_vs_{right.name}",
                "left_case": left.name,
                "right_case": right.name,
            }
        )
    return pairs


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
                    "method": case.method,
                    "lr_evaluations": case.lr_evaluations,
                    "mixed_evaluations": case.mixed_evaluations,
                    "hr_evaluations": case.hr_evaluations,
                    "total_evaluations": case.total_evaluations,
                    "handoff_step": case.handoff_step,
                    "refinement_steps": case.refinement_steps,
                    "reschedule_mode": case.reschedule_mode,
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
    environment.setdefault("LIGHTX2V_REPO", "../LightX2V")
    environment.setdefault("DIFFSYNTH_REPO", "../DiffSynth-Studio")
    roots = [environment["LIGHTX2V_REPO"], environment["DIFFSYNTH_REPO"], str(REPO_ROOT)]
    if environment.get("PYTHONPATH"):
        roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def validate_inputs(args: argparse.Namespace, cases: list[Case]) -> None:
    required = [Path(args.prompts), Path(args.model_root)]
    if any(case.method in {"talh", "endpoint"} for case in cases):
        required.extend([Path(args.stage2_checkpoint), Path(args.stage2_train_config)])
    case_names = {case.name for case in cases}
    if "talh40" in case_names:
        required.append(Path(args.lora40_checkpoint))
    if "talh45" in case_names:
        required.append(Path(args.lora45_checkpoint))
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
    parser = argparse.ArgumentParser(
        description="Run the unified Wan Pareto suite with LightX2V, Endpoint, TrajScale, and full RALU Quality."
    )
    parser.add_argument("mode", choices=["check", "run", "benchmark-spec"])
    parser.add_argument(
        "--out-root", default=str(REPO_ROOT / "outputs/aaai27_experiments/quality_efficiency_final")
    )
    parser.add_argument(
        "--prompts",
        default=str(REPO_ROOT / "changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt"),
    )
    parser.add_argument("--model-root", default="Wan-AI/Wan2.1-T2V-1.3B")
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
    parser.add_argument("--step40-strength", type=float, default=0.75)
    parser.add_argument("--step45-strength", type=float, default=0.75)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["native", "lightx2v", "talh", "endpoint", "ralu"],
        default=["native", "lightx2v", "talh", "endpoint", "ralu"],
        help="Method families to include in this run.",
    )
    parser.add_argument("--lightx2v-handoff-steps", nargs="+", type=int, default=[40, 45, 48])
    parser.add_argument("--endpoint-refinement-steps", nargs="+", type=int, default=[0, 1, 2, 5])
    parser.add_argument("--ralu-stage-steps", nargs=3, type=int, default=[5, 6, 7])
    parser.add_argument("--ralu-end-times", nargs=3, type=float, default=[0.30, 0.45, 1.0])
    parser.add_argument("--ralu-stage-shifts", nargs=3, type=float, default=[10.0, 8.8787, 5.3374])
    parser.add_argument("--ralu-z", type=float, default=100.0)
    parser.add_argument("--ralu-up-ratio", type=float, default=0.30)
    parser.add_argument("--ralu-canny-low", type=int, default=100)
    parser.add_argument("--ralu-canny-high", type=int, default=200)
    parser.add_argument("--ralu-edge-temporal-quantile", type=float, default=0.75)
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
    args = parser.parse_args()
    if args.ralu_stage_steps != [5, 6, 7]:
        parser.error("this entrypoint is fixed to the RALU Quality stage steps 5 6 7")
    if args.ralu_end_times != [0.3, 0.45, 1.0]:
        parser.error("RALU Quality is fixed to --ralu-end-times 0.3 0.45 1.0")
    expected_shifts = [10.0, 8.8787, 5.3374]
    if any(abs(value - expected) > 1e-6 for value, expected in zip(args.ralu_stage_shifts, expected_shifts)):
        parser.error("RALU Quality is fixed to --ralu-stage-shifts 10.0 8.8787 5.3374")
    if abs(args.ralu_z - 100.0) > 1e-12:
        parser.error("RALU Quality is fixed to --ralu-z 100")
    if abs(args.ralu_up_ratio - 0.3) > 1e-12:
        parser.error("RALU Quality is fixed to --ralu-up-ratio 0.3")
    if abs(args.ralu_edge_temporal_quantile - 0.75) > 1e-12:
        parser.error("RALU Quality is fixed to --ralu-edge-temporal-quantile 0.75")
    if (args.ralu_canny_low, args.ralu_canny_high) != (100, 200):
        parser.error("RALU Quality is fixed to Canny thresholds 100 200")
    return args


if __name__ == "__main__":
    main()
