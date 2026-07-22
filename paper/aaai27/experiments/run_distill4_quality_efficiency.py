from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
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
    refinement_steps: int | None = None
    resizer: str | None = None
    alignment: str = "base"


@dataclass(frozen=True)
class GPUAssignment:
    gpu: int
    cases: tuple[Case, ...]
    estimated_cost: float


def build_cases(args: argparse.Namespace) -> list[Case]:
    groups = set(getattr(args, "case_groups", ["native", "handoff", "endpoint"]))
    refinements = tuple(
        int(v) for v in getattr(args, "endpoint_refinement_steps", [0, 1, 2, 4])
    )
    resizers = tuple(getattr(args, "endpoint_resizers", ["stage2", "interp", "rgb"]))
    invalid_refinements = [value for value in refinements if value < 0 or value > 4]
    if invalid_refinements:
        raise SystemExit(
            f"Invalid endpoint refinement step(s) {invalid_refinements}; expected [0, 4]"
        )

    cases: list[Case] = []
    if "native" in groups:
        cases.append(Case("native_hr4", "native", "wan2.1_distill", 0, 4, 4))
    if "handoff" in groups:
        # Interp@1 is intentionally excluded: the Distill4 main experiment
        # concentrates on the useful 2+2 and 3+1 operating points.
        cases.extend(
            [
                Case(
                    "interp2",
                    "interp",
                    "wan2.1_distill_interp_bridge",
                    2,
                    2,
                    4,
                    2,
                    resizer="interp",
                ),
                Case(
                    "interp3",
                    "interp",
                    "wan2.1_distill_interp_bridge",
                    3,
                    1,
                    4,
                    3,
                    resizer="interp",
                ),
                Case(
                    "taa_interp3",
                    "taa_interp",
                    "wan2.1_distill_last_step_lora_interp_bridge",
                    3,
                    1,
                    4,
                    3,
                    resizer="interp",
                    alignment="taa",
                ),
                Case(
                    "cll3",
                    "cll",
                    "wan2.1_distill_clean_resizer_bridge",
                    3,
                    1,
                    4,
                    3,
                    resizer="stage2",
                ),
                Case(
                    "talh3",
                    "talh",
                    "wan2.1_distill_last_step_lora_clean_resizer_bridge",
                    3,
                    1,
                    4,
                    3,
                    resizer="stage2",
                    alignment="taa",
                ),
            ]
        )
    if "endpoint" in groups:
        model_classes = {
            "stage2": "wan2.1_distill_full_lr_stage2_k_hr",
            "interp": "wan2.1_distill_full_lr_interp_k_hr",
            "rgb": "wan2.1_distill_full_lr_rgb_k_hr",
        }
        for resizer in resizers:
            if resizer not in model_classes:
                raise SystemExit(f"Unsupported endpoint resizer: {resizer}")
            cases.extend(
                Case(
                    f"endpoint_{resizer}_{steps}hr",
                    f"endpoint_{resizer}",
                    model_classes[resizer],
                    4,
                    steps,
                    4 + steps,
                    handoff_step=4,
                    refinement_steps=steps,
                    resizer=resizer,
                )
                for steps in refinements
            )
    names = [case.name for case in cases]
    if not names:
        raise SystemExit("No Distill4 quality-efficiency cases selected")
    if len(names) != len(set(names)):
        raise SystemExit("Distill4 case names are not unique")
    return cases


def base_config(args: argparse.Namespace, case: Case) -> dict[str, Any]:
    return {
        "infer_steps": 4,
        "target_video_length": args.num_frames,
        "text_len": 512,
        "target_height": 720,
        "target_width": 1248,
        "self_attn_1_type": "flash_attn3",
        "cross_attn_1_type": "flash_attn3",
        "cross_attn_2_type": "flash_attn3",
        "sample_guide_scale": args.guide_scale,
        "sample_shift": 5,
        "enable_cfg": False,
        "cpu_offload": False,
        "feature_caching": "NoCaching",
        "denoising_step_list": [1000, 750, 500, 250],
        "dit_original_ckpt": str(Path(args.dit_ckpt).resolve()),
        "compare_name": case.name,
    }


def write_config(path: Path, args: argparse.Namespace, case: Case) -> None:
    config = base_config(args, case)
    if case.method != "native":
        config.update(
            {
                "changing_resolution": True,
                "resolution_rate": [368 / 720],
                "wan_lowres_latent_size": [46, 80],
                "changing_resolution_steps": [case.handoff_step],
                "wan_distill_bridge_renoise_mode": args.renoise_mode,
            }
        )
    if case.resizer == "stage2":
        config.update(
            {
                "wan_clean_resizer_repo": str(REPO_ROOT),
                "wan_clean_resizer_ckpt": str(Path(args.stage2_checkpoint).resolve()),
                "wan_clean_resizer_train_config": str(
                    Path(args.stage2_train_config).resolve()
                ),
                "wan_clean_resizer_model_class": "stage2",
                "wan_clean_resizer_use_ema": args.stage2_use_ema,
            }
        )
    if case.resizer == "rgb":
        config.update(
            {
                "wan_rgb_sr_backend": args.rgb_sr_backend,
                "wan_rgb_sr_scale": 2.0,
                "wan_rgb_sr_tile": args.rgb_sr_tile,
                "wan_rgb_sr_tile_pad": args.rgb_sr_tile_pad,
                "wan_rgb_sr_pre_pad": args.rgb_sr_pre_pad,
                "wan_rgb_sr_half": not args.rgb_sr_fp32,
                "wan_rgb_sr_gpu_id": 0,
            }
        )
        if args.rgb_sr_backend == "realesrgan":
            config["wan_rgb_sr_checkpoint"] = str(
                Path(args.realesrgan_x2_checkpoint).resolve()
            )
    if case.alignment == "taa":
        config.update(
            {
                "lora_dynamic_apply": True,
                "lora_active_steps": [3],
                "lora_configs": [
                    {
                        "name": "wan2.1",
                        "path": str(Path(args.lora_checkpoint).resolve()),
                        "strength": args.lora_strength,
                    }
                ],
            }
        )
    if case.refinement_steps is not None:
        config["wan_final_refine_steps"] = case.refinement_steps
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def analysis_pairs(cases: list[Case]) -> list[dict[str, str]]:
    by_name = {case.name: case for case in cases}
    pairs: list[dict[str, str]] = []
    if "native_hr4" in by_name:
        pairs.extend(
            {
                "comparison": f"native_hr4_vs_{case.name}",
                "left_case": "native_hr4",
                "right_case": case.name,
            }
            for case in cases
            if case.name != "native_hr4"
        )
    for left, right in (
        ("interp3", "taa_interp3"),
        ("interp3", "cll3"),
        ("cll3", "talh3"),
        ("interp3", "talh3"),
        ("cll3", "endpoint_stage2_1hr"),
        ("talh3", "endpoint_stage2_1hr"),
        ("talh3", "endpoint_rgb_1hr"),
    ):
        if left in by_name and right in by_name:
            pairs.append(
                {
                    "comparison": f"{left}_vs_{right}",
                    "left_case": left,
                    "right_case": right,
                }
            )
    refinement_steps = sorted(
        {
            int(case.refinement_steps)
            for case in cases
            if case.refinement_steps is not None
        }
    )
    for steps in refinement_steps:
        names = [
            f"endpoint_{resizer}_{steps}hr" for resizer in ("interp", "stage2", "rgb")
        ]
        available = [name for name in names if name in by_name]
        for left, right in zip(available, available[1:]):
            pairs.append(
                {
                    "comparison": f"{left}_vs_{right}",
                    "left_case": left,
                    "right_case": right,
                }
            )
    for resizer in ("stage2", "interp", "rgb"):
        names = [f"endpoint_{resizer}_{steps}hr" for steps in refinement_steps]
        available = [name for name in names if name in by_name]
        for left, right in zip(available, available[1:]):
            pairs.append(
                {
                    "comparison": f"{left}_vs_{right}",
                    "left_case": left,
                    "right_case": right,
                }
            )
    return pairs


def write_manifest(
    root: Path, args: argparse.Namespace, prompts: list[str], cases: list[Case]
) -> None:
    payload = {
        "schema_version": 1,
        "family": "distill4_quality_efficiency",
        "seed_base": args.seed,
        "prompt_offset": args.prompt_offset,
        "prompts": prompts,
        "cases": [asdict(case) for case in cases],
        "analysis_pairs": analysis_pairs(cases),
        "settings": {
            "model_root": str(Path(args.model_root).resolve()),
            "dit_ckpt": str(Path(args.dit_ckpt).resolve()),
            "prompts_file": str(Path(args.prompts).resolve()),
            "num_frames": args.num_frames,
            "guide_scale": args.guide_scale,
            "lora_strength": args.lora_strength,
            "renoise_mode": args.renoise_mode,
            "stage2_use_ema": args.stage2_use_ema,
            "endpoint_refinement_steps": args.endpoint_refinement_steps,
            "endpoint_resizers": args.endpoint_resizers,
            "rgb_sr_backend": args.rgb_sr_backend,
            "rgb_sr_protocol": "Wan VAE decode -> Real-ESRGAN x2 -> center crop -> same Wan VAE encode",
            "generation_gpus": args.gpus,
            "generation_parallelism": "one independent case process per physical GPU",
        },
        "artifacts": artifact_manifest(
            args, cases, root / "artifact_fingerprints.json"
        ),
    }
    (root / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def artifact_manifest(
    args: argparse.Namespace,
    cases: list[Case],
    cache_path: Path | None = None,
) -> dict[str, Any]:
    cache = load_artifact_cache(cache_path)
    artifact_paths = {"distill_dit": Path(args.dit_ckpt)}
    if any(case.resizer == "stage2" for case in cases):
        artifact_paths["stage2"] = Path(args.stage2_checkpoint)
    if any(case.alignment == "taa" for case in cases):
        artifact_paths["lora3"] = Path(args.lora_checkpoint)
    if (
        any(case.resizer == "rgb" for case in cases)
        and args.rgb_sr_backend == "realesrgan"
    ):
        artifact_paths["realesrgan_x2"] = Path(args.realesrgan_x2_checkpoint)
    artifacts = {}
    for name, path in artifact_paths.items():
        artifacts[name] = artifact_fingerprint(path, cache=cache)
        if cache_path is not None:
            write_artifact_cache(cache_path, cache)
    return artifacts


def run_case(
    root: Path, args: argparse.Namespace, case: Case, *, gpu: int
) -> list[str]:
    batch = (
        REPO_ROOT
        / "changing_resolution_distill/scripts/bridge/run_lightx2v_distill_bridge_batch_infer.py"
    )
    command = [
        args.python,
        str(batch),
        "--seed",
        str(args.seed),
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
        str(root / "videos" / case.name),
        "--name_prefix",
        case.name,
        "--prompt-offset",
        str(args.prompt_offset),
        "--limit",
        str(args.limit),
        "--target_video_length",
        str(args.num_frames),
        "--negative_prompt",
        args.negative_prompt,
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    print(
        f"[gpu:{gpu}] {case.name}: {args.limit} prompt(s), one model load",
        flush=True,
    )
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=inference_environment(gpu))
    return command


def estimated_case_cost(case: Case) -> float:
    """Approximate wall cost for deterministic longest-first GPU packing."""

    cost = float(case.lr_evaluations + 4 * case.hr_evaluations)
    if case.resizer == "stage2":
        cost += 1.0
    elif case.resizer == "rgb":
        cost += 4.0
    return cost


def assign_cases_to_gpus(cases: list[Case], gpus: list[int]) -> list[GPUAssignment]:
    if not gpus:
        raise ValueError("At least one GPU is required")
    if len(gpus) != len(set(gpus)) or any(gpu < 0 for gpu in gpus):
        raise ValueError(f"GPU ids must be unique non-negative integers, got {gpus}")

    buckets: list[list[Case]] = [[] for _ in gpus]
    loads = [0.0 for _ in gpus]
    ordered = sorted(cases, key=lambda case: (-estimated_case_cost(case), case.name))
    for case in ordered:
        bucket_index = min(range(len(gpus)), key=lambda index: (loads[index], index))
        buckets[bucket_index].append(case)
        loads[bucket_index] += estimated_case_cost(case)
    return [
        GPUAssignment(gpu=gpu, cases=tuple(bucket), estimated_cost=loads[index])
        for index, (gpu, bucket) in enumerate(zip(gpus, buckets))
        if bucket
    ]


def run_assigned_cases(
    root: Path,
    args: argparse.Namespace,
    assignment: GPUAssignment,
) -> list[list[str]]:
    commands = []
    for case in assignment.cases:
        commands.append(run_case(root, args, case, gpu=assignment.gpu))
    return commands


def run_cases_parallel(
    root: Path, args: argparse.Namespace, cases: list[Case]
) -> list[list[str]]:
    assignments = assign_cases_to_gpus(cases, args.gpus)
    schedule = {
        "schema_version": 1,
        "strategy": "case_parallel_longest_estimated_cost_first",
        "gpu_assignments": [
            {
                "gpu": assignment.gpu,
                "estimated_cost": assignment.estimated_cost,
                "cases": [case.name for case in assignment.cases],
            }
            for assignment in assignments
        ],
    }
    schedule_path = root / "generation_schedule.json"
    schedule_path.write_text(
        json.dumps(schedule, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for assignment in assignments:
        names = ", ".join(case.name for case in assignment.cases)
        print(
            f"[schedule] GPU {assignment.gpu}: cost={assignment.estimated_cost:.1f}; {names}",
            flush=True,
        )

    commands: list[list[str]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(assignments), thread_name_prefix="distill4-gpu"
    ) as executor:
        future_to_assignment = {
            executor.submit(run_assigned_cases, root, args, assignment): assignment
            for assignment in assignments
        }
        try:
            for future in concurrent.futures.as_completed(future_to_assignment):
                commands.extend(future.result())
        except BaseException:
            for future in future_to_assignment:
                future.cancel()
            raise
    return commands


def write_benchmark_spec(
    root: Path, args: argparse.Namespace, cases: list[Case]
) -> None:
    vbench = root / "metrics/vbench_v1_custom.json"
    if not vbench.is_file():
        raise SystemExit(f"Run VBench before preparing benchmark spec: {vbench}")
    spec_cases = []
    for case in cases:
        components = vbench_case_scores(vbench, case.name)
        spec_cases.append(
            {
                "family": "distill4",
                "name": case.name,
                "measurement": "warm_model_single_video_end_to_end",
                "quality_metric": "vbench_custom_quality5_mean",
                "quality_value": sum(components.values()) / len(components),
                "quality_components": components,
                "vbench_source": str(vbench),
                "protocol": asdict(case),
            }
        )
    payload = {
        "schema_version": 1,
        "protocol": {
            "measurement": "warm_model_single_video_end_to_end",
            "quality_dimensions": QUALITY_DIMENSIONS,
            "dynamic_degree_reported_separately": True,
            "same_model_prompt_seed_frames_resolution": True,
        },
        "cases": spec_cases,
    }
    output = (
        Path(args.benchmark_output)
        if args.benchmark_output
        else root / "benchmark_spec.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Distill4 benchmark spec: {output}")


def validate_inputs(args: argparse.Namespace, cases: list[Case]) -> None:
    required = [Path(args.prompts), Path(args.model_root), Path(args.dit_ckpt)]
    if any(case.resizer == "stage2" for case in cases):
        required.extend([Path(args.stage2_checkpoint), Path(args.stage2_train_config)])
    if any(case.alignment == "taa" for case in cases):
        required.append(Path(args.lora_checkpoint))
    if (
        any(case.resizer == "rgb" for case in cases)
        and args.rgb_sr_backend == "realesrgan"
    ):
        required.append(Path(args.realesrgan_x2_checkpoint))
    missing = sorted({str(path) for path in required if not path.exists()})
    if missing:
        raise SystemExit("Missing Distill4 suite inputs:\n  " + "\n  ".join(missing))


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


def artifact_fingerprint(
    path: Path, *, cache: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    cache_key = str(resolved)
    cached = (cache or {}).get(cache_key)
    if (
        cached
        and int(cached.get("size_bytes", -1)) == stat.st_size
        and int(cached.get("mtime_ns", -1)) == stat.st_mtime_ns
        and len(str(cached.get("sha256", ""))) == 64
    ):
        print(
            f"[fingerprint:cached] {resolved} ({stat.st_size / (1024**3):.2f} GiB)",
            flush=True,
        )
        return dict(cached)

    print(
        f"[fingerprint:start] {resolved} ({stat.st_size / (1024**3):.2f} GiB)",
        flush=True,
    )
    digest = hashlib.sha256()
    processed = 0
    report_interval = max(1024**3, stat.st_size // 20)
    next_report = report_interval
    started = time.perf_counter()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            processed += len(chunk)
            if processed >= next_report and processed < stat.st_size:
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(
                    f"[fingerprint:progress] {resolved.name}: "
                    f"{100.0 * processed / stat.st_size:.1f}% "
                    f"({processed / (1024**3):.1f} GiB, "
                    f"{processed / (1024**2) / elapsed:.0f} MiB/s)",
                    flush=True,
                )
                next_report += report_interval
    fingerprint = {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }
    if cache is not None:
        cache[cache_key] = dict(fingerprint)
    print(
        f"[fingerprint:done] {resolved.name}: {fingerprint['sha256'][:12]}... "
        f"in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return fingerprint


def load_artifact_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("entries", {})
        if not isinstance(entries, dict):
            raise ValueError("entries must be an object")
        return entries
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[fingerprint:cache-warning] Ignore invalid {path}: {exc}", flush=True)
        return {}


def write_artifact_cache(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps({"schema_version": 1, "entries": entries}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def inference_environment(gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("LIGHTX2V_REPO", "/mnt/afs_2/houze/LightX2V")
    environment.setdefault("DIFFSYNTH_REPO", "/mnt/afs_2/houze/DiffSynth-Studio")
    roots = [
        environment["LIGHTX2V_REPO"],
        environment["DIFFSYNTH_REPO"],
        str(REPO_ROOT),
    ]
    if environment.get("PYTHONPATH"):
        roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Distill4 quality-efficiency suite with 2/3-step handoffs and 4x3 endpoint lifting."
    )
    parser.add_argument("mode", choices=["check", "run", "benchmark-spec"])
    parser.add_argument(
        "--out-root",
        default=str(
            REPO_ROOT / "outputs/aaai27_experiments/quality_efficiency_distill4"
        ),
    )
    parser.add_argument(
        "--prompts",
        default=str(
            REPO_ROOT
            / "changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt"
        ),
    )
    parser.add_argument(
        "--model-root",
        default="/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill",
    )
    parser.add_argument(
        "--dit-ckpt",
        default="/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt",
    )
    parser.add_argument(
        "--stage2-checkpoint",
        default=str(
            REPO_ROOT
            / "outputs/changing_resolution_distill_clean_368x640_720x1248_stage2_14b_cfgdistill_5k_lmdb/latest.pt"
        ),
    )
    parser.add_argument(
        "--stage2-train-config",
        default=str(
            REPO_ROOT
            / "changing_resolution_distill/configs/train_clean_368x640_to_720x1248_lmdb_stage2_distill.yaml"
        ),
    )
    parser.add_argument(
        "--lora-checkpoint",
        default=str(
            REPO_ROOT
            / "outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3/step_0010000.safetensors"
        ),
    )
    parser.add_argument("--realesrgan-x2-checkpoint", default="")
    parser.add_argument(
        "--rgb-sr-backend", choices=["realesrgan", "bicubic"], default="realesrgan"
    )
    parser.add_argument("--rgb-sr-tile", type=int, default=0)
    parser.add_argument("--rgb-sr-tile-pad", type=int, default=10)
    parser.add_argument("--rgb-sr-pre-pad", type=int, default=0)
    parser.add_argument("--rgb-sr-fp32", action="store_true")
    parser.add_argument(
        "--case-groups",
        nargs="+",
        choices=["native", "handoff", "endpoint"],
        default=["native", "handoff", "endpoint"],
    )
    parser.add_argument(
        "--endpoint-refinement-steps", nargs="+", type=int, default=[0, 1, 2, 4]
    )
    parser.add_argument(
        "--endpoint-resizers",
        nargs="+",
        choices=["stage2", "interp", "rgb"],
        default=["stage2", "interp", "rgb"],
    )
    parser.add_argument("--lora-strength", type=float, default=0.75)
    parser.add_argument(
        "--renoise-mode", choices=["random", "resize_flow"], default="random"
    )
    parser.add_argument(
        "--stage2-use-ema", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--guide-scale", type=float, default=6.0)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=9800)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--benchmark-output", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="Physical GPU ids used for case-parallel generation.",
    )
    args = parser.parse_args()
    if args.limit < 1 or args.prompt_offset < 0 or args.num_frames < 1:
        parser.error(
            "--limit and --num-frames must be positive; --prompt-offset must be non-negative"
        )
    if args.lora_strength < 0:
        parser.error("--lora-strength must be non-negative")
    if args.rgb_sr_tile < 0 or args.rgb_sr_tile_pad < 0 or args.rgb_sr_pre_pad < 0:
        parser.error("RGB SR tile settings must be non-negative")
    if len(args.gpus) != len(set(args.gpus)) or any(gpu < 0 for gpu in args.gpus):
        parser.error("--gpus must contain unique non-negative GPU ids")
    return args


def main() -> None:
    args = parse_args()
    root = Path(args.out_root).resolve()
    cases = build_cases(args)
    validate_inputs(args, cases)
    prompts = load_prompts(Path(args.prompts), args.prompt_offset, args.limit)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "videos").mkdir(parents=True, exist_ok=True)
    for case in cases:
        write_config(root / "configs" / f"{case.name}.json", args, case)
    write_manifest(root, args, prompts, cases)

    if args.mode == "check":
        print(f"Validated Distill4 suite: {len(cases)} cases; root={root}")
        return
    if args.mode == "benchmark-spec":
        write_benchmark_spec(root, args, cases)
        return
    run_cases_parallel(root, args, cases)


if __name__ == "__main__":
    main()
