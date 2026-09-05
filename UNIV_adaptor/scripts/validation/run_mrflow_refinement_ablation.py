"""Run LR50 once, then compare explicit HR sigma and step-count combinations."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.data_protocol import canonical_sha256, sha256_file, write_json_atomic
from UNIV_adaptor.hr_refinement import direct_hr_sigmas, quantize_float32_timesteps
from UNIV_adaptor.schedule import action_from_config, resolve_schedule

DEFAULT_SIGMAS = (0.12, 0.20, 0.30)
DEFAULT_HR_STEPS = (1, 2, 4)
CONTROL_ID = "S0000_HR00"
DEFAULT_PROMPT = (
    "A cinematic tracking shot of a red fox walking steadily through a snowy forest. "
    "Its paws leave footprints in fresh snow, its detailed fur moves gently, "
    "and snowflakes drift in front of dark pine branches. The camera follows smoothly."
)


def parse_sigmas(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sigmas must be comma-separated floats") from exc
    if not result or any(not math.isfinite(v) or not 0 < v < 1 for v in result):
        raise argparse.ArgumentTypeError("sigmas must be finite values in (0, 1)")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("sigmas must be unique")
    return result


def parse_hr_steps(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hr_steps must be comma-separated integers") from exc
    if not result or any(v < 1 for v in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("hr_steps must be unique positive integers")
    return result


def case_id(sigma: float, steps: int) -> str:
    scaled = round(float(sigma) * 1000)
    if abs(scaled / 1000 - float(sigma)) > 1e-9:
        raise ValueError("sigma case IDs support at most three decimal places")
    return f"S{scaled:04d}_HR{steps:02d}"


def build_cases(sigmas, hr_steps, out: Path):
    cases = [{
        "id": CONTROL_ID,
        "refine_sigma": 0.0,
        "hr_steps": 0,
        "planned_sigmas": [0.0],
        "planned_model_timesteps": [],
        "video_path": str(out / f"{CONTROL_ID}.mp4"),
    }]
    for sigma in sigmas:
        for steps in hr_steps:
            identifier = case_id(sigma, steps)
            planned_sigmas = list(direct_hr_sigmas(
                start_sigma=float(sigma), hr_steps=int(steps)
            ))
            planned_timesteps = list(quantize_float32_timesteps(
                planned_sigmas[:-1], num_train_timesteps=1000
            ))
            if planned_timesteps[-1] <= 0 or any(
                left <= right for left, right in zip(planned_timesteps, planned_timesteps[1:])
            ):
                raise ValueError(
                    f"direct HR grid collapses after Wan timestep quantization: {identifier}"
                )
            cases.append({
                "id": identifier,
                "refine_sigma": float(sigma),
                "hr_steps": int(steps),
                "planned_sigmas": planned_sigmas,
                "planned_model_timesteps": planned_timesteps,
                "video_path": str(out / f"{identifier}.mp4"),
            })
    if len({row["id"] for row in cases}) != len(cases):
        raise ValueError("sigma and HR-step combinations produced duplicate case IDs")
    return cases


def build_plan(args):
    config = json.loads(Path(args.template_config).read_text(encoding="utf-8"))
    if int(config["infer_steps"]) != 50:
        raise ValueError("this comparison requires the 50-step Wan reference")
    action = action_from_config(config)
    shape = (
        16,
        (int(config["target_video_length"]) - 1) // 4 + 1,
        int(config["target_height"]) // 8,
        int(config["target_width"]) // 8,
    )
    schedule = resolve_schedule(action, reference_nfe=50, target_latent_shape=shape)
    if schedule.switch_step != 50 or schedule.hr_compute_steps:
        raise ValueError("use switch_ratio=1.0 so LR reaches the clean endpoint")
    if tuple(schedule.lr_compute_steps) != tuple(range(50)):
        raise ValueError("use lr_nfe_ratio=1.0 so all 50 LR positions run")
    if config.get("univ_transition_baseline") != "dvg_latent_anchor":
        raise ValueError("this comparison fixes transition=dvg_latent_anchor")
    if config.get("feature_caching") != "NoCaching":
        raise ValueError("this comparison requires feature_caching=NoCaching")
    if config.get("cpu_offload") or config.get("compile"):
        raise ValueError("this comparison requires resident, uncompiled weights")
    if not args.prompt.strip():
        raise ValueError("prompt must not be empty")

    out = Path(args.out_dir).resolve()
    config["univ_mrflow_boundary_path"] = str(out / "shared_clean_transition.pt")
    plan = {
        "schema": "univ_mrflow_refinement_plan_v1",
        "method": "MrFlow-style direct-sigma refinement over dvg_latent_anchor",
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "model_root": str(Path(args.model_root).resolve()),
        "config": config,
        "reference_schedule": schedule.as_dict(),
        "sigmas": list(args.sigmas),
        "hr_steps": list(args.hr_steps),
        "cases": build_cases(args.sigmas, args.hr_steps, out),
        "comparison": (
            "one completed LR50 endpoint, one clean HR transition and one HR noise tensor; "
            "branches vary only direct start sigma and HR solver evaluations"
        ),
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def prepare(args):
    plan = build_plan(args)
    out = Path(args.out_dir).resolve()
    path = out / "comparison_plan.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError(f"different comparison already planned under {out}; use a new OUT_DIR")
    else:
        write_json_atomic(path, plan)
    config_path = out / "resolved_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != plan["config"]:
            raise RuntimeError(f"resolved config changed: {config_path}")
    else:
        write_json_atomic(config_path, plan["config"])
    print(f"Prompt: {plan['prompt']}\nSeed: {args.seed}\nOutput: {out}")
    for case in plan["cases"]:
        print(f"{case['id']}: sigma grid = " + ", ".join(
            f"{value:.6f}" for value in case["planned_sigmas"]
        ))
    return plan, config_path


def runtime_args(args, config_path):
    return SimpleNamespace(
        seed=args.seed,
        model_cls="wan2.1_univ_mrflow_ablation",
        task="t2v",
        support_tasks=[],
        model_path=args.model_root,
        config_json=str(config_path),
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        use_prompt_enhancer=False,
        return_result_tensor=False,
        target_shape=[],
        target_video_length=81,
        save_result_path=None,
    )


def run(args, plan, config_path):
    lightx2v = Path(args.lightx2v_repo).resolve()
    if not lightx2v.is_dir():
        raise FileNotFoundError(f"LightX2V repository not found: {lightx2v}")
    sys.path.insert(0, str(lightx2v))
    os.environ.setdefault("DTYPE", "BF16")
    import torch
    from lightx2v.common import ops  # noqa: F401 -- operator registrations
    from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
    from lightx2v.utils.set_config import set_config
    from lightx2v.utils.utils import seed_all, validate_config_paths
    from UNIV_adaptor.model_contract import validate_wan21_t2v_model_root
    from UNIV_adaptor.mrflow_ablation_runner import WanMrFlowRefinementAblationRunner

    validate_wan21_t2v_model_root(args.model_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    execution_args = runtime_args(args, config_path)
    config = set_config(execution_args)
    validate_config_paths(config)
    if args.mode == "check":
        print("Model contract, runtime imports and CUDA check passed.")
        return

    out = Path(args.out_dir).resolve()
    outputs = [Path(case["video_path"]) for case in plan["cases"]]
    protected = [
        *outputs,
        Path(config["univ_mrflow_boundary_path"]),
        out / "comparison_summary.json",
    ]
    if any(path.exists() for path in protected):
        raise FileExistsError("comparison artifacts already exist; use a new OUT_DIR")

    torch.set_grad_enabled(False)
    seed_all(args.seed)
    runner = WanMrFlowRefinementAblationRunner(config)
    runner.init_modules()
    summary = {
        "schema": "univ_mrflow_refinement_results_v1",
        "plan_sha256": plan["plan_sha256"],
        "method": plan["method"],
        "prompt": plan["prompt"],
        "negative_prompt": plan["negative_prompt"],
        "seed": plan["seed"],
        "sigmas": plan["sigmas"],
        "hr_steps": plan["hr_steps"],
        "run_order": [case["id"] for case in plan["cases"]],
        "shared_boundary_path": str(config["univ_mrflow_boundary_path"]),
        "timing_note": (
            "candidate_denoise_seconds adds the one-time LR50 and transition timings to "
            "each branch HR timing; encoding, decoding and checkpoint I/O are excluded"
        ),
        "cases": [],
        "complete": False,
    }
    summary_path = out / "comparison_summary.json"
    for index, case in enumerate(plan["cases"]):
        runner.refine_sigma = case["refine_sigma"]
        runner.hr_steps = case["hr_steps"]
        input_info = init_empty_input_info("t2v", [])
        payload = vars(execution_args).copy()
        payload.update(
            save_result_path=case["video_path"],
            target_video_length=int(config["target_video_length"]),
        )
        update_input_info_from_dict(input_info, payload)
        seed_all(args.seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        runner.run_pipeline(input_info)
        torch.cuda.synchronize()
        pipeline_seconds = time.perf_counter() - started

        video = Path(case["video_path"])
        if not video.is_file() or video.stat().st_size < 1024:
            raise RuntimeError(f"missing or undersized video: {video}")
        runtime = runner.univ_runtime_record
        endpoint = runtime["lr_endpoint"]
        grid = runtime["hr_schedule"]
        shared = runtime["shared_clean_hr"]
        if endpoint.get("steps") != 50 or endpoint.get("sigma") != 0.0 \
                or endpoint.get("final_step_post_completed") is not True:
            raise RuntimeError("branch did not use a completed LR50 solver endpoint")
        if bool(shared.get("reused")) != (index > 0):
            raise RuntimeError("shared clean transition reuse does not match the run order")
        if grid.get("hr_steps") != case["hr_steps"] or len(grid.get("sigmas", [])) != case["hr_steps"] + 1:
            raise RuntimeError(f"wrong direct HR schedule for {case['id']}")
        timing = runtime["timing_seconds"]
        summary["cases"].append({
            "id": case["id"],
            "refine_sigma": case["refine_sigma"],
            "hr_steps": case["hr_steps"],
            "total_nfe": 50 + case["hr_steps"],
            "video_path": str(video),
            "video_sha256": sha256_file(video),
            "clean_lr_sha256": endpoint["clean_lr_sha256"],
            "clean_hr_sha256": shared["clean_hr_sha256"],
            "hr_noise_sha256": shared["hr_noise_sha256"],
            "branch_start_sha256": shared["branch_start_sha256"],
            "hr_schedule": grid,
            "hr_seconds": timing["hr_full_compute"],
            "candidate_denoise_seconds": timing["candidate_denoise"],
            "pipeline_seconds_this_branch": pipeline_seconds,
        })
        write_json_atomic(summary_path, summary)

    for key in ("clean_lr_sha256", "clean_hr_sha256", "hr_noise_sha256"):
        if len({row[key] for row in summary["cases"]}) != 1:
            raise RuntimeError(f"branches did not share one {key}")
    for sigma in plan["sigmas"]:
        hashes = {
            row["branch_start_sha256"]
            for row in summary["cases"]
            if row["refine_sigma"] == sigma
        }
        if len(hashes) != 1:
            raise RuntimeError(f"sigma={sigma} branches did not share one starting tensor")
    summary["complete"] = True
    write_json_atomic(summary_path, summary)
    print(f"{len(plan['cases'])} comparison videos completed. Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "check", "run"), nargs="?", default="plan")
    parser.add_argument(
        "--template-config",
        default=str(REPO_ROOT / "UNIV_adaptor/configs/univ_mrflow_refinement_ablation.json"),
    )
    parser.add_argument(
        "--model-root",
        default=os.environ.get("MODEL_ROOT", "/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B"),
    )
    parser.add_argument(
        "--lightx2v-repo",
        default=os.environ.get("LIGHTX2V_REPO", "/mnt/afs_2/houze/LightX2V"),
    )
    parser.add_argument(
        "--out-dir", default=str(REPO_ROOT / "outputs/univ_mrflow_refinement_v1")
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--negative-prompt",
        default="blurry details, static image, camera shake, watermark, subtitles, distorted anatomy",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigmas", type=parse_sigmas, default=DEFAULT_SIGMAS)
    parser.add_argument("--hr-steps", type=parse_hr_steps, default=DEFAULT_HR_STEPS)
    args = parser.parse_args()
    plan, config_path = prepare(args)
    if args.mode != "plan":
        run(args, plan, config_path)


if __name__ == "__main__":
    main()
