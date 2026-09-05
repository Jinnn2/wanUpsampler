"""Compare HR10/HR6/HR4/HR2 with a fixed boundary or a fixed 50-step total."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.data_protocol import canonical_sha256, sha256_file, write_json_atomic
from UNIV_adaptor.hr_refinement import resample_hr_sigmas
from UNIV_adaptor.schedule import action_from_config, resolve_schedule

HR_STEPS = (10, 6, 4, 2)
DEFAULT_PROMPT = (
    "A cinematic tracking shot of a red fox walking steadily through a snowy forest. "
    "Its paws leave footprints in fresh snow, its detailed fur moves gently, "
    "and snowflakes drift in front of dark pine branches. The camera follows smoothly."
)


def build_plan(args):
    comparison = getattr(args, "comparison", "fixed-boundary")
    config = json.loads(Path(args.template_config).read_text(encoding="utf-8"))
    if int(config["infer_steps"]) != 50:
        raise ValueError("this comparison uses the 50-step Wan reference")
    action = action_from_config(config)
    shape = (16, (int(config["target_video_length"]) - 1) // 4 + 1,
             int(config["target_height"]) // 8, int(config["target_width"]) // 8)
    schedule = resolve_schedule(action, reference_nfe=50, target_latent_shape=shape)
    if len(schedule.hr_compute_steps) != 10:
        raise ValueError("HR10 must be the unchanged reference suffix: use switch_ratio=0.8")
    if action.lr_nfe_ratio != 1.0:
        raise ValueError("use lr_nfe_ratio=1.0 to isolate HR discretization from LR caching")
    if config.get("univ_transition_baseline") != "dvg_latent_anchor":
        raise ValueError("this comparison fixes transition=dvg_latent_anchor")
    if config.get("cpu_offload") or config.get("compile"):
        raise ValueError("this comparison requires resident, uncompiled weights")
    if config.get("feature_caching") != "NoCaching":
        raise ValueError("this comparison requires feature_caching=NoCaching")
    if not args.prompt.strip():
        raise ValueError("prompt must not be empty")
    out = Path(args.out_dir).resolve()
    config["univ_hr_boundary_path"] = str(out / "shared_hr_boundary.pt")
    # Planning approximation only; runtime grids use the actual scheduler tensors.
    shift = float(config["sample_shift"])
    if shift <= 0:
        raise ValueError("sample_shift must be positive")
    raw = [0.999 * (1.0 - i / 50) for i in range(50)]
    reference = [shift * value / (1.0 + (shift - 1.0) * value) for value in raw] + [0.0]
    plan = {
        "schema": "univ_hr_refinement_ablation_plan_v1",
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "model_root": str(Path(args.model_root).resolve()),
        "config": config,
        "reference_schedule": schedule.as_dict(),
        "cases": [
            {
                "id": f"HR{steps:02d}",
                "hr_steps": steps,
                "planned_sigmas": list(resample_hr_sigmas(reference, boundary_step=40, hr_steps=steps)),
                "video_path": str(out / f"HR{steps:02d}.mp4"),
            }
            for steps in HR_STEPS
        ],
        "comparison": "same prompt, seed, prefix, transition tensor and boundary sigma",
    }
    if comparison == "fixed-total":
        plan["schema"] = "univ_hr_fixed_total_plan_v1"
        plan["comparison"] = "same prompt, seed, initial noise and original 50-step grid; independent transitions"
        for case in plan["cases"]:
            boundary = 50 - case["hr_steps"]
            case["lr_steps"] = boundary
            case["planned_sigmas"] = reference[boundary:]
            case["boundary_path"] = str(out / f"{case['id']}_boundary.pt")
            case_config = copy.deepcopy(config)
            case_config["univ_action"]["switch_ratio"] = boundary / 50
            case_config["univ_hr_boundary_path"] = case["boundary_path"]
            case["config"] = case_config
            case["reference_schedule"] = resolve_schedule(
                action_from_config(case_config), reference_nfe=50, target_latent_shape=shape,
            ).as_dict()
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def prepare(args):
    plan = build_plan(args)
    out = Path(args.out_dir).resolve()
    path = out / "comparison_plan.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != plan:
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
        print(f"{case['id']}: sigma grid = " + ", ".join(f"{s:.6f}" for s in case["planned_sigmas"]))
        if "config" in case:
            write_json_atomic(out / f"{case['id']}_config.json", case["config"])
    return plan, config_path


def runtime_args(args, config_path):
    return SimpleNamespace(
        seed=args.seed, model_cls="wan2.1_univ_hr_ablation", task="t2v", support_tasks=[],
        model_path=args.model_root, config_json=str(config_path),
        prompt=args.prompt, negative_prompt=args.negative_prompt,
        use_prompt_enhancer=False, return_result_tensor=False, target_shape=[],
        target_video_length=81, save_result_path=None,
    )


def run(args, plan, config_path):
    fixed_total = plan["schema"] == "univ_hr_fixed_total_plan_v1"
    lightx2v = Path(args.lightx2v_repo).resolve()
    if not lightx2v.is_dir():
        raise FileNotFoundError(f"LightX2V repository not found: {lightx2v}")
    sys.path.insert(0, str(lightx2v))
    os.environ.setdefault("DTYPE", "BF16")
    import torch
    from UNIV_adaptor.model_contract import validate_wan21_t2v_model_root
    validate_wan21_t2v_model_root(args.model_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    from lightx2v.common import ops  # noqa: F401 -- operator registrations
    from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
    from lightx2v.utils.set_config import set_config
    from lightx2v.utils.utils import seed_all, validate_config_paths
    from UNIV_adaptor.hr_ablation_runner import WanHRRefinementAblationRunner

    execution_args = runtime_args(args, config_path)
    config = set_config(execution_args)
    validate_config_paths(config)
    if args.mode == "check":
        print("Model contract, runtime imports and CUDA check passed.")
        return
    outputs = [Path(case["video_path"]) for case in plan["cases"]]
    protected = [*outputs, Path(config["univ_hr_boundary_path"]), Path(args.out_dir) / "comparison_summary.json"]
    protected += [Path(case["boundary_path"]) for case in plan["cases"] if "boundary_path" in case]
    if any(path.exists() for path in protected):
        raise FileExistsError("comparison artifacts already exist; use a new OUT_DIR")
    torch.set_grad_enabled(False)
    seed_all(args.seed)
    runner = WanHRRefinementAblationRunner(config)
    runner.init_modules()
    summary = {
        "schema": "univ_hr_refinement_ablation_results_v1",
        "plan_sha256": plan["plan_sha256"],
        "prompt": args.prompt, "seed": args.seed,
        "run_order": [case["id"] for case in plan["cases"]],
        "timing_note": "Single-pass quality ablation, no dedicated warmup. First branch includes shared prefix; compare synchronized hr_seconds, not whole pipeline times.",
        "cases": [],
    }
    summary_path = Path(args.out_dir) / "comparison_summary.json"
    if fixed_total:
        summary["schema"] = "univ_hr_fixed_total_results_v1"
        summary["timing_note"] = "Each branch executes its own LR prefix and transition; denoise_seconds sums synchronized LR, transition and HR times. Single pass, no warmup."
    for case in plan["cases"]:
        if fixed_total:
            configure_fixed_total_case(runner, case)
        runner.hr_steps = case["hr_steps"]
        input_info = init_empty_input_info("t2v", [])
        payload = vars(execution_args).copy()
        payload.update(save_result_path=case["video_path"], target_video_length=int(config["target_video_length"]))
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
        summary["cases"].append({
            "id": case["id"], "hr_steps": case["hr_steps"],
            "video_path": str(video), "video_sha256": sha256_file(video),
            "hr_seconds": runtime["timing_seconds"]["hr_full_compute"],
            "pipeline_seconds_this_branch": pipeline_seconds,
            "boundary_sha256": runtime["shared_boundary"]["tensor_sha256"],
            "hr_schedule": runtime["hr_schedule"],
        })
        if fixed_total:
            actual_schedule = runtime["reference_schedule"]
            if (actual_schedule["switch_step"] != case["lr_steps"]
                    or len(actual_schedule["lr_compute_steps"]) != case["lr_steps"]
                    or actual_schedule["lr_cache_steps"]
                    or runtime["shared_boundary"]["reused"]):
                raise RuntimeError("fixed-total branch used an incorrect prefix or reused a boundary")
            summary["cases"][-1].update(
                lr_steps=case["lr_steps"], reference_schedule=actual_schedule,
                denoise_seconds=sum(runtime["timing_seconds"][key] for key in
                                    ("lr_full_compute", "lr_cache_reuse", "transition", "hr_full_compute")),
            )
        summary["complete"] = len(summary["cases"]) == len(HR_STEPS)
        write_json_atomic(summary_path, summary)
    if not fixed_total and len({case["boundary_sha256"] for case in summary["cases"]}) != 1:
        raise RuntimeError("branches did not share the same HR input")
    print(f"Four comparison videos completed. Summary: {summary_path}")


def configure_fixed_total_case(runner, case):
    """Keep resident weights but discard all reusable experiment boundary state."""
    runner.shared_boundary = runner.shared_identity = runner.shared_record = None
    # LightX2V versions may copy configs, so update each consumer explicitly.
    seen = set()
    for owner in (runner, runner.scheduler, runner.model, runner.model.scheduler):
        config = owner.config
        if id(config) in seen:
            continue
        seen.add(id(config))
        temporarily_unlocked = getattr(config, "temporarily_unlocked", None)
        context = temporarily_unlocked() if callable(temporarily_unlocked) else nullcontext()
        with context:
            # Preserve the nested LockableDict so recursive locking is restored on exit.
            config["univ_action"].clear()
            config["univ_action"].update(copy.deepcopy(case["config"]["univ_action"]))
            config["univ_hr_boundary_path"] = case["boundary_path"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "check", "run"), nargs="?", default="plan")
    parser.add_argument("--template-config", default=str(REPO_ROOT / "UNIV_adaptor/configs/univ_hr_refinement_ablation.json"))
    parser.add_argument("--model-root", default=os.environ.get("MODEL_ROOT", "/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B"))
    parser.add_argument("--lightx2v-repo", default=os.environ.get("LIGHTX2V_REPO", "/mnt/afs_2/houze/LightX2V"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default="blurry details, static image, camera shake, watermark, subtitles, distorted anatomy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comparison", choices=("fixed-boundary", "fixed-total"), default="fixed-boundary")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = str(REPO_ROOT / "outputs" / (
            "univ_hr_fixed_total_v1" if args.comparison == "fixed-total" else "univ_hr_refinement_ablation_v1"
        ))
    plan, config_path = prepare(args)
    if args.mode != "plan":
        run(args, plan, config_path)


if __name__ == "__main__":
    main()
