from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Case:
    name: str
    step: int
    handoff: str
    resizer: str


def main() -> None:
    args = parse_args()
    family = family_config(args)
    prompts = load_prompts(Path(args.prompts), args.prompt_offset, args.limit)
    cases = build_cases(args.family, args.steps)
    out_root = Path(args.out_root)
    (out_root / "configs").mkdir(parents=True, exist_ok=True)
    (out_root / "videos").mkdir(parents=True, exist_ok=True)
    write_run_manifest(out_root, args, prompts, cases)

    for case in cases:
        config_path = out_root / "configs" / f"{case.name}.json"
        write_config(config_path, args, family, case)

    validate_inputs(args, family, cases)
    if args.mode == "check":
        print(f"Check passed: {len(cases)} cases, {len(prompts)} prompts; configs={out_root / 'configs'}")
        return

    for index, prompt in enumerate(prompts, start=args.prompt_offset):
        seed = args.seed + index
        label = f"{index:02d}"
        for case in cases:
            output = out_root / "videos" / case.name / f"{case.name}_{label}_seed{seed}.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.is_file() and output.stat().st_size > 0 and args.skip_existing:
                print(f"[reuse] {output}", flush=True)
                continue
            reused = reuse_existing(args, case, label, seed, output)
            if reused:
                print(f"[reuse] {reused} -> {output}", flush=True)
                continue
            run_inference(args, family, case, prompt, seed, output)


def family_config(args: argparse.Namespace) -> dict[str, object]:
    if args.family == "wan50":
        return {
            "infer_steps": 50,
            "sample_shift": 8,
            "enable_cfg": True,
            "model_root": args.model_root or "/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B",
            "bridge": REPO_ROOT / "changing_resolution/scripts/bridge/run_lightx2v_clean_bridge_infer.py",
            "base_interp_cls": "wan2.1_clean_interp_bridge",
            "base_stage2_cls": "wan2.1_clean_resizer_bridge",
            "lora_interp_cls": "wan2.1_tail_skip_lora_clean_interp_bridge",
            "lora_stage2_cls": "wan2.1_tail_skip_lora_clean_resizer_bridge",
        }
    return {
        "infer_steps": 4,
        "sample_shift": 5,
        "enable_cfg": False,
        "model_root": args.model_root or "/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill",
        "bridge": REPO_ROOT / "changing_resolution_distill/scripts/bridge/run_lightx2v_distill_bridge_infer.py",
        "base_interp_cls": "wan2.1_distill_interp_bridge",
        "base_stage2_cls": "wan2.1_distill_clean_resizer_bridge",
        "lora_interp_cls": "wan2.1_distill_last_step_lora_interp_bridge",
        "lora_stage2_cls": "wan2.1_distill_last_step_lora_clean_resizer_bridge",
    }


def build_cases(family: str, steps: list[int]) -> list[Case]:
    infer_steps = 50 if family == "wan50" else 4
    for step in steps:
        if step < 1 or step >= infer_steps:
            raise SystemExit(f"Invalid handoff step {step} for {family}")
    return [
        Case(f"step{step}_{handoff}_{resizer}", step, handoff, resizer)
        for step in steps
        for handoff in ("base", "lora")
        for resizer in ("interp", "stage2")
    ]


def write_config(path: Path, args: argparse.Namespace, family: dict[str, object], case: Case) -> None:
    config: dict[str, object] = {
        "infer_steps": family["infer_steps"],
        "target_video_length": args.num_frames,
        "text_len": 512,
        "target_height": 720,
        "target_width": 1248,
        "self_attn_1_type": "flash_attn3",
        "cross_attn_1_type": "flash_attn3",
        "cross_attn_2_type": "flash_attn3",
        "sample_guide_scale": args.guide_scale,
        "sample_shift": family["sample_shift"],
        "enable_cfg": family["enable_cfg"],
        "cpu_offload": False,
        "feature_caching": "NoCaching",
        "changing_resolution": True,
        "resolution_rate": [368 / 720],
        "wan_lowres_latent_size": [46, 80],
        "changing_resolution_steps": [case.step],
        "compare_name": case.name,
    }
    if args.family == "distill4":
        config["denoising_step_list"] = [1000, 750, 500, 250]
        config["dit_original_ckpt"] = args.dit_ckpt
        config["wan_distill_bridge_renoise_mode"] = args.renoise_mode
    if case.resizer == "stage2":
        config.update(
            {
                "wan_clean_resizer_repo": str(REPO_ROOT),
                "wan_clean_resizer_ckpt": args.stage2_checkpoint,
                "wan_clean_resizer_train_config": args.stage2_train_config,
                "wan_clean_resizer_model_class": "stage2",
                "wan_clean_resizer_use_ema": args.stage2_use_ema,
            }
        )
    if case.handoff == "lora":
        config.update(
            {
                "lora_dynamic_apply": True,
                "lora_active_steps": [case.step],
                "lora_configs": [
                    {"name": "wan2.1", "path": lora_checkpoint(args, case.step), "strength": args.lora_strength}
                ],
            }
        )
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def lora_checkpoint(args: argparse.Namespace, step: int) -> str:
    value = args.lora_checkpoint.get(step)
    if not value:
        raise SystemExit(f"Missing LoRA checkpoint for step {step}; use --lora-checkpoint {step}=PATH")
    return value


def run_inference(
    args: argparse.Namespace, family: dict[str, object], case: Case, prompt: str, seed: int, output: Path
) -> None:
    class_key = f"{case.handoff}_{case.resizer}_cls"
    command = [
        sys.executable,
        str(family["bridge"]),
        "--seed",
        str(seed),
        "--model_cls",
        str(family[class_key]),
        "--task",
        "t2v",
        "--model_path",
        str(family["model_root"]),
        "--config_json",
        str(Path(args.out_root) / "configs" / f"{case.name}.json"),
        "--prompt",
        prompt,
        "--negative_prompt",
        args.negative_prompt,
        "--save_result_path",
        str(output),
        "--target_video_length",
        str(args.num_frames),
    ]
    print(f"[run] {case.name} seed={seed}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def reuse_existing(args: argparse.Namespace, case: Case, label: str, seed: int, output: Path) -> Path | None:
    aliases = existing_aliases(args.family, case, label, seed)
    for root_text in args.reuse_root:
        root = Path(root_text)
        for relative in aliases:
            source = root / relative
            if source.is_file() and source.stat().st_size > 0:
                link_or_copy(source, output)
                return source
    return None


def existing_aliases(family: str, case: Case, label: str, seed: int) -> list[Path]:
    aliases = [Path("videos") / case.name / f"{case.name}_{label}_seed{seed}.mp4"]
    if family == "wan50" and case.step == 45 and case.resizer == "stage2":
        old = "ori45_stage2" if case.handoff == "base" else "lora45_stage2"
        aliases.append(Path("videos") / old / f"{old}_{label}_seed{seed}.mp4")
    if family == "distill4" and case.step == 3 and case.resizer == "stage2":
        old = "base3_stage2_hr4" if case.handoff == "base" else "lora3_stage2_hr4"
        aliases.append(Path("videos") / old / f"{old}_{label}_seed{seed}.mp4")
    return aliases


def link_or_copy(source: Path, output: Path) -> None:
    output.unlink(missing_ok=True)
    try:
        os.link(source, output)
    except OSError:
        try:
            output.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, output)


def validate_inputs(args: argparse.Namespace, family: dict[str, object], cases: list[Case]) -> None:
    required = [Path(args.prompts), Path(args.stage2_checkpoint), Path(args.stage2_train_config), Path(str(family["model_root"]))]
    if args.family == "distill4":
        required.append(Path(args.dit_ckpt))
    required.extend(Path(lora_checkpoint(args, case.step)) for case in cases if case.handoff == "lora")
    missing = sorted({str(path) for path in required if not path.exists()})
    if missing:
        raise SystemExit("Missing required paths:\n  " + "\n  ".join(missing))


def load_prompts(path: Path, offset: int, limit: int) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    selected = prompts[offset : offset + limit]
    if not selected:
        raise SystemExit(f"No prompts selected from {path}")
    return selected


def write_run_manifest(out_root: Path, args: argparse.Namespace, prompts: list[str], cases: list[Case]) -> None:
    payload = {
        "family": args.family,
        "seed_base": args.seed,
        "prompt_offset": args.prompt_offset,
        "prompts": prompts,
        "cases": [case.__dict__ for case in cases],
    }
    (out_root / "run_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_step_paths(values: list[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for value in values:
        try:
            step_text, path = value.split("=", 1)
            result[int(step_text)] = path
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Expected STEP=PATH, got {value}") from exc
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TALH Base/LoRA x interpolation/Stage2 factorial.")
    parser.add_argument("mode", choices=["check", "run"])
    parser.add_argument("--family", choices=["wan50", "distill4"], required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--prompts", default=str(REPO_ROOT / "changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt"))
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model-root")
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--stage2-train-config", required=True)
    parser.add_argument("--lora-checkpoint", action="append", default=[], metavar="STEP=PATH")
    parser.add_argument("--dit-ckpt", default="/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt")
    parser.add_argument("--reuse-root", action="append", default=[])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=9700)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--guide-scale", type=float, default=6.0)
    parser.add_argument("--lora-strength", type=float, default=0.75)
    parser.add_argument("--renoise-mode", default="random")
    parser.add_argument("--stage2-use-ema", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--negative-prompt", default="")
    args = parser.parse_args()
    args.lora_checkpoint = parse_step_paths(args.lora_checkpoint)
    return args


if __name__ == "__main__":
    main()
