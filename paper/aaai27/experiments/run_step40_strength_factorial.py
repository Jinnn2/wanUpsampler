from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Case:
    name: str
    step: int
    handoff: str
    resizer: str
    strength: float | None
    model_cls: str


def main() -> None:
    args = parse_args()
    strengths = validate_strengths(args.strengths)
    prompts = load_prompts(Path(args.prompts), args.prompt_offset, args.limit)
    cases = build_cases(args.step, strengths)
    validate_inputs(args)

    root = Path(args.out_root).resolve()
    (root / "configs").mkdir(parents=True, exist_ok=True)
    for case in cases:
        write_config(root / "configs" / f"{case.name}.json", args, case)
    write_manifest(root, args, prompts, cases)
    if args.mode == "check":
        print(f"Validated {len(cases)} cases and {len(prompts)} prompts under {root}")
        return

    for case in cases:
        run_case(root, args, case)


def build_cases(step: int, strengths: list[float]) -> list[Case]:
    cases = [
        Case(f"step{step}_base_interp", step, "base", "interp", None, "wan2.1_clean_interp_bridge"),
        Case(f"step{step}_base_stage2", step, "base", "stage2", None, "wan2.1_clean_resizer_bridge"),
    ]
    for strength in strengths:
        tag = strength_tag(strength)
        cases.extend(
            [
                Case(
                    f"step{step}_lora_s{tag}_interp",
                    step,
                    "lora",
                    "interp",
                    strength,
                    "wan2.1_tail_skip_lora_clean_interp_bridge",
                ),
                Case(
                    f"step{step}_lora_s{tag}_stage2",
                    step,
                    "lora",
                    "stage2",
                    strength,
                    "wan2.1_tail_skip_lora_clean_resizer_bridge",
                ),
            ]
        )
    return cases


def write_config(path: Path, args: argparse.Namespace, case: Case) -> None:
    config: dict[str, Any] = {
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
        "changing_resolution": True,
        "resolution_rate": [368 / 720],
        "wan_lowres_latent_size": [46, 80],
        "changing_resolution_steps": [case.step],
        "compare_name": case.name,
    }
    if case.resizer == "stage2":
        config.update(
            {
                "wan_clean_resizer_repo": str(REPO_ROOT),
                "wan_clean_resizer_ckpt": str(Path(args.stage2_checkpoint).resolve()),
                "wan_clean_resizer_train_config": str(Path(args.stage2_train_config).resolve()),
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
                    {
                        "name": "wan2.1",
                        "path": str(Path(args.lora_checkpoint).resolve()),
                        "strength": case.strength,
                    }
                ],
            }
        )
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_manifest(root: Path, args: argparse.Namespace, prompts: list[str], cases: list[Case]) -> None:
    base_stage2 = f"step{args.step}_base_stage2"
    review_pairs = [
        {
            "comparison": f"lora_s{strength_tag(strength)}_with_stage2",
            "step": args.step,
            "left_case": base_stage2,
            "right_case": f"step{args.step}_lora_s{strength_tag(strength)}_stage2",
        }
        for strength in validate_strengths(args.strengths)
    ]
    payload = {
        "schema_version": 1,
        "family": "wan50_step40_strength",
        "seed_base": args.seed,
        "prompt_offset": args.prompt_offset,
        "prompts": prompts,
        "cases": [asdict(case) for case in cases],
        "review_pairs": review_pairs,
        "lora_artifact": artifact_fingerprint(Path(args.lora_checkpoint)),
        "stage2_artifact": artifact_fingerprint(Path(args.stage2_checkpoint)),
        "settings": {
            "num_frames": args.num_frames,
            "guide_scale": args.guide_scale,
            "strengths": validate_strengths(args.strengths),
            "stage2_use_ema": args.stage2_use_ema,
        },
    }
    (root / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_case(root: Path, args: argparse.Namespace, case: Case) -> None:
    batch = REPO_ROOT / "changing_resolution/scripts/bridge/run_lightx2v_clean_bridge_batch_infer.py"
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
    print(f"[batch] {case.name}: {args.limit} prompt(s), one model load", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=inference_environment())


def inference_environment() -> dict[str, str]:
    environment = dict(os.environ)
    roots = [
        environment.get("LIGHTX2V_REPO", "/mnt/afs_2/houze/LightX2V"),
        environment.get("DIFFSYNTH_REPO", "/mnt/afs_2/houze/DiffSynth-Studio"),
        str(REPO_ROOT),
    ]
    existing = environment.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def validate_inputs(args: argparse.Namespace) -> None:
    required = [
        Path(args.prompts),
        Path(args.model_root),
        Path(args.stage2_checkpoint),
        Path(args.stage2_train_config),
        Path(args.lora_checkpoint),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required paths:\n  " + "\n  ".join(missing))
    if args.step < 1 or args.step >= 50:
        raise SystemExit("--step must be in [1, 49]")


def validate_strengths(values: list[float]) -> list[float]:
    if not values:
        raise SystemExit("At least one --strength is required")
    result = []
    for value in values:
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise SystemExit("LoRA strengths must be finite and non-negative")
        if value in result:
            raise SystemExit(f"Duplicate LoRA strength: {value}")
        result.append(value)
    return result


def strength_tag(value: float) -> str:
    return format(value, ".8g").replace("-", "m").replace(".", "p")


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
        "requested_path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Wan50 step40 LoRA-strength end-to-end factorial.")
    parser.add_argument("mode", choices=["check", "run"])
    parser.add_argument("--step", type=int, default=40)
    parser.add_argument("--strength", dest="strengths", type=float, action="append", default=[])
    parser.add_argument(
        "--prompts",
        default=str(REPO_ROOT / "changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt"),
    )
    parser.add_argument(
        "--out-root",
        default=str(REPO_ROOT / "outputs/aaai27_experiments/wan50_step40_strength"),
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
        "--lora-checkpoint",
        default=str(
            REPO_ROOT / "outputs/changing_resolution_tail_skip_lora_step40_to_step50_temporal/latest.safetensors"
        ),
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=9700)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--guide-scale", type=float, default=6.0)
    parser.add_argument("--stage2-use-ema", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if not args.strengths:
        args.strengths = [0.5, 0.75, 1.0]
    return args


if __name__ == "__main__":
    main()
