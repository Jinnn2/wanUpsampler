from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BATCH_RUNNER = REPO_ROOT / "changing_resolution/scripts/bridge/run_lightx2v_clean_bridge_batch_infer.py"


@dataclass(frozen=True)
class TimingCase:
    name: str
    model_cls: str
    config_kind: str


CASES = (
    TimingCase("interp45", "wan2.1_clean_interp_bridge", "interp"),
    TimingCase("cll_only45", "wan2.1_clean_resizer_bridge", "cll"),
    TimingCase("taa_interp45", "wan2.1_tail_skip_lora_clean_interp_bridge", "taa"),
    TimingCase("trajscale45", "wan2.1_tail_skip_lora_clean_resizer_bridge", "full"),
)

PAIRS = (
    ("interp45", "cll_only45", "CLL overhead over interpolation"),
    ("interp45", "taa_interp45", "TAA overhead over interpolation"),
    ("interp45", "trajscale45", "full TrajScale overhead over interpolation"),
    ("cll_only45", "trajscale45", "conditional TAA overhead with CLL"),
    ("taa_interp45", "trajscale45", "conditional CLL overhead with TAA"),
)


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else suite_root / "handoff_overhead_step45"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "configs").mkdir(exist_ok=True)
    (output_root / "raw").mkdir(exist_ok=True)
    (output_root / "videos").mkdir(exist_ok=True)

    manifest = load_json(suite_root / "run_manifest.json")
    settings = manifest.get("settings", {})
    model_root_value = args.model_root or settings.get("model_root")
    prompts_value = args.prompts or settings.get("prompts_file")
    if not model_root_value or not prompts_value:
        raise SystemExit(
            "Model/prompts paths are absent from run_manifest.json; pass --model-root and --prompts"
        )
    model_root = Path(model_root_value).resolve()
    prompts = Path(prompts_value).resolve()
    require_path(model_root, "model root")
    require_path(prompts, "prompts file")
    available_prompts = [
        line
        for line in prompts.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    required_prompts = args.prompt_offset + args.warmup + args.repeats
    if len(available_prompts) < required_prompts:
        raise SystemExit(
            f"Need at least {required_prompts} prompts for offset+warmup+repeats, "
            f"found {len(available_prompts)} in {prompts}"
        )

    talh_config_path = suite_root / "configs/talh45.json"
    require_path(talh_config_path, "TALH-45 config")
    talh_config = load_json(talh_config_path)
    validate_talh45(talh_config)

    selected_names = set(args.cases)
    unknown = selected_names - {case.name for case in CASES}
    if unknown:
        raise SystemExit(f"Unknown case(s): {sorted(unknown)}")
    selected = [case for case in CASES if case.name in selected_names]
    if not selected:
        raise SystemExit("No timing cases selected")

    config_paths: dict[str, Path] = {}
    for case in selected:
        config = derive_config(talh_config, case)
        path = output_root / "configs" / f"{case.name}.json"
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        config_paths[case.name] = path

    protocol = {
        "schema_version": 1,
        "purpose": "separate warm-process handoff overhead from cold-start model loading",
        "gpu_physical": args.gpu,
        "warmup_videos": args.warmup,
        "measured_videos": args.repeats,
        "prompt_offset": args.prompt_offset,
        "same_prompt_seed_resolution": True,
        "practical_overhead_criterion": (
            "upper bound of the paired 95% CI for steady-state denoise overhead "
            f"is <= {args.overhead_margin_pct:.3f}% of the baseline mean"
        ),
        "timing_boundaries": {
            "initialization": "runner construction plus init_modules, CUDA-synchronized",
            "pipeline": "runner.run_pipeline including text/VAE/save, CUDA-synchronized",
            "denoise": "sum of runner.run_segment calls, CUDA-synchronized",
        },
        "cases": [case.__dict__ for case in selected],
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    total_videos = args.warmup + args.repeats
    for case in selected:
        raw_path = output_root / "raw" / f"{case.name}.jsonl"
        if args.resume and valid_raw_result(raw_path, args.repeats):
            print(f"[resume] {case.name}: keep {raw_path}", flush=True)
            continue
        command = [
            args.python,
            str(BATCH_RUNNER),
            "--seed",
            str(args.seed),
            "--increment_seed",
            "--model_cls",
            case.model_cls,
            "--task",
            "t2v",
            "--model_path",
            str(model_root),
            "--config_json",
            str(config_paths[case.name]),
            "--prompts_file",
            str(prompts),
            "--out_dir",
            str(output_root / "videos" / case.name),
            "--name_prefix",
            case.name,
            "--prompt-offset",
            str(args.prompt_offset),
            "--limit",
            str(total_videos),
            "--target_video_length",
            str(args.num_frames or settings.get("num_frames", 81)),
            "--negative_prompt",
            args.negative_prompt,
            "--timing-jsonl",
            str(raw_path),
            "--timing-warmup",
            str(args.warmup),
        ]
        print(f"[run] {case.name}: {total_videos} video(s), physical GPU {args.gpu}", flush=True)
        print_command(command)
        if args.dry_run:
            continue
        environment = inference_environment(args.gpu)
        subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)

    if args.dry_run:
        print(f"Dry-run protocol and configs: {output_root}")
        return
    rows = summarize(selected, output_root, args.repeats)
    write_summary(rows, output_root)
    write_pair_deltas(selected, output_root, args.repeats, args.overhead_margin_pct)
    print(f"Raw timings : {output_root / 'raw'}")
    print(f"Summary CSV : {output_root / 'summary.csv'}")
    print(f"Pair deltas : {output_root / 'pair_deltas.csv'}")


def derive_config(source: dict[str, Any], case: TimingCase) -> dict[str, Any]:
    config = dict(source)
    config["compare_name"] = case.name
    lora_keys = {"lora_dynamic_apply", "lora_active_steps", "lora_configs"}
    cll_keys = {
        "wan_clean_resizer_repo",
        "wan_clean_resizer_ckpt",
        "wan_clean_resizer_train_config",
        "wan_clean_resizer_model_class",
        "wan_clean_resizer_use_ema",
        "wan_clean_resizer_residual_skip",
    }
    if case.config_kind in {"interp", "cll"}:
        for key in lora_keys:
            config.pop(key, None)
    if case.config_kind in {"interp", "taa"}:
        for key in cll_keys:
            config.pop(key, None)
    for key in ("wan_ralu_noise_c", "wan_ralu_suffix_shift", "wan_ralu_adaptation"):
        config.pop(key, None)
    return config


def validate_talh45(config: dict[str, Any]) -> None:
    if [int(x) for x in config.get("changing_resolution_steps", [])] != [45]:
        raise SystemExit("Expected configs/talh45.json to use changing_resolution_steps=[45]")
    lora_configs = config.get("lora_configs") or []
    if len(lora_configs) != 1 or float(lora_configs[0].get("strength", -1)) != 0.75:
        raise SystemExit("Expected TALH-45 LoRA strength=0.75")


def valid_raw_result(path: Path, repeats: int) -> bool:
    if not path.is_file():
        return False
    rows = read_jsonl(path)
    measured = [row for row in rows if row.get("kind") == "video" and row.get("phase") == "measured"]
    return len(measured) == repeats


def summarize(cases: list[TimingCase], output_root: Path, repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        raw = read_jsonl(output_root / "raw" / f"{case.name}.jsonl")
        init = [float(row["elapsed_s"]) for row in raw if row.get("kind") == "initialization"]
        measured = [row for row in raw if row.get("kind") == "video" and row.get("phase") == "measured"]
        if len(init) != 1 or len(measured) != repeats:
            raise RuntimeError(
                f"{case.name}: expected one initialization and {repeats} measured rows, "
                f"got {len(init)} and {len(measured)}"
            )
        pipeline = [float(row["pipeline_elapsed_s"]) for row in measured]
        denoise = [float(row["denoise_elapsed_s"]) for row in measured]
        overhead = [total - core for total, core in zip(pipeline, denoise)]
        rows.append(
            {
                "case": case.name,
                "model_cls": case.model_cls,
                "initialization_s": init[0],
                "pipeline_mean_s": statistics.mean(pipeline),
                "pipeline_std_s": statistics.stdev(pipeline) if len(pipeline) > 1 else 0.0,
                "denoise_mean_s": statistics.mean(denoise),
                "denoise_std_s": statistics.stdev(denoise) if len(denoise) > 1 else 0.0,
                "non_denoise_mean_s": statistics.mean(overhead),
                "repeats": repeats,
            }
        )
    return rows


def write_summary(rows: list[dict[str, Any]], output_root: Path) -> None:
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_pair_deltas(
    cases: list[TimingCase],
    output_root: Path,
    repeats: int,
    overhead_margin_pct: float,
) -> None:
    case_names = {case.name for case in cases}
    measured: dict[str, list[dict[str, Any]]] = {}
    initialization: dict[str, float] = {}
    for case in cases:
        raw = read_jsonl(output_root / "raw" / f"{case.name}.jsonl")
        initialization[case.name] = next(
            float(row["elapsed_s"]) for row in raw if row.get("kind") == "initialization"
        )
        case_rows = [
            row for row in raw if row.get("kind") == "video" and row.get("phase") == "measured"
        ]
        measured[case.name] = sorted(case_rows, key=lambda row: int(row["repeat"]))

    deltas: list[dict[str, Any]] = []
    for baseline_name, case_name, comparison in PAIRS:
        if baseline_name not in case_names or case_name not in case_names:
            continue
        baseline_rows = measured[baseline_name]
        case_rows = measured[case_name]
        if len(baseline_rows) != repeats or len(case_rows) != repeats:
            raise RuntimeError(f"Incomplete paired rows for {baseline_name} vs {case_name}")
        for left, right in zip(baseline_rows, case_rows):
            if (left["repeat"], left["prompt_index"], left["seed"]) != (
                right["repeat"],
                right["prompt_index"],
                right["seed"],
            ):
                raise RuntimeError(f"Pairing mismatch for {baseline_name} vs {case_name}")
        pipeline_base = [float(row["pipeline_elapsed_s"]) for row in baseline_rows]
        denoise_base = [float(row["denoise_elapsed_s"]) for row in baseline_rows]
        pipeline_delta = [
            float(right["pipeline_elapsed_s"]) - float(left["pipeline_elapsed_s"])
            for left, right in zip(baseline_rows, case_rows)
        ]
        denoise_delta = [
            float(right["denoise_elapsed_s"]) - float(left["denoise_elapsed_s"])
            for left, right in zip(baseline_rows, case_rows)
        ]
        pipeline_mean, pipeline_low, pipeline_high = mean_ci95(pipeline_delta)
        denoise_mean, denoise_low, denoise_high = mean_ci95(denoise_delta)
        deltas.append(
            {
                "comparison": comparison,
                "baseline": baseline_name,
                "case": case_name,
                "paired_repeats": repeats,
                "pipeline_delta_mean_s": pipeline_mean,
                "pipeline_delta_ci95_low_s": pipeline_low,
                "pipeline_delta_ci95_high_s": pipeline_high,
                "pipeline_overhead_pct": 100.0 * pipeline_mean / statistics.mean(pipeline_base),
                "denoise_delta_mean_s": denoise_mean,
                "denoise_delta_ci95_low_s": denoise_low,
                "denoise_delta_ci95_high_s": denoise_high,
                "denoise_overhead_pct": 100.0 * denoise_mean / statistics.mean(denoise_base),
                "denoise_ci95_high_overhead_pct": 100.0
                * denoise_high
                / statistics.mean(denoise_base),
                "overhead_margin_pct": overhead_margin_pct,
                "within_overhead_margin": (
                    100.0 * denoise_high / statistics.mean(denoise_base)
                    <= overhead_margin_pct
                ),
                "initialization_delta_s": initialization[case_name] - initialization[baseline_name],
            }
        )
    if not deltas:
        raise RuntimeError("Selected cases do not contain any registered comparison pair")
    with (output_root / "pair_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean, mean
    # Two-sided Student-t 97.5% quantiles for df=1..30; normal approximation above 30.
    t_critical = (
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
        2.042,
    )
    df = len(values) - 1
    critical = t_critical[df - 1] if df <= len(t_critical) else 1.96
    margin = critical * statistics.stdev(values) / (len(values) ** 0.5)
    return mean, mean - margin, mean + margin


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    require_path(path, path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def require_path(path: Path, label: str) -> None:
    if not str(path) or not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def inference_environment(gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("LIGHTX2V_REPO", "/path/to/LightX2V")
    environment.setdefault("DIFFSYNTH_REPO", "/path/to/DiffSynth-Studio")
    roots = [environment["LIGHTX2V_REPO"], environment["DIFFSYNTH_REPO"], str(REPO_ROOT)]
    if environment.get("PYTHONPATH"):
        roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def print_command(command: list[str]) -> None:
    import shlex

    print("  " + shlex.join(command), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm-process factorial timing for step-45 interpolation, TAA, CLL, and TrajScale."
    )
    parser.add_argument("--suite-root", required=True, help="Final-v2 root containing run_manifest.json and configs/talh45.json")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=2, help="Physical GPU exposed to the child process")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--model-root", default="")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--cases", nargs="+", default=[case.name for case in CASES])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overhead-margin-pct",
        type=float,
        default=5.0,
        help="Pre-registered maximum acceptable steady-state denoise overhead.",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.overhead_margin_pct <= 0:
        parser.error("--warmup must be >= 0, --repeats >= 1, and --overhead-margin-pct > 0")
    return args


if __name__ == "__main__":
    main()
