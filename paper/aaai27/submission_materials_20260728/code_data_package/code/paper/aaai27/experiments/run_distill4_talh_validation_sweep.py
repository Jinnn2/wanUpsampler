from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
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
from paper.aaai27.experiments.run_distill4_quality_efficiency import (  # noqa: E402
    Case,
    GPUAssignment,
    assign_cases_to_gpus,
    load_prompts,
    run_case,
    validate_inputs,
    write_config,
)


MAIN_PROMPTS = (
    REPO_ROOT / "changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt"
)


@dataclass(frozen=True)
class SweepPoint:
    name: str
    lora_strength: float
    renoise_mode: str


def strength_tag(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def build_sweep(strengths: list[float], renoise_modes: list[str]) -> list[SweepPoint]:
    points = [
        SweepPoint(
            name=f"talh3_s{strength_tag(strength)}_{renoise}",
            lora_strength=float(strength),
            renoise_mode=renoise,
        )
        for strength in strengths
        for renoise in renoise_modes
    ]
    if len({point.name for point in points}) != len(points):
        raise SystemExit("Sweep point names are not unique")
    return points


def talh_case(point: SweepPoint) -> Case:
    return Case(
        name=point.name,
        method="talh",
        model_cls="wan2.1_distill_last_step_lora_clean_resizer_bridge",
        lr_evaluations=3,
        hr_evaluations=1,
        total_evaluations=4,
        handoff_step=3,
        resizer="stage2",
        alignment="taa",
    )


def point_args(args: argparse.Namespace, point: SweepPoint) -> argparse.Namespace:
    values = vars(args).copy()
    values["lora_strength"] = point.lora_strength
    values["renoise_mode"] = point.renoise_mode
    return argparse.Namespace(**values)


def assert_validation_prompts_are_disjoint(prompts_path: Path) -> None:
    selected = set(load_prompts(prompts_path, 0, 1_000_000))
    main = set(load_prompts(MAIN_PROMPTS, 0, 1_000_000))
    overlap = sorted(selected & main)
    if overlap:
        raise SystemExit(
            "P3 validation prompts overlap the main 10-prompt test set:\n  "
            + "\n  ".join(overlap)
        )


def prepare(
    root: Path,
    args: argparse.Namespace,
    points: list[SweepPoint],
    prompts: list[str],
) -> None:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "videos").mkdir(parents=True, exist_ok=True)
    cases = []
    for point in points:
        case = talh_case(point)
        write_config(
            root / "configs" / f"{case.name}.json",
            point_args(args, point),
            case,
        )
        cases.append(
            {
                **asdict(case),
                "lora_strength": point.lora_strength,
                "renoise_mode": point.renoise_mode,
            }
        )
    manifest = {
        "schema_version": 1,
        "family": "distill4_talh_validation_sweep",
        "purpose": "P3 hyperparameter selection only; not final test reporting",
        "selection_rule": (
            "maximize validation VBench-5 mean; exact ties use lower LoRA strength "
            "then lexicographic renoise mode"
        ),
        "seed_base": args.seed,
        "prompt_offset": args.prompt_offset,
        "prompts": prompts,
        "settings": {
            "prompts_file": str(Path(args.prompts).resolve()),
            "main_test_prompts_file": str(MAIN_PROMPTS.resolve()),
            "prompt_sets_checked_disjoint": True,
            "strengths": args.strengths,
            "renoise_modes": args.renoise_modes,
            "num_frames": args.num_frames,
            "guide_scale": args.guide_scale,
            "generation_gpus": args.gpus,
        },
        "cases": cases,
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_assignment(
    root: Path,
    args: argparse.Namespace,
    assignment: GPUAssignment,
    point_by_name: dict[str, SweepPoint],
) -> list[list[str]]:
    commands = []
    for case in assignment.cases:
        commands.append(
            run_case(
                root,
                point_args(args, point_by_name[case.name]),
                case,
                gpu=assignment.gpu,
            )
        )
    return commands


def run_parallel(
    root: Path, args: argparse.Namespace, points: list[SweepPoint]
) -> None:
    cases = [talh_case(point) for point in points]
    point_by_name = {point.name: point for point in points}
    assignments = assign_cases_to_gpus(cases, args.gpus)
    schedule = {
        "schema_version": 1,
        "strategy": "case_parallel_four_gpu",
        "gpu_assignments": [
            {
                "gpu": assignment.gpu,
                "cases": [case.name for case in assignment.cases],
            }
            for assignment in assignments
        ],
    }
    (root / "generation_schedule.json").write_text(
        json.dumps(schedule, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(assignments), thread_name_prefix="talh-sweep-gpu"
    ) as executor:
        futures = [
            executor.submit(
                run_assignment, root, args, assignment, point_by_name
            )
            for assignment in assignments
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def aggregate_dimension(path: Path, case: str, dimension: str) -> float | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    numeric = payload.get("cases", {}).get(case, {}).get("numeric_metrics", {})
    candidates = [
        float(value)
        for key, value in numeric.items()
        if key.endswith(f".{dimension}.0")
    ]
    return sorted(candidates)[-1] if candidates else None


def select_best(root: Path) -> Path:
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    vbench_path = root / "metrics/vbench_v1_custom.json"
    rows: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        name = str(case["name"])
        components = vbench_case_scores(vbench_path, name)
        rows.append(
            {
                "case": name,
                "lora_strength": float(case["lora_strength"]),
                "renoise_mode": str(case["renoise_mode"]),
                **components,
                "quality5_mean": sum(components.values()) / len(QUALITY_DIMENSIONS),
                "temporal_flickering": aggregate_dimension(
                    vbench_path, name, "temporal_flickering"
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["quality5_mean"]),
            float(row["lora_strength"]),
            str(row["renoise_mode"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["selected"] = rank == 1
    metrics = root / "metrics"
    csv_path = metrics / "talh_validation_selection.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": 1,
        "selection_scope": "validation_only",
        "selection_rule": manifest["selection_rule"],
        "selected": rows[0],
        "ranking": rows,
    }
    output = metrics / "talh_validation_selection.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Selected {rows[0]['case']}: quality5={rows[0]['quality5_mean']:.6f}; "
        f"summary={output}"
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P3: independent-validation sweep for TALH3 strength and renoise mode."
    )
    parser.add_argument("action", choices=["check", "prepare", "run", "select"])
    parser.add_argument(
        "--out-root",
        default=str(
            REPO_ROOT
            / "outputs/aaai27_experiments/distill4_talh_validation_sweep"
        ),
    )
    parser.add_argument(
        "--prompts",
        default=str(
            REPO_ROOT
            / "paper/aaai27/experiments/distill4_talh_validation_prompts_8.txt"
        ),
    )
    parser.add_argument(
        "--model-root",
        default="/path/to/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill",
    )
    parser.add_argument(
        "--dit-ckpt",
        default="/path/to/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt",
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
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--renoise-modes",
        nargs="+",
        choices=["random", "resize_flow"],
        default=["random", "resize_flow"],
    )
    parser.add_argument(
        "--stage2-use-ema", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--guide-scale", type=float, default=6.0)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=16000)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()
    if not args.strengths or any(value < 0 for value in args.strengths):
        parser.error("--strengths must contain non-negative values")
    if len(args.gpus) != len(set(args.gpus)) or any(gpu < 0 for gpu in args.gpus):
        parser.error("--gpus must contain unique non-negative GPU ids")
    if args.limit < 1 or args.prompt_offset < 0:
        parser.error("--limit must be positive and --prompt-offset non-negative")
    return args


def main() -> None:
    args = parse_args()
    root = Path(args.out_root).resolve()
    if args.action == "select":
        select_best(root)
        return
    points = build_sweep(args.strengths, args.renoise_modes)
    cases = [talh_case(point) for point in points]
    assert_validation_prompts_are_disjoint(Path(args.prompts))
    validate_inputs(args, cases)
    prompts = load_prompts(Path(args.prompts), args.prompt_offset, args.limit)
    prepare(root, args, points, prompts)
    print(
        f"P3 TALH validation: {len(points)} cases x {len(prompts)} prompts; "
        f"output={root}"
    )
    if args.action == "run":
        run_parallel(root, args, points)


if __name__ == "__main__":
    main()
