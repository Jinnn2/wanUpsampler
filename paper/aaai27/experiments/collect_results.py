from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = Path(__file__).with_name("experiment_manifest.json")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    environment = build_environment(project_root, manifest.get("defaults", {}))
    results_root = Path(environment.get("AAAI_RESULTS", project_root / "outputs/aaai27_experiments"))
    if not results_root.is_absolute():
        results_root = project_root / results_root
    results_root = results_root.resolve()
    output_root = Path(args.output_root or results_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    inventory = collect_inventory(project_root, results_root, output_root, manifest, environment)
    write_outputs(output_root, inventory)
    print(f"Inventory: {output_root / 'result_inventory.json'}")
    print(f"Tables   : {output_root / 'paper_tables.md'}")
    print(f"CSV dir  : {output_root / 'compiled_tables'}")
    print(f"Issues   : {len(inventory['issues'])}")
    if args.strict and inventory["issues"]:
        raise SystemExit("Strict collection failed because missing or invalid evidence remains")


def collect_inventory(
    project_root: Path,
    results_root: Path,
    output_root: Path,
    manifest: dict[str, Any],
    environment: dict[str, str],
) -> dict[str, Any]:
    checkpoint_tag = Path(environment["DISTILL_LORA_480_CKPT"]).stem
    wan_factorial = Path(environment["WAN50_FACTORIAL"])
    distill_factorial = Path(environment["DISTILL_FACTORIAL"])
    if not wan_factorial.is_absolute():
        wan_factorial = project_root / wan_factorial
    if not distill_factorial.is_absolute():
        distill_factorial = project_root / distill_factorial

    sources: dict[str, Any] = {
        "operator_480p": load_json_source(
            project_root / "outputs/changing_resolution_operator_compare_stage2/tables/summary_val.json"
        ),
        "operator_368p": load_json_source(results_root / "operator_368p/tables/summary_val.json"),
        "wan50_endpoint_step45": load_csv_source(
            project_root
            / "outputs/changing_resolution_tail_skip_lora_clean_pred_compare_360p_368x640/metrics/original_lora_teacher_summary.csv"
        ),
        "wan50_endpoint_samples": load_csv_source(
            project_root
            / "outputs/changing_resolution_tail_skip_lora_clean_pred_compare_360p_368x640/metrics/original_lora_teacher_metrics.csv"
        ),
        "wan50_lora_strength": load_csv_source(
            project_root
            / "outputs/changing_resolution_tail_skip_lora_strength_sweep_360p_368x640/metrics/strength_sweep_summary.csv"
        ),
        "distill_checkpoint_metrics": load_csv_source(
            project_root / "outputs/eval_lora_ckpt_sweep_480p/checkpoint_metric_summary.csv"
        ),
        "distill_checkpoint_rank_l1": load_csv_source(
            project_root / "outputs/eval_lora_ckpt_sweep_480p/checkpoint_rank_by_l1.csv"
        ),
        "distill_368p_lora_strength": load_csv_source(
            project_root
            / f"outputs/changing_resolution_distill_480p_lora_strength_sweep_360p_368x640_{checkpoint_tag}/strength_metric_summary.csv"
        ),
        "distill_480lora_transfer_368p": load_csv_source(
            results_root / "distill_480p_lora_transfer_368p/transfer_sweep_summary.csv"
        ),
        "distill_480lora_transfer_368p_samples": load_csv_source(
            results_root
            / "distill_480p_lora_transfer_368p/strength_0p75/evaluation/distill_360p_metrics.csv"
        ),
        "timing_raw": load_csv_source(project_root / "outputs/changing_resolution_time_compare/time_summary.csv"),
        "lora_architecture_loss": load_csv_source(results_root / "ablations/lora_architecture_loss.csv"),
        "stage2_architecture_loss": load_csv_source(results_root / "ablations/stage2_architecture_loss.csv"),
        "quality_efficiency": load_csv_source(results_root / "efficiency/quality_efficiency.csv"),
        "generalization": load_csv_source(results_root / "generalization/summary.csv"),
    }

    sources["timing_summary"] = summarize_timing(sources["timing_raw"])
    sources["wan50_endpoint_paired_statistics"] = summarize_paired_metrics(sources["wan50_endpoint_samples"])
    sources["distill_transfer_paired_statistics"] = summarize_paired_metrics(
        sources["distill_480lora_transfer_368p_samples"]
    )
    factorials = {
        "wan50": inspect_factorial(wan_factorial, expected_family="wan50"),
        "distill4": inspect_factorial(distill_factorial, expected_family="distill4"),
    }
    ablations = {
        "distill_renoise": inspect_video_set(
            results_root / "ablation_distill_renoise/videos", "*.mp4", expected_min=2
        ),
        "wan50_handoff_sweep": inspect_video_set(
            project_root / "outputs/changing_resolution_clean_360p_stage2_three_way_step_sweep/compare",
            "*.mp4",
            expected_min=50,
        ),
        "direct_stage3_compare": inspect_video_set(
            project_root / "outputs/changing_resolution_stage3_three_model_compare/compare",
            "*.mp4",
            expected_min=10,
        ),
    }
    external = {
        "vbench": {
            "wan50": inspect_json_files(wan_factorial / "metrics", "vbench*.json"),
            "distill4": inspect_json_files(distill_factorial / "metrics", "vbench*.json"),
        },
        "human_review": {
            "wan50": inspect_human_review(wan_factorial),
            "distill4": inspect_human_review(distill_factorial),
        },
    }
    task_audit = audit_manifest_tasks(manifest, environment)
    factorial_task = next((task for task in manifest["tasks"] if task["id"] == "distill_factorial"), {})
    if "distill_480lora_transfer_sweep" in factorial_task.get("depends_on", []):
        for row in task_audit:
            if row["id"] in {"distill_lora3_lmdb", "distill_lora3_checkpoint"} and row["status"] != "complete":
                row["raw_status"] = row["status"]
                row["status"] = "not_required"
                row["evidence"] = "superseded by validated 480p-LoRA transfer route; " + row["evidence"]

    issues: list[dict[str, str]] = []
    for name, source in sources.items():
        if source.get("status") != "complete":
            issues.append(issue(f"sources.{name}", source.get("status", "invalid"), source.get("message", source["path"])))
    for name, result in factorials.items():
        if result["status"] != "complete":
            issues.append(issue(f"factorials.{name}", result["status"], "; ".join(result.get("issues", []))))
    for name, result in ablations.items():
        if result["status"] != "complete":
            issues.append(issue(f"ablations.{name}", result["status"], result.get("message", "")))
    for family, result in external["vbench"].items():
        if result["status"] != "complete":
            issues.append(issue(f"external.vbench.{family}", result["status"], result.get("message", "")))
    for family, result in external["human_review"].items():
        if result["completed_status"] != "complete":
            issues.append(
                issue(
                    f"external.human_review.{family}",
                    result["completed_status"],
                    result.get("completed_message", ""),
                )
            )

    return {
        "schema_version": 2,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_root": str(project_root),
        "canonical_results_root": str(results_root),
        "output_root": str(output_root),
        "final_configuration": {
            "distill_lora_checkpoint": environment["DISTILL_LORA_480_CKPT"],
            "distill_lora_strength": environment["DISTILL_LORA_STRENGTH"],
            "distill_stage2_checkpoint": environment["DISTILL_STAGE2_CKPT"],
        },
        "task_audit": task_audit,
        "sources": sources,
        "factorials": factorials,
        "ablations": ablations,
        "external": external,
        "issues": issues,
    }


def load_csv_source(path: Path) -> dict[str, Any]:
    base = {"path": str(path)}
    if not path.is_file():
        return {**base, "status": "missing", "row_count": 0, "rows": [], "message": "file not found"}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeError) as exc:
        return {**base, "status": "invalid", "row_count": 0, "rows": [], "message": str(exc)}
    if not rows:
        return {**base, "status": "invalid", "row_count": 0, "rows": [], "message": "CSV has no data rows"}
    return {**base, "status": "complete", "row_count": len(rows), "columns": list(rows[0]), "rows": rows}


def load_json_source(path: Path) -> dict[str, Any]:
    base = {"path": str(path)}
    if not path.is_file():
        return {**base, "status": "missing", "data": None, "message": "file not found"}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {**base, "status": "invalid", "data": None, "message": str(exc)}
    return {**base, "status": "complete", "data": data}


def summarize_timing(source: dict[str, Any]) -> dict[str, Any]:
    path = f"derived from {source['path']}"
    if source["status"] != "complete":
        return {"path": path, "status": source["status"], "rows": [], "row_count": 0, "message": source.get("message", "")}
    grouped: dict[str, list[float]] = defaultdict(list)
    invalid_rows = 0
    for row in source["rows"]:
        try:
            value = float(row["elapsed_sec"])
            if not math.isfinite(value) or value < 0:
                raise ValueError
            grouped[row["case"]].append(value)
        except (KeyError, TypeError, ValueError):
            invalid_rows += 1
    if not grouped:
        return {"path": path, "status": "invalid", "rows": [], "row_count": 0, "message": "no valid timing rows"}
    baseline = statistics.mean(grouped["direct_720p"]) if grouped.get("direct_720p") else None
    rows = []
    for case_name, values in sorted(grouped.items()):
        case_mean = statistics.mean(values)
        rows.append(
            {
                "case": case_name,
                "repeats": len(values),
                "mean_sec": case_mean,
                "std_sec": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "median_sec": statistics.median(values),
                "min_sec": min(values),
                "max_sec": max(values),
                "speedup_vs_direct": baseline / case_mean if baseline is not None and case_mean > 0 else None,
            }
        )
    status = "complete" if invalid_rows == 0 else "invalid"
    message = "" if invalid_rows == 0 else f"{invalid_rows} timing row(s) could not be parsed"
    return {"path": path, "status": status, "row_count": len(rows), "rows": rows, "message": message}


def summarize_paired_metrics(
    source: dict[str, Any], confidence: float = 0.95, bootstrap_samples: int = 10000, seed: int = 202707
) -> dict[str, Any]:
    path = f"derived from {source['path']}"
    if source["status"] != "complete":
        return {"path": path, "status": source["status"], "rows": [], "row_count": 0, "message": source.get("message", "")}
    columns = set(source.get("columns", []))
    metrics = sorted(
        column.removeprefix("original_")
        for column in columns
        if column.startswith("original_")
        and f"lora_{column.removeprefix('original_')}" in columns
        and column.removeprefix("original_") not in {"path", "wins"}
    )
    rows = []
    for metric in metrics:
        pairs = []
        for source_row in source["rows"]:
            try:
                original = float(source_row[f"original_{metric}"])
                lora = float(source_row[f"lora_{metric}"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(original) and math.isfinite(lora):
                pairs.append((original, lora))
        if not pairs:
            continue
        lower_is_better = metric not in {"psnr", "ssim", "vmaf"}
        deltas = [lora - original for original, lora in pairs]
        improvements = [-delta if lower_is_better else delta for delta in deltas]
        metric_seed = seed + sum((index + 1) * ord(char) for index, char in enumerate(metric))
        rng = random.Random(metric_seed)
        boot = [statistics.mean(rng.choices(improvements, k=len(improvements))) for _ in range(bootstrap_samples)]
        boot.sort()
        alpha = (1.0 - confidence) / 2.0
        wins = sum(value > 0 for value in improvements)
        losses = sum(value < 0 for value in improvements)
        rows.append(
            {
                "metric": metric,
                "better": "lower" if lower_is_better else "higher",
                "samples": len(pairs),
                "original_mean": statistics.mean(original for original, _ in pairs),
                "lora_mean": statistics.mean(lora for _, lora in pairs),
                "delta_lora_minus_original_mean": statistics.mean(deltas),
                "oriented_improvement_mean": statistics.mean(improvements),
                "oriented_improvement_std": statistics.pstdev(improvements) if len(improvements) > 1 else 0.0,
                "bootstrap_ci_low": quantile(boot, alpha),
                "bootstrap_ci_high": quantile(boot, 1.0 - alpha),
                "confidence": confidence,
                "wins": wins,
                "losses": losses,
                "ties": len(improvements) - wins - losses,
                "two_sided_sign_test_p": sign_test_p(wins, losses),
            }
        )
    if not rows:
        return {"path": path, "status": "invalid", "rows": [], "row_count": 0, "message": "no paired metric columns found"}
    return {"path": path, "status": "complete", "rows": rows, "row_count": len(rows)}


def quantile(values: list[float], probability: float) -> float:
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def sign_test_p(wins: int, losses: int) -> float:
    count = wins + losses
    if count == 0:
        return 1.0
    tail = sum(math.comb(count, k) for k in range(0, min(wins, losses) + 1)) / (2**count)
    return min(1.0, 2.0 * tail)


def inspect_factorial(root: Path, expected_family: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "root": str(root),
        "status": "missing",
        "family": expected_family,
        "expected_total": 0,
        "valid_total": 0,
        "cases": {},
        "issues": [],
        "provenance": {},
    }
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        result["issues"].append("run_manifest.json is missing")
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result["status"] = "invalid"
        result["issues"].append(f"invalid run manifest: {exc}")
        return result
    required = {"family", "seed_base", "prompt_offset", "prompts", "cases"}
    missing_keys = sorted(required - set(manifest))
    if missing_keys:
        result["status"] = "invalid"
        result["issues"].append(f"manifest keys missing: {', '.join(missing_keys)}")
        return result
    if manifest["family"] != expected_family:
        result["issues"].append(f"family mismatch: expected {expected_family}, found {manifest['family']}")
    prompts = manifest["prompts"]
    prompt_offset = int(manifest["prompt_offset"])
    seed_base = int(manifest["seed_base"])
    expected_total = 0
    valid_total = 0
    for case in manifest["cases"]:
        case_name = case["name"]
        expected_names = {
            f"{case_name}_{index:02d}_seed{seed_base + index}.mp4"
            for index in range(prompt_offset, prompt_offset + len(prompts))
        }
        case_dir = root / "videos" / case_name
        actual_paths = {path.name: path for path in case_dir.glob("*.mp4")} if case_dir.is_dir() else {}
        nonempty_names = {name for name, path in actual_paths.items() if path.stat().st_size > 0}
        zero_byte = sorted(set(actual_paths) - nonempty_names)
        missing = sorted(expected_names - nonempty_names)
        extra = sorted(nonempty_names - expected_names)
        config_issues, config_provenance = inspect_factorial_config(root / "configs" / f"{case_name}.json", case)
        complete = not missing and not zero_byte and not extra and not config_issues
        result["cases"][case_name] = {
            "status": "complete" if complete else "invalid",
            "expected": len(expected_names),
            "valid": len(expected_names & nonempty_names),
            "missing": missing,
            "extra": extra,
            "zero_byte": zero_byte,
            "config_issues": config_issues,
            "config_provenance": config_provenance,
        }
        expected_total += len(expected_names)
        valid_total += len(expected_names & nonempty_names)
        if missing:
            result["issues"].append(f"{case_name}: {len(missing)} expected video(s) missing")
        if extra:
            result["issues"].append(f"{case_name}: {len(extra)} unexpected video(s)")
        if zero_byte:
            result["issues"].append(f"{case_name}: {len(zero_byte)} zero-byte video(s)")
        result["issues"].extend(f"{case_name}: {message}" for message in config_issues)
    result["expected_total"] = expected_total
    result["valid_total"] = valid_total
    result["provenance"] = {
        "manifest_path": str(manifest_path),
        "seed_base": seed_base,
        "prompt_offset": prompt_offset,
        "prompt_count": len(prompts),
        "lora_artifacts": manifest.get("lora_artifacts", {}),
        "stage2_artifact": manifest.get("stage2_artifact"),
        "settings": manifest.get("settings", {}),
        "reuse_roots": manifest.get("reuse_roots", []),
    }
    result["status"] = "complete" if not result["issues"] and valid_total == expected_total else "invalid"
    return result


def inspect_factorial_config(path: Path, case: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    if not path.is_file():
        return ["config JSON is missing"], {"path": str(path)}
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"invalid config JSON: {exc}"], {"path": str(path)}
    issues = []
    expected_lora = case["handoff"] == "lora"
    expected_stage2 = case["resizer"] == "stage2"
    has_lora = bool(config.get("lora_configs"))
    has_stage2 = bool(config.get("wan_clean_resizer_ckpt"))
    if has_lora != expected_lora:
        issues.append(f"LoRA config mismatch (expected={expected_lora}, found={has_lora})")
    if has_stage2 != expected_stage2:
        issues.append(f"Stage2 config mismatch (expected={expected_stage2}, found={has_stage2})")
    if config.get("compare_name") != case["name"]:
        issues.append(f"compare_name mismatch: {config.get('compare_name')!r}")
    expected_step = int(case["step"])
    if list(config.get("changing_resolution_steps", [])) != [expected_step]:
        issues.append(f"handoff step mismatch: {config.get('changing_resolution_steps')!r}")
    if expected_lora and list(config.get("lora_active_steps", [])) != [expected_step]:
        issues.append(f"LoRA active step mismatch: {config.get('lora_active_steps')!r}")
    lora = config.get("lora_configs", [{}])[0] if has_lora else {}
    provenance = {
        "path": str(path),
        "lora_checkpoint": lora.get("path"),
        "lora_strength": lora.get("strength"),
        "stage2_checkpoint": config.get("wan_clean_resizer_ckpt"),
        "stage2_train_config": config.get("wan_clean_resizer_train_config"),
        "stage2_use_ema": config.get("wan_clean_resizer_use_ema"),
        "renoise_mode": config.get("wan_distill_bridge_renoise_mode"),
    }
    for key in ("lora_checkpoint", "stage2_checkpoint", "stage2_train_config"):
        value = provenance[key]
        if value and not Path(str(value)).is_file():
            issues.append(f"configured {key} does not exist: {value}")
    return issues, provenance


def inspect_video_set(root: Path, pattern: str, expected_min: int) -> dict[str, Any]:
    files = sorted(path for path in root.glob(pattern) if path.is_file() and path.stat().st_size > 0) if root.is_dir() else []
    status = "complete" if len(files) >= expected_min else "missing"
    return {
        "root": str(root),
        "status": status,
        "count": len(files),
        "expected_min": expected_min,
        "files": [str(path) for path in files],
        "message": f"found {len(files)}, expected at least {expected_min}",
    }


def inspect_json_files(root: Path, pattern: str) -> dict[str, Any]:
    files = sorted(root.glob(pattern)) if root.is_dir() else []
    parsed = []
    invalid = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            parsed.append({"path": str(path), "numeric_metrics": flatten_numeric(data)})
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            invalid.append({"path": str(path), "error": str(exc)})
    status = "complete" if parsed and not invalid else "invalid" if invalid else "missing"
    return {
        "root": str(root),
        "status": status,
        "files": parsed,
        "invalid": invalid,
        "message": "no canonical VBench JSON found" if not files else f"{len(parsed)} valid, {len(invalid)} invalid",
    }


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric(item, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        result[prefix or "value"] = float(value)
    return result


def inspect_human_review(root: Path) -> dict[str, Any]:
    ballot = load_csv_source(root / "review/human_ratings.csv")
    completed = load_csv_source(root / "review/human_ratings_completed.csv")
    required_suffix = "_winner_A_B_tie"
    completed_rows = 0
    if completed["status"] == "complete":
        rating_columns = [column for column in completed.get("columns", []) if column.endswith(required_suffix)]
        for row in completed["rows"]:
            if rating_columns and all(str(row.get(column, "")).strip() for column in rating_columns):
                completed_rows += 1
    completed_status = "complete" if completed["status"] == "complete" and completed_rows == completed["row_count"] else completed["status"]
    if completed["status"] == "complete" and completed_rows != completed["row_count"]:
        completed_status = "invalid"
    return {
        "root": str(root),
        "ballot_status": ballot["status"],
        "ballot_rows": ballot["row_count"],
        "completed_status": completed_status,
        "completed_rows": completed_rows,
        "completed_total_rows": completed["row_count"],
        "completed_message": completed.get("message", "")
        or f"{completed_rows}/{completed['row_count']} rows have every winner field filled",
    }


def audit_manifest_tasks(manifest: dict[str, Any], environment: dict[str, str]) -> list[dict[str, str]]:
    try:
        try:
            from .run_experiments import audit_task
        except ImportError:
            from run_experiments import audit_task
    except ImportError as exc:
        return [{"id": "manifest_audit", "group": "", "status": "invalid", "evidence": str(exc)}]
    return [audit_task(task, environment) for task in manifest["tasks"]]


def build_environment(project_root: Path, defaults: dict[str, str]) -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("PROJECT_ROOT", str(project_root))
    for key, value in defaults.items():
        environment.setdefault(key, expand(value, environment))
    return environment


def expand(value: str, environment: dict[str, str]) -> str:
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
    previous = None
    while value != previous:
        previous = value
        value = pattern.sub(lambda match: environment.get(match.group(1), match.group(0)), value)
    return value


def write_outputs(output_root: Path, inventory: dict[str, Any]) -> None:
    (output_root / "result_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "paper_tables.md").write_text(render_markdown(inventory), encoding="utf-8")
    compiled = output_root / "compiled_tables"
    compiled.mkdir(parents=True, exist_ok=True)
    source_names = {
        "operator_480p": operator_rows(inventory["sources"]["operator_480p"]),
        "operator_368p": operator_rows(inventory["sources"]["operator_368p"]),
        "wan50_endpoint_step45": inventory["sources"]["wan50_endpoint_step45"]["rows"],
        "wan50_lora_strength": inventory["sources"]["wan50_lora_strength"]["rows"],
        "distill_checkpoint_metrics": inventory["sources"]["distill_checkpoint_metrics"]["rows"],
        "distill_checkpoint_rank_l1": inventory["sources"]["distill_checkpoint_rank_l1"]["rows"],
        "distill_368p_lora_strength": inventory["sources"]["distill_368p_lora_strength"]["rows"],
        "distill_480lora_transfer_368p": inventory["sources"]["distill_480lora_transfer_368p"]["rows"],
        "wan50_endpoint_paired_statistics": inventory["sources"]["wan50_endpoint_paired_statistics"]["rows"],
        "distill_transfer_paired_statistics": inventory["sources"]["distill_transfer_paired_statistics"]["rows"],
        "timing_summary": inventory["sources"]["timing_summary"]["rows"],
        "lora_architecture_loss": inventory["sources"]["lora_architecture_loss"]["rows"],
        "stage2_architecture_loss": inventory["sources"]["stage2_architecture_loss"]["rows"],
        "quality_efficiency": inventory["sources"]["quality_efficiency"]["rows"],
        "generalization": inventory["sources"]["generalization"]["rows"],
        "factorial_coverage": factorial_coverage_rows(inventory["factorials"]),
        "task_audit": inventory["task_audit"],
    }
    for name, rows in source_names.items():
        write_csv(compiled / f"{name}.csv", rows)


def operator_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    if source["status"] != "complete" or not isinstance(source.get("data"), dict):
        return []
    return list(source["data"].get("metrics", []))


def factorial_coverage_rows(factorials: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for family, result in factorials.items():
        for case_name, case in result["cases"].items():
            rows.append(
                {
                    "family": family,
                    "case": case_name,
                    "status": case["status"],
                    "valid": case["valid"],
                    "expected": case["expected"],
                    "missing": len(case["missing"]),
                    "extra": len(case["extra"]),
                    "zero_byte": len(case["zero_byte"]),
                    "config_issues": len(case["config_issues"]),
                }
            )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("status\nMISSING\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(inventory: dict[str, Any]) -> str:
    lines = [
        "# AAAI-27 Experiment Data",
        "",
        f"Generated: `{inventory['generated_at_utc']}`",
        "",
        "Missing or invalid evidence is never imputed.",
        "",
        "## Final configuration",
        "",
        f"- Distill LoRA: `{inventory['final_configuration']['distill_lora_checkpoint']}`",
        f"- Distill LoRA strength: `{inventory['final_configuration']['distill_lora_strength']}`",
        f"- Distill Stage2: `{inventory['final_configuration']['distill_stage2_checkpoint']}`",
        "",
    ]
    lines.extend(render_task_summary(inventory["task_audit"]))
    lines.extend(render_operator("480p → 720p", inventory["sources"]["operator_480p"]))
    lines.extend(render_operator("368p → 720p", inventory["sources"]["operator_368p"]))
    lines.extend(render_metric_source("Wan50 endpoint correction", inventory["sources"]["wan50_endpoint_step45"]))
    lines.extend(
        render_paired_statistics(
            "Wan50 endpoint paired statistics", inventory["sources"]["wan50_endpoint_paired_statistics"]
        )
    )
    lines.extend(render_metric_source("Wan50 LoRA strength", inventory["sources"]["wan50_lora_strength"], group="strength"))
    lines.extend(
        render_metric_source(
            "Distill checkpoint selection (L1 ranking)", inventory["sources"]["distill_checkpoint_rank_l1"], group="checkpoint"
        )
    )
    lines.extend(
        render_paired_statistics(
            "Distill transfer paired statistics", inventory["sources"]["distill_transfer_paired_statistics"]
        )
    )
    lines.extend(render_csv_source("Distill checkpoint metrics (all metrics)", inventory["sources"]["distill_checkpoint_metrics"]))
    lines.extend(
        render_metric_source(
            "Distill 480p-LoRA strength at 368×640",
            inventory["sources"]["distill_368p_lora_strength"],
            group="strength",
        )
    )
    lines.extend(
        render_metric_source(
            "Distill 480p-LoRA + Stage2 transfer at 368×640",
            inventory["sources"]["distill_480lora_transfer_368p"],
            group="strength",
        )
    )
    lines.extend(render_factorials(inventory["factorials"]))
    lines.extend(render_timing(inventory["sources"]["timing_summary"]))
    lines.extend(render_csv_source("LoRA architecture/loss ablation", inventory["sources"]["lora_architecture_loss"]))
    lines.extend(render_csv_source("Stage2 architecture/loss ablation", inventory["sources"]["stage2_architecture_loss"]))
    lines.extend(render_csv_source("Peak memory and quality–efficiency", inventory["sources"]["quality_efficiency"]))
    lines.extend(render_csv_source("Generalization and failure cases", inventory["sources"]["generalization"]))
    lines.extend(render_ablations(inventory["ablations"]))
    lines.extend(render_external(inventory["external"]))
    lines.extend(render_issues(inventory["issues"]))
    return "\n".join(lines).rstrip() + "\n"


def render_task_summary(rows: list[dict[str, str]]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["status"]] += 1
    lines = ["## Manifest audit", "", "| Status | Tasks |", "|---|---:|"]
    for status in sorted(counts):
        lines.append(f"| {md(status)} | {counts[status]} |")
    missing = [row for row in rows if row["status"] not in {"complete", "not_required"}]
    if missing:
        lines.extend(["", "| Incomplete task | Group | Evidence |", "|---|---|---|"])
        for row in missing:
            lines.append(f"| {md(row['id'])} | {md(row['group'])} | {md(row['evidence'])} |")
    return lines + [""]


def render_operator(label: str, source: dict[str, Any]) -> list[str]:
    lines = [f"## Operator: {label}", ""]
    rows = operator_rows(source)
    if not rows:
        return lines + [f"**{source['status'].upper()}** — `{source['path']}`", ""]
    lines.extend(["| Metric | Better | Samples | Interpolation | Stage2 | Δ Stage2−interp | Win rate |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(
            f"| {md(row.get('metric'))} | {md(row.get('better'))} | {fmt(row.get('samples'))} | "
            f"{fmt(row.get('interp_mean'))} | {fmt(row.get('trained_mean'))} | {fmt(row.get('delta_mean'))} | "
            f"{fmt(row.get('win_rate'))} |"
        )
    return lines + [""]


def render_metric_source(title: str, source: dict[str, Any], group: str | None = None) -> list[str]:
    lines = [f"## {title}", ""]
    rows = source.get("rows", [])
    if source["status"] != "complete" or not rows:
        return lines + [f"**{source['status'].upper()}** — `{source['path']}`", ""]
    group_column = group if group and any(group in row for row in rows) else None
    columns = ([group_column] if group_column else []) + [
        "metric",
        "better",
        "samples",
        "original_mean",
        "lora_mean",
        "delta_lora_minus_original_mean",
        "lora_win_rate",
    ]
    # Checkpoint rank CSVs use the same core columns but omit samples in some versions.
    columns = [column for column in columns if any(column in row for row in rows)]
    lines.append("| " + " | ".join(md(column) for column in columns) + " |")
    lines.append("|" + "|".join("---:" if column not in {"metric", "better", group_column} else "---" for column in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(md(row.get(column, "")) if column in {"metric", "better", group_column} else fmt(row.get(column)) for column in columns) + " |")
    return lines + [""]


def render_factorials(factorials: dict[str, Any]) -> list[str]:
    lines = ["## Factorial coverage and provenance", "", "| Family | Case | Status | Valid | Expected | Missing | Extra | Config issues |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for family, result in factorials.items():
        if not result["cases"]:
            lines.append(f"| {md(family)} | MISSING | {md(result['status'])} | 0 | 0 | 0 | 0 | 0 |")
            continue
        for case_name, case in result["cases"].items():
            lines.append(
                f"| {md(family)} | {md(case_name)} | {md(case['status'])} | {case['valid']} | {case['expected']} | "
                f"{len(case['missing'])} | {len(case['extra'])} | {len(case['config_issues'])} |"
            )
    lines.extend(["", "### Factorial configuration", "", "| Family | LoRA checkpoint(s) | Strength(s) | Stage2 checkpoint(s) | Reuse roots |", "|---|---|---|---|---|"])
    for family, result in factorials.items():
        configs = [case.get("config_provenance", {}) for case in result["cases"].values()]
        loras = sorted({str(item["lora_checkpoint"]) for item in configs if item.get("lora_checkpoint")})
        strengths = sorted({str(item["lora_strength"]) for item in configs if item.get("lora_strength")})
        stage2 = sorted({str(item["stage2_checkpoint"]) for item in configs if item.get("stage2_checkpoint")})
        reuse = result.get("provenance", {}).get("reuse_roots", [])
        lines.append(
            f"| {md(family)} | {md(', '.join(loras) or 'MISSING')} | {md(', '.join(strengths) or 'MISSING')} | "
            f"{md(', '.join(stage2) or 'MISSING')} | {md(', '.join(map(str, reuse)) or 'unrecorded')} |"
        )
    return lines + [""]


def render_paired_statistics(title: str, source: dict[str, Any]) -> list[str]:
    lines = [f"## {title}", ""]
    rows = source.get("rows", [])
    if source["status"] != "complete" or not rows:
        return lines + [f"**{source['status'].upper()}** — {md(source.get('message', source['path']))}", ""]
    lines.extend(
        [
            "| Metric | Better | N | Improvement | 95% bootstrap CI | Wins–losses–ties | Sign-test p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        interval = f"[{fmt(row['bootstrap_ci_low'])}, {fmt(row['bootstrap_ci_high'])}]"
        record = f"{row['wins']}–{row['losses']}–{row['ties']}"
        lines.append(
            f"| {md(row['metric'])} | {row['better']} | {row['samples']} | {fmt(row['oriented_improvement_mean'])} | "
            f"{interval} | {record} | {fmt(row['two_sided_sign_test_p'])} |"
        )
    return lines + [""]


def render_csv_source(title: str, source: dict[str, Any]) -> list[str]:
    lines = [f"## {title}", ""]
    rows = source.get("rows", [])
    if source["status"] != "complete" or not rows:
        return lines + [f"**{source['status'].upper()}** — `{source['path']}`", ""]
    columns = list(source.get("columns", rows[0].keys()))
    lines.append("| " + " | ".join(md(column) for column in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    return lines + [""]


def render_timing(source: dict[str, Any]) -> list[str]:
    lines = ["## Quality–efficiency timing", ""]
    if source["status"] != "complete":
        return lines + [f"**{source['status'].upper()}** — {md(source.get('message', ''))}", ""]
    lines.extend(["| Case | Repeats | Mean s | Std s | Median s | Min s | Max s | Speedup vs direct |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in source["rows"]:
        lines.append(
            f"| {md(row['case'])} | {row['repeats']} | {fmt(row['mean_sec'])} | {fmt(row['std_sec'])} | "
            f"{fmt(row['median_sec'])} | {fmt(row['min_sec'])} | {fmt(row['max_sec'])} | {fmt(row['speedup_vs_direct'])} |"
        )
    return lines + [""]


def render_ablations(ablations: dict[str, Any]) -> list[str]:
    lines = ["## Executable ablation/baseline coverage", "", "| Experiment | Status | Files | Expected minimum |", "|---|---|---:|---:|"]
    for name, result in ablations.items():
        lines.append(f"| {md(name)} | {md(result['status'])} | {result['count']} | {result['expected_min']} |")
    return lines + [""]


def render_external(external: dict[str, Any]) -> list[str]:
    lines = ["## External evidence", "", "| Family | VBench | VBench files | Blind ballot rows | Completed rating rows | Human status |", "|---|---|---:|---:|---:|---|"]
    for family in ("wan50", "distill4"):
        vbench = external["vbench"][family]
        human = external["human_review"][family]
        lines.append(
            f"| {family} | {vbench['status']} | {len(vbench['files'])} | {human['ballot_rows']} | "
            f"{human['completed_rows']}/{human['completed_total_rows']} | {human['completed_status']} |"
        )
    numeric_rows = []
    for family in ("wan50", "distill4"):
        for file in external["vbench"][family]["files"]:
            for metric, value in sorted(file["numeric_metrics"].items()):
                numeric_rows.append((family, file["path"], metric, value))
    if numeric_rows:
        lines.extend(["", "### VBench numeric metrics", "", "| Family | File | Metric | Value |", "|---|---|---|---:|"])
        for family, path, metric, value in numeric_rows:
            lines.append(f"| {family} | {md(path)} | {md(metric)} | {fmt(value)} |")
    return lines + [""]


def render_issues(issues: list[dict[str, str]]) -> list[str]:
    lines = ["## Missing or invalid evidence", ""]
    if not issues:
        return lines + ["None.", ""]
    lines.extend(["| Item | Status | Detail |", "|---|---|---|"])
    for item in issues:
        lines.append(f"| {md(item['item'])} | {md(item['status'])} | {md(item['detail'])} |")
    return lines + [""]


def issue(item: str, status: str, detail: str) -> dict[str, str]:
    return {"item": item, "status": status, "detail": detail}


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return md(value)
    if not math.isfinite(numeric):
        return str(numeric)
    return f"{numeric:.6g}"


def md(value: Any) -> str:
    return str("" if value is None else value).replace("|", "\\|").replace("\n", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile strict AAAI-27 experiment inventory and paper tables.")
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-root")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any canonical evidence is missing or invalid.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
