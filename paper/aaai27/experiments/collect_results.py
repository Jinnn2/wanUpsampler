from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "outputs/aaai27_experiments"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    inventory = {
        "operator_480p": operator_summary(REPO_ROOT / "outputs/changing_resolution_operator_compare_stage2/tables/summary_val.json"),
        "operator_368p": operator_summary(OUT_ROOT / "operator_368p/tables/summary_val.json"),
        "endpoint_50step": csv_rows(
            REPO_ROOT
            / "outputs/changing_resolution_tail_skip_lora_clean_pred_compare_360p_368x640/metrics/original_lora_teacher_summary.csv"
        ),
        "factorial_wan50": factorial_coverage(OUT_ROOT / "factorial_wan50"),
        "factorial_distill4": factorial_coverage(OUT_ROOT / "factorial_distill4"),
        "timing": csv_rows(REPO_ROOT / "outputs/changing_resolution_time_compare/time_summary.csv"),
        "vbench": find_files(OUT_ROOT, "vbench*.json"),
        "human_completed": find_files(OUT_ROOT, "human_ratings_completed.csv"),
    }
    (OUT_ROOT / "result_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_ROOT / "paper_tables.md").write_text(render_markdown(inventory), encoding="utf-8")
    print(f"Inventory: {OUT_ROOT / 'result_inventory.json'}")
    print(f"Tables   : {OUT_ROOT / 'paper_tables.md'}")


def operator_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def factorial_coverage(root: Path) -> dict[str, Any] | None:
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = {}
    for case in manifest["cases"]:
        case_name = case["name"]
        files = sorted((root / "videos" / case_name).glob("*.mp4"))
        cases[case_name] = {
            "count": len([path for path in files if path.stat().st_size > 0]),
            "expected": len(manifest["prompts"]),
        }
    return {"family": manifest["family"], "cases": cases}


def find_files(root: Path, pattern: str) -> list[str]:
    return [str(path.relative_to(REPO_ROOT)) for path in sorted(root.rglob(pattern)) if path.is_file()]


def render_markdown(inventory: dict[str, Any]) -> str:
    lines = ["# AAAI-27 Experiment Data", "", "Generated from canonical result files. Missing values remain explicit.", ""]
    lines.extend(render_operator("480p -> 720p", inventory["operator_480p"]))
    lines.extend(render_operator("368p -> 720p", inventory["operator_368p"]))
    lines.extend(["## Factorial Coverage", "", "| Family | Case | Videos | Expected |", "|---|---|---:|---:|"])
    for key in ("factorial_wan50", "factorial_distill4"):
        item = inventory[key]
        if not item:
            lines.append(f"| {key} | MISSING | 0 | 0 |")
            continue
        for case, coverage in item["cases"].items():
            lines.append(f"| {item['family']} | {case} | {coverage['count']} | {coverage['expected']} |")
    lines.extend(["", "## Remaining External Evidence", ""])
    lines.append(f"- VBench result files: {len(inventory['vbench'])}")
    lines.append(f"- Completed human-rating files: {len(inventory['human_completed'])}")
    lines.append(f"- Timing rows: {len(inventory['timing'] or [])}")
    lines.append("")
    return "\n".join(lines)


def render_operator(label: str, summary: dict[str, Any] | None) -> list[str]:
    lines = [f"## Operator: {label}", ""]
    if not summary:
        return lines + ["MISSING", ""]
    lines.extend(["| Metric | Interpolation | Stage2 | Delta | Win rate |", "|---|---:|---:|---:|---:|"])
    for row in summary.get("metrics", []):
        lines.append(
            f"| {row['metric']} | {row['interp_mean']} | {row['trained_mean']} | {row['delta_mean']} | {row['win_rate']} |"
        )
    lines.append("")
    return lines


if __name__ == "__main__":
    main()
