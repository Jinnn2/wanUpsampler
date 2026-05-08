from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_COLUMNS = [
    "sample_index",
    "sample_id",
    "interp_latent_l1",
    "trained_latent_l1",
    "latent_l1_delta_trained_minus_interp",
    "interp_psnr",
    "trained_psnr",
    "psnr_delta_trained_minus_interp",
    "interp_ssim",
    "trained_ssim",
    "ssim_delta_trained_minus_interp",
    "interp_lpips",
    "trained_lpips",
    "lpips_delta_trained_minus_interp",
    "interp_temporal_l1",
    "trained_temporal_l1",
    "temporal_l1_delta_trained_minus_interp",
    "compare",
]

SUMMARY_METRICS = [
    ("latent_l1", "lower"),
    ("psnr", "higher"),
    ("ssim", "higher"),
    ("lpips", "lower"),
    ("temporal_l1", "lower"),
]


def main() -> None:
    args = parse_args()
    metrics_path = resolve_metrics_path(args.input, args.split)
    rows = load_rows(metrics_path)
    if not rows:
        raise SystemExit(f"No metric rows found in {metrics_path}")

    rows = [normalize_row(row) for row in rows]
    out_dir = Path(args.out_dir) if args.out_dir else metrics_path.parent / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = select_sample_rows(rows, args.top_k)
    sample_csv = out_dir / f"samples_{args.split}.csv"
    sample_md = out_dir / f"samples_{args.split}.md"
    summary_csv = out_dir / f"summary_{args.split}.csv"
    summary_md = out_dir / f"summary_{args.split}.md"
    summary_json = out_dir / f"summary_{args.split}.json"

    write_csv(sample_csv, sample_rows, DEFAULT_COLUMNS)
    write_markdown_table(sample_md, sample_rows, DEFAULT_COLUMNS, title="Operator Compare Samples")

    summary_rows = build_summary(rows)
    write_csv(summary_csv, summary_rows, list(summary_rows[0].keys()))
    write_markdown_table(summary_md, summary_rows, list(summary_rows[0].keys()), title="Operator Compare Summary")
    summary_json.write_text(json.dumps({"num_samples": len(rows), "metrics": summary_rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input metrics: {metrics_path}")
    print(f"Samples CSV : {sample_csv}")
    print(f"Samples MD  : {sample_md}")
    print(f"Summary CSV : {summary_csv}")
    print(f"Summary MD  : {summary_md}")
    print(f"Summary JSON: {summary_json}")


def resolve_metrics_path(input_path: str, split: str) -> Path:
    path = Path(input_path)
    if path.is_file():
        return path

    merged = path / f"metrics_{split}.jsonl"
    if merged.is_file():
        return merged

    part_files = sorted(path.glob(f"part_*/metrics_{split}_*.jsonl"))
    if not part_files:
        raise FileNotFoundError(f"No metrics_{split}.jsonl or part metrics found under {path}")

    merged.parent.mkdir(parents=True, exist_ok=True)
    with merged.open("w", encoding="utf-8") as out:
        for part in part_files:
            text = part.read_text(encoding="utf-8-sig")
            if text and not text.endswith("\n"):
                text += "\n"
            out.write(text)
    return merged


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["temporal_l1_delta_trained_minus_interp"] = finite_or_nan(row.get("trained_temporal_l1")) - finite_or_nan(
        row.get("interp_temporal_l1")
    )
    paths = row.get("paths")
    if isinstance(paths, dict):
        row["compare"] = paths.get("compare", "")
    else:
        row["compare"] = ""
    return row


def select_sample_rows(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: finite_or_nan(row.get("psnr_delta_trained_minus_interp")), reverse=True)
    if top_k > 0:
        sorted_rows = sorted_rows[:top_k]
    return [{column: format_value(row.get(column, "")) for column in DEFAULT_COLUMNS} for row in sorted_rows]


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    summary = []
    for name, direction in SUMMARY_METRICS:
        interp_key = f"interp_{name}" if name != "latent_l1" else "interp_latent_l1"
        trained_key = f"trained_{name}" if name != "latent_l1" else "trained_latent_l1"
        delta_key = f"{name}_delta_trained_minus_interp" if name != "latent_l1" else "latent_l1_delta_trained_minus_interp"

        interp_values = finite_values(row.get(interp_key) for row in rows)
        trained_values = finite_values(row.get(trained_key) for row in rows)
        delta_values = finite_values(row.get(delta_key) for row in rows)
        win_count = count_wins(rows, interp_key, trained_key, direction)

        summary.append(
            {
                "metric": name,
                "better": direction,
                "samples": str(len(rows)),
                "interp_mean": format_number(mean(interp_values)),
                "trained_mean": format_number(mean(trained_values)),
                "delta_mean": format_number(mean(delta_values)),
                "delta_std": format_number(pstdev(delta_values) if len(delta_values) > 1 else 0.0),
                "win_rate": format_number(win_count / len(rows)),
                "wins": f"{win_count}/{len(rows)}",
            }
        )
    return summary


def count_wins(rows: list[dict[str, Any]], interp_key: str, trained_key: str, direction: str) -> int:
    wins = 0
    for row in rows:
        interp = finite_or_nan(row.get(interp_key))
        trained = finite_or_nan(row.get(trained_key))
        if math.isnan(interp) or math.isnan(trained):
            continue
        if direction == "higher" and trained > interp:
            wins += 1
        elif direction == "lower" and trained < interp:
            wins += 1
    return wins


def finite_values(values: Any) -> list[float]:
    result = [finite_or_nan(value) for value in values]
    result = [value for value in result if math.isfinite(value)]
    if not result:
        raise ValueError("Expected at least one finite value")
    return result


def finite_or_nan(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[str], title: str) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = [escape_markdown(str(row.get(column, ""))) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def format_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return format_number(float(value))
    return str(value)


def format_number(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/changing_resolution_operator_compare_stage1",
        help="Operator compare output dir or metrics JSONL path.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--out_dir", help="Output table directory. Defaults to <metrics_dir>/tables.")
    parser.add_argument("--top_k", type=int, default=0, help="Limit sample table to top-k PSNR delta rows. 0 keeps all rows.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
