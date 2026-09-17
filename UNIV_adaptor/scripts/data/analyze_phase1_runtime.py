"""Summarize standalone phase1 runtime records without rerunning generation."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def load_records(root: Path, splits: list[str]) -> list[dict[str, Any]]:
    rows = []
    for split in splits:
        paths = sorted((root / "combined_records" / split).glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"no combined records under {root / 'combined_records' / split}")
        rows.extend(json.loads(path.read_text(encoding="utf-8")) for path in paths)
    return rows


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    index = (len(values) - 1) * q
    low, high = int(index), min(len(values) - 1, int(index) + 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def summarize(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    paired: dict[str, dict[str, dict[str, float]]] = {}
    for record in records:
        split = str(record["split"])
        for candidate in record["budget_candidates"]:
            action_id = str(candidate.get("artifact_id", candidate.get("budget_id", "")))
            cost = candidate.get("cost", {})
            stage = cost.get("stage_seconds", {}) or {}
            row = {
                "split": split,
                "action_id": action_id,
                "display_budget": candidate.get("display_budget", action_id),
                "proxy_compute_density": float(candidate.get("proxy_compute_density", 0.0)),
                "pipeline_seconds": float(cost.get("pipeline_seconds", 0.0)),
                "segment_seconds": float(cost.get("segment_seconds", 0.0)),
                "dit_seconds": float(stage.get("dit", stage.get("DiT", 0.0)) or 0.0),
                "vae_seconds": float(stage.get("vae_decode", stage.get("VAE Decoder", 0.0)) or 0.0),
                "peak_allocated_gib": float(cost.get("peak_allocated_gib", 0.0)),
            }
            groups.setdefault((split, action_id), []).append(row)
            paired.setdefault(record["trajectory_key"], {})[action_id] = row
    summary = []
    for (split, action_id), rows in sorted(groups.items()):
        out = {"split": split, "action_id": action_id, "display_budget": rows[0]["display_budget"], "count": len(rows), "proxy_compute_density": rows[0]["proxy_compute_density"]}
        for field in ("pipeline_seconds", "segment_seconds", "dit_seconds", "vae_seconds", "peak_allocated_gib"):
            values = [row[field] for row in rows]
            out[f"{field}_mean"] = statistics.fmean(values)
            out[f"{field}_median"] = statistics.median(values)
            out[f"{field}_p95"] = percentile(values, 0.95)
        summary.append(out)
    paired_rows = []
    for trajectory_key, actions in sorted(paired.items()):
        if len(actions) < 2:
            continue
        base_id = "P1_B15_BASE" if "P1_B15_BASE" in actions else sorted(actions)[0]
        base = actions[base_id]
        for action_id, row in sorted(actions.items()):
            paired_rows.append({
                "trajectory_key": trajectory_key,
                "action_id": action_id,
                "baseline_action_id": base_id,
                "pipeline_delta_seconds": row["pipeline_seconds"] - base["pipeline_seconds"],
                "pipeline_speedup_vs_baseline": base["pipeline_seconds"] / row["pipeline_seconds"] if row["pipeline_seconds"] > 0 else 0.0,
            })
    return summary, paired_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args()
    root = Path(args.dataset_root).resolve()
    out = Path(args.out_dir).resolve()
    records = load_records(root, args.splits)
    summary, paired = summarize(records)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase1_runtime_summary.json").write_text(json.dumps({"dataset_root": str(root), "record_count": len(records), "summary": summary, "paired": paired}, indent=2), encoding="utf-8")
    if summary:
        with (out / "phase1_runtime_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader(); writer.writerows(summary)
    print(json.dumps({"records": len(records), "summary_rows": len(summary), "paired_rows": len(paired), "out_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
