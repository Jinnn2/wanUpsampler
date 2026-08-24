#!/usr/bin/env python3
"""Combine isolated router benchmark outputs across utility lambdas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    combined: list[dict[str, Any]] = []
    for summary_path in sorted(
        runs_root.glob("lambda_*/router_benchmark_summary.json")
    ):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lambda_value = float(summary["primary_lambda"])
        for row in summary.get("results", []):
            combined.append(
                {
                    "lambda": lambda_value,
                    "run_dir": summary_path.parent.name,
                    **row,
                }
            )
    if not combined:
        raise RuntimeError(f"No completed lambda runs found under {runs_root}")

    combined.sort(key=lambda row: (float(row["lambda"]), str(row["Method"])))
    combined_path = runs_root / "lambda_model_summary.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    # Keep one predeclared architecture across lambda. Selecting the minimum
    # regret architecture here would select on test results and leak the test set.
    b4_rows = [
        row
        for row in combined
        if str(row["Method"]) == "Learned: Soft Distillation MLP (B4)"
    ]
    if not b4_rows:
        raise RuntimeError("No predeclared B4 rows found in completed lambda runs")
    b4_path = runs_root / "lambda_b4_results.csv"
    with b4_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(b4_rows[0]))
        writer.writeheader()
        writer.writerows(b4_rows)

    print(f"Combined lambda summary: {combined_path}")
    print(f"Predeclared B4 lambda results: {b4_path}")


if __name__ == "__main__":
    main()
