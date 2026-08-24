#!/usr/bin/env python3
"""
Publication-Ready Formatter and Visualizer for Router Benchmark Results & Token Attribution.
Reads outputs/router_benchmarks_1k/ and prints clean markdown tables and scientific insights.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_uni.scripts.data.oracle_record_schema import (  # noqa: E402
    FORMAL_STEPS,
    aggregate_prompt_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print summary report of router benchmarks."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Path to output directory (e.g. outputs/router_benchmarks_1k_lambda001).",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(
            REPO_ROOT / "data" / "changing_resolution_uni" / "oracle_dataset_1k"
        ),
        help="Path to dataset directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine out_dir: explicit or auto-detect most recently modified
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        candidates = list((REPO_ROOT / "outputs").glob("router_benchmarks_*"))
        if candidates:
            out_dir = max(candidates, key=lambda p: p.stat().st_mtime)
        else:
            out_dir = REPO_ROOT / "outputs" / "router_benchmarks_1k"

    dataset_dir = Path(args.dataset_dir).resolve()

    print("\n" + "=" * 95)
    print(
        f"   OPTIMAL TIMESTEP ROUTER BENCHMARK & TOKEN ATTRIBUTION REPORT ({out_dir.name})"
    )
    print("=" * 95)

    # 1. Master Benchmark Results Table
    csv_path = out_dir / "router_benchmark_results.csv"
    if csv_path.is_file():
        print(f"\n[1] Main Test Set Evaluation (from {csv_path.name}):")
        print("-" * 95)
        print(
            f"{'Method':<38} | {'Policy Regret':<14} | {'VBench-5':<9} | {'Latency':<9} | {'Speedup':<8} | {'MAE':<6} | {'Top-1'}"
        )
        print("-" * 95)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                reg = float(r.get("policy_regret", 0.0))
                vb = float(r.get("realized_vbench5", 0.0))
                lat = float(r.get("realized_latency_sec", 0.0))
                spd = float(r.get("speedup_vs_native", 0.0))
                mae = float(r.get("step_mae", 0.0))
                top1 = float(r.get("top1_acc", 0.0))
                print(
                    f"{r['Method']:<38} | "
                    f"{reg:<14.6f} | "
                    f"{vb:<9.4f} | "
                    f"{lat:<7.1f}s | "
                    f"{spd:<6.2f}x | "
                    f"{mae:<4.2f}st | "
                    f"{top1:.1f}%"
                )
        print("-" * 95)
    else:
        print(f"\n[!] Benchmark CSV not found at {csv_path}")

    summary_json = out_dir / "router_benchmark_summary.json"
    primary_lambda = 0.01
    if summary_json.is_file():
        primary_lambda = float(
            json.loads(summary_json.read_text(encoding="utf-8")).get(
                "primary_lambda", primary_lambda
            )
        )

    # 2. Prompt-level optimal step histogram (mean utility across seeds)
    records_dir = dataset_dir / "records"
    hist: dict[int, int] = {}
    if records_dir.is_dir():
        records_by_prompt: dict[int, list[dict]] = {}
        for r_file in records_dir.glob("*.json"):
            try:
                data = json.loads(r_file.read_text(encoding="utf-8"))
                records_by_prompt.setdefault(int(data["prompt_id"]), []).append(data)
            except Exception:
                pass
        if records_by_prompt:
            prompt_samples, _ = aggregate_prompt_records(
                records_by_prompt,
                candidate_steps=FORMAL_STEPS,
                primary_lambda=primary_lambda,
            )
            for sample in prompt_samples.values():
                opt_s = FORMAL_STEPS[int(np.argmax(sample["utilities"]))]
                hist[opt_s] = hist.get(opt_s, 0) + 1

    if hist:
        print(
            f"\n[2] Prompt Oracle Optimal Step Distribution "
            f"(mean across seeds, lambda={primary_lambda:.2f}):"
        )
        print(f"    - Step Histogram: {dict(sorted(hist.items()))}")

    # 3. Token Attribution Discovery (B4 by default; legacy directory as fallback)
    attribution_dir = out_dir / "token_attribution_b4"
    if not attribution_dir.is_dir():
        attribution_dir = out_dir / "token_attribution"
    late_csv = attribution_dir / "top_late_switch_words.csv"
    early_csv = attribution_dir / "top_early_switch_words.csv"

    if late_csv.is_file() and early_csv.is_file():
        print(
            "\n[3] Natural-Word Attribution: Semantic Keywords Driving Timestep Choice:"
        )
        print("-" * 95)
        print(
            f"{'Rank':<5} | {'Top Late-Switch Words (Stay in LR Longer)':<40} | {'Top Early-Switch Words (Switch to HR Earlier)':<40}"
        )
        print("-" * 95)
        with (
            open(late_csv, "r", encoding="utf-8") as f_l,
            open(early_csv, "r", encoding="utf-8") as f_e,
        ):
            r_l = list(csv.DictReader(f_l))
            r_e = list(csv.DictReader(f_e))
            for i in range(min(20, max(len(r_l), len(r_e)))):
                if i < len(r_l):
                    late_value = r_l[i].get(
                        "mean_expected_step_delta", r_l[i].get("mean_attribution")
                    )
                    w_l = f"{r_l[i]['word']} ({late_value})"
                else:
                    w_l = ""
                if i < len(r_e):
                    early_value = r_e[i].get(
                        "mean_expected_step_delta", r_e[i].get("mean_attribution")
                    )
                    w_e = f"{r_e[i]['word']} ({early_value})"
                else:
                    w_e = ""
                print(f"{i + 1:<5} | {w_l:<40} | {w_e:<40}")
        print("-" * 95)

    print("\n" + "=" * 95 + "\n")


if __name__ == "__main__":
    main()
