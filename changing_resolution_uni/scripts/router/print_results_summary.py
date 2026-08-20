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

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print summary report of router benchmarks.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Path to output directory (e.g. outputs/router_benchmarks_1k_lambda001).",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(REPO_ROOT / "data" / "changing_resolution_uni" / "oracle_dataset_1k"),
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
    print(f"   OPTIMAL TIMESTEP ROUTER BENCHMARK & TOKEN ATTRIBUTION REPORT ({out_dir.name})")
    print("=" * 95)

    # 1. Master Benchmark Results Table
    csv_path = out_dir / "router_benchmark_results.csv"
    if csv_path.is_file():
        print(f"\n[1] Main Test Set Evaluation (from {csv_path.name}):")
        print("-" * 95)
        print(f"{'Method':<38} | {'Policy Regret':<14} | {'VBench-5':<9} | {'Latency':<9} | {'Speedup':<8} | {'MAE':<6} | {'Top-1'}")
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

    # 2. Optimal Step Histogram (computed directly from records)
    records_dir = dataset_dir / "records"
    hist: dict[int, int] = {}
    if records_dir.is_dir():
        for r_file in records_dir.glob("*.json"):
            try:
                data = json.loads(r_file.read_text(encoding="utf-8"))
                cands = data.get("candidates", [])
                if cands:
                    # Find optimal step according to primary lambda or default
                    opt_s = data.get("optimal_step_lambda_001") or data.get("optimal_step_lambda_005") or data.get("optimal_step")
                    if opt_s is None:
                        opt_s = max(cands, key=lambda c: float(c.get("utilities", {}).get("u_0.01", c.get("vbench5", 0.0))))["step"]
                    hist[int(opt_s)] = hist.get(int(opt_s), 0) + 1
            except Exception:
                pass

    if hist:
        print("\n[2] Global Oracle Optimal Step Distribution (across dataset trajectories):")
        print(f"    - Step Histogram: {dict(sorted(hist.items()))}")

    # 3. Token Attribution Discovery (Top Late vs Early Switch Words)
    late_csv = out_dir / "token_attribution" / "top_late_switch_words.csv"
    early_csv = out_dir / "token_attribution" / "top_early_switch_words.csv"

    if late_csv.is_file() and early_csv.is_file():
        print("\n[3] Reverse Token Attribution: Discovered Semantic Keywords Driving Timestep Choice:")
        print("-" * 95)
        print(f"{'Rank':<5} | {'Top Late-Switch Words (Stay in LR Longer)':<40} | {'Top Early-Switch Words (Switch to HR Earlier)':<40}")
        print("-" * 95)
        with open(late_csv, "r", encoding="utf-8") as f_l, open(early_csv, "r", encoding="utf-8") as f_e:
            r_l = list(csv.DictReader(f_l))
            r_e = list(csv.DictReader(f_e))
            for i in range(min(20, max(len(r_l), len(r_e)))):
                w_l = f"{r_l[i]['word']} ({r_l[i]['mean_attribution']})" if i < len(r_l) else ""
                w_e = f"{r_e[i]['word']} ({r_e[i]['mean_attribution']})" if i < len(r_e) else ""
                print(f"{i+1:<5} | {w_l:<40} | {w_e:<40}")
        print("-" * 95)

    print("\n" + "=" * 95 + "\n")


if __name__ == "__main__":
    main()
