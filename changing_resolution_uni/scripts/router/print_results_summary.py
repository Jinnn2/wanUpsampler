#!/usr/bin/env python3
"""
Publication-Ready Formatter and Visualizer for Router Benchmark Results & Token Attribution.
Reads outputs/router_benchmarks_1k/ and prints clean markdown tables and scientific insights.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
out_dir = REPO_ROOT / "outputs" / "router_benchmarks_1k"
dataset_dir = REPO_ROOT / "data" / "changing_resolution_uni" / "oracle_dataset_1k"


def main() -> None:
    print("\n" + "=" * 95)
    print("           OPTIMAL TIMESTEP ROUTER & TOKEN ATTRIBUTION BENCHMARK REPORT")
    print("=" * 95)

    # 1. Master Benchmark Results Table
    csv_path = out_dir / "router_benchmark_results.csv"
    if csv_path.is_file():
        print("\n[1] Main Test Set Evaluation (100 Test Prompts, 300 Trajectories, Prompt-Disjoint):")
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

    # 2. Optimal Step Histogram
    manifest_path = dataset_dir / "dataset_manifest.json"
    if manifest_path.is_file():
        try:
            m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics = m_data.get("scientific_metrics", {})
            hist = metrics.get("oracle_optimal_step_distribution", {})
            print("\n[2] Global Oracle Optimal Step Distribution (1000 Prompts × 3 Seeds):")
            print(f"    - Theoretical Upper Bound Prompt Regret (R_prompt) : {metrics.get('prompt_explainable_regret_upper_bound_r_prompt', 'N/A')}")
            print(f"    - Mean Intra-Prompt Seed Std                     : {metrics.get('mean_intra_prompt_seed_step_std', 'N/A')} steps")
            print(f"    - Step Histogram: {hist}")
        except Exception:
            pass

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
