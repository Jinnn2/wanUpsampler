from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper.aaai27.experiments.paired_statistics import quantile, sign_test_p


def main() -> None:
    args = parse_args()
    root = Path(args.sweep_root)
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"^lora_(?P<step>\d+)_s(?P<tag>.+)$")
    for metrics_csv in sorted((root / "metrics").glob("lora_*_s*/original_lora_teacher_metrics.csv")):
        match = pattern.match(metrics_csv.parent.name)
        if not match:
            continue
        samples = read_csv(metrics_csv)
        metrics = sorted(
            field[len("original_") :]
            for field in samples[0]
            if field.startswith("original_")
            and f"lora_{field[len('original_') :]}" in samples[0]
            and f"{field[len('original_') :]}_delta_lora_minus_original" in samples[0]
        )
        for metric in metrics:
            a = [float(row[f"original_{metric}"]) for row in samples]
            b = [float(row[f"lora_{metric}"]) for row in samples]
            if not all(math.isfinite(value) for value in [*a, *b]):
                raise SystemExit(f"Non-finite {metric} value in {metrics_csv}")
            higher = metric in {"psnr", "ssim"}
            deltas = [right - left for left, right in zip(a, b)]
            oriented = deltas if higher else [-value for value in deltas]
            rng = random.Random(f"{args.seed}:{metrics_csv.parent.name}:{metric}")
            boot = [mean(rng.choices(oriented, k=len(oriented))) for _ in range(args.bootstrap_samples)]
            boot.sort()
            wins = sum(value > 0 for value in oriented)
            losses = sum(value < 0 for value in oriented)
            rows.append(
                {
                    "step": int(match.group("step")),
                    "strength_tag": match.group("tag"),
                    "case": metrics_csv.parent.name,
                    "metric": metric,
                    "better": "higher" if higher else "lower",
                    "samples": len(oriented),
                    "original_mean": mean(a),
                    "lora_mean": mean(b),
                    "oriented_improvement_mean": mean(oriented),
                    "oriented_improvement_std": pstdev(oriented) if len(oriented) > 1 else 0.0,
                    "bootstrap_ci_low": quantile(boot, 0.025),
                    "bootstrap_ci_high": quantile(boot, 0.975),
                    "wins": wins,
                    "losses": losses,
                    "ties": len(oriented) - wins - losses,
                    "two_sided_sign_test_p": sign_test_p(wins, losses),
                    "source": str(metrics_csv.resolve()),
                }
            )
    if not rows:
        raise SystemExit(f"No endpoint strength sample metrics found under {root / 'metrics'}")
    output = Path(args.output) if args.output else root / "metrics/strength_paired_statistics.csv"
    write_csv(output, rows)
    print(f"Endpoint strength paired statistics: {output}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile paired bootstrap/sign statistics for a LoRA strength sweep.")
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=202707)
    return parser.parse_args()


if __name__ == "__main__":
    main()
