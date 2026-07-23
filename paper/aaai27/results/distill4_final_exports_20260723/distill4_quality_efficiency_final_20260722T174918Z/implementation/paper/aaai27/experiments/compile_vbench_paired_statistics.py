from __future__ import annotations

import argparse
import csv
import json
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


DEFAULT_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
]


def main() -> None:
    args = parse_args()
    root = Path(args.factorial_root)
    manifest = load_json(root / "run_manifest.json")
    vbench = load_json(root / "metrics/vbench_v1_custom.json")
    contrasts = manifest.get("analysis_pairs") or manifest.get("review_pairs")
    if not contrasts:
        raise SystemExit("run_manifest.json must define analysis_pairs or review_pairs")
    dimensions = args.dimensions or DEFAULT_DIMENSIONS
    rows: list[dict[str, Any]] = []
    for contrast in contrasts:
        name = str(contrast["comparison"])
        case_a = str(contrast["left_case"])
        case_b = str(contrast["right_case"])
        for dimension in dimensions:
            values_a = per_video_values(vbench, case_a, dimension)
            values_b = per_video_values(vbench, case_b, dimension)
            common = sorted(set(values_a) & set(values_b))
            if not common:
                raise SystemExit(f"No paired {dimension} values for {case_a} and {case_b}")
            a = [values_a[index] for index in common]
            b = [values_b[index] for index in common]
            deltas = [right - left for left, right in zip(a, b)]
            rng = random.Random(f"{args.seed}:{name}:{dimension}")
            boot = [mean(rng.choices(deltas, k=len(deltas))) for _ in range(args.bootstrap_samples)]
            boot.sort()
            wins = sum(value > 0 for value in deltas)
            losses = sum(value < 0 for value in deltas)
            rows.append(
                {
                    "comparison": name,
                    "case_a": case_a,
                    "case_b": case_b,
                    "metric": dimension,
                    "samples": len(deltas),
                    "a_mean": mean(a),
                    "b_mean": mean(b),
                    "delta_b_minus_a_mean": mean(deltas),
                    "delta_std": pstdev(deltas) if len(deltas) > 1 else 0.0,
                    "bootstrap_ci_low": quantile(boot, 0.025),
                    "bootstrap_ci_high": quantile(boot, 0.975),
                    "wins": wins,
                    "losses": losses,
                    "ties": len(deltas) - wins - losses,
                    "two_sided_sign_test_p": sign_test_p(wins, losses),
                }
            )
    output = Path(args.output) if args.output else root / "metrics/vbench_paired_statistics.csv"
    write_csv(output, rows)
    print(f"VBench paired statistics: {output}")


def per_video_values(payload: dict[str, Any], case: str, dimension: str) -> dict[int, float]:
    numeric = payload.get("cases", {}).get(case, {}).get("numeric_metrics", {})
    pattern = re.compile(rf"\.{re.escape(dimension)}\.1\.(?P<index>\d+)\.video_results$")
    values: dict[int, float] = {}
    for key, raw_value in numeric.items():
        match = pattern.search(key)
        if not match:
            continue
        value = float(raw_value)
        if dimension == "imaging_quality" and value > 1.0:
            value /= 100.0
        values[int(match.group("index"))] = value
    return values


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid JSON {path}: {exc}") from exc


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap and sign statistics for VBench per-video outputs.")
    parser.add_argument("--factorial-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--dimension", dest="dimensions", action="append", default=[])
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=202707)
    return parser.parse_args()


if __name__ == "__main__":
    main()
