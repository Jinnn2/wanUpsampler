from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))
    pairs = [(float(row[args.a_field]), float(row[args.b_field])) for row in rows]
    pairs = [(a, b) for a, b in pairs if math.isfinite(a) and math.isfinite(b)]
    if not pairs:
        raise SystemExit("No finite paired observations")
    deltas = [b - a for a, b in pairs]
    oriented = [-delta if args.lower_is_better else delta for delta in deltas]
    rng = random.Random(args.seed)
    boot = [mean(rng.choices(oriented, k=len(oriented))) for _ in range(args.bootstrap_samples)]
    boot.sort()
    alpha = (1.0 - args.confidence) / 2.0
    wins = sum(value > 0 for value in oriented)
    losses = sum(value < 0 for value in oriented)
    result = {
        "input": args.input,
        "a_field": args.a_field,
        "b_field": args.b_field,
        "lower_is_better": args.lower_is_better,
        "samples": len(oriented),
        "a_mean": mean(a for a, _ in pairs),
        "b_mean": mean(b for _, b in pairs),
        "delta_b_minus_a_mean": mean(deltas),
        "oriented_improvement_mean": mean(oriented),
        "oriented_improvement_std": pstdev(oriented) if len(oriented) > 1 else 0.0,
        "bootstrap_confidence": args.confidence,
        "bootstrap_ci": [quantile(boot, alpha), quantile(boot, 1.0 - alpha)],
        "wins": wins,
        "losses": losses,
        "ties": len(oriented) - wins - losses,
        "two_sided_sign_test_p": sign_test_p(wins, losses),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def quantile(values: list[float], probability: float) -> float:
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap CI and exact sign test for two result columns.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--a-field", required=True)
    parser.add_argument("--b-field", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lower-is-better", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=202707)
    return parser.parse_args()


if __name__ == "__main__":
    main()
