from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WINNER_SUFFIX = "_winner_A_B_tie"
VALID_WINNERS = {"A", "B", "tie"}
VALID_FAILURES = {"A", "B", "neither"}


def main() -> None:
    args = parse_args()
    root = Path(args.factorial_root).resolve()
    if args.action == "merge":
        completed = merge(root, args.rater, args.review_name)
    else:
        review, _private = review_paths(root, args.review_name)
        completed = review / "human_ratings_completed.csv"
    rows, fields = read_csv(completed)
    validate_completed(root, rows, fields, args.min_raters, args.review_name)
    summary_csv, summary_json = summarize(root, rows, args.review_name)
    print(f"Validated ratings: {len(rows)} rows")
    print(f"Summary CSV     : {summary_csv}")
    print(f"Summary JSON    : {summary_json}")


def merge(root: Path, rater_specs: list[str], review_name: str = "default") -> Path:
    if not rater_specs:
        raise SystemExit("merge requires at least one --rater ID=/path/to/completed.csv")
    review, _private = review_paths(root, review_name)
    ballot_rows, ballot_fields = read_csv(review / "human_ratings.csv")
    ballot_ids = {row["blind_id"] for row in ballot_rows}
    merged: list[dict[str, str]] = []
    for spec in rater_specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid --rater {spec!r}; expected ID=/path/to/file.csv")
        rater_id, raw_path = spec.split("=", 1)
        rater_id = rater_id.strip()
        if not rater_id:
            raise SystemExit("rater ID cannot be empty")
        rows, fields = read_csv(Path(raw_path))
        if set(fields) != set(ballot_fields):
            raise SystemExit(f"Rater {rater_id} columns do not match the canonical ballot")
        ids = [row.get("blind_id", "") for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != ballot_ids:
            raise SystemExit(f"Rater {rater_id} must contain every blind_id exactly once")
        for row in rows:
            merged.append({"rater_id": rater_id, **row})
    output = review / "human_ratings_completed.csv"
    write_csv(output, merged, ["rater_id", *ballot_fields])
    return output


def validate_completed(
    root: Path,
    rows: list[dict[str, str]],
    fields: list[str],
    min_raters: int,
    review_name: str = "default",
) -> None:
    review, _private = review_paths(root, review_name)
    ballot, _ = read_csv(review / "human_ratings.csv")
    ballot_ids = {row["blind_id"] for row in ballot}
    winner_fields = [field for field in fields if field.endswith(WINNER_SUFFIX)]
    required = {"blind_id", "rater_id", "confidence_1_to_5", "severe_failure_A_B_neither"}
    missing_fields = sorted(required - set(fields))
    if missing_fields or not winner_fields:
        raise SystemExit("Completed ratings missing columns: " + ", ".join(missing_fields or ["winner columns"]))
    seen: set[tuple[str, str]] = set()
    raters_by_blind: dict[str, set[str]] = defaultdict(set)
    for line, row in enumerate(rows, start=2):
        blind_id = row["blind_id"].strip()
        rater = row["rater_id"].strip()
        if blind_id not in ballot_ids:
            raise SystemExit(f"line {line}: unknown blind_id {blind_id!r}")
        if not rater or (blind_id, rater) in seen:
            raise SystemExit(f"line {line}: missing or duplicate rater/blind_id")
        seen.add((blind_id, rater))
        raters_by_blind[blind_id].add(rater)
        for field in winner_fields:
            if row[field].strip() not in VALID_WINNERS:
                raise SystemExit(f"line {line}: {field} must be A, B, or tie")
        try:
            confidence = int(row["confidence_1_to_5"])
        except ValueError as exc:
            raise SystemExit(f"line {line}: confidence must be an integer 1..5") from exc
        if confidence not in range(1, 6):
            raise SystemExit(f"line {line}: confidence must be an integer 1..5")
        if row["severe_failure_A_B_neither"].strip() not in VALID_FAILURES:
            raise SystemExit(f"line {line}: severe failure must be A, B, or neither")
    insufficient = [blind_id for blind_id in ballot_ids if len(raters_by_blind[blind_id]) < min_raters]
    if insufficient:
        raise SystemExit(f"{len(insufficient)} pairs have fewer than {min_raters} independent ratings")


def summarize(root: Path, ratings: list[dict[str, str]], review_name: str = "default") -> tuple[Path, Path]:
    review, private = review_paths(root, review_name)
    keys, _ = read_csv(private / "human_review_key.csv")
    key_by_id = {row["blind_id"]: row for row in keys}
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    confidence: dict[tuple[str, str], list[int]] = defaultdict(list)
    prompt_votes: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    comparison_cases: dict[str, set[str]] = defaultdict(set)
    failures: dict[str, Counter[str]] = defaultdict(Counter)
    failure_exposures: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    winner_fields = [field for field in ratings[0] if field.endswith(WINNER_SUFFIX)]
    for row in ratings:
        key = key_by_id[row["blind_id"]]
        comparison = key["comparison"]
        sample_id = key.get("sample_index") or key["blind_id"]
        comparison_cases[comparison].update((key["case_A"], key["case_B"]))
        for field in winner_fields:
            dimension = field[: -len(WINNER_SUFFIX)]
            raw = row[field]
            preferred = "tie" if raw == "tie" else key[f"case_{raw}"]
            counts[(comparison, dimension)][preferred] += 1
            prompt_votes[(comparison, dimension, sample_id)][preferred] += 1
            confidence[(comparison, dimension)].append(int(row["confidence_1_to_5"]))
        failure = row["severe_failure_A_B_neither"]
        failures[comparison]["neither" if failure == "neither" else key[f"case_{failure}"]] += 1
        failure_exposures[key["case_A"]] += 1
        failure_exposures[key["case_B"]] += 1
        if failure != "neither":
            failure_counts[key[f"case_{failure}"]] += 1
    output_rows: list[dict[str, Any]] = []
    for (comparison, dimension), counter in sorted(counts.items()):
        total = sum(counter.values())
        for preferred_case, count in sorted(counter.items()):
            output_rows.append(
                {
                    "comparison": comparison,
                    "dimension": dimension,
                    "preferred_case": preferred_case,
                    "votes": count,
                    "total_votes": total,
                    "preference_rate": count / total,
                    "mean_confidence": sum(confidence[(comparison, dimension)]) / total,
                }
            )
    csv_path = review / "human_review_summary.csv"
    write_csv(csv_path, output_rows, list(output_rows[0]))

    prompt_rows, agreement_rows = summarize_prompt_majorities(prompt_votes, comparison_cases)
    prompt_csv_path = review / "human_review_prompt_summary.csv"
    agreement_csv_path = review / "human_review_agreement.csv"
    write_csv(prompt_csv_path, prompt_rows, list(prompt_rows[0]))
    write_csv(agreement_csv_path, agreement_rows, list(agreement_rows[0]))
    payload = {
        "schema_version": 2,
        "family": load_json(root / "run_manifest.json")["family"],
        "review_name": review_name,
        "raters": sorted({row["rater_id"] for row in ratings}),
        "rating_rows": len(ratings),
        "preferences": output_rows,
        "prompt_majority_preferences": prompt_rows,
        "inter_rater_agreement": agreement_rows,
        "severe_failures": {key: dict(value) for key, value in sorted(failures.items())},
        "severe_failure_rates": {
            case: {
                "failures": failure_counts[case],
                "exposures": exposure,
                "rate": failure_counts[case] / exposure,
            }
            for case, exposure in sorted(failure_exposures.items())
        },
        "artifacts": {
            "vote_summary_csv": str(csv_path),
            "prompt_summary_csv": str(prompt_csv_path),
            "agreement_csv": str(agreement_csv_path),
        },
    }
    json_path = review / "human_review_summary.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def summarize_prompt_majorities(
    prompt_votes: dict[tuple[str, str, str], Counter[str]],
    comparison_cases: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[Counter[str]]] = defaultdict(list)
    for (comparison, dimension, _sample_id), counter in sorted(prompt_votes.items()):
        grouped[(comparison, dimension)].append(counter)

    prompt_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    for (comparison, dimension), item_counters in sorted(grouped.items()):
        methods = sorted(comparison_cases[comparison])
        if len(methods) != 2:
            raise SystemExit(
                f"Human comparison {comparison!r} must contain exactly two cases, found {methods}"
            )
        outcomes = [majority_outcome(counter) for counter in item_counters]
        majority_counts = Counter(outcomes)
        method_a, method_b = methods
        sign_p = exact_sign_test_p(majority_counts[method_a], majority_counts[method_b])
        total = len(outcomes)
        for preferred in [method_a, method_b, "tie"]:
            rate = majority_counts[preferred] / total
            ci_low, ci_high = bootstrap_rate_ci(
                [outcome == preferred for outcome in outcomes],
                seed=f"{comparison}:{dimension}:{preferred}",
            )
            prompt_rows.append(
                {
                    "comparison": comparison,
                    "dimension": dimension,
                    "preferred_case": preferred,
                    "prompt_majorities": majority_counts[preferred],
                    "total_prompts": total,
                    "preference_rate": rate,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "method_a": method_a,
                    "method_b": method_b,
                    "method_a_wins": majority_counts[method_a],
                    "method_b_wins": majority_counts[method_b],
                    "prompt_ties": majority_counts["tie"],
                    "two_sided_sign_test_p": sign_p,
                }
            )
        agreement = fleiss_kappa(item_counters, [method_a, method_b, "tie"])
        agreement_rows.append(
            {
                "comparison": comparison,
                "dimension": dimension,
                **agreement,
            }
        )
    return prompt_rows, agreement_rows


def majority_outcome(counter: Counter[str]) -> str:
    if not counter:
        return "tie"
    highest = max(counter.values())
    winners = [label for label, count in counter.items() if count == highest]
    if len(winners) != 1 or winners[0] == "tie":
        return "tie"
    return winners[0]


def fleiss_kappa(item_counters: list[Counter[str]], categories: list[str]) -> dict[str, Any]:
    ratings_per_item = {sum(counter.values()) for counter in item_counters}
    if len(ratings_per_item) != 1:
        return {
            "items": len(item_counters),
            "ratings_per_item": "variable",
            "observed_agreement": "NA",
            "expected_agreement": "NA",
            "fleiss_kappa": "NA",
        }
    n = ratings_per_item.pop()
    if n < 2:
        return {
            "items": len(item_counters),
            "ratings_per_item": n,
            "observed_agreement": "NA",
            "expected_agreement": "NA",
            "fleiss_kappa": "NA",
        }
    observed_per_item = [
        (sum(counter[category] ** 2 for category in categories) - n) / (n * (n - 1))
        for counter in item_counters
    ]
    observed = sum(observed_per_item) / len(observed_per_item)
    total_ratings = len(item_counters) * n
    proportions = [sum(counter[category] for counter in item_counters) / total_ratings for category in categories]
    expected = sum(value**2 for value in proportions)
    if math.isclose(expected, 1.0):
        kappa = 1.0 if math.isclose(observed, 1.0) else "NA"
    else:
        kappa = (observed - expected) / (1.0 - expected)
    return {
        "items": len(item_counters),
        "ratings_per_item": n,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "fleiss_kappa": kappa,
    }


def bootstrap_rate_ci(values: list[bool], *, seed: str, samples: int = 10000) -> tuple[float, float]:
    numeric = [int(value) for value in values]
    rng = random.Random(seed)
    estimates = [sum(rng.choices(numeric, k=len(numeric))) / len(numeric) for _ in range(samples)]
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def percentile(values: list[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def exact_sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, index) for index in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise SystemExit(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows:
        raise SystemExit(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def review_paths(root: Path, review_name: str) -> tuple[Path, Path]:
    if review_name == "default":
        return root / "review", root / "_private"
    return root / "review" / review_name, root / "_private" / review_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge, validate, unblind, and summarize independent human ratings.")
    parser.add_argument("action", choices=["merge", "summarize"])
    parser.add_argument("--factorial-root", required=True)
    parser.add_argument("--rater", action="append", default=[])
    parser.add_argument("--min-raters", type=int, default=3)
    parser.add_argument("--review-name", default="default")
    args = parser.parse_args()
    if Path(args.review_name).name != args.review_name or args.review_name in {"", ".", ".."}:
        parser.error("--review-name must be one path-safe directory name")
    return args


if __name__ == "__main__":
    main()
