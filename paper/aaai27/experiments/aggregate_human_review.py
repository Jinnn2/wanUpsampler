from __future__ import annotations

import argparse
import csv
import json
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
        completed = merge(root, args.rater)
    else:
        completed = root / "review/human_ratings_completed.csv"
    rows, fields = read_csv(completed)
    validate_completed(root, rows, fields, args.min_raters)
    summary_csv, summary_json = summarize(root, rows)
    print(f"Validated ratings: {len(rows)} rows")
    print(f"Summary CSV     : {summary_csv}")
    print(f"Summary JSON    : {summary_json}")


def merge(root: Path, rater_specs: list[str]) -> Path:
    if not rater_specs:
        raise SystemExit("merge requires at least one --rater ID=/path/to/completed.csv")
    ballot_rows, ballot_fields = read_csv(root / "review/human_ratings.csv")
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
    output = root / "review/human_ratings_completed.csv"
    write_csv(output, merged, ["rater_id", *ballot_fields])
    return output


def validate_completed(root: Path, rows: list[dict[str, str]], fields: list[str], min_raters: int) -> None:
    ballot, _ = read_csv(root / "review/human_ratings.csv")
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


def summarize(root: Path, ratings: list[dict[str, str]]) -> tuple[Path, Path]:
    keys, _ = read_csv(root / "_private/human_review_key.csv")
    key_by_id = {row["blind_id"]: row for row in keys}
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    confidence: dict[tuple[str, str], list[int]] = defaultdict(list)
    failures: dict[str, Counter[str]] = defaultdict(Counter)
    failure_exposures: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    winner_fields = [field for field in ratings[0] if field.endswith(WINNER_SUFFIX)]
    for row in ratings:
        key = key_by_id[row["blind_id"]]
        comparison = key["comparison"]
        for field in winner_fields:
            dimension = field[: -len(WINNER_SUFFIX)]
            raw = row[field]
            preferred = "tie" if raw == "tie" else key[f"case_{raw}"]
            counts[(comparison, dimension)][preferred] += 1
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
    review = root / "review"
    csv_path = review / "human_review_summary.csv"
    write_csv(csv_path, output_rows, list(output_rows[0]))
    payload = {
        "schema_version": 1,
        "family": load_json(root / "run_manifest.json")["family"],
        "raters": sorted({row["rater_id"] for row in ratings}),
        "rating_rows": len(ratings),
        "preferences": output_rows,
        "severe_failures": {key: dict(value) for key, value in sorted(failures.items())},
        "severe_failure_rates": {
            case: {
                "failures": failure_counts[case],
                "exposures": exposure,
                "rate": failure_counts[case] / exposure,
            }
            for case, exposure in sorted(failure_exposures.items())
        },
    }
    json_path = review / "human_review_summary.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge, validate, unblind, and summarize independent human ratings.")
    parser.add_argument("action", choices=["merge", "summarize"])
    parser.add_argument("--factorial-root", required=True)
    parser.add_argument("--rater", action="append", default=[])
    parser.add_argument("--min-raters", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
