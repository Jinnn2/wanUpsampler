from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from aggregate_human_review import merge, read_csv, summarize, validate_completed, write_csv


def main() -> None:
    args = parse_args()
    root = Path(args.factorial_root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    available_steps = sorted({int(case["step"]) for case in manifest["cases"]})
    if len(available_steps) < 2:
        raise SystemExit("Legacy collision repair is only for multi-step factorial review packages")
    if args.step != available_steps[0]:
        raise SystemExit(
            f"Legacy blinded files contain the first generated step ({available_steps[0]}), not requested step {args.step}"
        )

    review = root / "review"
    private = root / "_private"
    ballot_path = review / "human_ratings.csv"
    key_path = private / "human_review_key.csv"
    ballot, ballot_fields = read_csv(ballot_path)
    keys, key_fields = read_csv(key_path)
    ballot = collapse_identical(ballot, "canonical ballot")
    keys = [row for row in keys if row.get("case_A", "").startswith(f"step{args.step}_")]
    ensure_unique(keys, "selected private key")
    if {row["blind_id"] for row in ballot} != {row["blind_id"] for row in keys}:
        raise SystemExit("Selected private key does not match the deduplicated legacy ballot")

    backup = review / "legacy_collision_backup"
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ballot_path, backup / "human_ratings.csv")
    shutil.copy2(key_path, backup / "human_review_key.csv")
    write_csv(ballot_path, ballot, ballot_fields)
    write_csv(key_path, keys, key_fields)

    corrected_dir = review / f"legacy_step{args.step}_raters"
    corrected_dir.mkdir(parents=True, exist_ok=True)
    corrected_specs = []
    for spec in args.rater:
        if "=" not in spec:
            raise SystemExit(f"Invalid --rater {spec!r}; expected ID=/path/to/file.csv")
        rater_id, raw_path = spec.split("=", 1)
        rows, fields = read_csv(Path(raw_path))
        rows = collapse_identical(rows, f"rater {rater_id}")
        output = corrected_dir / f"{rater_id}.csv"
        write_csv(output, rows, fields)
        corrected_specs.append(f"{rater_id}={output}")

    completed = merge(root, corrected_specs)
    completed_rows, completed_fields = read_csv(completed)
    validate_completed(root, completed_rows, completed_fields, args.min_raters)
    summary_csv, summary_json = summarize(root, completed_rows)
    print(f"Recovered legacy Wan review for step {args.step}: {len(ballot)} blind pairs")
    print(f"Completed ratings: {completed}")
    print(f"Summary CSV     : {summary_csv}")
    print(f"Summary JSON    : {summary_json}")
    print(f"Originals backed up under: {backup}")


def collapse_identical(rows: list[dict[str, str]], label: str) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("blind_id", ""), []).append(row)
    result = []
    for blind_id, copies in grouped.items():
        if not blind_id:
            raise SystemExit(f"{label}: empty blind_id")
        first = copies[0]
        if any(copy != first for copy in copies[1:]):
            raise SystemExit(f"{label}: duplicate blind_id {blind_id} has non-identical rows")
        result.append(first)
    return result


def ensure_unique(rows: list[dict[str, str]], label: str) -> None:
    ids = [row.get("blind_id", "") for row in rows]
    if not ids or len(ids) != len(set(ids)) or any(not blind_id for blind_id in ids):
        raise SystemExit(f"{label} does not contain unique non-empty blind IDs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely recover the step40 ratings from the legacy Wan blind-ID collision.")
    parser.add_argument("--factorial-root", required=True)
    parser.add_argument("--step", type=int, default=40)
    parser.add_argument("--rater", action="append", default=[], required=True)
    parser.add_argument("--min-raters", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
