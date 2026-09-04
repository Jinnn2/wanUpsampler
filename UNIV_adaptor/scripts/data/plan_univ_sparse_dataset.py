from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from UNIV_adaptor.data_protocol import (  # noqa: E402
    build_collection_plan,
    load_prompts,
    validate_collection_plan,
    validate_protocol,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or validate UNIV prompt-budget curve collection."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--protocol", required=True)
    plan_parser.add_argument("--prompts", required=True)
    plan_parser.add_argument("--output", required=True)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--plan", required=True)

    protocol_parser = subparsers.add_parser("check-protocol")
    protocol_parser.add_argument("--protocol", required=True)

    args = parser.parse_args()
    if args.command == "check-protocol":
        protocol = _load_json(args.protocol)
        validate_protocol(protocol)
        print(f"Protocol check passed: {Path(args.protocol).resolve()}")
        return
    if args.command == "check":
        plan = _load_json(args.plan)
        validate_collection_plan(plan)
        _print_summary(plan)
        print(f"Collection plan check passed: {Path(args.plan).resolve()}")
        return

    protocol = _load_json(args.protocol)
    prompts = load_prompts(args.prompts)
    plan = build_collection_plan(protocol, prompts)
    output = Path(args.output).resolve()
    if output.is_file():
        previous = _load_json(output)
        if previous.get("plan_sha256") != plan["plan_sha256"]:
            raise SystemExit(
                f"Refusing to replace a different immutable plan: {output}"
            )
    write_json_atomic(output, plan)
    _print_summary(plan)
    print(f"Collection plan written: {output}")


def _load_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _print_summary(plan: dict) -> None:
    assignments = plan["assignments"]
    splits = Counter(
        row["split"] for row in assignments if "split" in row
    )
    candidate_slots = sum(len(row["budget_candidates"]) for row in assignments)
    print(
        json.dumps(
            {
                "plan_sha256": plan["plan_sha256"],
                "prompt_count": plan["prompt_count"],
                "trajectory_count": len(assignments),
                "candidate_slot_count": candidate_slots,
                "budget_count": plan["budget_count"],
                "trajectories_by_split": dict(sorted(splits.items())) if splits else {},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
