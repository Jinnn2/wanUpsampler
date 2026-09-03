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
    build_probe_selection_plan,
    load_prompts,
    validate_collection_plan,
    validate_protocol,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or validate sparse UNIV controller data collection."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--protocol", required=True)
    plan_parser.add_argument("--prompts", required=True)
    plan_parser.add_argument("--output", required=True)

    probe_parser = subparsers.add_parser("plan-probes")
    probe_parser.add_argument("--protocol", required=True)
    probe_parser.add_argument("--prompts", required=True)
    probe_parser.add_argument("--output", required=True)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--plan", required=True)

    protocol_parser = subparsers.add_parser("check-protocol")
    protocol_parser.add_argument("--protocol", required=True)
    protocol_parser.add_argument("--allow-pending-probe", action="store_true")

    args = parser.parse_args()
    if args.command == "check-protocol":
        protocol = _load_json(args.protocol)
        validate_protocol(
            protocol,
            require_selected_probe=not args.allow_pending_probe,
        )
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
    if args.command == "plan-probes":
        plan = build_probe_selection_plan(protocol, prompts)
    else:
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
    budgets = Counter(row["budget_id"] for row in assignments)
    if plan["schema"] == "univ_probe_selection_plan_v1":
        candidate_slots = sum(
            sum(len(branch["downstream_candidates"]) for branch in row["probe_branches"])
            for row in assignments
        )
    else:
        candidate_slots = sum(len(row["candidate_slots"]) for row in assignments)
    print(
        json.dumps(
            {
                "plan_sha256": plan["plan_sha256"],
                "prompt_count": plan["prompt_count"],
                "trajectory_count": len(assignments),
                "candidate_slot_count": candidate_slots,
                "action_pool_size": plan["action_pool_size"],
                "trajectories_by_split": dict(sorted(splits.items())) if splits else {},
                "trajectories_by_budget": dict(sorted(budgets.items())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
