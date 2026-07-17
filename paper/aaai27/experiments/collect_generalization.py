from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for raw_root in args.factorial_root:
        root = Path(raw_root).resolve()
        manifest = load_json(root / "run_manifest.json")
        vbench = load_json(root / "metrics/vbench_v1_custom.json")
        human = load_json(root / "review/human_review_summary.json")
        rates = human.get("severe_failure_rates", {})
        for case_name, case_result in vbench.get("cases", {}).items():
            numeric = case_result.get("numeric_metrics", {})
            if not numeric:
                raise SystemExit(f"No VBench metrics for {root.name}/{case_name}")
            failure = rates.get(case_name)
            if not failure:
                raise SystemExit(f"No human severe-failure evidence for {root.name}/{case_name}")
            for metric, value in sorted(numeric.items()):
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    rows.append(
                        {
                            "family": manifest["family"],
                            "category": "all_unseen",
                            "case": case_name,
                            "samples": len(manifest["prompts"]),
                            "metric": metric,
                            "mean": value,
                            "severe_failures": failure["failures"],
                            "failure_exposures": failure["exposures"],
                            "severe_failure_rate": failure["rate"],
                            "factorial_root": str(root),
                        }
                    )
    if not rows:
        raise SystemExit("No generalization rows were compiled")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generalization summary: {output}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid or missing JSON {path}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile VBench and severe-failure evidence for unseen prompts.")
    parser.add_argument("--factorial-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
