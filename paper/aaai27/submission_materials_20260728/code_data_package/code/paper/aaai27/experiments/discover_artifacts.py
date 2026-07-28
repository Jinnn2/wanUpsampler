from __future__ import annotations

import argparse
import json
from pathlib import Path


PATTERNS = {
    "wan50_lora40": ("outputs", ("tail_skip", "lora", "step40"), ("latest.safetensors", "step_*.safetensors")),
    "distill_stage2_368p": ("outputs", ("distill", "clean", "368x640", "stage2"), ("latest.pt", "step_*.pt")),
    "distill_lora3_368p": (
        "outputs",
        ("distill", "last_step", "368x640"),
        ("latest.safetensors", "step_*.safetensors"),
    ),
    "wan50_lora40_lmdb": ("data", ("tail_skip", "lora", "step40"), ("data.mdb",)),
    "distill_stage2_lmdb": ("data", ("distill", "clean", "368x640"), ("data.mdb",)),
    "distill_lora3_lmdb": ("data", ("distill", "last_step", "368x640"), ("data.mdb",)),
    "distill_raw_videos": ("data", ("distill", "raw"), ("*.mp4",)),
    "operator_480_metrics": ("outputs", ("operator_compare_stage2",), ("metrics*.jsonl", "summary*.json")),
}


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    report = {}
    counts = {}
    for name, (top, required_tokens, file_patterns) in PATTERNS.items():
        candidates = []
        top_dir = root / top
        if top_dir.is_dir():
            for file_pattern in file_patterns:
                for path in top_dir.rglob(file_pattern):
                    normalized = path.as_posix().lower()
                    if not all(token in normalized for token in required_tokens):
                        continue
                    if not path.is_file() or path.stat().st_size == 0:
                        continue
                    stat = path.stat()
                    candidates.append(
                        {
                            "path": str(path),
                            "size_bytes": stat.st_size,
                            "mtime": stat.st_mtime,
                        }
                    )
        counts[name] = len(candidates)
        report[name] = sorted(candidates, key=lambda item: item["mtime"], reverse=True)[: args.limit]

    output = root / "outputs/aaai27_experiments/_state/artifact_candidates.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"counts": counts, "candidates": report}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, candidates in report.items():
        suffix = f"; showing newest {len(candidates)}" if counts[name] > len(candidates) else ""
        print(f"{name}: {counts[name]} candidate(s){suffix}")
        for candidate in candidates:
            print(f"  {candidate['size_bytes']:>14}  {candidate['path']}")
    print(f"Report: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find non-empty checkpoints and LMDB shards under legacy result names.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    main()
