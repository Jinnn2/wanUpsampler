from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    variants = registry.get("variants", [])
    if len(variants) < 6:
        raise SystemExit("A controlled ablation registry must contain at least six variants")
    axes = {str(variant.get("axis", "")) for variant in variants}
    required_axes = {"target_modules", "rank", "loss"} if args.kind == "lora" else {"architecture", "loss"}
    if not required_axes.issubset(axes):
        raise SystemExit("Ablation registry missing controlled axes: " + ", ".join(sorted(required_axes - axes)))
    rows: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    for variant in variants:
        checkpoint = resolve(registry_path, variant["checkpoint"])
        metrics_path = resolve(registry_path, variant["metrics_csv"])
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise SystemExit(f"Missing checkpoint for {variant.get('variant')}: {checkpoint}")
        metrics = read_csv(metrics_path)
        columns = variant.get("columns", {})
        metric_column = columns.get("metric", "metric")
        value_column = columns.get("value", "variant_mean" if args.kind == "lora" else "stage2_mean")
        samples_column = columns.get("samples", "samples")
        for source in metrics:
            for required in (metric_column, value_column, samples_column):
                if not str(source.get(required, "")).strip():
                    raise SystemExit(f"{metrics_path}: required column/value {required!r} is missing")
            row = {
                "axis": variant["axis"],
                "variant": variant["variant"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "metric": source[metric_column],
                "samples": source[samples_column],
                "better": source.get(columns.get("better", "better"), ""),
            }
            if args.kind == "lora":
                row.update(
                    {
                        "target_modules": variant["target_modules"],
                        "rank": variant["rank"],
                        "loss": variant["loss"],
                        "train_steps": variant["train_steps"],
                        "train_seed": variant["train_seed"],
                        "lora_strength": variant["lora_strength"],
                        "variant_mean": source[value_column],
                    }
                )
            else:
                row.update(
                    {
                        "architecture": variant["architecture"],
                        "prediction_mode": variant["prediction_mode"],
                        "loss": variant["loss"],
                        "train_steps": variant["train_steps"],
                        "train_seed": variant["train_seed"],
                        "stage2_mean": source[value_column],
                    }
                )
            rows.append(row)
        fingerprints[variant["variant"]] = row["checkpoint_sha256"]
    if len(fingerprints) != len(variants):
        raise SystemExit("variant names must be unique")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    provenance = output.with_suffix(".provenance.json")
    provenance.write_text(
        json.dumps({"kind": args.kind, "registry": str(registry_path), "checkpoint_sha256": fingerprints}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Ablation CSV       : {output}")
    print(f"Ablation provenance: {provenance}")


def resolve(registry: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (registry.parent / path).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Metrics CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Metrics CSV contains no rows: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile controlled ablation metrics with checkpoint provenance.")
    parser.add_argument("--kind", choices=["lora", "stage2"], required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
