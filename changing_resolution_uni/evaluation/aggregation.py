from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOWER_IS_BETTER = {
    "latent_charbonnier": True,
    "latent_mae": True,
    "latent_mse": True,
    "latent_rmse": True,
    "content_l1": True,
    "temporal_delta_l1": True,
    "pixel_mse": True,
    "rgb_temporal_delta_l1": True,
    "hf_energy_error": True,
    "lpips": True,
    "psnr": False,
    "ssim": False,
}


IDENTITY_COLUMNS = {
    "kind",
    "checkpoint",
    "checkpoint_path",
    "checkpoint_step",
    "checkpoint_sha256",
    "method",
    "weights",
    "variant",
    "precision",
    "vae_backend",
    "source_uid",
    "source_index",
    "shard",
    "row_id",
    "scale",
    "status",
    "error",
    "source_size",
    "target_size",
    "actual_scale_hw",
    "frames",
    "grid_unit",
    "latency_ms",
    "peak_memory_mb",
    "paths",
}


def read_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def numeric_metric_names(rows: Iterable[dict[str, Any]]) -> list[str]:
    names = set()
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        for key, value in row.items():
            if key not in IDENTITY_COLUMNS and _finite_number(value):
                names.add(key)
    return sorted(names)


def deduplicate_sample_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last completed row for a checkpoint/method/source/scale key."""

    indexed: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("kind", "")),
            str(row.get("checkpoint", "shared")),
            str(row.get("method", "")),
            str(row.get("variant", "default")),
            str(row.get("source_uid", "")),
            str(row.get("scale", "")),
        )
        indexed[key] = row
    return list(indexed.values())


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 1234,
) -> list[dict[str, Any]]:
    """Aggregate per-video rows by checkpoint/method and cluster-bootstrap sources."""

    rows = deduplicate_sample_rows(rows)
    metrics = numeric_metric_names(rows)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        key = (
            str(row.get("checkpoint", "shared")),
            str(row["method"]),
            str(row.get("variant", "default")),
            str(row["scale"]),
        )
        groups[key].append(row)

    output = []
    for (checkpoint, method, variant, scale), group in sorted(groups.items()):
        output.extend(
            _summarize_group(
                checkpoint,
                method,
                variant,
                scale,
                group,
                metrics,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        )

    expected_scales: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row.get("status", "ok") == "ok":
            expected_scales[
                (
                    str(row.get("checkpoint", "shared")),
                    str(row.get("variant", "default")),
                )
            ].add(str(row["scale"]))
    macro_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status", "ok") == "ok":
            macro_groups[
                (
                    str(row.get("checkpoint", "shared")),
                    str(row["method"]),
                    str(row.get("variant", "default")),
                )
            ].append(row)
    for (checkpoint, method, variant), group in sorted(macro_groups.items()):
        per_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            per_source[str(row["source_uid"])].append(row)
        macro_rows = []
        for source_uid, source_rows in per_source.items():
            present_scales = {str(row["scale"]) for row in source_rows}
            if present_scales != expected_scales[(checkpoint, variant)]:
                continue
            item: dict[str, Any] = {"source_uid": source_uid}
            for metric in metrics:
                values = [
                    float(row[metric])
                    for row in source_rows
                    if _finite_number(row.get(metric))
                ]
                if values:
                    item[metric] = float(np.mean(values))
            macro_rows.append(item)
        output.extend(
            _summarize_group(
                checkpoint,
                method,
                variant,
                "macro",
                macro_rows,
                metrics,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        )
    return output


def paired_comparisons(
    rows: list[dict[str, Any]],
    *,
    references: tuple[str, ...] = ("trilinear", "raw"),
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 1234,
) -> list[dict[str, Any]]:
    rows = deduplicate_sample_rows(rows)
    metrics = [
        name
        for name in numeric_metric_names(rows)
        if name not in {"target_hf_energy", "prediction_hf_energy"}
    ]
    indexed: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    methods_by_checkpoint: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        checkpoint = str(row.get("checkpoint", "shared"))
        method = str(row["method"])
        variant = str(row.get("variant", "default"))
        indexed[
            (checkpoint, method, variant, str(row["scale"]), str(row["source_uid"]))
        ] = row
        methods_by_checkpoint[(checkpoint, variant)].add(method)

    output = []
    for (checkpoint, variant), methods in sorted(methods_by_checkpoint.items()):
        for reference in references:
            if reference not in methods:
                continue
            for method in sorted(methods - {reference}):
                scales = sorted(
                    {
                        key[3]
                        for key in indexed
                        if key[0] == checkpoint
                        and key[2] == variant
                        and key[1] in {method, reference}
                    },
                    key=_scale_sort_key,
                )
                for scale in scales:
                    pairs = []
                    source_ids = sorted(
                        {
                            key[4]
                            for key in indexed
                            if key[0] == checkpoint
                            and key[2] == variant
                            and key[3] == scale
                            and key[1] == method
                        }
                    )
                    for source_uid in source_ids:
                        candidate = indexed.get(
                            (checkpoint, method, variant, scale, source_uid)
                        )
                        baseline = indexed.get(
                            (checkpoint, reference, variant, scale, source_uid)
                        )
                        if candidate is not None and baseline is not None:
                            pairs.append((source_uid, candidate, baseline))
                    for metric in metrics:
                        values = []
                        improvements = []
                        for _, candidate, baseline in pairs:
                            if not (
                                _finite_number(candidate.get(metric))
                                and _finite_number(baseline.get(metric))
                            ):
                                continue
                            delta = float(candidate[metric]) - float(baseline[metric])
                            values.append(delta)
                            improvements.append(
                                -delta if LOWER_IS_BETTER.get(metric, True) else delta
                            )
                        if not values:
                            continue
                        low, high = bootstrap_ci(
                            np.asarray(improvements, dtype=np.float64),
                            samples=bootstrap_samples,
                            seed=bootstrap_seed,
                        )
                        output.append(
                            {
                                "checkpoint": checkpoint,
                                "method": method,
                                "reference": reference,
                                "variant": variant,
                                "scale": scale,
                                "metric": metric,
                                "count": len(values),
                                "delta_method_minus_reference": float(np.mean(values)),
                                "improvement": float(np.mean(improvements)),
                                "improvement_ci95_low": low,
                                "improvement_ci95_high": high,
                                "win_rate": float(
                                    np.mean(np.asarray(improvements) > 0)
                                ),
                            }
                        )
    return output


def write_summary_files(
    rows: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    stem: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[Path, Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = aggregate_rows(
        rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    pairs = paired_comparisons(
        rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    json_path = out_dir / f"{stem}_summary.json"
    csv_path = out_dir / f"{stem}_summary.csv"
    pair_path = out_dir / f"{stem}_paired.csv"
    json_path.write_text(
        json.dumps(
            {"summary": summaries, "paired": pairs}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    _write_csv(csv_path, summaries)
    _write_csv(pair_path, pairs)
    return json_path, csv_path, pair_path


def bootstrap_ci(values: np.ndarray, *, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or samples <= 0:
        value = float(values.mean())
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = 1000
    for start in range(0, samples, chunk):
        end = min(samples, start + chunk)
        indices = rng.integers(0, values.size, size=(end - start, values.size))
        means[start:end] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _summarize_group(
    checkpoint: str,
    method: str,
    variant: str,
    scale: str,
    rows: list[dict[str, Any]],
    metrics: list[str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    output = []
    for metric in metrics:
        per_source: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if _finite_number(row.get(metric)):
                per_source[str(row["source_uid"])].append(float(row[metric]))
        values = np.asarray(
            [float(np.mean(source_values)) for source_values in per_source.values()],
            dtype=np.float64,
        )
        if not values.size:
            continue
        low, high = bootstrap_ci(values, samples=bootstrap_samples, seed=bootstrap_seed)
        output.append(
            {
                "checkpoint": checkpoint,
                "method": method,
                "variant": variant,
                "scale": scale,
                "metric": metric,
                "count": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _scale_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return 0, float(value)
    except ValueError:
        return 1, value
