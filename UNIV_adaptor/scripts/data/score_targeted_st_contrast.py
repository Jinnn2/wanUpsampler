"""Score and analyze the targeted equal-density S/T contrast."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
import os
from pathlib import Path
import shutil
import statistics as st
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.data_protocol import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    write_json_atomic,
)
from UNIV_adaptor.scripts.data.run_targeted_st_contrast_generation import (  # noqa: E402
    DATASET_SCHEMA,
    RECORD_SCHEMA,
    load_json,
)
from UNIV_adaptor.scripts.data.score_phase2_dataset import output_lock  # noqa: E402


SCORE_SCHEMA = "univ_targeted_st_vbench_scores_v1"
ANALYSIS_SCHEMA = "univ_targeted_st_analysis_v1"
QUALITY_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
)
DIAGNOSTIC_DIMENSIONS = ("dynamic_degree", "overall_consistency")
DIMENSIONS = (*QUALITY_DIMENSIONS, *DIAGNOSTIC_DIMENSIONS)
SPATIAL = "SPATIAL_ONLY_D050"
TEMPORAL = "TEMPORAL_ONLY_D050"
LOW_MOTION = "low_motion_high_detail_small_objects"
HIGH_MOTION = "high_motion_large_subject"


def validate_hashed_object(
    value: dict[str, Any], *, schema: str, hash_key: str
) -> None:
    if value.get("schema") != schema:
        raise ValueError(f"unsupported schema: {value.get('schema')}")
    body = {key: item for key, item in value.items() if key not in {"schema", hash_key}}
    if canonical_sha256(body) != value.get(hash_key):
        raise ValueError(f"{schema} hash mismatch")


def collect(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_path = root / "targeted_st_dataset.json"
    dataset = load_json(dataset_path)
    validate_hashed_object(dataset, schema=DATASET_SCHEMA, hash_key="dataset_sha256")
    if sha256_file(dataset["plan_path"]) != dataset["plan_file_sha256"]:
        raise ValueError("targeted plan file hash mismatch")
    if (
        sha256_file(dataset["generation_manifest_path"])
        != dataset["generation_manifest_file_sha256"]
    ):
        raise ValueError("targeted generation manifest file hash mismatch")
    rows = []
    group_ids = set()
    for entry in dataset["records"]:
        record_path = Path(entry["record_path"])
        if sha256_file(record_path) != entry["record_file_sha256"]:
            raise ValueError(f"targeted record file hash mismatch: {record_path}")
        record = load_json(record_path)
        validate_hashed_object(record, schema=RECORD_SCHEMA, hash_key="record_sha256")
        if record["record_sha256"] != entry["record_sha256"]:
            raise ValueError("targeted record identity mismatch")
        group_id = record["group_id"]
        if group_id in group_ids:
            raise ValueError(f"duplicate targeted group: {group_id}")
        group_ids.add(group_id)
        case_ids = set()
        for item in record["artifacts"]:
            case_id = item["case_id"]
            if case_id in case_ids:
                raise ValueError(f"duplicate targeted case: {group_id}/{case_id}")
            case_ids.add(case_id)
            artifact = item["artifact"]
            video = Path(artifact["video_path"])
            if not video.is_file() or video.stat().st_size != artifact["video_bytes"]:
                raise ValueError(f"targeted video missing or size mismatch: {video}")
            if sha256_file(video) != artifact["video_sha256"]:
                raise ValueError(f"targeted video hash mismatch: {video}")
            stem = f"{group_id}__{case_id}"
            rows.append(
                {
                    "observation_id": stem,
                    "stem": stem,
                    "group_id": group_id,
                    "prompt_index": int(record["prompt_index"]),
                    "prompt_group": record["prompt_group"],
                    "expected_preference": record["expected_preference"],
                    "prompt": record["prompt"],
                    "prompt_sha256": record["prompt_sha256"],
                    "base_seed": int(record["base_seed"]),
                    "seed": int(record["seed"]),
                    "case_id": case_id,
                    "axis": item["axis"],
                    "proxy_compute_density": float(item["proxy_compute_density"]),
                    "requested_action": item["requested_action"],
                    "resolved_schedule": item["resolved_schedule"],
                    "transition": item["transition"],
                    "video_path": str(video.resolve()),
                    "video_sha256": artifact["video_sha256"],
                    "pipeline_seconds": float(artifact["cost"]["pipeline_seconds"]),
                }
            )
        if case_ids != {SPATIAL, TEMPORAL}:
            raise ValueError(f"targeted group lacks the exact S/T pair: {group_id}")
    if len(rows) != dataset["counts"]["videos"]:
        raise ValueError("targeted collected video count differs from manifest")
    identity = {
        "dataset_manifest_path": str(dataset_path.resolve()),
        "dataset_manifest_file_sha256": sha256_file(dataset_path),
        "dataset_sha256": dataset["dataset_sha256"],
    }
    return sorted(rows, key=lambda row: row["observation_id"]), identity


def stage_inputs(rows: list[dict[str, Any]], out: Path) -> Path:
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    prompt_map = {}
    for row in rows:
        destination = inputs / f"{row['stem']}.mp4"
        if destination.is_symlink():
            raise ValueError(f"staged symlinks are unsupported: {destination}")
        if not destination.exists():
            try:
                os.link(row["video_path"], destination)
            except OSError:
                shutil.copy2(row["video_path"], destination)
        if sha256_file(destination) != row["video_sha256"]:
            raise ValueError(f"staged targeted video identity mismatch: {destination}")
        prompt_map[str(destination.resolve())] = row["prompt"]
    write_json_atomic(out / "prompt_map.json", prompt_map)
    return inputs


def score(
    rows: list[dict[str, Any]],
    out: Path,
    args: argparse.Namespace,
    input_sha256: str,
) -> dict[str, Any]:
    from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import (
        inspect_vbench_checkout,
        score_case_directory,
    )

    inputs = stage_inputs(rows, out)
    identity = inspect_vbench_checkout(
        Path(args.vbench_root).resolve(),
        expected_commit=args.expected_vbench_commit or None,
    )
    scores = {row["stem"]: {} for row in rows}
    provenance = {}
    for dimension in DIMENSIONS:
        print(
            f"[VBench] {dimension}: {len(rows)} videos, {args.ngpus} GPUs", flush=True
        )
        bundle = score_case_directory(
            Path(args.vbench_root).resolve(),
            args.vbench_python,
            inputs,
            out / "prompt_map.json",
            out / "vbench" / dimension,
            [dimension],
            [dimension] if dimension in QUALITY_DIMENSIONS else [],
            [dimension] if dimension in DIAGNOSTIC_DIMENSIONS else [],
            args.ngpus,
            args.force_rescore,
            identity,
        )
        if set(bundle.scores) != set(scores):
            raise ValueError(f"targeted VBench coverage mismatch: {dimension}")
        for stem in scores:
            scores[stem][dimension] = float(bundle.scores[stem][dimension])
        provenance[dimension] = bundle.provenance
    body = {
        "input_sha256": input_sha256,
        "dimensions": list(DIMENSIONS),
        "scores": scores,
        "provenance": provenance,
    }
    payload = {"schema": SCORE_SCHEMA, "payload_sha256": canonical_sha256(body), **body}
    path = out / "scores.json"
    if path.is_file() and load_json(path) != payload:
        raise RuntimeError(f"refusing to replace different targeted scores: {path}")
    if not path.is_file():
        write_json_atomic(path, payload)
    return payload


def validate_scores(value: dict[str, Any], *, input_sha256: str) -> dict[str, Any]:
    validate_hashed_object(value, schema=SCORE_SCHEMA, hash_key="payload_sha256")
    if value["input_sha256"] != input_sha256 or value["dimensions"] != list(DIMENSIONS):
        raise ValueError("targeted scores belong to different inputs or dimensions")
    return value


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty targeted CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_contrast(
    prompt_rows: list[dict[str, Any]], *, repetitions: int = 20000
) -> tuple[float, float]:
    rng = np.random.default_rng(20260921)
    by_class = {
        group: np.asarray(
            [
                row["mean_delta_t_minus_s_vbench5"]
                for row in prompt_rows
                if row["prompt_group"] == group
            ],
            dtype=np.float64,
        )
        for group in (LOW_MOTION, HIGH_MOTION)
    }
    boot = []
    for _ in range(repetitions):
        low = rng.choice(
            by_class[LOW_MOTION], size=len(by_class[LOW_MOTION]), replace=True
        ).mean()
        high = rng.choice(
            by_class[HIGH_MOTION], size=len(by_class[HIGH_MOTION]), replace=True
        ).mean()
        boot.append(low - high)
    return tuple(float(value) for value in np.quantile(boot, [0.025, 0.975]))


def report(
    rows: list[dict[str, Any]],
    score_payload: dict[str, Any],
    out: Path,
    input_sha256: str,
) -> dict[str, Any]:
    scores = score_payload["scores"]
    enriched = []
    for row in rows:
        values = {
            dimension: float(scores[row["stem"]][dimension]) for dimension in DIMENSIONS
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite targeted score: {row['stem']}")
        enriched.append(
            {
                **row,
                **values,
                "vbench5": st.fmean(
                    values[dimension] for dimension in QUALITY_DIMENSIONS
                ),
            }
        )
    video_rows = []
    for row in enriched:
        item = {
            key: row[key]
            for key in (
                "observation_id",
                "group_id",
                "prompt_index",
                "prompt_group",
                "expected_preference",
                "prompt",
                "prompt_sha256",
                "base_seed",
                "seed",
                "case_id",
                "axis",
                "proxy_compute_density",
                "pipeline_seconds",
                "transition",
                "vbench5",
                *DIMENSIONS,
            )
        }
        item.update(row["requested_action"])
        video_rows.append(item)
    csv_write(out / "quality_by_video.csv", video_rows)

    grouped = defaultdict(dict)
    for row in enriched:
        grouped[row["group_id"]][row["case_id"]] = row
    pairs = []
    for group_id, pair in sorted(grouped.items()):
        spatial = pair[SPATIAL]
        temporal = pair[TEMPORAL]
        item = {
            "group_id": group_id,
            "prompt_index": spatial["prompt_index"],
            "prompt_group": spatial["prompt_group"],
            "expected_preference": spatial["expected_preference"],
            "prompt": spatial["prompt"],
            "prompt_sha256": spatial["prompt_sha256"],
            "base_seed": spatial["base_seed"],
            "seed": spatial["seed"],
            "spatial_vbench5": spatial["vbench5"],
            "temporal_vbench5": temporal["vbench5"],
            "delta_t_minus_s_vbench5": temporal["vbench5"] - spatial["vbench5"],
            "spatial_seconds": spatial["pipeline_seconds"],
            "temporal_seconds": temporal["pipeline_seconds"],
            "time_ratio_t_over_s": temporal["pipeline_seconds"]
            / spatial["pipeline_seconds"],
        }
        for dimension in QUALITY_DIMENSIONS:
            item[f"delta_t_minus_s_{dimension}"] = (
                temporal[dimension] - spatial[dimension]
            )
        item["observed_preference"] = (
            TEMPORAL if item["delta_t_minus_s_vbench5"] > 0 else SPATIAL
        )
        item["matches_expected"] = (
            item["observed_preference"] == item["expected_preference"]
        )
        pairs.append(item)
    csv_write(out / "paired_st.csv", pairs)

    prompt_rows = []
    by_prompt = defaultdict(list)
    for row in pairs:
        by_prompt[row["prompt_index"]].append(row)
    for prompt_index, items in sorted(by_prompt.items()):
        deltas = [row["delta_t_minus_s_vbench5"] for row in items]
        expected = items[0]["expected_preference"]
        expected_sign = 1 if expected == TEMPORAL else -1
        prompt_rows.append(
            {
                "prompt_index": prompt_index,
                "prompt_group": items[0]["prompt_group"],
                "expected_preference": expected,
                "prompt": items[0]["prompt"],
                "seeds": len(items),
                "mean_delta_t_minus_s_vbench5": st.fmean(deltas),
                "stdev_delta_t_minus_s_vbench5": st.stdev(deltas),
                "all_seed_same_preference": all(value > 0 for value in deltas)
                or all(value < 0 for value in deltas),
                "expected_seed_fraction": st.fmean(
                    float(expected_sign * value > 0) for value in deltas
                ),
                "mean_matches_expected": expected_sign * st.fmean(deltas) > 0,
            }
        )
    csv_write(out / "prompt_summary.csv", prompt_rows)

    class_rows = []
    for prompt_group in (LOW_MOTION, HIGH_MOTION):
        class_prompts = [
            row for row in prompt_rows if row["prompt_group"] == prompt_group
        ]
        class_pairs = [row for row in pairs if row["prompt_group"] == prompt_group]
        class_rows.append(
            {
                "prompt_group": prompt_group,
                "prompts": len(class_prompts),
                "prompt_seed_groups": len(class_pairs),
                "mean_delta_t_minus_s_vbench5": st.fmean(
                    row["mean_delta_t_minus_s_vbench5"] for row in class_prompts
                ),
                "expected_seed_accuracy": st.fmean(
                    float(row["matches_expected"]) for row in class_pairs
                ),
                "expected_prompt_accuracy": st.fmean(
                    float(row["mean_matches_expected"]) for row in class_prompts
                ),
                "unanimous_prompt_fraction": st.fmean(
                    float(row["all_seed_same_preference"]) for row in class_prompts
                ),
            }
        )
    csv_write(out / "class_summary.csv", class_rows)
    class_map = {row["prompt_group"]: row for row in class_rows}
    contrast = (
        class_map[LOW_MOTION]["mean_delta_t_minus_s_vbench5"]
        - class_map[HIGH_MOTION]["mean_delta_t_minus_s_vbench5"]
    )
    ci_low, ci_high = bootstrap_contrast(prompt_rows)
    spatial_time = st.fmean(row["spatial_seconds"] for row in pairs)
    temporal_time = st.fmean(row["temporal_seconds"] for row in pairs)
    time_relative_gap = abs(temporal_time - spatial_time) / st.fmean(
        (spatial_time, temporal_time)
    )
    matched_latency = time_relative_gap <= 0.05
    hypothesis_pass = (
        class_map[LOW_MOTION]["mean_delta_t_minus_s_vbench5"] > 0
        and class_map[HIGH_MOTION]["mean_delta_t_minus_s_vbench5"] < 0
        and contrast > 0
        and matched_latency
    )
    body = {
        "input_sha256": input_sha256,
        "score_payload_sha256": score_payload["payload_sha256"],
        "quality_definition": "arithmetic_mean_of_five_vbench_dimensions",
        "primary_estimand": "Q_temporal_minus_Q_spatial_at_equal_proxy_density",
        "target_proxy_compute_density": 0.5,
        "class_summary": class_rows,
        "difference_in_differences": contrast,
        "difference_in_differences_prompt_bootstrap_ci": [ci_low, ci_high],
        "mean_spatial_seconds": spatial_time,
        "mean_temporal_seconds": temporal_time,
        "relative_latency_gap": time_relative_gap,
        "measured_latency_matched_within_5pct": matched_latency,
        "directional_hypothesis_pass": hypothesis_pass,
        "development_only": True,
        "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "analysis_sha256": canonical_sha256(body),
        **body,
    }
    write_json_atomic(out / "analysis.json", analysis)
    lines = [
        "# Targeted equal-density spatial vs temporal contrast",
        "",
        "Development-only prompt-group existence test. Both methods use proxy density 0.5.",
        "",
        "| Prompt group | Prompts | Groups | Mean Q(T)-Q(S) | Seed accuracy | Prompt accuracy | Unanimous |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in class_rows:
        lines.append(
            f"| {row['prompt_group']} | {row['prompts']} | {row['prompt_seed_groups']} | "
            f"{row['mean_delta_t_minus_s_vbench5']:.6f} | {row['expected_seed_accuracy']:.1%} | "
            f"{row['expected_prompt_accuracy']:.1%} | {row['unanimous_prompt_fraction']:.1%} |"
        )
    lines += [
        "",
        f"- Difference-in-differences: {contrast:.6f}; prompt bootstrap 95% CI [{ci_low:.6f}, {ci_high:.6f}].",
        f"- Mean seconds: spatial={spatial_time:.3f}, temporal={temporal_time:.3f}; relative gap={time_relative_gap:.2%}.",
        f"- Measured latency matched within 5%: {matched_latency}.",
        f"- Directional hypothesis pass: {hypothesis_pass}.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "score", "report", "all"))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--vbench-root", default="/mnt/afs_2/houze/VBench")
    parser.add_argument("--vbench-python", default=sys.executable)
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--expected-vbench-commit", default="")
    parser.add_argument("--force-rescore", action="store_true")
    args = parser.parse_args()
    root = Path(args.dataset_root).resolve()
    out = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else root / "metrics" / "targeted_st_vbench"
    )
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        rows, identity = collect(root)
        input_body = {"identity": identity, "rows": rows}
        input_sha256 = canonical_sha256(input_body)
        inputs = {"input_sha256": input_sha256, **input_body}
        input_path = out / "evaluation_inputs.json"
        if input_path.is_file() and load_json(input_path) != inputs:
            raise RuntimeError("refusing to replace changed targeted evaluation inputs")
        if not input_path.is_file():
            write_json_atomic(input_path, inputs)
        print(
            f"Verified {len(rows)} videos in {len({r['group_id'] for r in rows})} groups",
            flush=True,
        )
        if args.mode in {"score", "all"}:
            payload = score(rows, out, args, input_sha256)
        else:
            payload = None
        if args.mode in {"report", "all"}:
            if payload is None:
                payload = validate_scores(
                    load_json(out / "scores.json"), input_sha256=input_sha256
                )
            report(rows, payload, out, input_sha256)
            print(out / "report.md")


if __name__ == "__main__":
    main()
