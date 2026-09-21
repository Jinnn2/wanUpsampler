"""Score FULL/S/T/C controlled-factor videos and build prompt-mean targets."""

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


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    write_json_atomic,
)
from UNIV_adaptor.scripts.data.run_controlled_factor_generation import (  # noqa: E402
    DATASET_SCHEMA,
    RECORD_SCHEMA,
)
from UNIV_adaptor.scripts.data.run_prompt_budget_generation import load_json  # noqa: E402
from UNIV_adaptor.scripts.data.score_phase2_dataset import output_lock  # noqa: E402


SCORE_SCHEMA = "univ_controlled_factor_vbench_scores_v1"
ANALYSIS_SCHEMA = "univ_controlled_factor_analysis_v1"
QUALITY_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
)
DIAGNOSTIC_DIMENSIONS = ("dynamic_degree", "overall_consistency")
DIMENSIONS = (*QUALITY_DIMENSIONS, *DIAGNOSTIC_DIMENSIONS)
FULL = "FULL_NATIVE_50"
ACTIONS = (
    "SPATIAL_ONLY_D050",
    "TEMPORAL_ONLY_RT036_CAL",
    "CACHE_ONLY_D050",
)


def validate_hashed(value: dict[str, Any], schema: str, hash_key: str) -> None:
    if value.get("schema") != schema:
        raise ValueError(f"unsupported schema: {value.get('schema')}")
    body = {key: item for key, item in value.items() if key not in {"schema", hash_key}}
    if canonical_sha256(body) != value.get(hash_key):
        raise ValueError(f"{schema} hash mismatch")


def collect(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_path = root / "controlled_factor_dataset.json"
    dataset = load_json(dataset_path)
    validate_hashed(dataset, DATASET_SCHEMA, "dataset_sha256")
    if sha256_file(dataset["plan_path"]) != dataset["plan_file_sha256"]:
        raise ValueError("controlled plan file hash mismatch")
    if sha256_file(dataset["generation_manifest_path"]) != dataset["generation_manifest_file_sha256"]:
        raise ValueError("controlled generation manifest file hash mismatch")
    rows: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for entry in dataset["records"]:
        path = Path(entry["record_path"])
        if sha256_file(path) != entry["record_file_sha256"]:
            raise ValueError(f"controlled record file hash mismatch: {path}")
        record = load_json(path)
        validate_hashed(record, RECORD_SCHEMA, "record_sha256")
        if record["record_sha256"] != entry["record_sha256"]:
            raise ValueError("controlled record identity mismatch")
        if record["group_id"] in seen_groups:
            raise ValueError(f"duplicate controlled group: {record['group_id']}")
        seen_groups.add(record["group_id"])
        seen_cases: set[str] = set()
        for item in record["artifacts"]:
            case_id = item["case_id"]
            if case_id in seen_cases:
                raise ValueError(f"duplicate case in {record['group_id']}: {case_id}")
            seen_cases.add(case_id)
            artifact = item["artifact"]
            video = Path(artifact["video_path"])
            if not video.is_file() or video.stat().st_size != artifact["video_bytes"]:
                raise ValueError(f"controlled video missing or size mismatch: {video}")
            if sha256_file(video) != artifact["video_sha256"]:
                raise ValueError(f"controlled video hash mismatch: {video}")
            stem = f"{record['group_id']}__{case_id}"
            rows.append(
                {
                    "observation_id": stem,
                    "stem": stem,
                    "group_id": record["group_id"],
                    "prompt_id": int(record["prompt_id"]),
                    "family_id": record["family_id"],
                    "split": record["split"],
                    "motion_level": record["motion_level"],
                    "detail_level": record["detail_level"],
                    "factor_cell": record["factor_cell"],
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
        if seen_cases != {FULL, *ACTIONS}:
            raise ValueError(f"controlled group has incomplete FULL/S/T/C coverage: {record['group_id']}")
    if len(rows) != int(dataset["counts"]["videos"]):
        raise ValueError("controlled collected video count differs from dataset")
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
            raise ValueError(f"staged video identity mismatch: {destination}")
        prompt_map[str(destination.resolve())] = row["prompt"]
    write_json_atomic(out / "prompt_map.json", prompt_map)
    return inputs


def score(rows: list[dict[str, Any]], out: Path, args: argparse.Namespace, digest: str) -> dict[str, Any]:
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
        print(f"[VBench] {dimension}: {len(rows)} videos, {args.ngpus} GPUs", flush=True)
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
            raise ValueError(f"controlled VBench coverage mismatch: {dimension}")
        for stem in scores:
            scores[stem][dimension] = float(bundle.scores[stem][dimension])
        provenance[dimension] = bundle.provenance
    body = {"input_sha256": digest, "dimensions": list(DIMENSIONS), "scores": scores, "provenance": provenance}
    payload = {"schema": SCORE_SCHEMA, "payload_sha256": canonical_sha256(body), **body}
    path = out / "scores.json"
    if path.is_file() and load_json(path) != payload:
        raise RuntimeError(f"refusing to replace different controlled scores: {path}")
    if not path.is_file():
        write_json_atomic(path, payload)
    return payload


def validate_scores(value: dict[str, Any], digest: str) -> dict[str, Any]:
    validate_hashed(value, SCORE_SCHEMA, "payload_sha256")
    if value["input_sha256"] != digest or value["dimensions"] != list(DIMENSIONS):
        raise ValueError("controlled scores belong to different inputs or dimensions")
    return value


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prompt_bootstrap(values: dict[str, list[float]], repetitions: int = 10000) -> list[float]:
    family_means = np.asarray([st.fmean(items) for items in values.values()], dtype=np.float64)
    if len(family_means) < 2:
        return [float(family_means.mean()), float(family_means.mean())]
    rng = np.random.default_rng(20260922)
    boot = family_means[rng.integers(0, len(family_means), size=(repetitions, len(family_means)))].mean(axis=1)
    return [float(value) for value in np.quantile(boot, [0.025, 0.975])]


def report(rows: list[dict[str, Any]], payload: dict[str, Any], out: Path, digest: str) -> dict[str, Any]:
    enriched = []
    for row in rows:
        values = {dimension: float(payload["scores"][row["stem"]][dimension]) for dimension in DIMENSIONS}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite controlled score: {row['stem']}")
        enriched.append({**row, **values, "vbench5": st.fmean(values[d] for d in QUALITY_DIMENSIONS)})
    video_rows = []
    for row in enriched:
        item = {
            key: row[key]
            for key in (
                "observation_id", "group_id", "prompt_id", "family_id", "split",
                "motion_level", "detail_level", "factor_cell", "prompt", "prompt_sha256",
                "base_seed", "seed", "case_id", "axis", "proxy_compute_density",
                "pipeline_seconds", "transition", "vbench5", *DIMENSIONS,
            )
        }
        item.update(row["requested_action"])
        video_rows.append(item)
    csv_write(out / "quality_by_video.csv", video_rows)

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in enriched:
        grouped[row["group_id"]][row["case_id"]] = row
    relative = []
    for group_id, cases in sorted(grouped.items()):
        reference = cases[FULL]
        for action in ACTIONS:
            candidate = cases[action]
            item = {
                key: reference[key]
                for key in (
                    "group_id", "prompt_id", "family_id", "split", "motion_level",
                    "detail_level", "factor_cell", "prompt", "prompt_sha256", "base_seed", "seed",
                )
            }
            item.update(
                {
                    "action_id": action,
                    "axis": candidate["axis"],
                    "full_vbench5": reference["vbench5"],
                    "action_vbench5": candidate["vbench5"],
                    "delta_vbench5": candidate["vbench5"] - reference["vbench5"],
                    "quality_loss_vbench5": reference["vbench5"] - candidate["vbench5"],
                    "full_seconds": reference["pipeline_seconds"],
                    "action_seconds": candidate["pipeline_seconds"],
                    "time_ratio_to_full": candidate["pipeline_seconds"] / reference["pipeline_seconds"],
                }
            )
            for dimension in QUALITY_DIMENSIONS:
                item[f"delta_{dimension}"] = candidate[dimension] - reference[dimension]
            relative.append(item)
    csv_write(out / "relative_to_full.csv", relative)

    prompt_targets = []
    by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in relative:
        by_prompt[int(row["prompt_id"])].append(row)
    for prompt_id, items in sorted(by_prompt.items()):
        identity = items[0]
        row = {
            key: identity[key]
            for key in (
                "prompt_id", "family_id", "split", "motion_level", "detail_level",
                "factor_cell", "prompt", "prompt_sha256",
            )
        }
        for action in ACTIONS:
            action_rows = [item for item in items if item["action_id"] == action]
            if len(action_rows) != 3:
                raise ValueError(f"prompt {prompt_id}/{action} does not have three seeds")
            tag = action.lower()
            row[f"mean_delta_vbench5__{tag}"] = st.fmean(item["delta_vbench5"] for item in action_rows)
            row[f"stdev_delta_vbench5__{tag}"] = st.stdev(item["delta_vbench5"] for item in action_rows)
            row[f"mean_time_ratio__{tag}"] = st.fmean(item["time_ratio_to_full"] for item in action_rows)
            for dimension in QUALITY_DIMENSIONS:
                row[f"mean_delta_{dimension}__{tag}"] = st.fmean(item[f"delta_{dimension}"] for item in action_rows)
        prompt_targets.append(row)
    prompt_inputs = [
        {
            key: row[key]
            for key in (
                "prompt_id", "family_id", "split", "motion_level", "detail_level",
                "factor_cell", "prompt", "prompt_sha256",
            )
        }
        for row in prompt_targets
    ]
    train_validation_targets = [row for row in prompt_targets if row["split"] != "test"]
    test_targets = [row for row in prompt_targets if row["split"] == "test"]
    csv_write(out / "prompt_inputs.csv", prompt_inputs)
    csv_write(out / "prompt_targets_train_validation.csv", train_validation_targets)
    csv_write(out / "prompt_targets_test.csv", test_targets)

    factor_rows = []
    # Default analysis deliberately withholds locked-test factor conclusions.
    for split in ("train", "validation"):
        for motion in ("low", "high"):
            for detail in ("low", "high"):
                selected_prompts = [
                    row for row in prompt_targets
                    if row["split"] == split and row["motion_level"] == motion and row["detail_level"] == detail
                ]
                for action in ACTIONS:
                    tag = action.lower()
                    by_family: dict[str, list[float]] = defaultdict(list)
                    for row in selected_prompts:
                        by_family[row["family_id"]].append(float(row[f"mean_delta_vbench5__{tag}"]))
                    values = [value for family in by_family.values() for value in family]
                    factor_rows.append(
                        {
                            "split": split,
                            "motion_level": motion,
                            "detail_level": detail,
                            "action_id": action,
                            "prompts": len(values),
                            "families": len(by_family),
                            "mean_delta_vbench5": st.fmean(values),
                            "family_bootstrap_ci_low": prompt_bootstrap(by_family)[0],
                            "family_bootstrap_ci_high": prompt_bootstrap(by_family)[1],
                        }
                    )
    csv_write(out / "factor_summary.csv", factor_rows)
    action_times = {}
    for case_id in (FULL, *ACTIONS):
        times = [row["pipeline_seconds"] for row in enriched if row["case_id"] == case_id]
        action_times[case_id] = {"mean_seconds": st.fmean(times), "median_seconds": st.median(times)}
    body = {
        "input_sha256": digest,
        "score_payload_sha256": payload["payload_sha256"],
        "quality_definition": "arithmetic_mean_of_five_vbench_dimensions",
        "target_definition": "three_seed_mean_action_minus_full_quality",
        "prompt_count": len(prompt_targets),
        "prompt_seed_groups": len(grouped),
        "video_count": len(enriched),
        "action_timing": action_times,
        "test_labels_scored_but_not_authorized_for_model_selection": True,
        "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    analysis = {"schema": ANALYSIS_SCHEMA, "analysis_sha256": canonical_sha256(body), **body}
    write_json_atomic(out / "analysis.json", analysis)
    lines = [
        "# Controlled prompt-factor dataset", "",
        f"Scored {len(enriched)} videos, {len(grouped)} prompt-seed groups, and {len(prompt_targets)} prompts.", "",
        "Targets are three-seed mean quality deltas relative to FULL_NATIVE_50.",
        "The test rows are scored for one-pass infrastructure convenience but must not be used by training/model selection before explicit confirmation.",
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
    out = Path(args.out_dir).resolve() if args.out_dir else root / "metrics" / "controlled_factor_vbench"
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        rows, identity = collect(root)
        input_body = {"identity": identity, "rows": rows}
        digest = canonical_sha256(input_body)
        inputs = {"input_sha256": digest, **input_body}
        input_path = out / "evaluation_inputs.json"
        if input_path.is_file() and load_json(input_path) != inputs:
            raise RuntimeError("refusing to replace changed controlled evaluation inputs")
        if not input_path.is_file():
            write_json_atomic(input_path, inputs)
        print(f"Verified {len(rows)} videos in {len({row['group_id'] for row in rows})} groups", flush=True)
        score_payload = None
        if args.mode in {"score", "all"}:
            score_payload = score(rows, out, args, digest)
        if args.mode in {"report", "all"}:
            if score_payload is None:
                score_payload = validate_scores(load_json(out / "scores.json"), digest)
            report(rows, score_payload, out, digest)
            print(out / "report.md")


if __name__ == "__main__":
    main()
