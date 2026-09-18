from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import statistics as st
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.data_protocol import canonical_sha256, sha256_file  # noqa: E402
from UNIV_adaptor.sparse_action_protocol import (  # noqa: E402
    validate_sparse_plan,
    validate_sparse_record,
    write_json_atomic,
)
from UNIV_adaptor.scripts.data.run_sparse_action_generation import (  # noqa: E402
    load_json,
    validate_dataset_manifest,
)


QUALITY_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
)
DIAGNOSTIC_DIMENSIONS = ("dynamic_degree", "overall_consistency")
DIMENSIONS = QUALITY_DIMENSIONS + DIAGNOSTIC_DIMENSIONS
SCORE_SCHEMA = "univ_sparse_action_vbench_scores_v1"
SCORED_DATASET_SCHEMA = "univ_sparse_action_scored_dataset_v1"


def collect(dataset_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_path = dataset_root / "sparse_dataset_manifest.json"
    dataset = validate_dataset_manifest(load_json(dataset_path))
    if sha256_file(dataset["plan_path"]) != dataset["plan_file_sha256"]:
        raise ValueError("sparse plan file identity changed after finalization")
    plan = validate_sparse_plan(load_json(dataset["plan_path"]))
    if dataset["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("dataset/plan hash mismatch")
    expected_groups = {group["group_id"]: group for group in plan["groups"]}
    index_groups = {row["group_id"] for row in dataset["records"]}
    if index_groups != set(expected_groups) or len(index_groups) != len(
        dataset["records"]
    ):
        raise ValueError(
            "dataset record index does not exactly cover sparse plan groups"
        )

    rows = []
    video_paths = set()
    artifact_keys = set()
    for item in dataset["records"]:
        path = Path(item["path"]).resolve()
        if sha256_file(path) != item["file_sha256"]:
            raise ValueError(f"sparse record file identity changed: {path}")
        record = validate_sparse_record(
            load_json(path), expected_plan_sha256=plan["plan_sha256"]
        )
        expected = expected_groups[record["group_id"]]
        for key in (
            "group_id",
            "prompt_key",
            "cohort",
            "prompt",
            "prompt_sha256",
            "source_prompt_id",
            "base_seed",
            "seed",
        ):
            if record[key] != expected[key]:
                raise ValueError(f"record identity mismatch: {path}: {key}")
        expected_actions = {row["observation_id"]: row for row in expected["actions"]}
        if {row["observation_id"] for row in record["actions"]} != set(
            expected_actions
        ):
            raise ValueError(f"record action coverage mismatch: {path}")
        for action_row in record["actions"]:
            observation_id = action_row["observation_id"]
            planned = expected_actions[observation_id]
            if (
                action_row["action_id"] != planned["action_id"]
                or action_row["artifact_mode"] != planned["artifact_mode"]
                or action_row["action"]
                != plan["action_catalog"][action_row["action_id"]]
            ):
                raise ValueError(
                    f"record action differs from sparse plan: {observation_id}"
                )
            artifact = action_row["artifact"]
            video = Path(artifact["video_path"]).resolve()
            if observation_id in artifact_keys:
                raise ValueError(f"duplicate observation id: {observation_id}")
            artifact_keys.add(observation_id)
            if str(video) in video_paths:
                raise ValueError(
                    f"video path reused by two sparse observations: {video}"
                )
            video_paths.add(str(video))
            if (
                not video.is_file()
                or video.stat().st_size != int(artifact["video_bytes"])
                or sha256_file(video) != artifact["video_sha256"]
            ):
                raise ValueError(f"video identity mismatch: {video}")
            sidecar_path = artifact.get("runtime_sidecar_path")
            sidecar_hash = artifact.get("runtime_sidecar_sha256")
            if bool(sidecar_path) != bool(sidecar_hash):
                raise ValueError("incomplete runtime sidecar identity")
            if sidecar_path and sha256_file(sidecar_path) != sidecar_hash:
                raise ValueError(f"runtime sidecar identity mismatch: {sidecar_path}")
            seconds = float(artifact["cost"]["pipeline_seconds"])
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("pipeline_seconds must be finite and positive")
            action = action_row["action"]
            rows.append(
                {
                    "observation_id": observation_id,
                    "stem": observation_id,
                    "group_id": record["group_id"],
                    "prompt_key": record["prompt_key"],
                    "cohort": record["cohort"],
                    "prompt": record["prompt"],
                    "prompt_sha256": record["prompt_sha256"],
                    "source_prompt_id": record["source_prompt_id"],
                    "base_seed": int(record["base_seed"]),
                    "seed": int(record["seed"]),
                    "action_id": action_row["action_id"],
                    "role": action["role"],
                    "levels": action["levels"],
                    "active_mask": action["active_mask"],
                    "requested_action": action["requested_action"],
                    "transition": action["transition"],
                    "action_key": action["action_key"],
                    "proxy_compute_density": action["proxy_compute_density"],
                    "resolved_schedule": action["resolved_schedule"],
                    "artifact_mode": action_row["artifact_mode"],
                    "video_path": str(video),
                    "video_sha256": artifact["video_sha256"],
                    "pipeline_seconds": seconds,
                    "record_file_sha256": item["file_sha256"],
                }
            )
    if len(rows) != int(dataset["counts"]["scored_videos"]):
        raise ValueError("collected video count differs from sparse dataset manifest")
    validate_pairing(rows, plan["protocol"])
    identity = {
        "dataset_manifest_path": str(dataset_path.resolve()),
        "dataset_manifest_file_sha256": sha256_file(dataset_path),
        "dataset_sha256": dataset["dataset_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "protocol_sha256": plan["protocol_sha256"],
    }
    return sorted(rows, key=lambda row: row["observation_id"]), identity


def validate_pairing(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_probes: dict[str, tuple[str, ...]] = {}
    for row in rows:
        groups[row["group_id"]].append(row)
    for group_rows in groups.values():
        references = [row for row in group_rows if row["action_id"] == "REFERENCE"]
        probes = sorted(
            row["action_id"] for row in group_rows if row["action_id"] != "REFERENCE"
        )
        if len(references) != 1 or len(probes) != protocol["probes_per_prompt"]:
            raise ValueError(
                "each sparse prompt-seed group needs one reference and all probes"
            )
        prompt_key = group_rows[0]["prompt_key"]
        previous = prompt_probes.setdefault(prompt_key, tuple(probes))
        if previous != tuple(probes):
            raise ValueError("probe assignment changed across seeds")
        if (
            len({row["prompt"] for row in group_rows}) != 1
            or len({row["seed"] for row in group_rows}) != 1
        ):
            raise ValueError("inconsistent prompt or seed within sparse group")
    seeds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if row["action_id"] == "REFERENCE":
            seeds[row["prompt_key"]].add(row["base_seed"])
    expected_seeds = set(protocol["base_seeds"])
    if any(value != expected_seeds for value in seeds.values()):
        raise ValueError("every sparse prompt must have all declared base seeds")


def stage_inputs(rows: list[dict[str, Any]], out: Path) -> Path:
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    expected = {f"{row['stem']}.mp4" for row in rows}
    extras = {path.name for path in inputs.iterdir()} - expected
    if extras:
        raise ValueError(
            f"unexpected staged files; use a separate score output directory: {sorted(extras)[:5]}"
        )
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
    vbench_root = Path(args.vbench_root).resolve()
    identity = inspect_vbench_checkout(
        vbench_root, expected_commit=args.expected_vbench_commit or None
    )
    scores = {row["stem"]: {} for row in rows}
    provenance = {}
    for dimension in DIMENSIONS:
        print(
            f"[VBench] {dimension}: {len(rows)} sparse observations, {args.ngpus} GPUs",
            flush=True,
        )
        bundle = score_case_directory(
            vbench_root,
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
            raise ValueError(f"VBench coverage mismatch: {dimension}")
        for stem in scores:
            scores[stem][dimension] = float(bundle.scores[stem][dimension])
        provenance[dimension] = bundle.provenance
    body = {
        "input_sha256": input_sha256,
        "dimensions": list(DIMENSIONS),
        "scores": scores,
        "provenance": provenance,
    }
    payload = {
        "schema": SCORE_SCHEMA,
        "payload_sha256": canonical_sha256(body),
        **body,
    }
    score_path = out / "scores.json"
    if score_path.is_file() and load_json(score_path) != payload:
        raise RuntimeError(f"refusing to replace different sparse scores: {score_path}")
    if not score_path.is_file():
        write_json_atomic(score_path, payload)
    return payload


def validate_scores(value: dict[str, Any], *, input_sha256: str) -> dict[str, Any]:
    if value.get("schema") != SCORE_SCHEMA:
        raise ValueError("unsupported sparse score bundle")
    body = {
        key: item
        for key, item in value.items()
        if key not in {"schema", "payload_sha256"}
    }
    if canonical_sha256(body) != value.get("payload_sha256"):
        raise ValueError("sparse score payload hash mismatch")
    if value.get("input_sha256") != input_sha256:
        raise ValueError("scores belong to different sparse evaluation inputs")
    if value.get("dimensions") != list(DIMENSIONS):
        raise ValueError("sparse score dimensions mismatch")
    return value


def report(
    rows: list[dict[str, Any]],
    score_payload: dict[str, Any],
    out: Path,
    input_sha256: str,
) -> dict[str, Any]:
    scores = score_payload["scores"]
    if set(scores) != {row["stem"] for row in rows}:
        raise ValueError("score coverage does not exactly match sparse observations")
    dimension_ids = sorted(
        {dimension_id for row in rows for dimension_id in (row["levels"] or {}).keys()}
    )
    enriched = []
    for row in rows:
        values = {
            dimension: float(scores[row["stem"]][dimension]) for dimension in DIMENSIONS
        }
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in values.values()
        ):
            raise ValueError(f"invalid VBench value: {row['stem']}")
        enriched.append(
            {
                **row,
                **values,
                "vbench5": st.fmean(
                    values[dimension] for dimension in QUALITY_DIMENSIONS
                ),
            }
        )
    reference = {
        row["group_id"]: row for row in enriched if row["action_id"] == "REFERENCE"
    }
    relative = []
    for row in enriched:
        if row["action_id"] == "REFERENCE":
            continue
        ref = reference[row["group_id"]]
        item = {
            "observation_id": row["observation_id"],
            "group_id": row["group_id"],
            "prompt_key": row["prompt_key"],
            "cohort": row["cohort"],
            "prompt": row["prompt"],
            "prompt_sha256": row["prompt_sha256"],
            "source_prompt_id": row["source_prompt_id"],
            "base_seed": row["base_seed"],
            "seed": row["seed"],
            "action_id": row["action_id"],
            "reference_action_id": "REFERENCE",
            "vbench5": row["vbench5"],
            "reference_vbench5": ref["vbench5"],
            "delta_vbench5": row["vbench5"] - ref["vbench5"],
            "pipeline_seconds": row["pipeline_seconds"],
            "reference_pipeline_seconds": ref["pipeline_seconds"],
            "delta_pipeline_seconds": row["pipeline_seconds"] - ref["pipeline_seconds"],
            "time_ratio_to_reference": row["pipeline_seconds"]
            / ref["pipeline_seconds"],
            "transition": row["transition"],
        }
        for field, value in row["requested_action"].items():
            item[field] = value
        for dimension_id in dimension_ids:
            item[f"level_{dimension_id}"] = row["levels"][dimension_id]
            item[f"active_{dimension_id}"] = row["active_mask"][dimension_id]
        for dimension in QUALITY_DIMENSIONS:
            item[f"delta_{dimension}"] = row[dimension] - ref[dimension]
        relative.append(item)

    video_csv_rows = []
    for row in enriched:
        item = {
            key: row[key]
            for key in (
                "observation_id",
                "group_id",
                "prompt_key",
                "cohort",
                "prompt",
                "prompt_sha256",
                "source_prompt_id",
                "base_seed",
                "seed",
                "action_id",
                "role",
                "artifact_mode",
                "pipeline_seconds",
                "proxy_compute_density",
                "transition",
                "vbench5",
                *DIMENSIONS,
            )
        }
        for dimension_id in dimension_ids:
            item[f"level_{dimension_id}"] = (
                "" if row["levels"] is None else row["levels"][dimension_id]
            )
            item[f"active_{dimension_id}"] = row["active_mask"][dimension_id]
        for field, value in row["requested_action"].items():
            item[field] = value
        video_csv_rows.append(item)
    csv_write(out / "quality_by_video.csv", video_csv_rows)
    csv_write(out / "relative_quality_pairs.csv", relative)

    action_summary = []
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in relative:
        by_action[row["action_id"]].append(row)
    for action_id, action_rows in sorted(by_action.items()):
        prompt_deltas: dict[str, list[float]] = defaultdict(list)
        prompt_times: dict[str, list[float]] = defaultdict(list)
        for row in action_rows:
            prompt_deltas[row["prompt_key"]].append(row["delta_vbench5"])
            prompt_times[row["prompt_key"]].append(row["time_ratio_to_reference"])
        action_summary.append(
            {
                "action_id": action_id,
                "prompts": len(prompt_deltas),
                "videos": len(action_rows),
                "mean_prompt_delta_vbench5": st.fmean(
                    st.fmean(values) for values in prompt_deltas.values()
                ),
                "mean_prompt_time_ratio_to_reference": st.fmean(
                    st.fmean(values) for values in prompt_times.values()
                ),
                "negative_delta_rate": st.fmean(
                    float(row["delta_vbench5"] < 0.0) for row in action_rows
                ),
            }
        )
    csv_write(out / "action_summary.csv", action_summary)

    prompt_seed_stability = []
    by_prompt_action: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in relative:
        by_prompt_action[(row["prompt_key"], row["action_id"])].append(row)
    for (prompt_key, action_id), action_rows in sorted(by_prompt_action.items()):
        deltas = [row["delta_vbench5"] for row in action_rows]
        prompt_seed_stability.append(
            {
                "prompt_key": prompt_key,
                "cohort": action_rows[0]["cohort"],
                "action_id": action_id,
                "seed_count": len(deltas),
                "mean_delta_vbench5": st.fmean(deltas),
                "stdev_delta_vbench5": st.stdev(deltas) if len(deltas) > 1 else 0.0,
                "positive_seed_count": sum(value > 0.0 for value in deltas),
                "negative_seed_count": sum(value < 0.0 for value in deltas),
                "unanimous_sign": all(value > 0.0 for value in deltas)
                or all(value < 0.0 for value in deltas),
            }
        )
    csv_write(out / "seed_stability.csv", prompt_seed_stability)

    output_files = {}
    for name in (
        "quality_by_video.csv",
        "relative_quality_pairs.csv",
        "action_summary.csv",
        "seed_stability.csv",
    ):
        path = out / name
        output_files[name] = {
            "path": str(path.resolve()),
            "file_sha256": sha256_file(path),
        }
    catalog = {
        row["action_id"]: {
            "role": row["role"],
            "levels": row["levels"],
            "active_mask": row["active_mask"],
            "requested_action": row["requested_action"],
            "transition": row["transition"],
            "action_key": row["action_key"],
            "resolved_schedule": row["resolved_schedule"],
        }
        for row in enriched
    }
    body = {
        "input_sha256": input_sha256,
        "score_payload_sha256": score_payload["payload_sha256"],
        "quality_profile": "strict_vbench5_arithmetic_mean_v1",
        "lambda_bound": False,
        "hard_oracle_labels_created": False,
        "video_count": len(enriched),
        "relative_pair_count": len(relative),
        "prompt_count": len({row["prompt_key"] for row in enriched}),
        "group_count": len({row["group_id"] for row in enriched}),
        "action_catalog": catalog,
        "action_summary": action_summary,
        "output_files": output_files,
    }
    scored_dataset = {
        "schema": SCORED_DATASET_SCHEMA,
        "dataset_sha256": canonical_sha256(body),
        **body,
    }
    scored_path = out / "scored_dataset.json"
    if scored_path.is_file() and load_json(scored_path) != scored_dataset:
        raise RuntimeError(
            f"refusing to replace different scored dataset: {scored_path}"
        )
    if not scored_path.is_file():
        write_json_atomic(scored_path, scored_dataset)
    lines = [
        "# Sparse prompt-action scoring report",
        "",
        f"- Videos: {len(enriched)}",
        f"- Prompt-seed groups: {body['group_count']}",
        f"- Reference-relative probe pairs: {len(relative)}",
        "- Quality: arithmetic mean of the five declared VBench dimensions.",
        "- Lambda: not bound; quality and measured time remain separate.",
        "- Labels: no per-prompt oracle class was created.",
        "",
        "| Action | Prompts | Videos | Mean delta VBench5 | Mean time/reference | Negative rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in action_summary:
        lines.append(
            f"| {row['action_id']} | {row['prompts']} | {row['videos']} | "
            f"{row['mean_prompt_delta_vbench5']:.6f} | "
            f"{row['mean_prompt_time_ratio_to_reference']:.4f} | "
            f"{row['negative_delta_rate']:.3f} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return scored_dataset


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def output_lock(out: Path):
    lock = out / ".evaluation.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(
            f"evaluation running or stale lock: {lock}; inspect owner.json/process"
        ) from None
    try:
        write_json_atomic(
            lock / "owner.json",
            {"pid": os.getpid(), "host": __import__("socket").gethostname()},
        )
        yield
    finally:
        (lock / "owner.json").unlink(missing_ok=True)
        lock.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score sparse prompt-action observations with strict paired provenance"
    )
    parser.add_argument("mode", choices=("check", "score", "report", "all"))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--vbench-root", default="/mnt/afs_2/houze/VBench")
    parser.add_argument("--vbench-python", default=sys.executable)
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--expected-vbench-commit", default="")
    parser.add_argument("--force-rescore", action="store_true")
    args = parser.parse_args()
    if args.ngpus < 1:
        parser.error("ngpus must be positive")
    root = Path(args.dataset_root).resolve()
    out = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else root / "metrics" / "sparse_action_vbench"
    )
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        rows, identity = collect(root)
        input_body = {"identity": identity, "rows": rows}
        input_sha256 = canonical_sha256(input_body)
        inputs = {"input_sha256": input_sha256, **input_body}
        input_path = out / "evaluation_inputs.json"
        if input_path.is_file() and load_json(input_path) != inputs:
            raise RuntimeError(
                f"refusing to replace changed evaluation inputs: {input_path}"
            )
        if not input_path.is_file():
            write_json_atomic(input_path, inputs)
        print(
            f"Verified {len(rows)} videos, "
            f"{len({row['group_id'] for row in rows})} prompt-seed groups; output: {out}",
            flush=True,
        )
        if args.mode in {"score", "all"}:
            score(rows, out, args, input_sha256)
        if args.mode in {"report", "all"}:
            payload = validate_scores(
                load_json(out / "scores.json"), input_sha256=input_sha256
            )
            report(rows, payload, out, input_sha256)
            print(out / "report.md")


if __name__ == "__main__":
    main()
