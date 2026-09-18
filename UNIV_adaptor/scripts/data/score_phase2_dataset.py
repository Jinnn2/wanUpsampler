"""Score finalized Phase 2 v2 records without changing generation artifacts."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import (
    build_collection_plan, canonical_sha256, sha256_file,
    validate_collection_plan, validate_trajectory_record, write_json_atomic,
)
from UNIV_adaptor.scripts.data.run_prompt_budget_generation import load_json, validate_manifest
from UNIV_adaptor.scripts.data.phase2_analysis import (
    DIMENSIONS, QUALITY_DIMENSIONS, DIAGNOSTICS, NATIVE, report, runtime_report,
)

SCORE_SCHEMA = "univ_phase2_vbench_scores_v1"
P2_IDS = {"P2_B30_CACHE", "P2_B25_SPATIAL", "P2_B25_SKIP",
          "P2_B25_TEMPORAL", "P2_B30_SPATIAL"}


def collect(root: Path):
    manifest = validate_manifest(load_json(root / "generation_manifest.json"))
    # Use the immutable local copy; no dependency on today's config/git revision.
    plan = load_json(root / "collection_plan.json")
    validate_collection_plan(plan)
    for key in ("plan_sha256", "protocol_sha256"):
        if manifest[key] != plan[key]:
            raise ValueError(f"manifest/plan mismatch: {key}")
    prompt_map = {}
    for assignment in plan["assignments"]:
        pid = assignment["prompt_id"]
        if pid in prompt_map and prompt_map[pid] != assignment["prompt"]:
            raise ValueError("inconsistent prompt identity in plan")
        prompt_map[pid] = assignment["prompt"]
    rebuilt = build_collection_plan(plan["protocol"], [prompt_map[k] for k in sorted(prompt_map)])
    if rebuilt != plan:
        raise ValueError("plan assignments do not match frozen protocol/prompts")
    assignments = [a for a in plan["assignments"] if a["split"] in ("train", "validation")]
    if not assignments or {a["split"] for a in assignments} != {"train", "validation"}:
        raise ValueError("Phase2 requires train and validation")
    if {p["id"] for p in plan["protocol"]["budget_presets"]} != P2_IDS:
        raise ValueError("expected five Phase2 candidates")
    for split in ("train", "validation"):
        expected = {a["trajectory_key"] for a in assignments if a["split"] == split}
        actual = {p.stem for p in (root / "records" / split).glob("*.json")}
        if actual != expected:
            raise ValueError(f"record coverage mismatch in {split}: missing={sorted(expected-actual)[:5]}, extra={sorted(actual-expected)[:5]}")
    rows = []
    seen_paths = set()
    for index, assignment in enumerate(assignments):
        path = root / "records" / assignment["split"] / (assignment["trajectory_key"] + ".json")
        record = load_json(path)
        validate_trajectory_record(record, expected_plan_sha256=plan["plan_sha256"], require_scores=False)
        for key in ("trajectory_key", "split", "prompt_id", "prompt", "prompt_sha256", "seed", "base_seed"):
            if record[key] != assignment[key]:
                raise ValueError(f"record identity mismatch: {path}: {key}")
        for key, expected in (("generation_manifest_sha256", manifest["manifest_sha256"]),
                              ("protocol_sha256", plan["protocol_sha256"])):
            if record["provenance"].get(key) != expected:
                raise ValueError(f"record provenance mismatch: {path}: {key}")
        expected_candidates = {c["budget_id"]: c for c in assignment["budget_candidates"]}
        if {c["budget_id"] for c in record["budget_candidates"]} != set(expected_candidates):
            raise ValueError(f"candidate coverage mismatch: {path}")
        artifacts = [(NATIVE, record["native_teacher"])]
        for candidate in record["budget_candidates"]:
            expected = expected_candidates[candidate["budget_id"]]
            if any(candidate.get(k) != v for k, v in expected.items()):
                raise ValueError(f"candidate differs from plan: {path}: {candidate['budget_id']}")
            artifacts.append((candidate["budget_id"], candidate))
        for action, artifact in artifacts:
            video = Path(artifact["video_path"]).resolve()
            if str(video) in seen_paths:
                raise ValueError(f"video path reused across records/actions: {video}")
            seen_paths.add(str(video))
            if sha256_file(video) != artifact["video_sha256"] or video.stat().st_size != artifact["video_bytes"]:
                raise ValueError(f"video identity mismatch: {video}")
            sidecar = artifact.get("runtime_sidecar_path")
            if bool(sidecar) != bool(artifact.get("runtime_sidecar_sha256")):
                raise ValueError("incomplete sidecar identity")
            if sidecar and sha256_file(sidecar) != artifact["runtime_sidecar_sha256"]:
                raise ValueError(f"sidecar identity mismatch: {sidecar}")
            seconds = float(artifact["cost"]["pipeline_seconds"])
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("pipeline_seconds must be finite and positive")
            rows.append({**{k: record[k] for k in ("split", "prompt_id", "prompt", "prompt_sha256", "seed", "base_seed", "trajectory_key")},
                         "action_id": action, "stem": record["trajectory_key"] + "__" + action,
                         "video_path": str(video), "video_sha256": artifact["video_sha256"],
                         "record_file_sha256": sha256_file(path), "pipeline_seconds": seconds,
                         "requested_action": artifact.get("requested_action"),
                         "resolved_schedule": artifact.get("resolved_schedule"),
                         "transition": artifact.get("transition"),
                         "proxy_compute_density": artifact.get("proxy_compute_density", 1.0)})
        if (index + 1) % 20 == 0:
            print(f"Verified {index+1}/{len(assignments)} trajectories", flush=True)
    return rows, {"generation_manifest_sha256": manifest["manifest_sha256"], "plan_sha256": plan["plan_sha256"]}


def stage_inputs(rows, out):
    inputs = out / "inputs"
    inputs.mkdir(exist_ok=True)
    expected = {r["stem"] + ".mp4" for r in rows}
    if {p.name for p in inputs.iterdir()} - expected:
        raise ValueError("unexpected staged files; use a separate evaluation directory")
    prompts = {}
    for row in rows:
        dst = inputs / (row["stem"] + ".mp4")
        if dst.is_symlink():
            raise ValueError(f"staged symlinks are unsupported: {dst}")
        if not dst.exists():
            try:
                os.link(row["video_path"], dst)
            except OSError:
                shutil.copy2(row["video_path"], dst)
        if sha256_file(dst) != row["video_sha256"]:
            raise ValueError(f"staged content mismatch: {dst}")
        prompts[str(dst.resolve())] = row["prompt"]
    write_json_atomic(out / "prompt_map.json", prompts)
    return inputs


def score(rows, out, args, input_digest):
    from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import inspect_vbench_checkout, score_case_directory
    inputs = stage_inputs(rows, out)
    vroot = Path(args.vbench_root).resolve()
    identity = inspect_vbench_checkout(vroot, expected_commit=None)
    scores = {row["stem"]: {} for row in rows}
    provenance = {}
    # One distributed run per dimension: completed dimensions are reusable on restart.
    for dimension in DIMENSIONS:
        print(f"[VBench] {dimension}: {len(rows)} videos, {args.ngpus} GPUs", flush=True)
        bundle = score_case_directory(
            vroot, args.vbench_python, inputs, out / "prompt_map.json", out / "vbench" / dimension,
            [dimension], [dimension] if dimension in QUALITY_DIMENSIONS else [],
            [dimension] if dimension in DIAGNOSTICS else [], args.ngpus, args.force_rescore, identity)
        if set(bundle.scores) != set(scores):
            raise ValueError(f"score coverage mismatch: {dimension}")
        for stem in scores:
            scores[stem][dimension] = bundle.scores[stem][dimension]
        provenance[dimension] = bundle.provenance
    payload = {"schema": SCORE_SCHEMA, "input_sha256": input_digest,
               "dimensions": list(DIMENSIONS), "scores": scores, "provenance": provenance}
    write_json_atomic(out / "scores.json", {**payload, "payload_sha256": canonical_sha256(payload)})


@contextmanager
def output_lock(out):
    lock = out / ".evaluation.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f"Evaluation running or stale lock: {lock}; inspect owner.json/process before removing") from None
    try:
        write_json_atomic(lock / "owner.json", {"pid": os.getpid(), "host": __import__("socket").gethostname()})
        yield
    finally:
        (lock / "owner.json").unlink(missing_ok=True)
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "score", "report", "all"))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", help="default: DATASET_ROOT/metrics/phase2_quality")
    parser.add_argument("--vbench-root", default="/mnt/afs_2/houze/VBench")
    parser.add_argument("--vbench-python", default=sys.executable)
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--force-rescore", action="store_true")
    parser.add_argument("--budgets-seconds", type=float, nargs="+", help="optional absolute caps; default: train candidate mean latency knots")
    parser.add_argument("--tie-epsilon", type=float, default=0.001, help="quality tie threshold for preference diagnostics")
    args = parser.parse_args()
    if args.ngpus < 1 or not math.isfinite(args.tie_epsilon) or args.tie_epsilon < 0:
        parser.error("ngpus must be positive and tie-epsilon finite/nonnegative")
    if args.budgets_seconds and any(not math.isfinite(x) or x <= 0 for x in args.budgets_seconds):
        parser.error("budgets must be finite positive seconds")
    root = Path(args.dataset_root).resolve()
    out = Path(args.out_dir).resolve() if args.out_dir else root / "metrics" / "phase2_quality"
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        rows, identity = collect(root)
        digest = canonical_sha256({"identity": identity, "rows": rows})
        write_json_atomic(out / "evaluation_inputs.json", {"input_sha256": digest, **identity, "rows": rows})
        runtime_report(rows, out)
        print(f"Verified {len(rows)} videos, {len({r['trajectory_key'] for r in rows})} trajectories; output: {out}", flush=True)
        if args.mode in ("score", "all"):
            score(rows, out, args, digest)
        if args.mode in ("report", "all"):
            payload = load_json(out / "scores.json")
            body = {k: v for k, v in payload.items() if k != "payload_sha256"}
            if payload.get("payload_sha256") != canonical_sha256(body):
                raise ValueError("score payload hash mismatch")
            if payload.get("schema") != SCORE_SCHEMA or payload.get("dimensions") != list(DIMENSIONS) or payload.get("input_sha256") != digest:
                raise ValueError("scores belong to different inputs/schema/dimensions")
            report(rows, payload["scores"], out, budgets=args.budgets_seconds, epsilon=args.tie_epsilon)
            print(out / "report.md", flush=True)


if __name__ == "__main__":
    main()
