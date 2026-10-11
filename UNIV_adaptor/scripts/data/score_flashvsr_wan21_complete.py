"""Four-GPU scoring of full FlashVSR clips, reusing verified full-study scores.

Independent scoring directory; never modify frozen generation/diagnostic outputs.
Only 24 Flash clips run inference. Ninety-six existing baseline scores must pass
CSV, raw-result, prompt, video, scorer and VBench-version provenance checks.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
from pathlib import Path
import statistics
import sys
import tarfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import complete_flashvsr_wan21 as generation
import evaluate_flashvsr_spatial_baseline as helpers

diag = generation.diag
QUALITY = ["subject_consistency", "background_consistency", "motion_smoothness", "aesthetic_quality", "imaging_quality"]
DIAGNOSTIC = ["dynamic_degree", "overall_consistency"]
DIMENSIONS = QUALITY + DIAGNOSTIC
VBENCH_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
SCORER = ROOT / "changing_resolution_uni/scripts/data/batch_vbench_score_dataset.py"


def implementation():
    return {path.name: diag.file_hash(path) for path in
            (Path(__file__), Path(generation.__file__), Path(diag.__file__), Path(helpers.__file__), SCORER)}


def collect(source):
    p = generation.load_plan(source)
    manifest_path = source / "dataset_manifest.json"
    manifest = diag.read(manifest_path)
    if (manifest["schema"] != "flashvsr_wan21_complete_dataset_v1" or
            manifest["plan_sha256"] != p["plan_sha256"] or len(manifest["videos"]) != 24):
        raise ValueError("Run full 24-clip generation/finalize first; prefix diagnostics are not compatible")
    lookup = {(r["prompt_id"], r["seed"]): r for r in manifest["videos"]}
    if len(lookup) != 24:
        raise ValueError("Duplicate Flash prompt/seed")
    study = Path(p["source_root"])
    source_plan = diag.read(study / "plan.json")
    if (source_plan["plan_sha256"] != p["source_plan_sha256"] or
            source_plan["config"]["evaluation"]["quality_dimensions"] != QUALITY or
            source_plan["config"]["evaluation"]["diagnostic_dimensions"] != DIAGNOSTIC):
        raise ValueError("Original study scoring dimensions/protocol changed")
    diag.verify_files([p["source_plan"]])
    clips, bindings = [], [diag.bound_file(source / "plan.json"), diag.bound_file(manifest_path), p["source_plan"]]
    for pair in p["pairs"]:
        row = lookup[pair["prompt_id"], pair["seed"]]
        result = generation.completed_result(source, p, pair)
        if (row["method"] != generation.METHOD or row["asset"] != result["asset"] or
                row["comparison"] != pair["comparison"] or row["prompt"] != pair["prompt"] or
                row["source_endpoint"] != pair["endpoint"] or
                (row["frames"], row["width"], row["height"], row["fps"]) != (81, 832, 480, 16)):
            raise ValueError("Full Flash manifest/result mismatch")
        bindings.append(diag.bound_file(generation.result_path(source, pair)))
        for arm in (*generation.BASE_ARMS, generation.METHOD):
            if arm == generation.METHOD:
                video = result["asset"]["video"]
            else:
                binding = pair["comparison"][arm]
                diag.verify_files([binding["receipt"], binding["video"]])
                bindings.append(binding["receipt"])
                video = binding["video"]
            clips.append({"clip_id": f"pilot_{pair['prompt_id']}_s{pair['seed']}_{arm}",
                          "group_id": f"{pair['prompt_id']}_s{pair['seed']}",
                          "prompt_id": pair["prompt_id"], "seed": pair["seed"], "prompt": pair["prompt"],
                          "method": arm, "video": video, "frames": 81, "width": 832, "height": 480, "fps": 16,
                          "restoration_seconds": result["sr_preprocess_model_postprocess_seconds"] if arm == generation.METHOD else None})
    csv_path, prov_path = study / "metrics/quality_by_video.csv", study / "metrics/score_provenance.json"
    provenance = diag.read(prov_path)
    if provenance["plan_sha256"] != p["source_plan_sha256"] or provenance["quality_csv_sha256"] != diag.file_hash(csv_path):
        raise ValueError("Original scoring CSV/provenance changed")
    bindings += [diag.bound_file(csv_path), diag.bound_file(prov_path)]
    return p, clips, bindings, provenance


def plan(args):
    diag.outside(args.out, args.source)
    if (args.out / "plan.json").exists():
        p = load(args)
        if p["source_root"] != str(args.source.resolve()):
            raise ValueError("Existing evaluation belongs to another source")
        print("Resuming frozen full-length scoring plan", flush=True)
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Unplanned nonempty scoring output; use a NEW directory")
    gen, clips, bindings, provenance = collect(args.source)
    diag.outside(args.out, gen["source_root"])
    body = {"schema": "flashvsr_wan21_complete_score_plan_v1", "source_root": str(args.source.resolve()),
            "study_root": gen["source_root"], "generation_plan_sha256": gen["plan_sha256"],
            "study_plan_sha256": gen["source_plan_sha256"], "bindings": bindings, "clips": clips,
            "baseline_score_provenance": provenance, "implementation": implementation(),
            "quality_dimensions": QUALITY, "diagnostic_dimensions": DIAGNOSTIC,
            "expected_vbench_commit": VBENCH_COMMIT}
    body["plan_sha256"] = diag.digest(body)
    diag.write_new(args.out / "plan.json", body)
    print("Frozen 120 full-length clips: score 24 Flash clips; reuse 96 verified baseline results", flush=True)


def load(args):
    p = diag.read(args.out / "plan.json")
    if (p["schema"] != "flashvsr_wan21_complete_score_plan_v1" or
            diag.digest({k: v for k, v in p.items() if k != "plan_sha256"}) != p["plan_sha256"] or
            p["implementation"] != implementation() or len(p["clips"]) != 120 or
            len({r["clip_id"] for r in p["clips"]}) != 120):
        raise ValueError("Scoring plan/code/coverage mismatch")
    for source in (p["source_root"], p["study_root"]):
        diag.outside(args.out, source)
    diag.verify_files(p["bindings"])
    diag.verify_files([r["video"] for r in p["clips"]])
    return p


def stage(args, p):
    folder = args.out / "inputs"
    mapping = {}
    for row in p["clips"]:
        if row["method"] != generation.METHOD:
            continue
        target = folder / (row["clip_id"] + ".mp4")
        helpers.copy_verified(Path(row["video"]["path"]), target, row["video"])
        mapping[str(target.resolve())] = row["prompt"]
    if {x.name for x in folder.glob("*.mp4")} != {Path(x).name for x in mapping}:
        raise ValueError("Unexpected Flash scoring video; refuse mixed inventory")
    diag.write_new(args.out / "prompt_map.json", mapping)
    return folder


def check(args):
    p = load(args)
    stage(args, p)
    observed = {}
    for clip in p["clips"]:
        geometry = helpers.probe(clip["video"]["path"], args.ffprobe)
        if any(geometry[k] != clip[k] for k in ("width", "height", "frames")) or abs(geometry["fps"] - clip["fps"]) > .01:
            raise ValueError(f"Not a matched 81-frame clip: {clip['clip_id']}")
        observed[clip["clip_id"]] = geometry
    diag.write_new(args.out / "check.json", {"plan_sha256": p["plan_sha256"], "geometry": observed})
    print("Verified 120 original full-length MP4s; only 24 Flash clips staged for inference", flush=True)


def validate_scores(scores, p):
    if set(scores) != {r["clip_id"] for r in p["clips"]}:
        raise ValueError("Expected all 120 matched scores")
    for values in scores.values():
        if set(values) != set(DIMENSIONS) or any(type(v) not in (int, float) or not math.isfinite(v) or
                not (-1 if dimension == "overall_consistency" else 0) <= v <= 1 for dimension, v in values.items()):
            raise ValueError("Invalid/missing normalized per-video dimensions")


def baseline_scores(args, p, backend, identity):
    study = Path(p["study_root"])
    with (study / "metrics/quality_by_video.csv").open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    lookup = {r["observation_id"]: r for r in csv_rows}
    if len(lookup) != len(csv_rows):
        raise ValueError("Duplicate original score row")
    combined, provenance, files = {}, {}, []
    for arm in generation.BASE_ARMS:
        selected = [r for r in p["clips"] if r["method"] == arm]
        old = p["baseline_score_provenance"]["vbench"][arm]
        manifest_path = Path(old["run_manifest_path"])
        root = manifest_path.parent.resolve()
        manifest = diag.read(manifest_path)
        for key, value in manifest.items():
            if old.get(key) != value:
                raise ValueError("Original run manifest/provenance mismatch")
        paths = {"manifest": manifest_path}
        for key in ("request_file", "result_file", "full_info_file"):
            path = (root / manifest[key]).resolve()
            if path.parent != root:
                raise ValueError("Original score artifact escapes run directory")
            paths[key] = path
        request = diag.read(paths["request_file"])
        request_hash = backend.canonical_json_sha256({k: v for k, v in request.items() if k != "request_sha256"})
        if (request["request_sha256"] != request_hash or request_hash != old["request_sha256"] or
                request["vbench"] != identity or request["scorer_sha256"] != diag.file_hash(SCORER) or
                request["mode"] != "custom_input" or request["imaging_quality_preprocessing_mode"] != "longer" or
                request["dimensions"] != DIMENSIONS or request["quality_dimensions"] != QUALITY or
                request["diagnostic_dimensions"] != DIAGNOSTIC):
            raise ValueError("Original scoring settings/version differ; do not silently combine incompatible scores")
        expected_inventory = sorted([{"name": r["clip_id"] + ".mp4", "size_bytes": r["video"]["bytes"],
                                      "sha256": r["video"]["sha256"]} for r in selected], key=lambda x: x["name"])
        if sorted(request["videos"], key=lambda x: x["name"]) != expected_inventory:
            raise ValueError("Original raw score inventory differs from matched videos")
        prompt_map = root / "prompt_map.json"
        if diag.file_hash(prompt_map) != request["prompt_map_sha256"]:
            raise ValueError("Original prompt map hash mismatch")
        prompts = diag.read(prompt_map)
        if len(prompts) != 24 or {Path(k).stem: v for k, v in prompts.items()} != {r["clip_id"]: r["prompt"] for r in selected}:
            raise ValueError("Original prompt text/coverage mismatch")
        if (diag.file_hash(paths["result_file"]) != old["result_sha256"] or
                diag.file_hash(paths["full_info_file"]) != old["full_info_sha256"]):
            raise ValueError("Original raw VBench artifacts changed")
        raw = backend.parse_vbench_eval_result(paths["result_file"], dimensions=DIMENSIONS,
                                               expected_stems={r["clip_id"] for r in selected})
        for clip in selected:
            row = lookup[clip["clip_id"]]
            if (row["prompt"] != clip["prompt"] or row["action_id"] != arm or
                    row["video_sha256"] != clip["video"]["sha256"] or int(row["seed"]) != clip["seed"] or
                    row["prompt_key"] != clip["prompt_id"]):
                raise ValueError("Original CSV prompt/seed/video identity mismatch")
            if any(abs(float(row[d]) - raw[clip["clip_id"]][d]) > 1e-12 for d in DIMENSIONS):
                raise ValueError("Original CSV values disagree with verified raw scores")
            combined[clip["clip_id"]] = raw[clip["clip_id"]]
        for path in (*paths.values(), prompt_map):
            binding = diag.bound_file(path)
            files.append(binding)
            helpers.copy_verified(path, args.out / "reused_vbench" / arm / path.name, binding)
        provenance[arm] = old
    return combined, provenance, files


def load_scores(args, p):
    saved = diag.read(args.out / "scores.json")
    helpers.verify_seal(saved, "scores_sha256")
    if saved["plan_sha256"] != p["plan_sha256"] or saved["vbench_identity"]["git_commit"] != VBENCH_COMMIT:
        raise ValueError("Score plan/VBench mismatch")
    diag.verify_files(saved["reused_raw_bindings"])
    diag.verify_files(saved["new_raw_bindings"])
    validate_scores(saved["scores"], p)
    return saved


def score(args):
    p = load(args)
    checked = diag.read(args.out / "check.json")
    if checked["plan_sha256"] != p["plan_sha256"] or set(checked["geometry"]) != {r["clip_id"] for r in p["clips"]}:
        raise ValueError("Run check for the frozen full-length inventory before scoring")
    backend = helpers.score_module()
    identity = backend.inspect_vbench_checkout(args.vbench_root, expected_commit=VBENCH_COMMIT)
    if (args.out / "scores.json").exists():
        if load_scores(args, p)["vbench_identity"] != identity:
            raise ValueError("Current VBench identity differs from saved scores")
        print("Reused complete verified scores; no inference", flush=True)
        return
    combined, reused, raw_files = baseline_scores(args, p, backend, identity)
    folder = stage(args, p)
    flash_ids = {r["clip_id"] for r in p["clips"] if r["method"] == generation.METHOD}
    for clip_id in flash_ids:
        combined[clip_id] = {}
    provenance, new_raw_files = {}, []
    # Sequential dimensions; four torchrun ranks each. Reuse working model cache.
    # Never call the old warmup helper that can delete shared torch.hub cache.
    prior_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(generation.gpu_ids(args.gpu_ids))
    try:
        for dimension in DIMENSIONS:
            quality = dimension in QUALITY
            print(f"Scoring {dimension}: 24 full FlashVSR clips, GPUs {args.gpu_ids}", flush=True)
            bundle = backend.score_case_directory(args.vbench_root, args.vbench_python, folder,
                        args.out / "prompt_map.json", args.out / "vbench" / dimension,
                        [dimension], [dimension] if quality else [], [] if quality else [dimension], 4, False, identity)
            if set(bundle.scores) != flash_ids or any(set(v) != {dimension} for v in bundle.scores.values()):
                raise ValueError("Incomplete Flash per-dimension scores")
            for clip_id in flash_ids:
                combined[clip_id][dimension] = bundle.scores[clip_id][dimension]
            provenance[dimension] = bundle.provenance
            run_manifest = Path(bundle.provenance["run_manifest_path"])
            run_root = run_manifest.parent.resolve()
            new_raw_files.append(diag.bound_file(run_manifest))
            for key in ("request_file", "result_file", "full_info_file"):
                path = (run_root / bundle.provenance[key]).resolve()
                if path.parent != run_root:
                    raise ValueError("New score artifact escapes its run directory")
                expected_hash = bundle.provenance.get(key.replace("_file", "_sha256")) if key != "request_file" else None
                new_raw_files.append(diag.bound_file(path, expected_hash))
    finally:
        if prior_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prior_visible
    validate_scores(combined, p)
    diag.write_new(args.out / "scores.json", helpers.sealed({"plan_sha256": p["plan_sha256"],
        "scores": combined, "vbench_identity": identity, "new_flash_provenance": provenance,
        "reused_baseline_provenance": reused, "reused_raw_bindings": raw_files,
        "new_raw_bindings": new_raw_files}, "scores_sha256"))


def write_csv(path, rows):
    # Canonical LF: read_text normalizes CRLF, so the legacy CRLF writer is not
    # idempotent when report/export is called twice, even on Linux.
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    helpers.write_text_new(path, buffer.getvalue())


def report(args):
    p = load(args)
    saved = load_scores(args, p)
    values = saved["scores"]
    per_video, deltas, flash_pairs, by_prompt, summaries = [], [], [], [], []
    lookup = {(r["group_id"], r["method"]): r for r in p["clips"]}
    for clip in p["clips"]:
        per_video.append({k: clip[k] for k in ("clip_id", "group_id", "prompt_id", "seed", "method", "prompt")} |
                         {"video_sha256": clip["video"]["sha256"]} | values[clip["clip_id"]])
        full = lookup[clip["group_id"], "FULL50"]
        if clip["method"] != "FULL50":
            deltas.append({k: clip[k] for k in ("group_id", "prompt_id", "seed", "method")} |
                          {"reference": "FULL50"} |
                          {"delta_" + d: values[clip["clip_id"]][d] - values[full["clip_id"]][d] for d in DIMENSIONS})
        if clip["method"] == generation.METHOD:
            for arm in generation.BASE_ARMS:
                baseline = lookup[clip["group_id"], arm]
                flash_pairs.append({k: clip[k] for k in ("group_id", "prompt_id", "seed")} |
                                   {"left": arm, "right": generation.METHOD} |
                                   {"delta_" + d: values[clip["clip_id"]][d] - values[baseline["clip_id"]][d] for d in DIMENSIONS})
    for arm in (*generation.BASE_ARMS, generation.METHOD):
        selected = [r for r in per_video if r["method"] == arm]
        summaries.append({"method": arm, "n_videos": len(selected)} |
                         {"mean_" + d: statistics.mean(r[d] for r in selected) for d in DIMENSIONS})
    for arm in (*generation.BASE_ARMS[1:], generation.METHOD):
        for prompt_id in sorted({r["prompt_id"] for r in p["clips"]}):
            selected = [r for r in deltas if r["method"] == arm and r["prompt_id"] == prompt_id]
            by_prompt.append({"method": arm, "prompt_id": prompt_id, "n_seeds": len(selected)} |
                             {"mean_delta_" + d: statistics.mean(r["delta_" + d] for r in selected) for d in DIMENSIONS})
    for name, rows in (("per_video_vbench.csv", per_video), ("paired_deltas_vs_full.csv", deltas),
                       ("flash_vs_methods.csv", flash_pairs), ("per_prompt_deltas.csv", by_prompt), ("method_summary.csv", summaries)):
        write_csv(args.out / name, rows)
    diag.write_new(args.out / "report.json", {"plan_sha256": p["plan_sha256"], "scores_sha256": saved["scores_sha256"],
        "n_videos": 120, "newly_scored": 24, "reused_scores": 96, "n_matched_groups": 24,
        "dimensions": DIMENSIONS, "summary": summaries, "delta_direction": "method/Flash minus reference",
        "end_to_end_speedup": None, "claim_status": "Matched exploratory evidence only; await visual/human validation.",
        "caveats": ["Seven custom-input dimensions, NOT complete 16-dimension VBench or official Total.",
                    "No composite quality score is invented; dynamic_degree is not uniformly higher-is-better.",
                    "No 33-frame diagnostic score enters this 81-frame comparison.",
                    "Same seed does not make FULL and native LR pixel-aligned ground truth.",
                    "Similar scores alone do not prove metric insufficiency or prompt-routing benefit.",
                    "SR-only time is not a measured complete-pipeline speedup."]})
    print(f"Saved five-method scores and matched deltas: {args.out}", flush=True)


def export(args):
    report(args)
    p = load(args)
    archive = args.out.parent / (args.out.name + "_analysis.tgz")
    receipt = args.out / "export_receipt.json"
    if archive.exists():
        if not receipt.exists():
            raise FileExistsError(f"Unreceipted/partial export; inspect/move before resume: {archive}")
        prior = diag.read(receipt)
        if prior["plan_sha256"] != p["plan_sha256"]:
            raise ValueError("Export plan mismatch")
        diag.verify_files([prior["archive"]])
        print(f"Verified existing export: {archive}")
        return
    if receipt.exists():
        raise FileNotFoundError("Export receipt exists but archive missing")
    with tarfile.open(archive, "x:gz") as bundle:
        for path in sorted(args.out.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".csv", ".log") and not path.is_relative_to(args.out / "inputs"):
                bundle.add(path, arcname=(Path(args.out.name) / path.relative_to(args.out)).as_posix())
    diag.write_new(receipt, {"plan_sha256": p["plan_sha256"], "archive": diag.bound_file(archive)})
    print(f"Download analysis: {archive} ({archive.stat().st_size / 1024**2:.1f} MiB); no video/weights", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("all", "plan", "check", "score", "report", "export"))
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/flashvsr_wan21_complete_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/flashvsr_wan21_complete_score_v1")
    parser.add_argument("--vbench-root", type=Path, default=Path("/mnt/afs_2/houze/VBench"))
    parser.add_argument("--vbench-python", default="/opt/conda/bin/python")
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--ffprobe", type=Path)
    args = parser.parse_args()
    args.source, args.out, args.vbench_root = args.source.resolve(), args.out.resolve(), args.vbench_root.resolve()
    generation.gpu_ids(args.gpu_ids)
    if args.mode == "all":
        for function in (plan, check, score, report, export):
            function(args)
    else:
        globals()[args.mode](args)


if __name__ == "__main__":
    main()
