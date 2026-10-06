"""Read-only asset reuse; independent VBench and offline, method-anonymous review.

No inference, setup, tensor decoding, or modification of the frozen diagnostic.
Only score imports the existing strict VBench scorer. Other modes use stdlib.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import flashvsr_asset_diagnostic as diagnostic

CONFIG = ROOT / "UNIV_adaptor/configs/flashvsr_spatial_baseline_eval_v1.json"
TEMPLATE = Path(__file__).with_name("flashvsr_spatial_blind.html")
TRACKS = ("FULL", "HR_DOWN4_BICUBIC", "HR_DOWN4_FLASH", "NATIVE_LR_BICUBIC", "NATIVE_LR_FLASH")
AXES = ("prompt_match", "detail_correctness", "sharpness", "temporal_stability", "overall_quality")
FLAGS = ("blur", "crop", "subject_count", "texture", "structure", "flicker", "other")
read, digest, file_hash, write_new = diagnostic.read, diagnostic.digest, diagnostic.file_hash, diagnostic.write_new


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
        raise ValueError(f"Unsafe identifier: {value!r}")
    return value


def sealed(body, key):
    return {**body, key: digest(body)}


def verify_seal(body, key):
    if body.get(key) != digest({k: v for k, v in body.items() if k != key}):
        raise ValueError(f"Invalid content seal: {key}")


def implementation():
    return {p.name: file_hash(p) for p in (Path(__file__), TEMPLATE, Path(diagnostic.__file__),
                                         ROOT / "changing_resolution_uni/scripts/data/batch_vbench_score_dataset.py")}


def collect(source, cfg):
    """Accept remote output OR exported package; never follow original absolute paths."""
    source = source.resolve()
    original = read(source / "plan.json")
    verify_seal(original, "plan_sha256")
    if original["schema"] != "flashvsr_asset_diagnostic_plan_v1":
        raise ValueError("Unexpected source plan schema")
    source_cfg = original["config"]
    expected = {(p, s, a) for p in source_cfg["prompt_ids"] for s in source_cfg["seeds"]
                for a in source_cfg["spatial_arms"]}
    seen, identities, rows, prompts = set(), {}, [], {}

    def identity(relative):
        path = source / relative
        if not path.resolve().is_relative_to(source):
            raise ValueError("Source asset escapes source root")
        result = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        identities[relative.as_posix()] = result
        return result

    identity(Path("plan.json"))
    for pair in original["pairs"]:
        pair_id = safe_id(pair["id"])
        key = (pair["prompt_id"], pair["seed"], pair["arm"])
        if key in seen or any(r["pair_id"] == pair_id for r in rows):
            raise ValueError("Duplicate source group")
        seen.add(key)
        if (pair["frames"] != cfg["frames"] or pair["frames"] != source_cfg["frames"]
                or pair["fps"] <= 0 or pair["width"] <= 0 or pair["height"] <= 0):
            raise ValueError("Source prefix/geometry mismatch")
        if not isinstance(pair["prompt"], str) or not pair["prompt"].strip():
            raise ValueError("Empty prompt")
        if prompts.setdefault(pair["prompt_id"], pair["prompt"]) != pair["prompt"]:
            raise ValueError("Prompt identity differs across seeds")
        prepared_path = Path("prepared") / f"{pair_id}.json"
        prepared = read(source / prepared_path)
        identity(prepared_path)
        if prepared["plan_sha256"] != original["plan_sha256"] or prepared["pair_id"] != pair_id:
            raise ValueError("Prepared receipt mismatch")
        assets = dict(prepared["assets"])
        timing = {}
        for kind in ("HR_DOWN4", "NATIVE_LR"):
            relative = Path("results") / f"{pair_id}_{kind}.json"
            result = read(source / relative)
            identity(relative)
            diagnostic.validate_result(result, original, pair, kind, prepared)
            assets[kind + "_FLASH"] = result["asset"]
            timing[kind] = {"sr_model_seconds": result["timing"]["sr_model_seconds"],
                           "sr_preprocess_model_postprocess_seconds": result["sr_preprocess_model_postprocess_seconds"]}
        for track in TRACKS:
            asset = assets[track]
            if (asset["shape"] != [pair["frames"], pair["height"], pair["width"], 3]
                    or not math.isfinite(asset["fps"]) or abs(asset["fps"] - pair["fps"]) > .01):
                raise ValueError("Target asset geometry mismatch")
            basename = asset["video"]["path"].replace("\\", "/").split("/")[-1]
            if basename != track + ".mp4":
                raise ValueError("Unexpected source video basename")
            relative = Path("media") / pair_id / basename
            video = identity(relative)
            if any(video[k] != asset["video"][k] for k in video):
                raise ValueError(f"Source video hash/size mismatch: {relative}")
            rows.append({"clip_id": digest([original["plan_sha256"], pair_id, track])[:24],
                         "pair_id": pair_id, "prompt_id": pair["prompt_id"], "prompt": pair["prompt"],
                         "seed": pair["seed"], "arm": pair["arm"], "track": track,
                         "relative_video": relative.as_posix(), "video": video,
                         "frames": pair["frames"], "fps": pair["fps"],
                         "width": pair["width"], "height": pair["height"],
                         "main_spatial_density": pair["lr_width"] * pair["lr_height"] / (pair["width"] * pair["height"]),
                         "restoration_timings": timing, "end_to_end_speedup": None})
    if not seen or seen != expected:
        raise ValueError("Incomplete source prompt/seed/arm coverage; no score-driven subset allowed")
    if len({r["clip_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate clip identity")
    return {"source_plan_sha256": original["plan_sha256"], "source_config": source_cfg,
            "inventory": identities, "clips": rows}


def plan(args):
    diagnostic.outside(args.out, args.source)
    cfg = read(args.config)
    if cfg["schema"] != "flashvsr_spatial_eval_config_v1" or cfg["frames"] != 33:
        raise ValueError("This protocol is explicitly a 33-frame screening")
    dims = cfg["quality_dimensions"] + cfg["diagnostic_dimensions"]
    if len(dims) != len(set(dims)) or cfg["minimum_raters_per_pair"] < 1:
        raise ValueError("Invalid evaluation configuration")
    comparisons = cfg["comparisons"]
    if len({c["id"] for c in comparisons}) != len(comparisons):
        raise ValueError("Duplicate comparison ID")
    for c in comparisons:
        safe_id(c["id"])
        if c["left"] not in TRACKS or c["right"] not in TRACKS or c["left"] == c["right"]:
            raise ValueError("Invalid comparison")
    body = {"schema": "flashvsr_spatial_eval_plan_v1", "config": cfg,
            # Do not bind machine-specific root paths: the exported archive and
            # the server must produce the same public bundle for rating import.
            "implementation": implementation(),
            **collect(args.source, cfg)}
    write_new(args.out / "plan.json", sealed(body, "plan_sha256"))
    print(f"Frozen {len(body['clips'])} reused clips; no generation. Output: {args.out}")


def load(args):
    diagnostic.outside(args.out, args.source)
    p = read(args.out / "plan.json")
    verify_seal(p, "plan_sha256")
    if p["schema"] != "flashvsr_spatial_eval_plan_v1" or p["implementation"] != implementation():
        raise ValueError("Evaluation implementation changed; use a new evaluation directory")
    # An exported source package may be relocated; identities, not original paths, bind it.
    current = collect(args.source, p["config"])
    for key in current:
        if current[key] != p[key]:
            raise ValueError(f"Frozen source changed: {key}")
    return p


def require_check(args, p):
    checked = read(args.out / "check.json")
    if checked["plan_sha256"] != p["plan_sha256"] or set(checked["geometry"]) != {r["clip_id"] for r in p["clips"]}:
        raise ValueError("Run check first")
    for row in p["clips"]:
        observed = checked["geometry"][row["clip_id"]]
        if (any(observed[k] != row[k] for k in ("frames", "width", "height"))
                or not math.isfinite(observed["fps"]) or abs(observed["fps"] - row["fps"]) > .01):
            raise ValueError("Checked geometry differs from frozen plan")


def copy_verified(source, target, identity):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    if target.stat().st_size != identity["bytes"] or file_hash(target) != identity["sha256"]:
        raise ValueError(f"Staged bytes differ; inspect/move before resuming: {target}")


def stage(args, p):
    folder = args.out.resolve() / "inputs"
    mapping = {}
    for row in p["clips"]:
        target = folder / (row["clip_id"] + ".mp4")
        copy_verified(args.source / row["relative_video"], target, row["video"])
        mapping[str(target)] = row["prompt"]
    if {f.stem for f in folder.glob("*.mp4")} != {r["clip_id"] for r in p["clips"]}:
        raise ValueError("Unexpected staged scoring video")
    write_new(args.out / "prompt_map.json", mapping)
    return folder


def probe(path, ffprobe=None):
    binary = ffprobe or shutil.which("ffprobe")
    if binary:
        data = json.loads(subprocess.check_output([str(binary), "-v", "error", "-select_streams", "v:0",
                                                  "-count_frames", "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
                                                  "-of", "json", str(path)], text=True))["streams"]
        if len(data) != 1:
            raise ValueError("Expected one video stream")
        s = data[0]
        return {"width": int(s["width"]), "height": int(s["height"]),
                "frames": int(s["nb_read_frames"]), "fps": float(Fraction(s["avg_frame_rate"]))}
    # The existing VBench/diagnostic environments normally provide imageio-ffmpeg.
    import imageio_ffmpeg
    stream = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    metadata = next(stream)
    count = sum(1 for _ in stream)
    return {"width": metadata["size"][0], "height": metadata["size"][1], "frames": count, "fps": metadata["fps"]}


def check(args):
    p = load(args)
    stage(args, p)
    geometry = {}
    for row in p["clips"]:
        observed = probe(args.out / "inputs" / (row["clip_id"] + ".mp4"), args.ffprobe)
        if any(observed[k] != row[k] for k in ("frames", "width", "height")) or abs(observed["fps"] - row["fps"]) > .01:
            raise ValueError(f"Decoded video geometry mismatch: {row['clip_id']}")
        geometry[row["clip_id"]] = observed
    write_new(args.out / "check.json", {"plan_sha256": p["plan_sha256"], "geometry": geometry})
    print(f"Verified actual geometry and original bytes for {len(geometry)} equal-length videos")


def pairs(p):
    result = []
    groups = sorted({r["pair_id"] for r in p["clips"]})
    lookup = {(r["pair_id"], r["track"]): r for r in p["clips"]}
    for c in p["config"]["comparisons"]:
        for group in groups:
            left, right = lookup[group, c["left"]], lookup[group, c["right"]]
            result.append({"id": digest([p["plan_sha256"], group, c["id"]])[:24],
                           "comparison": c["id"], "pair_id": group,
                           "prompt_id": left["prompt_id"], "seed": left["seed"],
                           "left": left["clip_id"], "right": right["clip_id"], "prompt": left["prompt"]})
    return result


def write_text_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise ValueError(f"Existing file differs: {path}")
    else:
        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(value)


def blind(args):
    p = load(args)
    require_check(args, p)
    rng = random.Random(p["config"]["random_seed"])
    trials = []
    primary = pairs(p)
    for c in p["config"]["comparisons"]:
        selected = [r for r in primary if r["comparison"] == c["id"]]
        flips = [i % 2 == 0 for i in range(len(selected))]
        rng.shuffle(flips)
        for row, flip in zip(selected, flips):
            trials.append({**row, "A": row["right"] if flip else row["left"],
                           "B": row["left"] if flip else row["right"], "repeat_of": None})
        original = rng.choice(trials[-len(selected):])
        trials.append({**original, "id": digest([original["id"], "repeat"])[:24],
                       "A": original["B"], "B": original["A"], "repeat_of": original["id"]})
    rng.shuffle(trials)
    public = sealed({"schema": "flashvsr_spatial_blind_bundle_v1", "axes": list(AXES), "flags": list(FLAGS),
                     "trials": [{"id": r["id"], "prompt": r["prompt"],
                                 "A": f"media/{r['A']}.mp4", "B": f"media/{r['B']}.mp4"} for r in trials]}, "bundle_id")
    private = {"plan_sha256": p["plan_sha256"], "bundle_id": public["bundle_id"], "trials": trials}
    folder = args.out / "blind"
    write_new(args.out / "blind_private.json", private)
    write_new(folder / "study.json", public)
    payload = json.dumps(public, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    write_text_new(folder / "index.html", TEMPLATE.read_text(encoding="utf-8").replace("__STUDY_JSON__", payload))
    for row in p["clips"]:
        copy_verified(args.source / row["relative_video"], folder / "media" / (row["clip_id"] + ".mp4"), row["video"])
    print(f"Anonymous offline review: {len(primary)} primary + {len(trials)-len(primary)} reliability pairs. {folder / 'index.html'}")


def score_module():
    sys.path.insert(0, str(ROOT / "changing_resolution_uni/scripts/data"))
    import batch_vbench_score_dataset
    return batch_vbench_score_dataset


def validate_scores(scores, p):
    dims = p["config"]["quality_dimensions"] + p["config"]["diagnostic_dimensions"]
    if set(scores) != {r["clip_id"] for r in p["clips"]}:
        raise ValueError("VBench clip coverage mismatch")
    for values in scores.values():
        if set(values) != set(dims):
            raise ValueError("VBench dimension coverage mismatch")
        for value in values.values():
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid normalized per-video VBench score")


def load_scores(args, p):
    s = read(args.out / "scores.json")
    verify_seal(s, "scores_sha256")
    if s["plan_sha256"] != p["plan_sha256"]:
        raise ValueError("Scores belong to another plan")
    if s["vbench_identity"]["git_commit"] != p["config"]["expected_vbench_commit"]:
        raise ValueError("VBench commit mismatch")
    validate_scores(s["scores"], p)
    return s


def score(args):
    p = load(args)
    require_check(args, p)
    if args.ngpus < 1:
        raise ValueError("ngpus must be positive")
    folder = stage(args, p)
    backend = score_module()
    identity = backend.inspect_vbench_checkout(args.vbench_root, expected_commit=p["config"]["expected_vbench_commit"])
    if (args.out / "scores.json").exists():
        cached = load_scores(args, p)
        if cached["vbench_identity"] != identity:
            raise ValueError("VBench identity changed")
        print("Reused verified complete scores")
        return
    # Do not call the legacy warmup helper: it can delete shared torch.hub cache
    # entries. This follow-up reuses the user's already-working VBench setup.
    combined = {r["clip_id"]: {} for r in p["clips"]}
    provenance = {}
    for d in p["config"]["quality_dimensions"] + p["config"]["diagnostic_dimensions"]:
        quality = d in p["config"]["quality_dimensions"]
        print(f"Scoring {d}: {len(combined)} videos on {args.ngpus} GPUs", flush=True)
        bundle = backend.score_case_directory(args.vbench_root, args.vbench_python, folder,
                                              (args.out / "prompt_map.json").resolve(), args.out / "vbench" / d,
                                              [d], [d] if quality else [], [] if quality else [d],
                                              args.ngpus, False, identity)
        if set(bundle.scores) != set(combined) or any(set(values) != {d} for values in bundle.scores.values()):
            raise ValueError("Incomplete per-dimension result")
        for clip_id in combined:
            combined[clip_id][d] = bundle.scores[clip_id][d]
        provenance[d] = bundle.provenance
    validate_scores(combined, p)
    write_new(args.out / "scores.json", sealed({"plan_sha256": p["plan_sha256"], "scores": combined,
                                               "vbench_identity": identity, "provenance": provenance}, "scores_sha256"))


def write_csv(path, rows):
    if not rows:
        return
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    write_text_new(path, buffer.getvalue())


def report(args):
    p = load(args)
    s = load_scores(args, p)
    dimensions = p["config"]["quality_dimensions"] + p["config"]["diagnostic_dimensions"]
    rows = []
    for pair in pairs(p):
        rows.append({k: pair[k] for k in ("id", "comparison", "pair_id", "prompt_id", "seed")} |
                    {f"delta_{d}": s["scores"][pair["right"]][d] - s["scores"][pair["left"]][d] for d in dimensions})
    write_csv(args.out / "paired_vbench.csv", rows)
    write_csv(args.out / "per_video_vbench.csv", [{k: r[k] for k in ("clip_id", "pair_id", "prompt_id", "seed", "track")} |
                                                   s["scores"][r["clip_id"]] for r in p["clips"]])
    summary = []
    for c in p["config"]["comparisons"]:
        selected = [r for r in rows if r["comparison"] == c["id"]]
        for prompt_id in sorted({r["prompt_id"] for r in selected}):
            prompt_rows = [r for r in selected if r["prompt_id"] == prompt_id]
            summary.append({"comparison": c["id"], "prompt_id": prompt_id, "n_seed_groups": len(prompt_rows)} |
                           {f"mean_delta_{d}": statistics.mean(r[f"delta_{d}"] for r in prompt_rows) for d in dimensions})
    write_csv(args.out / "per_prompt_vbench.csv", summary)
    write_new(args.out / "report.json", {"plan_sha256": p["plan_sha256"], "scores_sha256": s["scores_sha256"],
              "claim": p["config"]["claim"], "limitations": p["config"]["limitations"],
              "n_clips": len(p["clips"]), "n_primary_pairs": len(rows), "end_to_end_speedup": None,
              "delta_direction": "right minus left; dynamic_degree is diagnostic, not uniformly higher-is-better",
              "restoration_timings": {r["pair_id"]: r["restoration_timings"] for r in p["clips"]},
              "claim_status": "Exploratory screening only. Await independent human ratings; no learnability or metric-failure conclusion."})
    print(f"Saved seven separate VBench dimensions and matched deltas: {args.out}")


def validate_ratings(data, private):
    if data.get("schema") != "flashvsr_spatial_blind_ratings_v1" or data.get("bundle_id") != private["bundle_id"]:
        raise ValueError("Ratings schema/bundle mismatch")
    rater = safe_id(data["rater_id"])
    if type(data.get("prior_exposure")) is not bool:
        raise ValueError("Declare prior exposure")
    known = {r["id"] for r in private["trials"]}
    seen = set()
    for row in data["ratings"]:
        if row["id"] not in known or row["id"] in seen:
            raise ValueError("Unknown/duplicate rating trial")
        seen.add(row["id"])
        if row["preference"] not in ("A", "B", "tie", "uncertain") or not isinstance(row.get("notes", ""), str):
            raise ValueError("Invalid preference/notes")
        for side in ("A", "B"):
            if set(row[side]) != set(AXES):
                raise ValueError("Missing/extra human axis")
            if any(v is not None and (type(v) is not int or not 1 <= v <= 5) for v in row[side].values()):
                raise ValueError("Human scores must be integers 1..5 or explicit unknown")
            flags = row.get(side + "_flags")
            if not isinstance(flags, list) or len(set(flags)) != len(flags) or not set(flags).issubset(FLAGS):
                raise ValueError("Invalid artifact flags")
    return rater


def human_report(args):
    p = load(args)
    # Recompute the score-independent frozen randomization; verify all public
    # bytes and the private mapping instead of trusting a manually edited map.
    blind(args)
    private = read(args.out / "blind_private.json")
    public = read(args.out / "blind/study.json")
    verify_seal(public, "bundle_id")
    if private["plan_sha256"] != p["plan_sha256"] or private["bundle_id"] != public["bundle_id"]:
        raise ValueError("Blind mapping mismatch")
    trials = {r["id"]: r for r in private["trials"]}
    votes, raters, identities = {}, {}, {}
    for path in sorted(args.ratings.glob("*.json")):
        data = read(path)
        rater = validate_ratings(data, private)
        if rater in raters and raters[rater] != data["prior_exposure"]:
            raise ValueError("Conflicting prior-exposure declaration")
        raters[rater] = data["prior_exposure"]
        identities[path.name] = file_hash(path)
        for row in data["ratings"]:
            key = (rater, row["id"])
            if key in votes and votes[key] != row:
                raise ValueError("Conflicting exports from the same rater; inspect, do not silently overwrite")
            votes[key] = row
    if not votes:
        raise ValueError(f"No completed human ratings in {args.ratings}")
    scored = load_scores(args, p) if (args.out / "scores.json").exists() else None

    def aligned(trial, vote):
        left = "A" if trial["A"] == trial["left"] else "B"
        right = "B" if left == "A" else "A"
        preference = vote["preference"]
        preference = ("left" if preference == left else "right") if preference in ("A", "B") else preference
        deltas = {a: vote[right][a] - vote[left][a] if vote[right][a] is not None and vote[left][a] is not None else None for a in AXES}
        return preference, deltas

    rows, reliability = [], []
    for trial in private["trials"]:
        selected = [(rater, v) for (rater, tid), v in votes.items() if tid == trial["id"]]
        if trial["repeat_of"]:
            for rater, vote in selected:
                original = votes.get((rater, trial["repeat_of"]))
                if original:
                    first = aligned(trials[trial["repeat_of"]], original)[0]
                    second = aligned(trial, vote)[0]
                    reliability.append({"rater_id": rater, "trial": trial["id"],
                                        "comparable": first != "uncertain" and second != "uncertain",
                                        "preference_agrees": first == second,
                                        "note": "QC only; no automatic rater exclusion"})
            continue
        aligned_votes = [aligned(trial, v) for _, v in selected]
        row = {k: trial[k] for k in ("id", "comparison", "pair_id", "prompt_id", "seed")}
        row.update(n_raters=len(selected), n_declared_unexposed=sum(not raters[r] for r, _ in selected))
        for preference in ("left", "right", "tie", "uncertain"):
            row["votes_" + preference] = sum(v[0] == preference for v in aligned_votes)
        for axis in AXES:
            values = [v[1][axis] for v in aligned_votes if v[1][axis] is not None]
            row["n_" + axis] = len(values)
            row["delta_" + axis] = statistics.mean(values) if values else None
        if scored:
            for dimension in p["config"]["quality_dimensions"] + p["config"]["diagnostic_dimensions"]:
                row["vbench_delta_" + dimension] = scored["scores"][trial["right"]][dimension] - scored["scores"][trial["left"]][dimension]
        rows.append(row)
    seed_summary = []
    for comparison, prompt_id in sorted({(r["comparison"], r["prompt_id"]) for r in rows}):
        selected = [r for r in rows if r["comparison"] == comparison and r["prompt_id"] == prompt_id]
        summary = {"comparison": comparison, "prompt_id": prompt_id, "n_seed_groups": len(selected)}
        for axis in AXES:
            values = [r["delta_" + axis] for r in selected if r["delta_" + axis] is not None]
            summary["n_rated_seeds_" + axis] = len(values)
            summary["mean_seed_delta_" + axis] = statistics.mean(values) if values else None
        seed_summary.append(summary)
    body = {"plan_sha256": p["plan_sha256"], "bundle_id": private["bundle_id"], "rating_files": identities,
            "scores_sha256": scored["scores_sha256"] if scored else None,
            "raters_prior_exposure": raters, "n_raters": len(raters), "n_primary_pairs": len(rows),
            "coverage_minimum_met": all(r["n_raters"] >= p["config"]["minimum_raters_per_pair"] for r in rows),
            "independent_unexposed_coverage_met": all(r["n_declared_unexposed"] >= p["config"]["minimum_raters_per_pair"] for r in rows),
            "ready_for_confirmatory_claim": False, "pairs": rows, "per_prompt_seed_means": seed_summary,
            "raw_ratings": [{"rater_id": rater, **vote} for (rater, _), vote in sorted(votes.items())],
            "reliability": reliability, "claim": p["config"]["claim"], "limitations": p["config"]["limitations"],
            "caution": "Self-reported IDs/exposure do not verify independence. Partial coverage and unknowns are explicit; repeated trials excluded. No automatic human composite, metric calibration, training labels, p-values, or claim of VBench failure."}
    result = sealed(body, "analysis_sha256")
    folder = args.out / "human_reports" / result["analysis_sha256"][:16]
    write_new(folder / "report.json", result)
    write_csv(folder / "paired_human_vbench.csv", rows)
    write_csv(folder / "per_prompt_human.csv", seed_summary)
    print(f"Analyzed {len(raters)} declared raters; coverage={body['coverage_minimum_met']}. {folder}")


def export(args, public_only):
    p = load(args)
    if public_only:
        blind(args)
        # Build a whitelist, never recursively include private mapping or scoring files.
        public = read(args.out / "blind/study.json")
        verify_seal(public, "bundle_id")
        if read(args.out / "blind_private.json")["plan_sha256"] != p["plan_sha256"]:
            raise ValueError("Foreign blind bundle")
        names = ["index.html", "study.json"] + [f"media/{r['clip_id']}.mp4" for r in p["clips"]]
        files = [(args.out / "blind" / name, Path("video_review") / name) for name in names]
        label = "raters"
    else:
        allowed = ["plan.json", "check.json", "scores.json", "report.json", "blind_private.json",
                   "paired_vbench.csv", "per_video_vbench.csv", "per_prompt_vbench.csv"]
        files = [(args.out / name, Path("flashvsr_spatial_analysis") / name) for name in allowed if (args.out / name).is_file()]
        for folder_name in ("blind", "vbench", "ratings", "human_reports"):
            folder = args.out / folder_name
            for path in sorted(folder.rglob("*")) if folder.exists() else []:
                if path.is_file() and (folder_name == "blind" or path.suffix in (".json", ".csv")):
                    files.append((path, Path("flashvsr_spatial_analysis") / path.relative_to(args.out)))
        label = "analysis_PRIVATE"
    inventory = []
    for path, relative in files:
        if not path.resolve().is_relative_to(args.out.resolve()):
            raise ValueError("Export symlink escapes evaluation root")
        inventory.append({"path": relative.as_posix(), "sha256": file_hash(path), "bytes": path.stat().st_size})
    manifest = {"plan_sha256": p["plan_sha256"], "scope": label, "files": inventory}
    target = args.out / "exports" / f"flashvsr_spatial_{label}_{digest(manifest)[:12]}.tgz"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with tarfile.open(target, "r:gz") as archive:
            members = archive.getmembers()
            expected = {r["path"]: r for r in inventory}
            if {m.name for m in members} != set(expected) | {"manifest.json"} or any(not m.isfile() for m in members):
                raise ValueError("Existing export inventory mismatch")
            if json.load(archive.extractfile("manifest.json")) != manifest:
                raise ValueError("Existing export manifest mismatch")
            for name, row in expected.items():
                with archive.extractfile(name) as stream:
                    if archive.getmember(name).size != row["bytes"] or hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                        raise ValueError("Existing export payload mismatch")
    else:
        with tarfile.open(target, "x:gz") as archive:
            for path, relative in files:
                archive.add(path, arcname=relative.as_posix(), recursive=False)
            payload = json.dumps(manifest, ensure_ascii=False).encode()
            info = tarfile.TarInfo("manifest.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    print(f"Packaged {label}: {target} ({target.stat().st_size / 1024**2:.1f} MiB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "check", "blind", "score", "report", "human-report", "export-rater", "export-analysis"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/flashvsr_asset_diagnostic_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/flashvsr_spatial_baseline_eval_v1")
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--vbench-root", type=Path, default=Path("/mnt/afs_2/houze/VBench"))
    parser.add_argument("--vbench-python", default="/opt/conda/bin/python")
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--ffprobe")
    args = parser.parse_args()
    args.source, args.out = args.source.resolve(), args.out.resolve()
    if args.ratings is None:
        args.ratings = args.out / "ratings"
    if args.mode.startswith("export-"):
        export(args, args.mode == "export-rater")
    else:
        globals()[args.mode.replace("-", "_")](args)


if __name__ == "__main__":
    main()
