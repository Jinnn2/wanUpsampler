"""Synthetic-degradation control: every metric layer on FULL clips degraded by a known kind and amount.

A metric that measures quality loss should fall monotonically with the severity of a degradation it claims to
see. Here the ground truth is the construction, not a human: for each selected FULL clip, six degradations at
three severities are applied, plus an identity control (decode and re-encode only, the noise floor of both the
encoder and the paired judges). The degraded clips are written as one more "source" in the fixed-layer plan
format, so the fixed layer, the dynamic layer (requirement retention), the quality layer and VBench all score
them with their unchanged code; `report` then puts every metric against every degradation.

Degradations (level 1..3; level 0 = identity re-encode):
- blur:    Gaussian blur, sigma 1, 2, 4 px
- downup:  area downscale by 2, 3, 4 then bicubic upscale (what a low-resolution pipeline without SR loses)
- noise:   additive Gaussian noise, sd 4, 8, 16 (0-255 scale)
- jpeg:    per-frame JPEG at quality 40, 20, 8 (blocking and ringing)
- hold:    each frame repeated 2, 3, 4 times (judder at unchanged length)
- flicker: independent per-frame brightness gain 1 +- 3%, 6%, 12%

Usage (see run_univ_degradation.sh):
  python synthetic_degradation.py plan  --fixed-root OUT/fixed_metrics_v1 --out OUT/degradation_v1
  python synthetic_degradation.py make  --out OUT/degradation_v1 --shard-index I --shard-count N
  python synthetic_degradation.py finalize --out OUT/degradation_v1      # writes OUT/.../fixed/plan.json
  python synthetic_degradation.py vbench --out OUT/degradation_v1 --ngpus 4
  python synthetic_degradation.py report --out OUT/degradation_v1
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic  # noqa: E402
from UNIV_adaptor.scripts.data.score_controlled_factor_requirements import shard  # noqa: E402

SCHEMA = "univ_synthetic_degradation_v1"
SOURCE = "synthetic_degradation_v1"
SEED = 20261026
FULL = "FULL_NATIVE_50"
BASE_SEED = 42          # one seed per prompt keeps the run small; prompts carry the variance
PER_CELL = 16           # FULL clips per motion x detail cell: 64 clips x 19 variants = 1216 videos
DEGRADATIONS = {
    "blur": (1.0, 2.0, 4.0),
    "downup": (2, 3, 4),
    "noise": (4.0, 8.0, 16.0),
    "jpeg": (40, 20, 8),
    "hold": (2, 3, 4),
    "flicker": (0.03, 0.06, 0.12),
}
CRF = 12  # near-transparent re-encode; the identity variant measures what it costs
FIELDS = ("group_id", "prompt_id", "family_id", "split", "motion_level", "detail_level", "base_seed")
DEFAULT_PROMPTS = [ROOT / "prompts/univ_controlled_factor_v1.jsonl", ROOT / "prompts/univ_controlled_factor_dev2.jsonl"]


def variants():
    """(case_id, kind, level index, parameter); identity first."""
    out = [("DEG_IDENTITY_0", "identity", 0, None)]
    for kind, params in DEGRADATIONS.items():
        out += [(f"DEG_{kind.upper()}_{i + 1}", kind, i + 1, p) for i, p in enumerate(params)]
    return out


# ---------------------------------------------------------------- plan

def plan(args):
    fixed = json.loads((Path(args.fixed_root) / "plan.json").read_text(encoding="utf-8"))
    by_prompt = {}
    for v in sorted(fixed["videos"], key=lambda v: (v["prompt_id"], v["source"])):
        if v["case_id"] == FULL and v["split"] != "test" and int(v["base_seed"]) == BASE_SEED:
            by_prompt.setdefault(v["prompt_id"], v)  # FULL clips of one prompt and seed are the same draw per source
    cells = collections.defaultdict(list)
    for v in by_prompt.values():
        cells[(v["motion_level"], v["detail_level"])].append(v)
    rng = random.Random(SEED)
    clips = []
    for cell in sorted(cells):
        pool = sorted(cells[cell], key=lambda v: v["prompt_id"])
        rng.shuffle(pool)
        if len(pool) < PER_CELL:
            raise ValueError(f"only {len(pool)} FULL clips in cell {cell}")
        clips += sorted(pool[:PER_CELL], key=lambda v: v["prompt_id"])
    out = Path(args.out)
    if (out / "plan.json").exists():
        raise ValueError(f"{out / 'plan.json'} exists; use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "plan.json", {
        "schema": SCHEMA, "seed": SEED, "crf": CRF, "degradations": {k: list(v) for k, v in DEGRADATIONS.items()},
        "fixed_plan_sha256": sha256_file(Path(args.fixed_root) / "plan.json"),
        "clips": [{**{k: v[k] for k in FIELDS}, "video_path": v["video_path"], "video_sha256": v["video_sha256"],
                   "origin_source": v["source"]} for v in clips]})
    print(f"planned {len(clips)} FULL clips x {len(variants())} variants = {len(clips) * len(variants())} videos; "
          f"cells: {dict(collections.Counter((c['motion_level'], c['detail_level']) for c in clips))}")


def load_plan(out):
    value = json.loads((Path(out) / "plan.json").read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value["degradations"] != {k: list(v) for k, v in DEGRADATIONS.items()}:
        raise ValueError("unsupported or changed plan; use a new --out")
    return value


def stem(clip, case):
    return f"p{clip['prompt_id']}_s{clip['base_seed']}_{case}"


# ---------------------------------------------------------------- degrade

def degrade(frames, kind, param, rng):
    """frames: uint8 (T, H, W, 3) RGB. Returns the same shape and dtype."""
    import cv2

    if kind == "identity":
        return frames
    if kind == "blur":
        return np.stack([cv2.GaussianBlur(f, (0, 0), param) for f in frames])
    if kind == "downup":
        h, w = frames.shape[1:3]
        small = (max(1, round(w / param)), max(1, round(h / param)))
        return np.stack([cv2.resize(cv2.resize(f, small, interpolation=cv2.INTER_AREA), (w, h),
                                    interpolation=cv2.INTER_CUBIC) for f in frames])
    if kind == "noise":
        return np.clip(frames + rng.normal(0.0, param, frames.shape), 0, 255).astype(np.uint8)
    if kind == "jpeg":
        out = []
        for f in frames:
            ok, buf = cv2.imencode(".jpg", f[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(param)])
            if not ok:
                raise ValueError("JPEG encode failed")
            out.append(cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1])
        return np.stack(out)
    if kind == "hold":
        return frames[(np.arange(len(frames)) // int(param)) * int(param)]
    if kind == "flicker":
        gain = 1.0 + rng.uniform(-param, param, len(frames))
        return np.clip(frames * gain[:, None, None, None], 0, 255).astype(np.uint8)
    raise ValueError(f"unknown degradation: {kind}")


def read_video(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < 2:
        raise ValueError(f"could not decode video: {path}")
    return np.stack(frames), float(fps)


def encode(frames, fps, dst, ffmpeg):
    t, h, w, _ = frames.shape
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF),
           "-pix_fmt", "yuv420p", "-threads", "2", "-map_metadata", "-1", str(dst)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


def make(args):
    value = load_plan(args.out)
    vdir = Path(args.out) / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    clips = shard(value["clips"], args.shard_index, args.shard_count)
    for n, clip in enumerate(clips):
        todo = [v for v in variants() if not (vdir / f"{stem(clip, v[0])}.mp4").exists()]
        if not todo:
            continue
        if sha256_file(clip["video_path"]) != clip["video_sha256"]:
            raise ValueError(f"FULL clip changed since planning: {clip['video_path']}")
        frames, fps = read_video(clip["video_path"])
        for case, kind, _, param in todo:
            # Seeded per clip and variant, so a rerun or another shard layout reproduces the same video.
            rng = np.random.default_rng(int(hashlib.sha256(f"{SEED}-{clip['video_sha256']}-{case}".encode())
                                            .hexdigest()[:16], 16))
            dst = vdir / f"{stem(clip, case)}.mp4"
            tmp = dst.with_name(dst.stem + ".tmp.mp4")
            encode(degrade(frames, kind, param, rng), fps, tmp, args.ffmpeg)
            tmp.replace(dst)
        print(f"[make {args.shard_index}/{args.shard_count}] {n + 1}/{len(clips)} p{clip['prompt_id']}", flush=True)


def finalize(args):
    """The degraded clips (and their FULL clips) as one source in the fixed-layer plan format."""
    from UNIV_adaptor.scripts.data.score_fixed_metrics import PLAN_SCHEMA, SIGNAL_VERSION

    value = load_plan(args.out)
    out = Path(args.out)
    vdir = out / "videos"
    rows = []
    for clip in value["clips"]:
        base = {k: clip[k] for k in FIELDS} | {"source": SOURCE, "group_id": f"p{clip['prompt_id']}_s{clip['base_seed']}"}
        rows.append(base | {"stem": stem(clip, FULL), "case_id": FULL, "video_path": clip["video_path"],
                            "video_sha256": clip["video_sha256"]})
        for case, _, _, _ in variants():
            path = vdir / f"{stem(clip, case)}.mp4"
            if not path.exists():
                raise ValueError(f"missing degraded video (run make first): {path}")
            rows.append(base | {"stem": stem(clip, case), "case_id": case, "video_path": str(path.resolve()),
                                "video_sha256": sha256_file(path)})
    fixed = out / "fixed"
    fixed.mkdir(parents=True, exist_ok=True)
    write_json_atomic(fixed / "plan.json", {
        "schema": PLAN_SCHEMA, "signal_version": SIGNAL_VERSION,
        "sources": {SOURCE: {"kind": "synthetic", "root": str(out.resolve()), "identity": {"plan_sha256": sha256_file(out / "plan.json")},
                             "videos": len(rows), "excluded_test_videos": 0}},
        "videos": rows})
    print(f"wrote {fixed / 'plan.json'}: {len(rows)} videos")


# ---------------------------------------------------------------- VBench

def vbench(args):
    import os
    import shutil

    from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import (
        inspect_vbench_checkout, score_case_directory, warmup_vbench_cache)
    from UNIV_adaptor.scripts.data import requirements_v2 as rv2

    out = Path(args.out)
    rows = json.loads((out / "fixed" / "plan.json").read_text(encoding="utf-8"))["videos"]
    prompts = rv2.load_prompts(args.prompts)
    identity = inspect_vbench_checkout(Path(args.vbench_root), expected_commit=args.expected_vbench_commit)
    warmup_vbench_cache(args.vbench_python, Path(args.vbench_root))
    result = []
    for case in sorted({r["case_id"] for r in rows}):
        sub = [r for r in rows if r["case_id"] == case]
        directory = out / "vbench_inputs" / case
        directory.mkdir(parents=True, exist_ok=True)
        for r in sub:
            target = directory / f"{r['stem']}.mp4"
            if not target.exists():
                try:
                    os.link(r["video_path"], target)
                except OSError:
                    shutil.copy2(r["video_path"], target)
        prompt_map = directory / "prompt_map.json"
        prompt_map.write_text(json.dumps({str((directory / f"{r['stem']}.mp4").resolve()): prompts[r["prompt_id"]]["prompt"]
                                          for r in sub}, ensure_ascii=False), encoding="utf-8")
        bundle = score_case_directory(Path(args.vbench_root), args.vbench_python, directory, prompt_map,
                                      out / "vbench" / case, args.quality + args.diagnostic, args.quality,
                                      args.diagnostic, args.ngpus, False, identity)
        for r in sub:
            s = bundle.scores[r["stem"]]
            result.append({"stem": r["stem"], "case_id": case, **{d: s[d] for d in args.quality + args.diagnostic},
                           "vbench5": float(np.mean([s[d] for d in args.quality]))})
        print(f"[vbench] {case}: {len(sub)} videos", flush=True)
    with (out / "vbench_by_video.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f"wrote {out / 'vbench_by_video.csv'} ({len(result)} rows)")


# ---------------------------------------------------------------- report

# metric -> (file, column, kind): "delta" = minus the same clip's FULL value; "kept" = already relative to FULL.
# Every metric is oriented so that lower = worse after the sign in `orient` (fixed-layer artifact signals rise).
METRICS = {
    "vbench5": ("vbench", "vbench5", "delta", 1),
    "aesthetic_quality": ("vbench", "aesthetic_quality", "delta", 1),
    "imaging_quality": ("vbench", "imaging_quality", "delta", 1),
    "motion_smoothness": ("vbench", "motion_smoothness", "delta", 1),
    "subject_consistency": ("vbench", "subject_consistency", "delta", 1),
    "retention": ("dynamic", "retention", "kept", 1),
    "quality_kept": ("quality", "quality_kept", "kept", 1),
    "kept_sharp": ("quality", "kept_sharp", "kept", 1),
    "kept_artifact": ("quality", "kept_artifact", "kept", 1),
    "kept_aesthetic": ("quality", "kept_aesthetic", "kept", 1),
    "kept_temporal": ("quality", "kept_temporal", "kept", 1),
    "kept_acceptable": ("quality", "kept_acceptable", "kept", 1),
    "d_sharpness": ("fixed", "d_sharpness", "kept", 1),
    "d_flicker": ("fixed", "d_flicker", "kept", -1),
    "d_flow_alternation": ("fixed", "d_flow_alternation", "kept", -1),
    "d_boil": ("fixed", "d_boil", "kept", -1),
}


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def flt(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def avg_rank(a):
    """Ranks with ties sharing their mean rank."""
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(avg_rank(x), avg_rank(y))[0, 1])


def collect(out):
    """stem -> {metric: value oriented so that higher = better, relative to FULL}."""
    out = Path(out)
    tables = {"fixed": out / "fixed" / "fixed_by_video.csv", "dynamic": out / "dynamic" / "dynamic_by_video.csv",
              "quality": out / "quality" / "quality_by_video.csv", "vbench": out / "vbench_by_video.csv"}
    loaded = {k: {r["stem"]: r for r in read_csv(p)} for k, p in tables.items() if p.exists()}
    missing = sorted(set(tables) - set(loaded))
    value = load_plan(out)
    result = {}
    for clip in value["clips"]:
        full_stem = stem(clip, FULL)
        for case, kind, level, _ in variants():
            s = stem(clip, case)
            row = {"kind": kind, "level": level, "prompt_id": clip["prompt_id"],
                   "cell": clip["motion_level"][0] + clip["detail_level"][0]}
            for name, (table, col, how, sign) in METRICS.items():
                t = loaded.get(table)
                if t is None or s not in t:
                    row[name] = float("nan")
                    continue
                v = flt(t[s].get(col))
                if how == "delta":
                    v = v - flt(t[full_stem].get(col)) if full_stem in t else float("nan")
                row[name] = sign * v
            result[s] = row
    return result, missing


def report(args):
    rows, missing = collect(args.out)
    if missing:
        print(f"NOTE: no results yet for: {', '.join(missing)} (those columns are empty)")
    names = [m for m in METRICS if any(math.isfinite(r[m]) for r in rows.values())]
    kinds = list(DEGRADATIONS)
    lines = ["# Synthetic-degradation control", "",
             "Each metric is oriented so that higher = closer to FULL (fixed-layer flicker, judder and boil signals "
             "are sign-flipped). Severity 0 is the identity re-encode.", "",
             "## Spearman correlation of each metric with severity (0-3), pooled over clips",
             "A metric that sees a degradation should be clearly negative; near 0 = blind to it.", "",
             "| metric | " + " | ".join(kinds) + " |", "| --- |" + " --- |" * len(kinds)]
    for m in names:
        cells = []
        for k in kinds:
            pts = [(r["level"], r[m]) for r in rows.values() if r["kind"] in (k, "identity")]
            rho = spearman([p[0] for p in pts], [p[1] for p in pts])
            cells.append("" if math.isnan(rho) else f"{rho:+.2f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "## Mean by severity (identity, level 1, 2, 3)", "",
              "| metric | identity | " + " | ".join(f"{k} 1/2/3" for k in kinds) + " |",
              "| --- | --- |" + " --- |" * len(kinds)]
    for m in names:
        ident = np.nanmean([r[m] for r in rows.values() if r["kind"] == "identity"])
        cells = []
        for k in kinds:
            means = [np.nanmean([r[m] for r in rows.values() if r["kind"] == k and r["level"] == lv]) for lv in (1, 2, 3)]
            cells.append(" / ".join(f"{x:.3g}" for x in means))
        lines.append(f"| {m} | {ident:.3g} | " + " | ".join(cells) + " |")
    lines += ["", "## Detection rate at the strongest level",
              "Share of clips whose level-3 value is below the same clip's identity value (0.5 = chance).", "",
              "| metric | " + " | ".join(kinds) + " |", "| --- |" + " --- |" * len(kinds)]
    by_clip = collections.defaultdict(dict)
    for r in rows.values():
        by_clip[r["prompt_id"]][(r["kind"], r["level"])] = r
    for m in names:
        cells = []
        for k in kinds:
            pts = [(c[(k, 3)][m], c[("identity", 0)][m]) for c in by_clip.values()
                   if (k, 3) in c and ("identity", 0) in c and math.isfinite(c[(k, 3)][m]) and math.isfinite(c[("identity", 0)][m])]
            cells.append(f"{np.mean([(a < b) + 0.5 * (a == b) for a, b in pts]):.2f}" if pts else "")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (Path(args.out) / "degradation_report.md").write_text(text, encoding="utf-8")
    with (Path(args.out) / "degradation_by_video.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", *next(iter(rows.values()))])
        writer.writeheader()
        writer.writerows({"stem": s, **r} for s, r in rows.items())
    print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--fixed-root", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("make")
    p.add_argument("--out", required=True)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p = sub.add_parser("finalize")
    p.add_argument("--out", required=True)
    p = sub.add_parser("vbench")
    p.add_argument("--out", required=True)
    p.add_argument("--prompts", nargs="+", default=[str(x) for x in DEFAULT_PROMPTS])
    p.add_argument("--vbench-root", default="/mnt/afs_2/houze/VBench")
    p.add_argument("--vbench-python", default="/opt/conda/bin/python")
    p.add_argument("--expected-vbench-commit", default="fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490")
    p.add_argument("--ngpus", type=int, default=1)
    p.add_argument("--quality", nargs="+", default=["subject_consistency", "background_consistency",
                                                     "motion_smoothness", "aesthetic_quality", "imaging_quality"])
    p.add_argument("--diagnostic", nargs="*", default=["dynamic_degree", "overall_consistency"])
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"plan": plan, "make": make, "finalize": finalize, "vbench": vbench, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
