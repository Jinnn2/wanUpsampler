"""Representative clips for the thesis figures, chosen by fixed rules from the finished scoring layers, then cut
into frame strips and x-t slices with FFmpeg. No generation, no GPU, no new scores.

Figure A (thesis step 1, "the average score is not enough"): clips where VBench does not penalize the method
but another signal shows a clear failure. Each case has three parts:

    blind      VBench-5 does not drop (relative to the same-seed FULL clip, or to a contrast arm)
    fail       a non-VBench signal crosses a fixed threshold
    severity   how bad the failure is (larger = worse)

Among the clips that are both blind and failing (and whose FULL clip the quality layer calls acceptable), the
clip at the MEDIAN severity is chosen, not the worst one, so the figure shows a typical failure. The caption
numbers (how many clips fail, how many of those VBench misses) are written next to it.

Figure B (thesis step 2, "each method has its own region"): one prompt x seed group per motion x detail cell.
Each group has a score vector over the compared arms (joint score of retention and quality_kept, both relative to
the same-seed FULL clip); the chosen group is the one whose vector is closest (L2) to its cell's mean vector, so
the clips show the cell's average ranking rather than the most extreme prompt.

Every rule is fixed here before looking at the clips. candidates.csv lists the next clips in the same order, so a
human swap (e.g. an unreadable scene) is recorded as a rank, never as a free pick.

Pins (--pins): clips outside the fixed plan, e.g. DVG's short-VBench-prompt collapse, which comes from a separate
study. They are rendered the same way and marked as pinned in the manifest.

Usage (see run_univ_figure_samples.sh):
  python figure_samples.py select --fixed-root F --dynamic D/dynamic_by_video.csv --quality Q2/quality_by_video.csv --out OUT
  python figure_samples.py pins-dvg --dvg-study DVG_ROOT --out OUT/pins.json   # optional, before select
  python figure_samples.py render --out OUT
"""
from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic  # noqa: E402

MANIFEST_SCHEMA = "univ_figure_samples_v1"
FULL = "FULL_NATIVE_50"
CELLS = ("hh", "hl", "lh", "ll")  # motion, detail
DEFAULT_PROMPTS = [ROOT / "prompts/univ_controlled_factor_v1.jsonl", ROOT / "prompts/univ_controlled_factor_dev2.jsonl"]
BLIND_PP = -0.5  # VBench-5 drop smaller than 0.5pp counts as "VBench does not penalize"
LOG2 = math.log(2.0)

# Figure A. Fields: arm, contrast arm (blind is judged against it instead of FULL), motion level filter,
# fail and severity on the clip record (see clip_records).
CASES_A = {
    "A1_freeze": dict(arm="DVG_D050", motion="high", note="high-motion clip loses more than half its flow",
                      fail=lambda c: c["lr_flow"] <= -LOG2, severity=lambda c: -c["lr_flow"]),
    "A2_judder": dict(arm="F41_DUP", contrast="F41_RIFE", note="frame-hold judder flagged by the fixed layer",
                      fail=lambda c: c["judder"] == 1, severity=lambda c: c["d_flow_alternation"]),
    "A3_static_motion": dict(arm="RCM4", motion="low", note="low-motion prompt gets more than twice FULL's flow",
                             fail=lambda c: c["lr_flow"] >= LOG2, severity=lambda c: c["lr_flow"]),
    "A4_semantic_loss": dict(arm="RCM4", note="keeps at most half of the requirements FULL satisfies",
                             fail=lambda c: c["retention"] <= 0.5, severity=lambda c: -c["retention"]),
}
ARMS_B = ("STEP20", "TEA015", "RCM4", "F41_RIFE", "LR_FLASH")
JOINT = {"min": min, "avg": lambda r, q: (r + q) / 2, "geo": lambda r, q: math.sqrt(max(r, 0.0) * max(q, 0.0))}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def flt(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def clean(x):
    """JSON-safe copy: NaN -> None, numpy scalars -> float."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


# ---------------------------------------------------------------- select

def vbench_by_video(fixed):
    """(source, stem) -> VBench-5 raw mean, from each source's own VBench CSV."""
    out = {}
    for name, src in fixed["sources"].items():
        root = Path(src["root"])
        path = root / "metrics" / "quality_by_video.csv" if src["kind"] == "published" else root / "quality_by_video.csv"
        for r in read_csv(path):
            out[(name, r["observation_id"])] = flt(r["vbench5"])
    return out


def clip_records(fixed, fixed_rows, dyn_rows, q_rows, vbench, prompts):
    sig = {(r["source"], r["stem"]): r for r in fixed_rows}
    dyn = {(r["source"], r["stem"]): r for r in dyn_rows}
    qual = {(r["source"], r["stem"]): r for r in q_rows}
    has_judder = bool(q_rows) and "judder" in q_rows[0]
    full = {(v["source"], v["group_id"]): v for v in fixed["videos"] if v["case_id"] == FULL}
    clips = []
    for v in fixed["videos"]:
        key = (v["source"], v["stem"])
        f = full[(v["source"], v["group_id"])]
        s, d, q = sig.get(key, {}), dyn.get(key, {}), qual.get(key, {})
        vb, vb_full = vbench.get(key, float("nan")), vbench.get((f["source"], f["stem"]), float("nan"))
        clips.append({
            "source": v["source"], "stem": v["stem"], "group_id": v["group_id"], "prompt_id": int(v["prompt_id"]),
            "split": v["split"], "cell": v["motion_level"][0] + v["detail_level"][0],
            "motion_level": v["motion_level"], "detail_level": v["detail_level"], "seed": int(v["base_seed"]),
            "case_id": v["case_id"], "video_path": v["video_path"], "video_sha256": v["video_sha256"],
            "prompt": prompts.get(int(v["prompt_id"]), ""), "full_stem": f["stem"],
            "full_video_path": f["video_path"], "full_video_sha256": f["video_sha256"],
            "vbench5": vb, "d_vbench5_pp": 100 * (vb - vb_full),
            "lr_flow": flt(s.get("logratio_flow_mag")), "flow_mag": flt(s.get("flow_mag")),
            "d_flow_alternation": flt(s.get("d_flow_alternation")),
            "retention": flt(d.get("retention")), "quality_kept": flt(q.get("quality_kept")),
            "full_acceptable": flt(q.get("full_acceptable")),
            "judder": int(flt(q.get("judder")) == 1) if has_judder else None})
    return clips, has_judder


def select_a(clips, has_judder):
    index = {(c["source"], c["group_id"], c["case_id"]): c for c in clips}
    panels, candidates, notes = [], [], []
    for case_id, rule in CASES_A.items():
        if rule["arm"] == "F41_DUP" and not has_judder:
            notes.append(f"{case_id}: skipped, the quality CSV has no judder flag (use the quality v2 CSV)")
            continue
        pool = [c for c in clips if c["case_id"] == rule["arm"] and c["full_acceptable"] == 1
                and (rule.get("motion") is None or c["motion_level"] == rule["motion"])]
        rows = []
        for c in pool:
            if "contrast" in rule:
                other = index.get((c["source"], c["group_id"], rule["contrast"]))
                if other is None:
                    continue
                blind = c["vbench5"] - other["vbench5"] >= BLIND_PP / 100
            else:
                other, blind = None, c["d_vbench5_pp"] >= BLIND_PP
            fail = bool(rule["fail"](c))
            rows.append((c, other, blind, fail))
        failing = [r for r in rows if r[3]]
        hits = sorted((r for r in failing if r[2]), key=lambda r: (-rule["severity"](r[0]), r[0]["stem"]))
        stats = {"arm_clips": len(rows), "failing": len(failing), "failing_and_vbench_blind": len(hits),
                 "fail_rule": rule["note"], "blind_rule": f"VBench-5 drop vs {rule.get('contrast', 'FULL')} > {BLIND_PP}pp"}
        if not hits:
            notes.append(f"{case_id}: no clip is both failing and VBench-blind ({stats})")
            continue
        pick = len(hits) // 2
        for rank, (c, other, _, _) in enumerate(hits):
            candidates.append({"panel": case_id, "rank": rank, "chosen": int(rank == pick),
                               "severity": rule["severity"](c), **summary(c)})
        c, other = hits[pick][0], hits[pick][1]
        rows_out = [row("FULL", c, full=True), row(c["case_id"], c)] + ([row(other["case_id"], other)] if other else [])
        panels.append({"id": case_id, "figure": "A", "cell": c["cell"], "group_id": c["group_id"], "seed": c["seed"],
                       "prompt_id": c["prompt_id"], "prompt": c["prompt"], "rule": "median severity among failing "
                       "and VBench-blind clips", "rank": pick, **stats, "rows": rows_out})
    return panels, candidates, notes


def select_b(clips, arms, metric):
    """One group per cell: the score vector closest to the cell mean. Published sources only (shared FULL)."""
    joint = JOINT[metric]
    by_group = collections.defaultdict(dict)
    for c in clips:
        if c["source"].startswith("dvg_"):
            continue
        if c["case_id"] in arms and c["full_acceptable"] == 1:
            score = joint(c["retention"], c["quality_kept"])
            if math.isfinite(score):
                by_group[c["group_id"]][c["case_id"]] = (score, c)
    panels, candidates, notes = [], [], []
    for cell in CELLS:
        groups = {g: a for g, a in by_group.items() if len(a) == len(arms) and next(iter(a.values()))[1]["cell"] == cell}
        if not groups:
            notes.append(f"B_{cell}: no group has all of {arms}")
            continue
        mat = np.array([[groups[g][a][0] for a in arms] for g in sorted(groups)])
        mean = mat.mean(0)
        dist = np.sqrt(((mat - mean) ** 2).sum(1))
        order = np.argsort(dist, kind="stable")
        names = sorted(groups)
        for rank, i in enumerate(order[:10]):
            g = groups[names[i]]
            candidates.append({"panel": f"B_{cell}", "rank": rank, "chosen": int(rank == 0), "severity": float(dist[i]),
                               **summary(next(iter(g.values()))[1]),
                               **{f"score_{a}": g[a][0] for a in arms}})
        g = groups[names[order[0]]]
        first = g[arms[0]][1]
        fulls = {g[a][1]["full_video_sha256"] for a in arms}
        if len(fulls) > 1:
            notes.append(f"B_{cell}: FULL clip differs across sources for {names[order[0]]}; the row uses {arms[0]}'s")
        panels.append({"id": f"B_{cell}", "figure": "B", "cell": cell, "group_id": names[order[0]], "seed": first["seed"],
                       "prompt_id": first["prompt_id"], "prompt": first["prompt"], "metric": metric,
                       "rule": "score vector closest to the cell mean", "rank": 0, "groups_in_cell": len(groups),
                       "cell_mean": {a: float(m) for a, m in zip(arms, mean)}, "distance": float(dist[order[0]]),
                       "rows": [row("FULL", first, full=True)] + [row(a, g[a][1], score=g[a][0]) for a in arms]})
    return panels, candidates, notes


def summary(c):
    return {k: c[k] for k in ("source", "stem", "group_id", "cell", "prompt_id", "seed", "case_id", "d_vbench5_pp",
                              "lr_flow", "d_flow_alternation", "retention", "quality_kept", "judder")}


def row(label, c, full=False, score=None):
    if full:
        return {"label": label, "source": c["source"], "stem": c["full_stem"], "video_path": c["full_video_path"],
                "video_sha256": c["full_video_sha256"]}
    out = {"label": label, "source": c["source"], "stem": c["stem"], "video_path": c["video_path"],
           "video_sha256": c["video_sha256"], **{k: c[k] for k in ("d_vbench5_pp", "lr_flow", "retention", "quality_kept")}}
    if score is not None:
        out["score"] = score
    return out


def load_pins(path):
    """[{"id", "note", "rows": [{"label", "video_path"}]}] -> pinned panels with hashes recorded now."""
    panels = []
    for p in json.loads(Path(path).read_text(encoding="utf-8")):
        rows = [{"label": r["label"], "video_path": str(Path(r["video_path"]).resolve()),
                 "video_sha256": sha256_file(r["video_path"])} for r in p["rows"]]
        panels.append({"id": p["id"], "figure": p.get("figure", "A"), "pinned": True, "note": p.get("note", ""),
                       "prompt": p.get("prompt", ""), "rule": "pinned by hand (outside the fixed plan)", "rows": rows})
    return panels


COLLAPSE_PROMPTS = ("a bench", "downtown", "shower", "A person is clapping")  # eyeballed 2026-10-18


def pins_dvg(args):
    """Pins for DVG's short-prompt collapse from the DVG study's repro set (VBench prompts, seed 42):
    FULL / DVG_D050 on the short prompt, plus DVG_D050 on the extended prompt when that set was generated."""
    from UNIV_adaptor.scripts.data import dvg_wan21_study as dvg

    value = dvg.load_plan(args.dvg_study)
    receipts = {r["stem"]: r for r in dvg.receipts(args.dvg_study, value)}
    jobs = [j for j in value["jobs"] if j["set"] in ("repro", "repro_ext")]
    wanted = {p.strip().lower(): p for p in args.prompts}
    pins = []
    for j in jobs:
        if j["set"] != "repro" or j["case_id"] != "DVG_D050" or j["prompt"].strip().lower() not in wanted:
            continue
        i = j["prompt_id"]
        rows = [("FULL", f"repro_v{i:04d}_s42__{FULL}"), ("DVG_D050", j["stem"]),
                ("DVG_D050_ext", f"reproext_v{i:04d}_s42__DVG_D050")]
        pins.append({"id": f"A5_collapse_v{i:04d}", "figure": "A", "prompt": j["prompt"],
                     "note": "DVG short-prompt collapse (VBench prompt, seed 42); _ext = same prompt extended",
                     "rows": [{"label": label, "video_path": receipts[stem]["video_path"]}
                              for label, stem in rows if stem in receipts]})
    missing = set(wanted) - {p["prompt"].strip().lower() for p in pins}
    if missing:
        short = sorted({j["prompt"] for j in jobs if j["set"] == "repro" and len(j["prompt"].split()) <= 3})
        raise ValueError(f"prompts not in the repro set: {sorted(missing)}; short repro prompts: {short}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(pins, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(pins)} pinned panels)")


def select(args):
    fixed_path = Path(args.fixed_root) / "plan.json"
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    prompts = {}
    for p in args.prompts or DEFAULT_PROMPTS:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                prompts[int(r["prompt_id"])] = r["prompt"]
    clips, has_judder = clip_records(fixed, read_csv(Path(args.fixed_root) / "fixed_by_video.csv"), read_csv(args.dynamic),
                                     read_csv(args.quality), vbench_by_video(fixed), prompts)
    clips = [c for c in clips if c["split"] != "test"]
    pa, ca, na = select_a(clips, has_judder)
    pb, cb, nb = select_b(clips, tuple(args.arms_b), args.metric)
    panels = pa + pb + (load_pins(args.pins) if args.pins else [])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "manifest.json", clean({
        "schema": MANIFEST_SCHEMA, "fixed_plan_sha256": sha256_file(fixed_path),
        "dynamic_sha256": sha256_file(args.dynamic), "quality_sha256": sha256_file(args.quality),
        "pins_sha256": sha256_file(args.pins) if args.pins else None, "arms_b": list(args.arms_b), "metric": args.metric,
        "blind_pp": BLIND_PP, "notes": na + nb, "panels": panels}))
    rows = ca + cb
    with (out / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(clean(rows))
    for p in panels:
        stat = (f"{p['failing_and_vbench_blind']}/{p['failing']} failing clips VBench-blind, {p['arm_clips']} clips"
                if "failing" in p else p.get("note") or f"{p.get('groups_in_cell')} groups, distance {p.get('distance', 0):.3f}")
        print(f"{p['id']:18s} {p.get('group_id', '-'):18s} {' / '.join(r['label'] for r in p['rows']):40s} {stat}")
    for n in na + nb:
        print("note:", n)
    print(f"wrote {out / 'manifest.json'} ({len(panels)} panels), {out / 'candidates.csv'} ({len(rows)} rows)")


# ---------------------------------------------------------------- render

def frame_count(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def slice_row(path, width, height):
    """Row (in the scaled frame) with the most temporal change in this clip: where an x-t slice shows motion."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (width, height), interpolation=cv2.INTER_AREA))
    cap.release()
    if len(frames) < 2:
        raise ValueError(f"could not decode video: {path}")
    change = np.abs(np.diff(np.stack(frames).astype(np.float32), axis=0)).mean((0, 2))
    change = np.convolve(change, np.ones(9) / 9, mode="same")
    margin = height // 10  # keep away from the borders
    return int(margin + np.argmax(change[margin:height - margin]))


def ffmpeg(args, *cmd):
    subprocess.run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *cmd], check=True)


def render(args):
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise ValueError("unsupported manifest")
    w, h = args.width, args.width * 480 // 832 // 2 * 2
    scale = f"scale={w}:{h}:flags=lanczos,setsar=1"
    receipts = []
    for p in manifest["panels"]:
        if args.panels and p["id"] not in args.panels:
            continue
        pdir = out / "render" / p["id"]
        pdir.mkdir(parents=True, exist_ok=True)
        for r in p["rows"]:
            if sha256_file(r["video_path"]) != r["video_sha256"]:
                raise ValueError(f"video hash mismatch: {r['video_path']}")
        y = slice_row(p["rows"][0]["video_path"], w, h)  # from the reference row (FULL), same line for every row
        files = []
        for i, r in enumerate(p["rows"]):
            name = f"{i}_{r['label']}"
            n = frame_count(r["video_path"])
            picks = sorted({round(k * (n - 1) / (args.frames - 1)) for k in range(args.frames)})
            sel = "+".join(f"eq(n\\,{k})" for k in picks)
            ffmpeg(args, "-i", r["video_path"], "-vf", f"{scale},select={sel},tile={len(picks)}x1",
                   "-frames:v", "1", str(pdir / f"{name}_strip.png"))
            ffmpeg(args, "-i", r["video_path"], "-vf", f"{scale},crop={w}:1:0:{y},tile=1x{n}",
                   "-frames:v", "1", str(pdir / f"{name}_xt.png"))
            files += [f"{name}_strip.png", f"{name}_xt.png"]
        ffmpeg(args, "-i", p["rows"][0]["video_path"], "-vf", f"{scale},drawbox=x=0:y={y - 1}:w={w}:h=3:color=red@0.9:t=fill",
               "-frames:v", "1", str(pdir / "slice_line.png"))
        inputs = [x for r in p["rows"] for x in ("-i", r["video_path"])]
        graph = ";".join(f"[{i}:v]{scale}[v{i}]" for i in range(len(p["rows"])))
        graph += ";" + "".join(f"[v{i}]" for i in range(len(p["rows"]))) + f"vstack=inputs={len(p['rows'])}[out]"
        ffmpeg(args, *inputs, "-filter_complex", graph, "-map", "[out]", "-c:v", "libx264", "-crf", str(args.crf),
               "-pix_fmt", "yuv420p", "-an", str(pdir / "compare.mp4"))
        files += ["slice_line.png", "compare.mp4"]
        receipts.append({"panel": p["id"], "slice_y": y, "size": [w, h], "frames_per_strip": args.frames, "files": files})
        print(f"[render] {p['id']}: {len(p['rows'])} rows, x-t row y={y}", flush=True)
    write_json_atomic(out / "render" / "receipts.json", {"manifest_sha256": sha256_file(out / "manifest.json"),
                                                         "panels": receipts})
    write_index(out, manifest, {r["panel"]: r for r in receipts})
    print(f"wrote {out / 'render' / 'index.html'}")


def write_index(out, manifest, receipts):
    """Offline review page: each panel's strips, x-t slices and side-by-side video, with the selection numbers."""
    e = html.escape
    parts = ["<!doctype html><meta charset='utf-8'><title>Figure samples</title><style>body{font:14px sans-serif;"
             "margin:16px;background:#fff;color:#111}img,video{max-width:100%;display:block;margin:4px 0}"
             ".p{border-top:1px solid #ccc;padding:12px 0}.m{color:#555}td,th{padding:2px 8px;text-align:right}"
             "</style><h1>Figure samples</h1>"]
    for p in manifest["panels"]:
        rec = receipts.get(p["id"])
        if rec is None:
            continue
        stat = ", ".join(f"{k}={p[k]}" for k in ("cell", "group_id", "rank", "arm_clips", "failing",
                                                    "failing_and_vbench_blind", "groups_in_cell") if k in p)
        parts.append(f"<div class='p'><h2>{e(p['id'])}</h2><p class='m'>{e(p['rule'])}; {e(stat)}; "
                     f"x-t row y={rec['slice_y']}</p><p>{e(p.get('prompt', ''))}</p>")
        parts.append("<table><tr><th>row</th><th>&Delta;VBench5 pp</th><th>log flow ratio</th><th>retention</th>"
                     "<th>quality_kept</th><th>score</th></tr>")
        for r in p["rows"]:
            cells = "".join(f"<td>{'' if r.get(k) is None else f'{r[k]:.3f}'}</td>"
                            for k in ("d_vbench5_pp", "lr_flow", "retention", "quality_kept", "score"))
            parts.append(f"<tr><th>{e(r['label'])}</th>{cells}</tr>")
        parts.append("</table>")
        base = f"{p['id']}/"
        for i, r in enumerate(p["rows"]):
            name = f"{i}_{r['label']}"
            parts.append(f"<h3>{e(r['label'])}</h3><img src='{base}{name}_strip.png'><img src='{base}{name}_xt.png'>")
        parts.append(f"<h3>x-t row</h3><img src='{base}slice_line.png' style='max-width:420px'>"
                     f"<video src='{base}compare.mp4' controls loop muted></video></div>")
    (out / "render" / "index.html").write_text("".join(parts), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("select")
    p.add_argument("--fixed-root", required=True)
    p.add_argument("--dynamic", required=True)
    p.add_argument("--quality", required=True, help="quality v2 CSV (has the judder flag)")
    p.add_argument("--prompts", nargs="*")
    p.add_argument("--pins", help="JSON list of pinned panels outside the fixed plan")
    p.add_argument("--arms-b", nargs="+", default=list(ARMS_B))
    p.add_argument("--metric", choices=sorted(JOINT), default="min")
    p.add_argument("--out", required=True)
    p = sub.add_parser("pins-dvg")
    p.add_argument("--dvg-study", required=True, help="dvg_wan21_study root with the repro set")
    p.add_argument("--prompts", nargs="+", default=list(COLLAPSE_PROMPTS))
    p.add_argument("--out", required=True)
    p = sub.add_parser("render")
    p.add_argument("--out", required=True)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--frames", type=int, default=6, help="frames per strip, evenly spaced, first and last included")
    p.add_argument("--width", type=int, default=832, help="every row is scaled to this width (aspect 832x480)")
    p.add_argument("--crf", type=int, default=12)
    p.add_argument("--panels", nargs="*", help="render only these panel ids")
    args = parser.parse_args(argv)
    {"select": select, "pins-dvg": pins_dvg, "render": render}[args.mode](args)


if __name__ == "__main__":
    main()
