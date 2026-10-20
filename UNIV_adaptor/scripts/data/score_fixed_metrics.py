"""Fixed-layer metrics (REQUIREMENTS_V2_SPEC.md, layer 1): prompt-independent signals for every video,
then paired against the same-seed FULL clip of the same source. No VLM, no checklist.

Per video (signals mode, cached by video sha256 so shared FULL clips are computed once):
- temporal artifacts: frame-hold fraction and flow alternation (judder), luma flicker, high-frequency
  "boil" in near-static regions after flow compensation, warp-error spikes (flow breaks);
- motion: RAFT on every consecutive pair; global magnitude, camera (median flow vector) and local
  (residual after removing the median) components;
- detail and colour: per-frame Laplacian sharpness (level and temporal stability), saturation, value,
  colourfulness and saturation-weighted hue.
The report pairs each accelerated clip with its FULL clip: log motion ratios (more AND less motion both
count as infidelity), detail and colour shifts, and artifact differences.

Locked test rows are excluded at plan time and never decoded.

Usage (see run_univ_fixed_metrics.sh):
  python score_fixed_metrics.py plan --published ROOT [ROOT ...] --dvg-study ROOT [ROOT ...] --out OUT
  python score_fixed_metrics.py signals --out OUT --shard-index I --shard-count N
  python score_fixed_metrics.py report --out OUT
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic  # noqa: E402
from UNIV_adaptor.scripts.data.score_controlled_factor_requirements import (  # noqa: E402
    FLOW_SIZE, SIGNAL_FRAMES, pick, published_rows, read_frames, shard)

PLAN_SCHEMA = "univ_fixed_metrics_plan_v1"
SIGNAL_VERSION = "fixed_signals_v1"
FULL = "FULL_NATIVE_50"
STATIC_FLOW_PX = 1.0  # pixels at FLOW_SIZE; below this a pixel counts as near-static for boil
EPS = 0.05  # px, floor for log motion ratios
FIELDS = ("source", "stem", "group_id", "prompt_id", "family_id", "split", "motion_level", "detail_level",
          "base_seed", "case_id", "video_path", "video_sha256")


# ---------------------------------------------------------------- plan

def dvg_rows(root):
    from UNIV_adaptor.scripts.data import dvg_wan21_study as dvg

    rows = [r for r in dvg.receipts(root, dvg.load_plan(root)) if r["set"] == "main"]
    return rows, {"dvg_study_plan_sha256": sha256_file(Path(root) / "plan.json")}


def plan(args):
    videos, sources = [], {}
    for kind, roots in (("published", args.published or []), ("dvg", args.dvg_study or [])):
        for root in roots:
            name = Path(root).resolve().name
            if name in sources:
                raise ValueError(f"duplicate source name: {name}")
            rows, identity = published_rows(root)[:2] if kind == "published" else dvg_rows(root)
            keep = [r for r in rows if r["split"] != "test"]
            if not any(r["case_id"] == FULL for r in keep):
                raise ValueError(f"{name}: no FULL clips to pair against")
            sources[name] = {"kind": kind, "root": str(Path(root).resolve()), "identity": identity,
                             "videos": len(keep), "excluded_test_videos": len(rows) - len(keep)}
            videos += [{**{k: r[k] for k in FIELDS if k != "source"}, "source": name} for r in keep]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "plan.json", {"schema": PLAN_SCHEMA, "signal_version": SIGNAL_VERSION,
                                          "sources": sources, "videos": videos})
    unique = len({v["video_sha256"] for v in videos})
    print(f"planned {len(videos)} videos from {len(sources)} sources ({unique} unique clips)")


def load_plan(out):
    value = json.loads((Path(out) / "plan.json").read_text(encoding="utf-8"))
    if value.get("schema") != PLAN_SCHEMA or value.get("signal_version") != SIGNAL_VERSION:
        raise ValueError("unsupported plan; re-plan into a new output root")
    return value


def unique_clips(value):
    seen = {}
    for v in value["videos"]:
        seen.setdefault(v["video_sha256"], v["video_path"])
    return sorted(seen.items())


# ---------------------------------------------------------------- signals

def frame_stats(frames):
    import cv2

    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]).astype(np.float32)
    luma = gray.mean((1, 2))
    flicker = float(np.abs(luma[1:-1] - 0.5 * (luma[:-2] + luma[2:])).mean())
    diffs = np.abs(np.diff(gray, axis=0)).mean((1, 2))
    med = float(np.median(diffs))
    hold = float((diffs < 0.25 * med).mean()) if med > 0.5 else 0.0
    sharp = np.log(np.array([cv2.Laplacian(g / 255.0, cv2.CV_32F).var() for g in gray]) + 1e-8)
    sat, val, colorful, hue_x, hue_y = [], [], [], [], []
    for f in frames[pick(len(frames), SIGNAL_FRAMES)]:
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.float32)
        s, v = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
        angle = hsv[..., 0] / 180.0 * 2 * math.pi
        sat.append(s.mean())
        val.append(v.mean())
        hue_x.append((s * np.cos(angle)).mean())
        hue_y.append((s * np.sin(angle)).mean())
        r, g, b = (f[..., i].astype(np.float32) for i in range(3))
        rg, yb = r - g, 0.5 * (r + g) - b
        colorful.append(math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean()))
    return {"luma_mean": float(luma.mean()), "flicker": flicker, "pixel_hold_fraction": hold,
            "sharpness": float(sharp.mean()), "sharpness_temporal_std": float(sharp.std()),
            "saturation": float(np.mean(sat)), "value": float(np.mean(val)), "colorfulness": float(np.mean(colorful)),
            "hue_deg": float(math.degrees(math.atan2(np.mean(hue_y), np.mean(hue_x))) % 360.0)}


def flow_series(frames, model, device):
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)
    x = F.interpolate(x, size=FLOW_SIZE, mode="bilinear", align_corners=False, antialias=True)
    gray = x.mean(1, keepdim=True)
    highpass = gray - F.avg_pool2d(F.pad(gray, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
    h, w = FLOW_SIZE
    gy, gx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    mag, cam, loc, warp, boil = [], [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(x) - 1, 8):
            stop = min(start + 8, len(x) - 1)
            ia, ib = x[start:stop].to(device), x[start + 1:stop + 1].to(device)
            fwd = model(ia * 2 - 1, ib * 2 - 1)[-1]
            median = fwd.flatten(2).median(dim=2).values[:, :, None, None]
            speed = fwd.pow(2).sum(1).sqrt()
            mag.append(speed.mean((1, 2)))
            cam.append(median.flatten(1).pow(2).sum(1).sqrt())
            loc.append((fwd - median).pow(2).sum(1).sqrt().mean((1, 2)))
            grid = torch.stack(((gx + fwd[:, 0]) / (w - 1) * 2 - 1, (gy + fwd[:, 1]) / (h - 1) * 2 - 1), dim=-1)
            warp.append((F.grid_sample(gray[start + 1:stop + 1].to(device), grid, align_corners=True)
                         - gray[start:stop].to(device)).abs().mean((1, 2, 3)))
            ha = highpass[start:stop].to(device)
            hb = F.grid_sample(highpass[start + 1:stop + 1].to(device), grid, align_corners=True)
            still = (speed < STATIC_FLOW_PX).unsqueeze(1).float()
            resid = ((ha - hb).abs() * still).sum((1, 2, 3))
            energy = (ha.abs() * still).sum((1, 2, 3)).clamp_min(1e-6)
            boil.append(torch.where(still.sum((1, 2, 3)) > 0.05 * h * w, resid / energy, torch.full_like(resid, float("nan"))))
    m, c, l, e, b = (torch.cat(s).cpu().numpy().astype(np.float64) for s in (mag, cam, loc, warp, boil))
    med = float(np.median(m))
    return {"flow_mag": float(m.mean()), "flow_camera": float(c.mean()), "flow_local": float(l.mean()),
            "flow_alternation": float(np.abs(np.diff(m)).mean() / (m.mean() + 1e-3)),
            "flow_hold_fraction": float((m < 0.2 * med).mean()) if med > 0.2 else 0.0,
            "warp_error": float(e.mean()), "warp_spike": float(e.max() / (np.median(e) + 1e-6)),
            "boil": float(np.nanmean(b)) if np.isfinite(b).any() else float("nan"),
            "flow_mag_series": [round(float(v), 4) for v in m]}


def signals(args):
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    value = load_plan(args.out)
    device = torch.device(args.device)
    model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(device).eval()
    cache = Path(args.out) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [c for c in shard(unique_clips(value), args.shard_index, args.shard_count)
            if not (cache / f"{c[0]}.json").exists()]
    for i, (sha, path) in enumerate(todo):
        if sha256_file(path) != sha:
            raise ValueError(f"video hash mismatch: {path}")
        frames = read_frames(path)
        result = {"video_sha256": sha, "signal_version": SIGNAL_VERSION, "frames": int(len(frames)),
                  **frame_stats(frames), **flow_series(frames, model, device)}
        write_json_atomic(cache / f"{sha}.json", result)
        print(f"[signals {args.shard_index}/{args.shard_count}] {i + 1}/{len(todo)} {Path(path).name}", flush=True)


# ---------------------------------------------------------------- report

RAW = ("flow_mag", "flow_camera", "flow_local", "flow_alternation", "flow_hold_fraction", "pixel_hold_fraction",
       "flicker", "boil", "warp_error", "warp_spike", "sharpness", "sharpness_temporal_std", "saturation", "value",
       "colorfulness", "luma_mean", "hue_deg")
DIFF = ("flow_alternation", "flow_hold_fraction", "pixel_hold_fraction", "flicker", "boil", "warp_spike",
        "sharpness", "sharpness_temporal_std", "saturation", "value", "colorfulness")


def paired(row, full):
    out = {}
    for k in ("flow_mag", "flow_camera", "flow_local"):
        out[f"logratio_{k}"] = math.log((row[k] + EPS) / (full[k] + EPS))
    for k in DIFF:
        out[f"d_{k}"] = row[k] - full[k]
    out["hue_shift_deg"] = abs((row["hue_deg"] - full["hue_deg"] + 180.0) % 360.0 - 180.0)
    return out


def report(args):
    value = load_plan(args.out)
    cache = Path(args.out) / "cache"
    sig = {}
    for sha, _ in unique_clips(value):
        s = json.loads((cache / f"{sha}.json").read_text(encoding="utf-8"))
        if s["signal_version"] != SIGNAL_VERSION:
            raise ValueError(f"stale signal cache: {sha}")
        sig[sha] = s
    full = {(v["source"], v["group_id"]): sig[v["video_sha256"]] for v in value["videos"] if v["case_id"] == FULL}
    rows = []
    for v in value["videos"]:
        s = sig[v["video_sha256"]]
        row = {**{k: v[k] for k in FIELDS if k != "video_path"}, **{k: s[k] for k in RAW}}
        ref = full.get((v["source"], v["group_id"]))
        if ref is None:
            raise ValueError(f"no FULL clip for {v['source']} {v['group_id']}")
        if v["case_id"] != FULL:
            row.update(paired(s, ref))
        rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    path = Path(args.out) / "fixed_by_video.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")
    keys = ("logratio_flow_mag", "logratio_flow_camera", "d_flow_alternation", "d_flicker", "d_boil",
            "d_sharpness", "d_saturation", "hue_shift_deg")
    print("median paired deltas by case (all sources):")
    print("case".ljust(18) + "".join(k.replace("logratio_", "lr_").replace("d_", "")[:12].rjust(13) for k in keys))
    for case in sorted({r["case_id"] for r in rows} - {FULL}):
        sub = [r for r in rows if r["case_id"] == case]
        print(case.ljust(18) + "".join(f"{np.nanmedian([r[k] for r in sub]):13.3f}" for k in keys))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--published", nargs="*", help="finalized + scored published_wan21 roots")
    p.add_argument("--dvg-study", nargs="*", help="dvg_wan21_study roots (main set only)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("signals")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"plan": plan, "signals": signals, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
