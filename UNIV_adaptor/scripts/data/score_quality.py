"""Visual-quality layer: prompt-independent "does it look right" judgments for every video, next to the
dynamic layer's requirement retention (which only asks whether what the prompt asked for survived).

Five fixed aspects (QUALITY), each asked of every clip and, pairwise, of every accelerated clip against its
same-seed FULL clip in both orders:
- sharp, artifact, aesthetic: Qwen3-VL P(Yes) on 5 native-resolution frames, averaged;
- temporal, acceptable: P(Yes) on 32 uniformly sampled frames given as one video.
Optional: DOVER (aesthetic and technical video quality, trained on human ratings) when its repo is installed.
The fixed layer's Laplacian sharpness is joined in the report. All thresholds here are provisional; the
quality questions of the human kit (annotation_v2.py quality-kit) use the same five aspects to calibrate them.

Usage (see run_univ_quality.sh):
  python score_quality.py plan  --fixed-root OUT/fixed_metrics_v1 --out OUT
  python score_quality.py vlm   --out OUT --model M --shard-index I --shard-count N
  python score_quality.py pair  --out OUT --model M --shard-index I --shard-count N
  python score_quality.py dover --out OUT --dover-root DOVER --shard-index I --shard-count N
  python score_quality.py report --out OUT
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic  # noqa: E402
from UNIV_adaptor.scripts.data.score_controlled_factor_requirements import pick, shard  # noqa: E402
from UNIV_adaptor.scripts.data.score_dynamic import (  # noqa: E402
    FRAME_COUNT, Judge, SINGLE, decode, video_clip)

PLAN_SCHEMA = "univ_quality_plan_v1"
PROTOCOL = "quality_v1"  # bump when frames, prompts or aspects change
FULL = "FULL_NATIVE_50"
FIELDS = ("source", "stem", "group_id", "prompt_id", "family_id", "split", "motion_level", "detail_level",
          "base_seed", "case_id")

# id, judge, English question (asked of the model), Chinese question (shown to annotators), annotator hint
QUALITY = (
    ("sharp", "frames", "Is the image sharp and clear, without blur or loss of fine detail?",
     "画面清晰吗？没有模糊，细节没有糊掉。", "画质：可暂停逐帧看，注意纹理、边缘、文字、毛发等细节"),
    ("artifact", "frames", "Is the image free of visual artifacts such as distorted or melted shapes, smeared "
     "textures, noise, blocky patches or broken structures?",
     "画面没有伪影吗？（形状扭曲或融化、纹理涂抹、噪点、色块、结构破碎）", "画质：可暂停逐帧看，人脸、手、物体边缘最容易出问题"),
    ("aesthetic", "frames", "Is the image aesthetically pleasing, with good composition, lighting and colour?",
     "画面好看吗？（构图、光线、色彩）", "画质：凭第一观感判断，不看是否符合文字描述"),
    ("temporal", "video", "Is the video temporally stable, without flicker, jitter or stutter, and without shapes "
     "or textures that warp or boil over time?",
     "视频在时间上稳定吗？没有闪烁、抖动、卡顿，物体形状和纹理不会随时间扭动或沸腾。", "画质：正常速度播放看整体，再用 0.25× 看细节"),
    ("acceptable", "video", "Overall, is the visual quality of this video acceptable, with no obvious defect "
     "that a viewer would notice?",
     "整体看，这个视频的画质达到可用标准吗？没有观众一眼就能看出的明显缺陷。", "画质：总判断，像在审一条要发布的视频"),
)
ASPECTS = tuple(a[0] for a in QUALITY)
PAIRED = ("{A} and {B} come from two videos generated from the same description.\nQuestion: {q}\n"
          "Is {X} at least as good as {Y} in this respect? Answer with only Yes or No.")


# ---------------------------------------------------------------- plan

def plan(args):
    fixed = Path(args.fixed_root)
    value = json.loads((fixed / "plan.json").read_text(encoding="utf-8"))
    videos = [v for v in value["videos"] if v["split"] != "test"]  # the fixed plan already drops test rows
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "plan.json", {"schema": PLAN_SCHEMA, "protocol": PROTOCOL, "fixed_root": str(fixed.resolve()),
                                          "fixed_plan_sha256": sha256_file(fixed / "plan.json"),
                                          "aspects": [list(a) for a in QUALITY], "videos": videos})
    print(f"planned {len(videos)} videos ({len({v['video_sha256'] for v in videos})} unique clips)")


def load_plan(out):
    value = json.loads((Path(out) / "plan.json").read_text(encoding="utf-8"))
    if value.get("schema") != PLAN_SCHEMA or value.get("protocol") != PROTOCOL:
        raise ValueError("unsupported plan; re-plan into a new output root")
    if tuple(tuple(a) for a in value["aspects"]) != QUALITY:
        raise ValueError("quality aspects changed since planning; re-plan into a new output root")
    return value


def full_of(value):
    return {(v["source"], v["group_id"]): v for v in value["videos"] if v["case_id"] == FULL}


def clips(value):
    return sorted({v["video_sha256"]: v for v in value["videos"]}.items())


def verified(v):
    if sha256_file(v["video_path"]) != v["video_sha256"]:
        raise ValueError(f"video changed since the fixed-layer plan: {v['video_path']}")
    return decode(v["video_path"])


# ---------------------------------------------------------------- VLM

def score_single(judge, frames, fps):
    fq = [a for a in QUALITY if a[1] == "frames"]
    vq = [a for a in QUALITY if a[1] == "video"]
    out = {}
    per = [judge.p_yes([["image", SINGLE.format(what="frame", q=a[2])] for a in fq], images=[frames[i]] * len(fq))
           for i in pick(len(frames), FRAME_COUNT)]
    for k, a in enumerate(fq):
        ps = [p[k] for p in per]
        out[a[0]] = {"p_yes": float(np.mean(ps)), "p_frames": [round(p, 5) for p in ps]}
    clip, meta = video_clip(frames, fps)
    for a, p in zip(vq, judge.p_yes([["video", SINGLE.format(what="video", q=a[2])] for a in vq],
                                    videos=[clip] * len(vq), metadata=[meta] * len(vq))):
        out[a[0]] = {"p_yes": float(p)}
    return out


def paired_prompt(kind, q, accel_first):
    """A is shown first; the question always asks whether the accelerated clip is at least as good as FULL."""
    asked, other = ("A", "B") if accel_first else ("B", "A")
    noun = kind.capitalize()
    return [f"{noun} A:", kind, f"{noun} B:", kind,
            PAIRED.format(A=f"{noun} A", B=f"{kind} B", q=q, X=f"{kind} {asked}", Y=f"{kind} {other}")]


def score_pair(judge, accel, full):
    (fa, fps_a), (ff, fps_f) = accel, full
    fq = [a for a in QUALITY if a[1] == "frames"]
    vq = [a for a in QUALITY if a[1] == "video"]
    out = {}
    ff_first, ac_first = [], []
    for i, j in zip(pick(len(fa), FRAME_COUNT), pick(len(ff), FRAME_COUNT)):
        ff_first.append(judge.p_yes([paired_prompt("image", a[2], False) for a in fq], images=[ff[j], fa[i]] * len(fq)))
        ac_first.append(judge.p_yes([paired_prompt("image", a[2], True) for a in fq], images=[fa[i], ff[j]] * len(fq)))
    for k, a in enumerate(fq):
        out[a[0]] = {"p_full_first": float(np.mean([p[k] for p in ff_first])),
                     "p_accel_first": float(np.mean([p[k] for p in ac_first]))}
    (ca, ma), (cf, mf) = video_clip(fa, fps_a), video_clip(ff, fps_f)
    for a in vq:  # two 32-frame videos per call; one aspect at a time bounds memory
        x, = judge.p_yes([paired_prompt("video", a[2], False)], videos=[cf, ca], metadata=[mf, ma])
        y, = judge.p_yes([paired_prompt("video", a[2], True)], videos=[ca, cf], metadata=[ma, mf])
        out[a[0]] = {"p_full_first": float(x), "p_accel_first": float(y)}
    return out


def vlm(args, judge=None):
    value = load_plan(args.out)
    cache = Path(args.out) / "vlm"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [c for c in shard(clips(value), args.shard_index, args.shard_count) if not (cache / f"{c[0]}.json").exists()]
    if not todo:
        return
    judge = judge or Judge(args.model, args.device)
    for i, (sha, v) in enumerate(todo):
        frames, fps = verified(v)
        write_json_atomic(cache / f"{sha}.json", {"video_sha256": sha, "protocol": PROTOCOL, "model": str(args.model),
                                                  "answers": score_single(judge, frames, fps)})
        print(f"[vlm {args.shard_index}/{args.shard_count}] {i + 1}/{len(todo)} {Path(v['video_path']).name}", flush=True)


def pair_jobs(value):
    fulls = full_of(value)
    jobs = {}
    for v in value["videos"]:
        if v["case_id"] != FULL:
            f = fulls[(v["source"], v["group_id"])]
            jobs.setdefault(f"{v['video_sha256']}__{f['video_sha256']}", (v, f))
    return sorted(jobs.items())


def pair(args, judge=None):
    value = load_plan(args.out)
    cache = Path(args.out) / "pair"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [j for j in shard(pair_jobs(value), args.shard_index, args.shard_count) if not (cache / f"{j[0]}.json").exists()]
    if not todo:
        return
    judge = judge or Judge(args.model, args.device)
    for i, (key, (v, f)) in enumerate(todo):
        write_json_atomic(cache / f"{key}.json", {"pair": key, "protocol": PROTOCOL, "model": str(args.model),
                                                   "answers": score_pair(judge, verified(v), verified(f))})
        print(f"[pair {args.shard_index}/{args.shard_count}] {i + 1}/{len(todo)} {Path(v['video_path']).name}", flush=True)


# ---------------------------------------------------------------- DOVER (optional)

class Dover:
    """DOVER (Wu et al., ICCV 2023) from its official repo: raw aesthetic and technical scores plus the
    repo's sigmoid-rescaled values (0-1). Needs DOVER_ROOT with dover.yml and pretrained_weights/DOVER.pth."""

    def __init__(self, root, device="cuda"):
        import torch
        import yaml

        sys.path.insert(0, str(root))
        from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
        from dover.models import DOVER

        self.torch, self.decompose = torch, spatial_temporal_view_decomposition
        opt = yaml.safe_load((Path(root) / "dover.yml").read_text(encoding="utf-8"))
        self.dopt = opt["data"]["val-l1080p"]["args"]
        self.model = DOVER(**opt["model"]["args"]).to(device).eval()
        weights = Path(root) / opt["test_load_path"]
        self.model.load_state_dict(torch.load(weights, map_location=device))
        self.samplers = {}
        for stype, s in self.dopt["sample_types"].items():
            if "t_frag" not in s:
                self.samplers[stype] = UnifiedFrameSampler(s["clip_len"], s["num_clips"], s["frame_interval"])
            else:
                self.samplers[stype] = UnifiedFrameSampler(s["clip_len"] // s["t_frag"], s["t_frag"],
                                                           s["frame_interval"], s["num_clips"])
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        self.device = device

    def __call__(self, path):
        torch = self.torch
        views, _ = self.decompose(str(path), self.dopt["sample_types"], self.samplers)
        for k, v in views.items():
            n = self.dopt["sample_types"][k].get("num_clips", 1)
            views[k] = (((v.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)
                        .reshape(v.shape[0], n, -1, *v.shape[2:]).transpose(0, 1).to(self.device))
        with torch.inference_mode():
            aes, tech = [float(r.mean()) for r in self.model(views)]
        t, a = (tech - 0.1107) / 0.07355, (aes + 0.08285) / 0.03774  # the repo's rescaling (fuse_results)
        sig = lambda x: float(1 / (1 + np.exp(-x)))  # noqa: E731
        return {"aesthetic_raw": aes, "technical_raw": tech, "aesthetic": sig(a), "technical": sig(t),
                "overall": sig(0.6104 * t + 0.3896 * a)}


def dover(args, model=None):
    value = load_plan(args.out)
    cache = Path(args.out) / "dover"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [c for c in shard(clips(value), args.shard_index, args.shard_count) if not (cache / f"{c[0]}.json").exists()]
    if not todo:
        return
    model = model or Dover(args.dover_root, args.device)
    for i, (sha, v) in enumerate(todo):
        if sha256_file(v["video_path"]) != sha:
            raise ValueError(f"video changed since the fixed-layer plan: {v['video_path']}")
        write_json_atomic(cache / f"{sha}.json", {"video_sha256": sha, "protocol": PROTOCOL, **model(v["video_path"])})
        print(f"[dover {args.shard_index}/{args.shard_count}] {i + 1}/{len(todo)} {Path(v['video_path']).name}", flush=True)


# ---------------------------------------------------------------- report

def read(path):
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    if r["protocol"] != PROTOCOL:
        raise ValueError(f"stale cache entry: {path}")
    return r


def report(args):
    value = load_plan(args.out)
    out = Path(args.out)
    fixed_cache = Path(value["fixed_root"]) / "cache"
    single, dov, sharp = {}, {}, {}
    for sha, _ in clips(value):
        single[sha] = read(out / "vlm" / f"{sha}.json")["answers"]
        if (out / "dover" / f"{sha}.json").exists():
            dov[sha] = read(out / "dover" / f"{sha}.json")
        sharp[sha] = json.loads((fixed_cache / f"{sha}.json").read_text(encoding="utf-8"))["sharpness"]
    fulls = full_of(value)
    rows = []
    for v in value["videos"]:
        f = fulls[(v["source"], v["group_id"])]
        s, fs = single[v["video_sha256"]], single[f["video_sha256"]]
        row = {**{k: v[k] for k in FIELDS}}
        for a in ASPECTS:
            row[f"p_{a}"] = round(s[a]["p_yes"], 5)
            row[f"full_p_{a}"] = round(fs[a]["p_yes"], 5)
        row["acceptable"] = int(s["acceptable"]["p_yes"] >= 0.5)
        row["full_acceptable"] = int(fs["acceptable"]["p_yes"] >= 0.5)
        row["d_sharpness"] = round(sharp[v["video_sha256"]] - sharp[f["video_sha256"]], 5)
        if v["case_id"] != FULL:
            pans = read(out / "pair" / f"{v['video_sha256']}__{f['video_sha256']}.json")["answers"]
            for a in ASPECTS:  # kept: at least as good as FULL in both presentation orders
                row[f"pair_{a}"] = round((pans[a]["p_full_first"] + pans[a]["p_accel_first"]) / 2, 5)
                row[f"kept_{a}"] = int(pans[a]["p_full_first"] >= 0.5 and pans[a]["p_accel_first"] >= 0.5)
            row["quality_kept"] = float(np.mean([row[f"kept_{a}"] for a in ASPECTS]))
        if v["video_sha256"] in dov and f["video_sha256"] in dov:
            for k in ("aesthetic", "technical", "overall"):
                row[f"dover_{k}"] = round(dov[v["video_sha256"]][k], 5)
                row[f"d_dover_{k}"] = round(dov[v["video_sha256"]][k] - dov[f["video_sha256"]][k], 5)
        rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "quality_by_video.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out / 'quality_by_video.csv'} ({len(rows)} rows)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        summarize(rows)


def summarize(rows):
    def m(sub, key):
        xs = [r[key] for r in sub if r.get(key, "") != ""]
        return float(np.mean(xs)) if xs else float("nan")

    cells = sorted({(r["motion_level"], r["detail_level"]) for r in rows})
    full_ok = m([r for r in rows if r["case_id"] == FULL], "acceptable")
    print(f"\nFULL acceptable rate (single-video VLM): {full_ok:.2f}")
    print("\nper case: share kept vs FULL (pairwise, both orders) per aspect; acceptable rate; DOVER and sharpness deltas")
    head = "case".ljust(18) + "".join(a[:6].rjust(8) for a in ASPECTS) + "all".rjust(7) + \
        "".join(f"{x[0]}{y[0]}".rjust(6) for x, y in cells) + "accept".rjust(8) + "dDOVER".rjust(8) + "dSharp".rjust(8)
    print(head)
    for case in sorted({r["case_id"] for r in rows} - {FULL}):
        sub = [r for r in rows if r["case_id"] == case]
        by_cell = [m([r for r in sub if (r["motion_level"], r["detail_level"]) == c], "quality_kept") for c in cells]
        print(case.ljust(18) + "".join(f"{m(sub, f'kept_{a}'):8.2f}" for a in ASPECTS) + f"{m(sub, 'quality_kept'):7.2f}"
              + "".join(f"{x:6.2f}" for x in by_cell) + f"{m(sub, 'acceptable'):8.2f}"
              + f"{m(sub, 'd_dover_overall'):8.3f}" + f"{m(sub, 'd_sharpness'):8.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--fixed-root", required=True, help="fixed-layer output root (plan.json + cache/)")
    p.add_argument("--out", required=True)
    for mode in ("vlm", "pair", "dover"):
        p = sub.add_parser(mode)
        p.add_argument("--out", required=True)
        if mode == "dover":
            p.add_argument("--dover-root", required=True)
        else:
            p.add_argument("--model", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--shard-index", type=int, default=0)
        p.add_argument("--shard-count", type=int, default=1)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"plan": plan, "vlm": vlm, "pair": pair, "dover": dover, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
