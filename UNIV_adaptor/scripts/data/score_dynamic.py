"""Dynamic-layer scoring (REQUIREMENTS_V2_SPEC.md sections 4-5, milestone M2) against the frozen v2 checklist.

Each question is routed by its type (requirements_v2.JUDGE):
- vlm_frames: P(Yes) of a large VLM on 5 native-resolution frames, averaged;
- vlm_video: P(Yes) on 32 uniformly sampled frames (short side 448) given as one video with timestamps;
- flow / sharpness / luma: measured, from the fixed-layer cache (RAFT flow components, sharpness) plus a
  per-frame luma series computed here.
Every VLM question is also asked pairwise against the same-seed FULL clip, in both orders, but only where FULL
itself satisfies it. Dependencies gate: a question counts as unsatisfied when a question it depends on is.

The flow/luma/sharpness thresholds below are provisional; M3 calibrates them against human labels.

Usage (see run_univ_dynamic.sh):
  python score_dynamic.py plan --fixed-root OUT/fixed_metrics_v1 --checklist prompts/univ_requirements_v2_frozen.jsonl \
      --prompts P1.jsonl P2.jsonl --out OUT [--prompt-ids ...]
  python score_dynamic.py vlm  --out OUT --model M --shard-index I --shard-count N
  python score_dynamic.py pair --out OUT --model M --shard-index I --shard-count N
  python score_dynamic.py report --out OUT
"""
from __future__ import annotations

import argparse
import csv
import collections
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic  # noqa: E402
from UNIV_adaptor.scripts.data.score_controlled_factor_requirements import pick, shard  # noqa: E402
from UNIV_adaptor.scripts.data import requirements_v2 as rv2  # noqa: E402

PLAN_SCHEMA = "univ_dynamic_plan_v1"
PROTOCOL = "dynamic_v1"  # bump when frames, prompts or the decision rule of VLM calls change
FULL = "FULL_NATIVE_50"
FRAME_COUNT = 5
VIDEO_FRAMES = 32
VIDEO_SIZE = (448, 768)  # H, W; multiples of 32 (Qwen3-VL patch 16 x merge 2), aspect close to 832x480
PAIR_GATE = 0.5  # FULL P(Yes) at or above this: the requirement is asked pairwise

SINGLE = "Look carefully at this {what} from a generated video and answer with only Yes or No.\nQuestion: {q}"
PAIRED = ("{A} and {B} come from two videos generated from the same description.\nRequirement: {q}\n"
          "Does {X} satisfy this requirement at least as well as {Y}? Answer with only Yes or No.")

# Measured judges. Flow in pixels per frame pair at the fixed layer's FLOW_SIZE (360x624); luma on 0-255.
FLOW_KEY = {"global": "flow_mag", "subject": "flow_local", "background": "flow_camera"}
PROVISIONAL = {
    "still_px": 0.5,           # static / locked camera: region flow below this
    "moving_px": 0.5,          # motion "moving" / moving camera: region flow above this
    "fast_px": 2.0,            # motion "fast"
    "still_ratio": 2.0,        # paired: at most this times FULL's flow (with the fixed layer's 0.05 px floor)
    "moving_ratio": 0.5,       # paired: at least this fraction of FULL's flow
    "focus_std": 0.15,         # focus: temporal std of log Laplacian sharpness
    "focus_margin": 0.10,      # paired: at most FULL + margin
    "luma_range": 8.0,         # lighting: max-min of smoothed per-frame luma
    "luma_flicker": 1.0,       # lighting: mean second difference of per-frame luma
    "luma_margin": 4.0, "flicker_margin": 0.5,
}
EPS = 0.05


# ---------------------------------------------------------------- plan

def plan(args):
    fixed = Path(args.fixed_root)
    fixed_plan = json.loads((fixed / "plan.json").read_text(encoding="utf-8"))
    checklist = {r["prompt_id"]: r for r in rv2.load_jsonl(args.checklist)}
    prompts = rv2.load_prompts(args.prompts)
    for pid, rec in checklist.items():
        errs = rv2.validate_record(rec, prompts.get(pid))
        if errs:
            raise ValueError("checklist is not valid under the current rules: " + "; ".join(errs[:3]))
    keep = set(args.prompt_ids) if args.prompt_ids else set(checklist)
    videos = [v for v in fixed_plan["videos"] if v["prompt_id"] in keep and v["prompt_id"] in checklist
              and v["split"] != "test"]
    groups = collections.defaultdict(set)
    for v in videos:
        groups[(v["source"], v["group_id"])].add(v["case_id"])
    missing = [g for g, cases in groups.items() if FULL not in cases]
    if missing:
        raise ValueError(f"{len(missing)} groups have no FULL clip, e.g. {missing[0]}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "plan.json", {
        "schema": PLAN_SCHEMA, "protocol": PROTOCOL, "fixed_root": str(fixed.resolve()),
        "fixed_plan_sha256": sha256_file(fixed / "plan.json"), "checklist": str(Path(args.checklist).resolve()),
        "checklist_sha256": sha256_file(args.checklist), "prompt_ids": sorted(keep & set(checklist)),
        "videos": videos})
    print(f"planned {len(videos)} videos ({len({v['video_sha256'] for v in videos})} unique clips), "
          f"{len(keep & set(checklist))} prompts, {len(groups)} source/seed groups")


def load_plan(out):
    value = json.loads((Path(out) / "plan.json").read_text(encoding="utf-8"))
    if value.get("schema") != PLAN_SCHEMA or value.get("protocol") != PROTOCOL:
        raise ValueError("unsupported plan; re-plan into a new output root")
    if sha256_file(value["checklist"]) != value["checklist_sha256"]:
        raise ValueError("checklist changed since planning")
    value["checklist_by_prompt"] = {r["prompt_id"]: r for r in rv2.load_jsonl(value["checklist"])}
    return value


def full_of(value):
    return {(v["source"], v["group_id"]): v for v in value["videos"] if v["case_id"] == FULL}


# ---------------------------------------------------------------- video access

def decode(path):
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


def video_clip(frames, fps):
    """32 uniformly sampled frames at VIDEO_SIZE and the metadata Qwen3-VL needs to timestamp them."""
    import cv2

    idx = pick(len(frames), VIDEO_FRAMES)
    h, w = VIDEO_SIZE
    clip = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames[idx]])
    return clip, {"total_num_frames": int(len(frames)), "fps": fps, "frames_indices": [int(i) for i in idx],
                  "height": h, "width": w}


def luma_series(frames):
    import cv2

    return [round(float(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).mean()), 3) for f in frames]


# ---------------------------------------------------------------- VLM judge

class Judge:
    """P(Yes) from the first answer token of a Qwen3-VL-style model; batches several prompts per call."""

    def __init__(self, model_id, device="cuda"):
        import torch
        import transformers

        self.torch = torch
        self.processor = transformers.AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"  # the last position is then the next-token position for every row
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device).eval()
        self.yes = sorted({self.tokenizer.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes")})
        self.no = sorted({self.tokenizer.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No")})
        h, w = VIDEO_SIZE
        self.image_size = {"shortest_edge": 32 * 32, "longest_edge": 4096 * 4096}  # keep native resolution
        self.video_size = {"shortest_edge": 32 * 32, "longest_edge": VIDEO_FRAMES * h * w}

    def _text(self, parts):
        content = [{"type": p} if p in ("image", "video") else {"type": "text", "text": p} for p in parts]
        return self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                  add_generation_prompt=True)

    def p_yes(self, prompts, images=None, videos=None, metadata=None):
        """prompts: list of part lists (strings, "image", "video"); images/videos in placeholder order."""
        kw = {"text": [self._text(p) for p in prompts], "padding": True, "return_tensors": "pt"}
        if images:
            kw.update(images=images, size=self.image_size)
        if videos:
            # Frames are pre-sampled: no re-sampling at the processor's default 2 fps; timestamps come from
            # the metadata. The per-frame cap (default from transformers 5.22) equals VIDEO_SIZE here.
            kw.update(videos=videos, video_metadata=metadata, do_sample_frames=False, size=self.video_size,
                      cap_pixels_per_frame=True)
        inputs = self.processor(**kw).to(self.model.device)
        with self.torch.inference_mode():
            # Only the next-token position: full-sequence logits of a few video prompts (~5.5k tokens each,
            # 150k vocabulary) alone take over 6 GB.
            logits = self.model(**inputs, logits_to_keep=1).logits[:, -1].float()
        yes = self.torch.logsumexp(logits[:, self.yes], 1)
        no = self.torch.logsumexp(logits[:, self.no], 1)
        return self.torch.sigmoid(yes - no).cpu().tolist()


def p_yes_split(judge, prompts, per_prompt):
    """judge.p_yes over prompts, halving the batch on CUDA OOM (a clip with many video questions batches them all).
    per_prompt: {"images"/"videos"/"metadata": one item list per prompt}. A batch that fits is scored as before."""
    try:
        return judge.p_yes(prompts, **{k: v for k, v in per_prompt.items() if v})
    except Exception as e:  # torch.OutOfMemoryError, without importing torch here
        if type(e).__name__ != "OutOfMemoryError" or len(prompts) == 1:
            raise
    torch = getattr(judge, "torch", None)
    if torch is not None:
        torch.cuda.empty_cache()
    h = len(prompts) // 2
    return [*p_yes_split(judge, prompts[:h], {k: v[:h] if v else v for k, v in per_prompt.items()}),
            *p_yes_split(judge, prompts[h:], {k: v[h:] if v else v for k, v in per_prompt.items()})]


def vlm_questions(record):
    return [q for q in record["questions"] if rv2.JUDGE[q["type"]].startswith("vlm")]


def score_single(judge, frames, fps, questions):
    """{qid: {"p_yes", "p_frames"?}} for one video."""
    out = {}
    fq = [q for q in questions if rv2.JUDGE[q["type"]] == "vlm_frames"]
    vq = [q for q in questions if rv2.JUDGE[q["type"]] == "vlm_video"]
    if fq:
        per = []
        for i in pick(len(frames), FRAME_COUNT):
            prompts = [["image", SINGLE.format(what="frame", q=q["question"])] for q in fq]
            per.append(p_yes_split(judge, prompts, {"images": [frames[i]] * len(fq)}))
        for k, q in enumerate(fq):
            ps = [p[k] for p in per]
            out[q["id"]] = {"p_yes": float(np.mean(ps)), "p_frames": [round(p, 5) for p in ps]}
    if vq:
        clip, meta = video_clip(frames, fps)
        prompts = [["video", SINGLE.format(what="video", q=q["question"])] for q in vq]
        for q, p in zip(vq, p_yes_split(judge, prompts, {"videos": [clip] * len(vq), "metadata": [meta] * len(vq)})):
            out[q["id"]] = {"p_yes": float(p)}
    return out


def paired_prompt(kind, q, accel_first):
    """A is shown first. The question always asks about the accelerated clip: B when FULL is A, else A."""
    noun = kind  # "image" or "video"
    asked, other = ("A", "B") if accel_first else ("B", "A")
    return [f"{noun.capitalize()} A:", kind, f"{noun.capitalize()} B:", kind,
            PAIRED.format(A=f"{noun.capitalize()} A", B=f"{noun} B", q=q["question"],
                          X=f"{noun} {asked}", Y=f"{noun} {other}")]


def score_pair(judge, accel, full, questions):
    """Both orders. p_full_first: FULL shown as A, asked whether B (accelerated) is at least as good;
    p_accel_first: accelerated shown as A, asked whether A is at least as good as B (FULL)."""
    (fa, fps_a), (ff, fps_f) = accel, full
    out = {}
    fq = [q for q in questions if rv2.JUDGE[q["type"]] == "vlm_frames"]
    vq = [q for q in questions if rv2.JUDGE[q["type"]] == "vlm_video"]
    if fq:
        ia, iff = pick(len(fa), FRAME_COUNT), pick(len(ff), FRAME_COUNT)
        ff_first, ac_first = [], []
        for i, j in zip(ia, iff):
            ff_first.append(judge.p_yes([paired_prompt("image", q, False) for q in fq], images=[ff[j], fa[i]] * len(fq)))
            ac_first.append(judge.p_yes([paired_prompt("image", q, True) for q in fq], images=[fa[i], ff[j]] * len(fq)))
        for k, q in enumerate(fq):
            out[q["id"]] = {"p_full_first": float(np.mean([p[k] for p in ff_first])),
                            "p_accel_first": float(np.mean([p[k] for p in ac_first]))}
    if vq:
        (ca, ma), (cf, mf) = video_clip(fa, fps_a), video_clip(ff, fps_f)
        for q in vq:  # two 32-frame videos per call already; one question at a time bounds memory
            x, = judge.p_yes([paired_prompt("video", q, False)], videos=[cf, ca], metadata=[mf, ma])
            y, = judge.p_yes([paired_prompt("video", q, True)], videos=[ca, cf], metadata=[ma, mf])
            out[q["id"]] = {"p_full_first": float(x), "p_accel_first": float(y)}
    return out


def vlm(args, judge=None):
    value = load_plan(args.out)
    cache = Path(args.out) / "vlm"
    cache.mkdir(parents=True, exist_ok=True)
    clips = sorted({v["video_sha256"]: v for v in value["videos"]}.items())
    todo = [c for c in shard(clips, args.shard_index, args.shard_count) if not (cache / f"{c[0]}.json").exists()]
    if not todo:
        return
    judge = judge or Judge(args.model, args.device)
    for i, (sha, v) in enumerate(todo):
        frames, fps = decode(v["video_path"])
        record = value["checklist_by_prompt"][v["prompt_id"]]
        result = {"video_sha256": sha, "protocol": PROTOCOL, "checklist_sha256": value["checklist_sha256"],
                  "model": str(args.model), "fps": fps, "frames": int(len(frames)), "luma_series": luma_series(frames),
                  "answers": score_single(judge, frames, fps, vlm_questions(record))}
        write_json_atomic(cache / f"{sha}.json", result)
        print(f"[vlm {args.shard_index}/{args.shard_count}] {i + 1}/{len(todo)} {Path(v['video_path']).name}", flush=True)


def load_cached(path, value):
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    if r["protocol"] != PROTOCOL or r["checklist_sha256"] != value["checklist_sha256"]:
        raise ValueError(f"stale cache entry: {path}")
    return r


def pair_jobs(value):
    """(accel video, FULL video, questions FULL satisfies) for every accelerated clip, deduplicated by sha pair."""
    cache = Path(value["out"]) / "vlm"
    fulls = full_of(value)
    jobs = {}
    for v in value["videos"]:
        if v["case_id"] == FULL:
            continue
        f = fulls[(v["source"], v["group_id"])]
        key = f"{v['video_sha256']}__{f['video_sha256']}"
        if key in jobs:
            continue
        answers = load_cached(cache / f"{f['video_sha256']}.json", value)["answers"]
        qs = [q for q in vlm_questions(value["checklist_by_prompt"][v["prompt_id"]])
              if answers[q["id"]]["p_yes"] >= PAIR_GATE]
        jobs[key] = (v, f, qs)
    return sorted(jobs.items())


def pair(args, judge=None):
    value = load_plan(args.out)
    value["out"] = args.out
    cache = Path(args.out) / "pair"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [j for j in shard(pair_jobs(value), args.shard_index, args.shard_count)
            if not (cache / f"{j[0]}.json").exists()]
    if not todo:
        return
    judge = judge or Judge(args.model, args.device)
    for i, (key, (v, f, qs)) in enumerate(todo):
        answers = score_pair(judge, decode(v["video_path"]), decode(f["video_path"]), qs) if qs else {}
        write_json_atomic(cache / f"{key}.json", {"pair": key, "protocol": PROTOCOL, "model": str(args.model),
                                                   "checklist_sha256": value["checklist_sha256"], "answers": answers})
        print(f"[pair {args.shard_index}/{args.shard_count}] {i + 1}/{len(todo)} {Path(v['video_path']).name} "
              f"({len(qs)} questions)", flush=True)


# ---------------------------------------------------------------- decisions

def lighting_stats(luma):
    x = np.asarray(luma, dtype=np.float64)
    smooth = np.convolve(x, np.ones(5) / 5, mode="valid") if len(x) >= 5 else x
    return {"luma_range": float(smooth.max() - smooth.min()),
            "luma_flicker": float(np.abs(x[1:-1] - 0.5 * (x[:-2] + x[2:])).mean()) if len(x) >= 3 else 0.0}


def measure(q, sig):
    """The measured value a question is decided on, from one clip's signals."""
    t = q["type"]
    if t in ("motion", "static", "camera"):
        return sig[FLOW_KEY[q["region"]]]
    if t == "focus":
        return sig["sharpness_temporal_std"]
    if t == "lighting":
        return lighting_stats(sig["luma_series"])
    raise ValueError(f"not a measured type: {t}")


def measured_pass(q, x, th=PROVISIONAL):
    t = q["type"]
    if t == "lighting":
        return x["luma_range"] < th["luma_range"] and x["luma_flicker"] < th["luma_flicker"]
    if t == "focus":
        return x < th["focus_std"]
    if q.get("expect") == "still":
        return x < th["still_px"]
    return x > (th["fast_px"] if q.get("expect") == "fast" else th["moving_px"])


def measured_kept(q, x, full, th=PROVISIONAL):
    """Paired decision against the same-seed FULL value: not more motion where stillness is required, not
    less where motion is required, no more lighting change or focus drift than FULL plus a margin."""
    t = q["type"]
    if t == "lighting":
        return (x["luma_range"] <= full["luma_range"] + th["luma_margin"]
                and x["luma_flicker"] <= full["luma_flicker"] + th["flicker_margin"])
    if t == "focus":
        return x <= full + th["focus_margin"]
    ratio = (x + EPS) / (full + EPS)
    return ratio <= th["still_ratio"] if q.get("expect") == "still" else ratio >= th["moving_ratio"]


def gate(questions, raw):
    """Final verdicts: a question holds only if it holds raw and every question it depends on holds."""
    final = {}
    for q in questions:  # ids run in order and depends_on only names earlier questions
        final[q["id"]] = bool(raw[q["id"]]) and all(final[d] for d in q.get("depends_on", []))
    return final


def type_balanced(questions, verdict, among=None):
    """Mean over types of the within-type mean; `among` restricts to a subset of question ids."""
    by_type = collections.defaultdict(list)
    for q in questions:
        if among is None or q["id"] in among:
            by_type[q["type"]].append(float(verdict[q["id"]]))
    return float(np.mean([np.mean(v) for v in by_type.values()])) if by_type else float("nan")


# ---------------------------------------------------------------- report

FIELDS = ("source", "stem", "group_id", "prompt_id", "family_id", "split", "motion_level", "detail_level",
          "base_seed", "case_id")


def report(args):
    value = load_plan(args.out)
    out = Path(args.out)
    fixed_cache = Path(value["fixed_root"]) / "cache"
    fulls = full_of(value)
    sig, vlm_ans = {}, {}
    for sha in {v["video_sha256"] for v in value["videos"]}:
        cached = load_cached(out / "vlm" / f"{sha}.json", value)
        sig[sha] = {**json.loads((fixed_cache / f"{sha}.json").read_text(encoding="utf-8")),
                    "luma_series": cached["luma_series"]}
        vlm_ans[sha] = cached["answers"]

    def single(v):
        record = value["checklist_by_prompt"][v["prompt_id"]]
        raw, detail = {}, {}
        for q in record["questions"]:
            if rv2.JUDGE[q["type"]].startswith("vlm"):
                p = vlm_ans[v["video_sha256"]][q["id"]]["p_yes"]
                raw[q["id"]], detail[q["id"]] = p >= 0.5, p
            else:
                x = measure(q, sig[v["video_sha256"]])
                raw[q["id"]], detail[q["id"]] = measured_pass(q, x), x
        return record["questions"], raw, gate(record["questions"], raw), detail

    q_rows, v_rows = [], []
    for v in value["videos"]:
        questions, raw, final, detail = single(v)
        f = fulls[(v["source"], v["group_id"])]
        _, _, f_final, f_detail = single(f)
        kept_raw, pair_p = {}, {}
        if v["case_id"] != FULL:
            pans = json.loads((out / "pair" / f"{v['video_sha256']}__{f['video_sha256']}.json").read_text(encoding="utf-8"))["answers"]
            for q in questions:
                if rv2.JUDGE[q["type"]].startswith("vlm"):
                    a = pans.get(q["id"])
                    pair_p[q["id"]] = a
                    kept_raw[q["id"]] = bool(a) and a["p_full_first"] >= 0.5 and a["p_accel_first"] >= 0.5
                else:
                    kept_raw[q["id"]] = measured_kept(q, detail[q["id"]], f_detail[q["id"]])
            # Retention is defined on what FULL satisfies; a kept question also needs its prerequisites kept.
            kept = gate(questions, {k: kept_raw[k] and f_final[k] for k in kept_raw})
            among = {k for k, ok in f_final.items() if ok}
        for q in questions:
            x = detail[q["id"]]
            q_rows.append({**{k: v[k] for k in FIELDS}, "qid": q["id"], "type": q["type"], "judge": rv2.JUDGE[q["type"]],
                           "value": json.dumps(x) if isinstance(x, dict) else round(float(x), 5),
                           "pass_raw": int(raw[q["id"]]), "pass": int(final[q["id"]]), "full_pass": int(f_final[q["id"]]),
                           "kept": "" if v["case_id"] == FULL or not f_final[q["id"]] else int(kept[q["id"]]),
                           "p_full_first": (pair_p.get(q["id"]) or {}).get("p_full_first", ""),
                           "p_accel_first": (pair_p.get(q["id"]) or {}).get("p_accel_first", "")})
        row = {**{k: v[k] for k in FIELDS}, "questions": len(questions),
               "satisfaction": type_balanced(questions, final),
               "full_satisfaction": type_balanced(questions, f_final)}
        if v["case_id"] != FULL:
            row["retention"] = type_balanced(questions, kept, among)
            for judge in ("vlm_frames", "vlm_video", "flow", "luma", "sharpness"):
                ids = {q["id"] for q in questions if rv2.JUDGE[q["type"]] == judge} & among
                row[f"retention_{judge}"] = float(np.mean([kept[i] for i in ids])) if ids else float("nan")
        v_rows.append(row)
    for name, rows in (("dynamic_by_question.csv", q_rows), ("dynamic_by_video.csv", v_rows)):
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out / name} ({len(rows)} rows)")
    summarize(v_rows)


def summarize(rows):
    with warnings.catch_warnings():  # a judge with no questions in a cell averages to nan, by design
        warnings.simplefilter("ignore", RuntimeWarning)
        _summarize(rows)


def _summarize(rows):
    cells = sorted({(r["motion_level"], r["detail_level"]) for r in rows})
    print("\nmean retention by case (type-balanced, among requirements FULL satisfies); satisfaction minus FULL's")
    print("case".ljust(18) + "retain".rjust(8) + "d_sat".rjust(8) + "".join(f"{m[0]}{d[0]}".rjust(7) for m, d in cells)
          + "frames".rjust(8) + "video".rjust(7) + "flow".rjust(7))
    for case in sorted({r["case_id"] for r in rows} - {FULL}):
        sub = [r for r in rows if r["case_id"] == case]
        cols = [np.nanmean([r["retention"] for r in sub]),
                np.nanmean([r["satisfaction"] - r["full_satisfaction"] for r in sub])]
        cols += [np.nanmean([r["retention"] for r in sub if (r["motion_level"], r["detail_level"]) == c] or [np.nan])
                 for c in cells]
        cols += [np.nanmean([r[f"retention_{j}"] for r in sub]) for j in ("vlm_frames", "vlm_video", "flow")]
        print(case.ljust(18) + "".join(f"{c:8.3f}" if i < 2 else f"{c:7.3f}" for i, c in enumerate(cols)))
    fsat = [r["full_satisfaction"] for r in rows if r["case_id"] == FULL]
    print(f"\nFULL satisfaction: mean {np.mean(fsat):.3f} over {len(fsat)} clips")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--fixed-root", required=True, help="fixed-layer output root (plan.json + cache/)")
    p.add_argument("--checklist", required=True)
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--prompt-ids", nargs="*", type=int, default=[])
    p.add_argument("--out", required=True)
    for mode in ("vlm", "pair"):
        p = sub.add_parser(mode)
        p.add_argument("--out", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--shard-index", type=int, default=0)
        p.add_argument("--shard-count", type=int, default=1)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"plan": plan, "vlm": vlm, "pair": pair, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
