"""Requirement-aware scoring of controlled-factor development videos (no generation).

Three VBench-independent measurements, each compared to the matched FULL clip:

- signals: detail statistics at native resolution (Laplacian variance, spectral
  high-frequency share, Canny edge density) and RAFT motion (mean flow
  magnitude, occlusion-masked warp error). These are distribution statistics,
  NOT pixel fidelity, so FULL/accelerated layouts need not align.
- vlm: a frozen per-prompt checklist (prompts/*_requirements.jsonl) of atomic
  yes/no requirements typed subject/detail/motion/static/plain. Each is scored
  as P(Yes) from the first-token logits of a VLM. subject/motion/static use the
  clip; detail/plain use three native-resolution frames, averaged.
- report: merges with VBench quality_by_video.csv and reruns the
  prompt-versus-seed variance analysis on the new metrics.

Locked test rows are excluded at plan time and never decoded.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic  # noqa: E402

PLAN_SCHEMA = "univ_controlled_factor_requirements_plan_v1"
SIGNAL_FRAMES = 9
FLOW_SIZE = (360, 624)  # H, W; both divisible by 8 for RAFT
VIDEO_FRAMES = 32
VIDEO_SIZE = (252, 448)  # H, W; multiples of 28 for Qwen2.5-VL patches
DETAIL_FRAMES = (0.0, 0.5, 1.0)
VISUAL_OF_TYPE = {"subject": "video", "motion": "video", "static": "video", "detail": "frames", "plain": "frames"}
QUESTION = "Look carefully at this {kind} and answer the question with only Yes or No.\nQuestion: {question}"


# ---------------------------------------------------------------- plan

PUBLISHED_FULL = "FULL50"
VBENCH_DIMS = ("subject_consistency", "background_consistency", "motion_smoothness", "aesthetic_quality",
               "imaging_quality", "dynamic_degree", "overall_consistency")


def published_rows(root):
    """Finalized published-pipeline platform (controlled-factor prompts) in the requirement/variance schema.

    Group ids follow the DVG study (main_p###_s<seed>) so both sources join on prompt and seed;
    FULL50 is renamed to the shared FULL case id. Also converts the platform's VBench CSV.
    """
    from UNIV_adaptor.scripts.data.analyze_controlled_factor_variance import FULL as FULL_CASE

    root = Path(root)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if manifest["plan_sha256"] != frozen["plan_sha256"] or manifest["partial_exploratory"]:
        raise ValueError("requirement scoring needs the complete finalized platform manifest")
    prompts = {p["id"]: p for p in frozen["config"]["prompts"]}
    rows, by_job = [], {}
    for record in manifest["records"]:
        job = record["job"]
        p = prompts[job["prompt_id"]]
        if p["origin"] not in ("controlled_factor_v1", "controlled_factor_dev2"):
            raise ValueError(f"not a controlled-factor prompt: {p['id']}")
        pid = int(p["id"][2:])
        arm = job["arm"]["id"]
        row = {"stem": job["id"], "group_id": f"main_p{pid:03d}_s{job['seed']}", "prompt_id": pid,
               "family_id": p["family_id"], "split": p["split"], "motion_level": p["motion"],
               "detail_level": p["detail"], "base_seed": job["seed"],
               "case_id": FULL_CASE if arm == PUBLISHED_FULL else arm,
               "video_path": record["video_path"], "video_sha256": record["video_sha256"]}
        rows.append(row)
        by_job[job["id"]] = row
    quality_path = root / "metrics" / "quality_by_video.csv"
    with quality_path.open(encoding="utf-8-sig", newline="") as handle:
        scored = list(csv.DictReader(handle))
    if {r["observation_id"] for r in scored} != set(by_job):
        raise ValueError("platform VBench CSV does not cover the finalized manifest; run its score mode first")
    quality = []
    for r in scored:
        v = by_job[r["observation_id"]]
        if r["video_sha256"] != v["video_sha256"]:
            raise ValueError(f"VBench row identity mismatch: {r['observation_id']}")
        quality.append({"observation_id": v["stem"], "set": "main",
                        **{k: v[k] for k in ("group_id", "prompt_id", "family_id", "split", "motion_level",
                                             "detail_level", "base_seed", "case_id")},
                        "pipeline_seconds": r["pipeline_seconds"], "vbench5": r["vbench5"],
                        **{k: r[k] for k in VBENCH_DIMS}})
    identity = {"published_plan_sha256": frozen["plan_sha256"], "published_quality_sha256": sha256_file(quality_path)}
    return rows, identity, quality


def plan(args):
    quality = None
    if getattr(args, "published", None):
        rows, identity, quality = published_rows(args.published)
    elif args.dvg_study:
        from UNIV_adaptor.scripts.data import dvg_wan21_study as dvg

        rows = [r for r in dvg.receipts(args.dvg_study, dvg.load_plan(args.dvg_study)) if r["set"] == "main"]
        identity = {"dvg_study_plan_sha256": sha256_file(Path(args.dvg_study) / "plan.json")}
    else:
        from UNIV_adaptor.scripts.data.score_controlled_factor_dataset import collect

        rows, identity = collect(Path(args.root))
    keep = [r for r in rows if r["split"] != "test"]
    checklist = load_checklist(args.checklist)
    missing = {r["prompt_id"] for r in keep} - set(checklist)
    if missing:
        raise ValueError(f"checklist lacks development prompts: {sorted(missing)}")
    videos = [{k: r[k] for k in ("stem", "group_id", "prompt_id", "family_id", "split", "motion_level",
                                 "detail_level", "base_seed", "case_id", "video_path", "video_sha256")}
              for r in keep]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if quality is not None:  # converted platform VBench rows, for `report --quality-csv`
        with (out / "quality_by_video.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(quality[0]))
            writer.writeheader()
            writer.writerows(q for q in quality if q["split"] != "test")
    body = {
        "schema": PLAN_SCHEMA, "source": identity,
        "checklist_path": str(Path(args.checklist).resolve()), "checklist_sha256": sha256_file(args.checklist),
        "excluded_test_videos": len(rows) - len(keep), "videos": videos,
    }
    write_json_atomic(out / "plan.json", body)
    print(f"planned {len(videos)} development videos; excluded {body['excluded_test_videos']} locked test videos")


def load_plan(out):
    value = json.loads((Path(out) / "plan.json").read_text(encoding="utf-8"))
    if value.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported plan schema")
    if sha256_file(value["checklist_path"]) != value["checklist_sha256"]:
        raise ValueError("checklist changed after planning; use a new output root")
    return value


def load_checklist(path):
    out = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            out[int(item["prompt_id"])] = item["requirements"]
    return out


def shard(videos, index, count):
    return [v for i, v in enumerate(videos) if i % count == index]


def read_frames(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < 2:
        raise ValueError(f"could not decode video: {path}")
    return np.stack(frames)


def pick(n, count):
    return np.unique(np.linspace(0, n - 1, count).round().astype(int))


def verified_frames(video):
    if sha256_file(video["video_path"]) != video["video_sha256"]:
        raise ValueError(f"video hash mismatch: {video['video_path']}")
    return read_frames(video["video_path"])


# ---------------------------------------------------------------- signals

def detail_stats(frames):
    import cv2

    lap, hf, edges = [], [], []
    for frame in frames[pick(len(frames), SIGNAL_FRAMES)]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        lap.append(float(cv2.Laplacian(gray.astype(np.float32) / 255.0, cv2.CV_32F).var()))
        power = np.abs(np.fft.fft2(gray.astype(np.float32) / 255.0)) ** 2
        fy, fx = np.meshgrid(np.fft.fftfreq(gray.shape[0]), np.fft.fftfreq(gray.shape[1]), indexing="ij")
        radius = np.hypot(fx, fy)
        power[0, 0] = 0.0
        hf.append(float(power[radius > 0.25].sum() / power.sum()))
        edges.append(float((cv2.Canny(gray, 100, 200) > 0).mean()))
    return {"log_laplacian_var": math.log(np.mean(lap)), "hf_share": float(np.mean(hf)),
            "edge_density": float(np.mean(edges))}


def flow_stats(frames, model, device):
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)
    x = F.interpolate(x, size=FLOW_SIZE, mode="bilinear", align_corners=False, antialias=True)
    a, b = x[:-1:2], x[1::2]
    mags, errs, valid_fracs = [], [], []
    h, w = FLOW_SIZE
    gy, gx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    with torch.inference_mode():
        for start in range(0, len(a), 8):
            ia, ib = a[start:start + 8].to(device), b[start:start + 8].to(device)
            fwd = model(ia * 2 - 1, ib * 2 - 1)[-1]
            bwd = model(ib * 2 - 1, ia * 2 - 1)[-1]
            grid = torch.stack(((gx + fwd[:, 0]) / (w - 1) * 2 - 1, (gy + fwd[:, 1]) / (h - 1) * 2 - 1), dim=-1)
            warped_b = F.grid_sample(ib, grid, align_corners=True)
            bwd_w = F.grid_sample(bwd, grid, align_corners=True)
            cycle = (fwd + bwd_w).pow(2).sum(1)
            valid = cycle < 0.01 * (fwd.pow(2).sum(1) + bwd_w.pow(2).sum(1)) + 0.5
            err = (warped_b - ia).abs().mean(1)
            mags.append(fwd.pow(2).sum(1).sqrt().mean((1, 2)).cpu())
            errs.append(((err * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)).cpu())
            valid_fracs.append(valid.float().mean((1, 2)).cpu())
    mag = float(torch.cat(mags).mean())
    return {"log_flow_magnitude": math.log(mag + 1e-3), "warp_error": float(torch.cat(errs).mean()),
            "flow_valid_fraction": float(torch.cat(valid_fracs).mean())}


def signals(args):
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    value = load_plan(args.out)
    device = torch.device(args.device)
    model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(device).eval()
    out = Path(args.out) / "signals"
    out.mkdir(parents=True, exist_ok=True)
    todo = [v for v in shard(value["videos"], args.shard_index, args.shard_count) if not (out / f"{v['stem']}.json").exists()]
    for i, video in enumerate(todo):
        frames = verified_frames(video)
        result = {"stem": video["stem"], "video_sha256": video["video_sha256"], "frames": int(len(frames)),
                  "height": int(frames.shape[1]), "width": int(frames.shape[2]),
                  **detail_stats(frames), **flow_stats(frames, model, device)}
        write_json_atomic(out / f"{video['stem']}.json", result)
        print(f"[signals {args.shard_index}/{args.shard_count}] {i + 1}/{len(todo)} {video['stem']}", flush=True)


# ---------------------------------------------------------------- vlm

class YesNoScorer:
    def __init__(self, model_id, device):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        tok = self.processor.tokenizer
        self.yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")]
        self.no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")]
        self.device = device

    def p_yes(self, kind, visual, question):
        """P(Yes) renormalized over the Yes/No first tokens."""
        from PIL import Image

        content = [{"type": kind}, {"type": "text", "text": QUESTION.format(kind=kind, question=question)}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True)
        if kind == "image":
            inputs = self.processor(text=[text], images=[Image.fromarray(visual)], return_tensors="pt")
        else:
            inputs = self.processor(text=[text], videos=[visual], return_tensors="pt")
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        yes = self.torch.logsumexp(logits[self.yes], 0)
        no = self.torch.logsumexp(logits[self.no], 0)
        return float(self.torch.sigmoid(yes - no))


def clip_for_vlm(frames):
    import cv2

    h, w = VIDEO_SIZE
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames[pick(len(frames), VIDEO_FRAMES)]])


def vlm(args):
    value = load_plan(args.out)
    checklist = load_checklist(value["checklist_path"])
    scorer = YesNoScorer(args.model, args.device)
    out = Path(args.out) / "vlm"
    out.mkdir(parents=True, exist_ok=True)
    todo = [v for v in shard(value["videos"], args.shard_index, args.shard_count) if not (out / f"{v['stem']}.json").exists()]
    for i, video in enumerate(todo):
        frames = verified_frames(video)
        clip = clip_for_vlm(frames)
        stills = [frames[int(round(q * (len(frames) - 1)))] for q in DETAIL_FRAMES]
        answers = []
        for req in checklist[int(video["prompt_id"])]:
            if VISUAL_OF_TYPE[req["type"]] == "video":
                p = scorer.p_yes("video", clip, req["question"])
            else:
                p = float(np.mean([scorer.p_yes("image", still, req["question"]) for still in stills]))
            answers.append({**req, "p_yes": p})
        write_json_atomic(out / f"{video['stem']}.json", {
            "stem": video["stem"], "video_sha256": video["video_sha256"], "model": args.model,
            "checklist_sha256": value["checklist_sha256"], "answers": answers})
        print(f"[vlm {args.shard_index}/{args.shard_count}] {i + 1}/{len(todo)} {video['stem']}", flush=True)


def check(args):
    """Score ONE planned FULL video end to end; prints values for eyeballing."""
    value = load_plan(args.out)
    video = next(v for v in value["videos"] if v["case_id"] == "FULL_NATIVE_50")
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    frames = verified_frames(video)
    model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(args.device).eval()
    print(video["stem"], frames.shape, detail_stats(frames), flow_stats(frames, model, torch.device(args.device)))
    del model
    scorer = YesNoScorer(args.model, args.device)
    clip = clip_for_vlm(frames)
    for req in load_checklist(value["checklist_path"])[int(video["prompt_id"])]:
        visual = clip if VISUAL_OF_TYPE[req["type"]] == "video" else frames[0]
        kind = "video" if VISUAL_OF_TYPE[req["type"]] == "video" else "image"
        print(f"  {req['type']:8s} p_yes={scorer.p_yes(kind, visual, req['question']):.3f}  {req['question']}")
    print("sanity (should be low):", scorer.p_yes("image", frames[0], "Is there a giraffe wearing a top hat?"))


# ---------------------------------------------------------------- report

REQ_TYPES = ("subject", "detail", "motion", "static", "plain")
SIGNAL_METRICS = ("log_laplacian_var", "hf_share", "edge_density", "log_flow_magnitude", "warp_error")
REPORT_METRICS = ("vbench5", "imaging_quality", "req_all", *SIGNAL_METRICS)


def report(args):
    from UNIV_adaptor.scripts.data.analyze_controlled_factor_variance import FULL, run

    value = load_plan(args.out)
    out = Path(args.out)
    sig, vlm_ans = {}, {}
    for video in value["videos"]:
        stem = video["stem"]
        s = json.loads((out / "signals" / f"{stem}.json").read_text(encoding="utf-8"))
        q = json.loads((out / "vlm" / f"{stem}.json").read_text(encoding="utf-8"))
        if s["video_sha256"] != video["video_sha256"] or q["video_sha256"] != video["video_sha256"]:
            raise ValueError(f"receipt identity mismatch: {stem}")
        if q["checklist_sha256"] != value["checklist_sha256"]:
            raise ValueError(f"checklist mismatch: {stem}")
        sig[stem], vlm_ans[stem] = s, q["answers"]

    merged = []
    with Path(args.quality_csv).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                continue
            stem = row["observation_id"]
            answers = vlm_ans[stem]
            row.update({k: sig[stem][k] for k in SIGNAL_METRICS})
            row["req_all"] = float(np.mean([a["p_yes"] for a in answers]))
            for t in REQ_TYPES:
                vals = [a["p_yes"] for a in answers if a["type"] == t]
                row[f"req_{t}"] = float(np.mean(vals)) if vals else ""
            merged.append(row)
    if len(merged) != len(value["videos"]):
        raise ValueError("VBench CSV and plan cover different development videos")
    analysis = out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    merged_csv = analysis / "quality_with_requirements.csv"
    with merged_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)

    # requirement-level retention: among requirements FULL satisfies, how many does the action keep?
    by_group = defaultdict(dict)
    for video in value["videos"]:
        by_group[video["group_id"]][video["case_id"]] = video
    counts = defaultdict(lambda: [0, 0, 0.0])
    for cases in by_group.values():
        full = {a["id"]: a for a in vlm_ans[cases[FULL]["stem"]]}
        for case_id, video in cases.items():
            if case_id == FULL:
                continue
            for a in vlm_ans[video["stem"]]:
                ref = full[a["id"]]
                key = (case_id, a["type"], video["motion_level"], video["detail_level"])
                counts[key][2] += a["p_yes"] - ref["p_yes"]
                if ref["p_yes"] > 0.5:
                    counts[key][0] += 1
                    counts[key][1] += a["p_yes"] > 0.5
    n_all = defaultdict(int)
    for cases in by_group.values():
        for case_id, video in cases.items():
            if case_id != FULL:
                for a in vlm_ans[video["stem"]]:
                    n_all[(case_id, a["type"], video["motion_level"], video["detail_level"])] += 1
    retention = [{"action": k[0], "type": k[1], "motion": k[2], "detail": k[3], "instances": n_all[k],
                  "full_satisfied": v[0], "retained": v[1],
                  "retention_rate": v[1] / v[0] if v[0] else float("nan"), "mean_delta_p_yes": v[2] / n_all[k]}
                 for k, v in sorted(counts.items())]
    with (analysis / "requirement_retention.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(retention[0]))
        writer.writeheader()
        writer.writerows(retention)

    _, summary = run(merged_csv, analysis, REPORT_METRICS)
    print(f"{summary['prompts']} prompts / {summary['families']} families")
    for metric in REPORT_METRICS:
        print(f"\n### {metric}\n{summary['headline'][metric]}")
    print("\n### requirement retention (FULL-satisfied -> kept), by action/type")
    agg = defaultdict(lambda: [0, 0])
    for r in retention:
        agg[r["action"], r["type"], r["detail"]][0] += r["full_satisfied"]
        agg[r["action"], r["type"], r["detail"]][1] += r["retained"]
    for (action, t, detail), (n, k) in sorted(agg.items()):
        if n:
            print(f"{action:26s} {t:8s} detail={detail:4s} {k}/{n} = {k / n:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", help="controlled-factor OUT_ROOT")
    source.add_argument("--dvg-study", help="dvg_wan21_study output root (main set only)")
    source.add_argument("--published", help="finalized + scored published_wan21 platform root")
    p.add_argument("--checklist", required=True)
    p.add_argument("--out", required=True)
    for name in ("signals", "vlm", "check"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--shard-index", type=int, default=0)
        p.add_argument("--shard-count", type=int, default=1)
        p.add_argument("--model", default=os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.add_argument("--quality-csv", required=True)
    args = parser.parse_args()
    {"plan": plan, "signals": signals, "vlm": vlm, "check": check, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
