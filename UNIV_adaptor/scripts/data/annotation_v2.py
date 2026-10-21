"""M3 human annotation of the v2 dynamic layer (REQUIREMENTS_V2_SPEC.md): sample, package, analyze.

A group is one accelerated clip next to its same-seed FULL clip (sides randomized, method hidden) with up to
four of the prompt's checklist questions. For each question an annotator answers: does the left clip satisfy
it, does the right clip, and which satisfies it better. The answers test the judges:
- single-video VLM P(Yes) and the flow/luma thresholds against human single labels;
- pairwise VLM retention against human retention (accelerated satisfies the requirement and is not worse);
- the judges' method-level conclusions against the same conclusions recomputed from human labels.

The sample is drawn from finished dynamic-layer results before anyone annotates, with a fixed seed, and is
committed with this analysis code (pre-registration). Locked test prompts never enter: the dynamic plan
already excludes them and sampling checks again.

Usage (see run_univ_annotation.sh):
  python annotation_v2.py sample --dyn-root OUT/dynamic_v1_all --out OUT/annotation_v1
  python annotation_v2.py kit --out OUT/annotation_v1 --annotator A    # then B
  python annotation_v2.py analyze --out OUT/annotation_v1 --labels labels_A.json labels_B.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import shutil
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic  # noqa: E402
from UNIV_adaptor.scripts.data import requirements_v2 as rv2  # noqa: E402

SCHEMA = "univ_annotation_v2_sample_v1"
SEED = 20261021
FULL = "FULL_NATIVE_50"
FAMILY = {
    "SCALING10": "cache_step", "STEP20": "cache_step", "STEP25": "cache_step", "TEA008": "cache_step",
    "TEA015": "cache_step", "F41_DUP": "f41", "F41_RIFE": "f41", "LR_FLASH": "lowres", "LR_BICUBIC": "lowres",
    "LR41_FLASH_RIFE": "lowres", "DVG_D050": "dvg", "DVG_D070": "dvg", "RCM1": "rcm", "RCM2": "rcm", "RCM4": "rcm",
}
CELLS = ("hh", "hl", "lh", "ll")
PER_CELL = 10          # groups per family x cell: 5 x 4 x 10 = 200
QUESTIONS_PER_GROUP = 4
OVERLAP = 50           # groups both annotators label (inter-annotator agreement)
ANSWERS = {"yes", "no", "unsure"}
COMPARE = {"left", "same", "right", "unsure"}


def flt(x):
    return float(x) if x not in ("", None) else None


# ---------------------------------------------------------------- sample

def load_dynamic(dyn_root):
    dyn = Path(dyn_root)
    plan = json.loads((dyn / "plan.json").read_text(encoding="utf-8"))
    with (dyn / "dynamic_by_question.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return plan, rows


def candidate_groups(plan, rows):
    """One candidate per accelerated clip, with the judge results the question picker stratifies on."""
    full_clip = {(v["source"], v["group_id"]): v for v in plan["videos"] if v["case_id"] == FULL}
    by_video = collections.defaultdict(dict)
    for r in rows:
        by_video[(r["source"], r["stem"])][r["qid"]] = r
    out = []
    for v in plan["videos"]:
        if v["case_id"] == FULL:
            continue
        if v["split"] == "test":
            raise ValueError(f"locked test clip in the dynamic plan: {v['stem']}")
        f = full_clip[(v["source"], v["group_id"])]
        out.append({"video": v, "full": f, "answers": by_video[(v["source"], v["stem"])],
                    "full_answers": by_video[(f["source"], f["stem"])]})
    return out


def pick_questions(record, cand, rng, k=QUESTIONS_PER_GROUP):
    """Up to k questions: one where single and pairwise VLM verdicts disagree, one video question, one flow
    question, then random others. Returns [(question, reason)]."""
    qs = list(record["questions"])
    rng.shuffle(qs)
    a, fa = cand["answers"], cand["full_answers"]

    def disagree(q):
        r = a.get(q["id"], {})
        pf, pa = flt(r.get("p_full_first")), flt(r.get("p_accel_first"))
        return (rv2.JUDGE[q["type"]].startswith("vlm") and pf is not None and pa is not None
                and int(r["pass_raw"]) == 1 and int(fa[q["id"]]["pass_raw"]) == 1 and min(pf, pa) < 0.5)

    chosen = []
    for reason, test in (("single_pair_disagree", disagree),
                         ("video", lambda q: rv2.JUDGE[q["type"]] == "vlm_video"),
                         ("flow", lambda q: rv2.JUDGE[q["type"]] == "flow")):
        q = next((q for q in qs if test(q) and q not in [c for c, _ in chosen]), None)
        if q is not None:
            chosen.append((q, reason))
    for q in qs:
        if len(chosen) >= k:
            break
        if q not in [c for c, _ in chosen]:
            chosen.append((q, "random"))
    order = {q["id"]: i for i, q in enumerate(record["questions"])}
    return sorted(chosen[:k], key=lambda c: order[c[0]["id"]])


def sample(args):
    plan, rows = load_dynamic(args.dyn_root)
    checklist = {r["prompt_id"]: r for r in rv2.load_jsonl(plan["checklist"])}
    prompts = rv2.load_prompts(args.prompts)
    rng = random.Random(SEED)
    pools = collections.defaultdict(list)
    for c in candidate_groups(plan, rows):
        v = c["video"]
        pools[(FAMILY[v["case_id"]], v["motion_level"][0] + v["detail_level"][0])].append(c)
    groups = []
    for family in sorted(set(FAMILY.values())):
        used_prompts = set()
        for cell in CELLS:
            pool = sorted(pools[(family, cell)], key=lambda c: (c["video"]["source"], c["video"]["stem"]))
            rng.shuffle(pool)
            # Spread over prompts: take unused prompts first, then allow repeats if the cell runs short.
            pool.sort(key=lambda c: c["video"]["prompt_id"] in used_prompts)
            picked = []
            for c in pool:
                if len(picked) == PER_CELL:
                    break
                if any(p["video"]["prompt_id"] == c["video"]["prompt_id"] and p["video"]["base_seed"] == c["video"]["base_seed"]
                       for p in picked):
                    continue
                picked.append(c)
                used_prompts.add(c["video"]["prompt_id"])
            if len(picked) < PER_CELL:
                raise ValueError(f"only {len(picked)} candidates for {family}/{cell}")
            groups += picked
    rng.shuffle(groups)
    out_groups = []
    for i, c in enumerate(groups):
        v, f = c["video"], c["full"]
        rec = checklist[v["prompt_id"]]
        out_groups.append({
            "gid": f"g{i + 1:03d}", "source": v["source"], "prompt_id": v["prompt_id"], "base_seed": v["base_seed"],
            "case_id": v["case_id"], "family": FAMILY[v["case_id"]], "cell": v["motion_level"][0] + v["detail_level"][0],
            "prompt": prompts[v["prompt_id"]]["prompt"], "accel_left": rng.random() < 0.5,
            "accel": {k: v[k] for k in ("stem", "video_path", "video_sha256")},
            "full": {k: f[k] for k in ("stem", "video_path", "video_sha256")},
            "questions": [{**{k: q[k] for k in ("id", "type", "question", "region", "expect") if k in q},
                           "judge": rv2.JUDGE[q["type"]], "reason": reason}
                          for q, reason in pick_questions(rec, c, rng)]})
    gids = [g["gid"] for g in out_groups]
    overlap, rest = gids[:OVERLAP], gids[OVERLAP:]
    value = {"schema": SCHEMA, "seed": SEED, "dyn_root": str(Path(args.dyn_root).resolve()),
             "dyn_plan_sha256": sha256_file(Path(args.dyn_root) / "plan.json"),
             "dyn_questions_sha256": sha256_file(Path(args.dyn_root) / "dynamic_by_question.csv"),
             "checklist_sha256": plan["checklist_sha256"],
             "assignment": {"A": overlap + rest[0::2], "B": overlap + rest[1::2], "overlap": overlap},
             "groups": out_groups}
    out = Path(args.out)
    if (out / "sample.json").exists():
        raise ValueError(f"{out / 'sample.json'} exists: a pre-registered sample is never redrawn; use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "sample.json", value)
    describe(value)


def describe(value):
    gs = value["groups"]
    qs = [q for g in gs for q in g["questions"]]
    print(f"{len(gs)} groups, {len(qs)} questions; annotator A {len(value['assignment']['A'])} groups, "
          f"B {len(value['assignment']['B'])}, overlap {len(value['assignment']['overlap'])}")
    print("groups by family:", dict(collections.Counter(g["family"] for g in gs)))
    print("questions by judge:", dict(collections.Counter(q["judge"] for q in qs)))
    print("questions by reason:", dict(collections.Counter(q["reason"] for q in qs)))
    print("distinct prompts:", len({g["prompt_id"] for g in gs}), " accelerated clip on the left:",
          sum(g["accel_left"] for g in gs))


# ---------------------------------------------------------------- kit

HINT = {
    "entity": "画面中是否有这个物体", "count": "数量是否符合", "attribute": "颜色、材质、形状是否符合",
    "detail": "细小结构是否清晰可辨（可暂停、放大看）", "spatial": "位置、构图、视角是否符合",
    "negative": "不该出现的东西确实没有出现", "action": "整段视频里是否做出这个动作",
    "integrity": "运动时形状是否保持完整，没有融化、撕裂、变形", "motion": "运动的幅度和速度是否达到描述",
    "static": "是否保持静止（忽略极轻微的噪点闪动）", "camera": "镜头是否按描述运动或保持锁定",
    "focus": "对焦是否始终不变", "lighting": "光照是否始终不变（无明显闪烁或明暗变化）",
}


def encode(src, dst, ffmpeg):
    """Same encoder and settings for both sides, native size and frame rate, no metadata."""
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-map_metadata", "-1", "-an",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(dst)], check=True)


def kit(args):
    out = Path(args.out)
    value = json.loads((out / "sample.json").read_text(encoding="utf-8"))
    gids = value["assignment"][args.annotator]
    by_gid = {g["gid"]: g for g in value["groups"]}
    order = list(gids)
    random.Random(f"{SEED}-{args.annotator}").shuffle(order)  # each annotator sees their own order
    kdir = out / f"kit_{args.annotator}"
    (kdir / "videos").mkdir(parents=True, exist_ok=True)
    tasks = []
    for n, gid in enumerate(order):
        g = by_gid[gid]
        sides = {"L": g["accel"] if g["accel_left"] else g["full"], "R": g["full"] if g["accel_left"] else g["accel"]}
        for side, clip in sides.items():
            dst = kdir / "videos" / f"{gid}_{side}.mp4"
            if not dst.exists():
                if sha256_file(clip["video_path"]) != clip["video_sha256"]:
                    raise ValueError(f"video changed since the dynamic run: {clip['video_path']}")
                encode(clip["video_path"], dst.with_suffix(".tmp.mp4"), args.ffmpeg)
                dst.with_suffix(".tmp.mp4").replace(dst)
        tasks.append({"gid": gid, "prompt": g["prompt"],
                      "questions": [{"id": q["id"], "type": q["type"], "question": q["question"], "hint": HINT[q["type"]]}
                                    for q in g["questions"]]})
        print(f"[kit {args.annotator}] {n + 1}/{len(order)} {gid}", flush=True)
    payload = {"annotator": args.annotator, "sample_sha256": sha256_file(out / "sample.json"), "tasks": tasks}
    (kdir / "tasks.js").write_text("window.ANNOTATION = " + json.dumps(payload, ensure_ascii=False) + ";\n",
                                   encoding="utf-8")
    shutil.copy(ROOT / "UNIV_adaptor/annotation/annotate.html", kdir / "annotate.html")
    print(f"kit ready: {kdir} ({len(tasks)} groups)")


# ---------------------------------------------------------------- analyze

def kappa(pairs):
    """Cohen's kappa for a list of (a, b) labels."""
    pairs = list(pairs)
    if not pairs:
        return float("nan"), 0
    labels = sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return ((po - pe) / (1 - pe) if pe < 1 else 1.0), n


def human_items(value, labels):
    """(annotator, gid, qid) -> labels mapped from left/right to FULL/accelerated."""
    by_gid = {g["gid"]: g for g in value["groups"]}
    items = {}
    for lab in labels:
        if lab["sample_sha256"] != value["sample_sha256"]:
            raise ValueError(f"labels of annotator {lab['annotator']} were made on a different sample")
        for gid, answers in lab["answers"].items():
            g = by_gid[gid]
            acc, ful = ("left", "right") if g["accel_left"] else ("right", "left")
            for qid, a in answers.items():
                if a.get(acc) not in ANSWERS or a.get(ful) not in ANSWERS or a.get("better") not in COMPARE:
                    continue  # incomplete
                better = {acc: "accel", ful: "full", "same": "same", "unsure": "unsure"}[a["better"]]
                items[(lab["annotator"], gid, qid)] = {"accel": a[acc], "full": a[ful], "better": better}
    return items


def consensus(items):
    """One label per (gid, qid): the single annotator's, or the shared one where both agree."""
    grouped = collections.defaultdict(list)
    for (_, gid, qid), x in items.items():
        grouped[(gid, qid)].append(x)
    out = {}
    for key, xs in grouped.items():
        merged = {}
        for field in ("accel", "full", "better"):
            vals = {x[field] for x in xs}
            merged[field] = vals.pop() if len(vals) == 1 else "unsure"
        out[key] = merged
    return out


def calib_key(q):
    return q["judge"] if q["judge"].startswith("vlm") else f"flow {q['type']} {q.get('expect', '')}".strip()


def calibrate(points, lower_is_yes=False):
    """(AUC, best cut, accuracy at it) for scores predicting a yes/no label; ties count half."""
    pts = [(-v if lower_is_yes else v, y) for v, y in points]
    pos, neg = [v for v, y in pts if y], [v for v, y in pts if not y]
    auc = (sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
           if pos and neg else float("nan"))
    best = (float("nan"), 0.0)
    for cut in sorted({v for v, _ in pts}):
        acc = sum((v >= cut) == y for v, y in pts) / len(pts)
        if acc > best[1]:
            best = (cut, acc)
    cut = -best[0] if lower_is_yes else best[0]
    return auc, cut, best[1]


def human_kept(x):
    """Human retention: the accelerated clip satisfies the requirement and is not worse than FULL."""
    if x["accel"] == "unsure" or x["better"] == "unsure":
        return None
    return x["accel"] == "yes" and x["better"] != "full"


def analyze(args):
    out = Path(args.out)
    value = json.loads((out / "sample.json").read_text(encoding="utf-8"))
    value["sample_sha256"] = sha256_file(out / "sample.json")
    labels = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.labels]
    _, rows = load_dynamic(value["dyn_root"])
    if sha256_file(Path(value["dyn_root"]) / "dynamic_by_question.csv") != value["dyn_questions_sha256"]:
        print("NOTE: dynamic results changed since sampling (e.g. thresholds); judges are compared as they are now")
    judged = {(r["source"], r["stem"], r["qid"]): r for r in rows}
    by_gid = {g["gid"]: g for g in value["groups"]}
    items = human_items(value, labels)
    lines = ["# M3 annotation analysis", "",
             f"Annotators: {', '.join(l['annotator'] for l in labels)}; labelled question items: {len(items)}", ""]

    # 1. Agreement between annotators on the overlap.
    lines += ["## Annotator agreement (overlap groups)", "", "| judge | label | kappa | n |", "| --- | --- | --- | --- |"]
    names = sorted({a for a, _, _ in items})
    if len(names) >= 2:
        for judge in sorted({q["judge"] for g in value["groups"] for q in g["questions"]}):
            for field in ("full", "accel", "better"):
                pairs = []
                for (a, gid, qid), x in items.items():
                    if a != names[0]:
                        continue
                    y = items.get((names[1], gid, qid))
                    q = next(q for q in by_gid[gid]["questions"] if q["id"] == qid)
                    if y and q["judge"] == judge and "unsure" not in (x[field], y[field]):
                        pairs.append((x[field], y[field]))
                k, n = kappa(pairs)
                lines.append(f"| {judge} | {field} | {k:.2f} | {n} |")

    # 2. Judges against the consensus label.
    cons = consensus(items)
    single, kept_single, kept_pair, measured = [], [], [], []
    calib = collections.defaultdict(list)
    by_family = collections.defaultdict(lambda: collections.defaultdict(list))
    for (gid, qid), x in cons.items():
        g = by_gid[gid]
        q = next(q for q in g["questions"] if q["id"] == qid)
        ra = judged[(g["source"], g["accel"]["stem"], qid)]
        rf = judged[(g["source"], g["full"]["stem"], qid)]
        for who, r in (("accel", ra), ("full", rf)):
            if x[who] != "unsure":
                single.append((q["judge"], q["type"], x[who] == "yes", int(r["pass_raw"]) == 1))
                if q["judge"] in ("vlm_frames", "vlm_video", "flow"):
                    calib[calib_key(q)].append((float(r["value"]), x[who] == "yes"))
        hk = human_kept(x)
        if hk is None or x["full"] != "yes":
            continue
        model_single = int(ra["pass_raw"]) == 1
        if q["judge"].startswith("vlm"):
            pf, pa = flt(ra["p_full_first"]), flt(ra["p_accel_first"])
            model_pair = pf is not None and pa is not None and pf >= 0.5 and pa >= 0.5
            kept_single.append((q["judge"], g["family"], hk, model_single))
            kept_pair.append((q["judge"], g["family"], hk, model_pair))
            by_family[g["family"]]["human"].append(hk)
            by_family[g["family"]]["single"].append(model_single)
            by_family[g["family"]]["pair"].append(model_pair)
        else:
            measured.append((q["type"], g["family"], hk, ra["kept"] == "1"))

    def agree(rows_, key):
        lines_ = ["| group | n | accuracy | kappa | human yes | judge yes |", "| --- | --- | --- | --- | --- | --- |"]
        groups = collections.defaultdict(list)
        for r in rows_:
            groups[key(r)].append((r[-2], r[-1]))
        groups["all"] = [(r[-2], r[-1]) for r in rows_]
        for k_, ps in sorted(groups.items(), key=lambda kv: str(kv[0])):
            if not ps:
                continue
            acc = sum(a == b for a, b in ps) / len(ps)
            kp, _ = kappa(ps)
            lines_.append(f"| {k_} | {len(ps)} | {acc:.2f} | {kp:.2f} | {sum(a for a, _ in ps) / len(ps):.2f} | "
                          f"{sum(b for _, b in ps) / len(ps):.2f} |")
        return lines_

    lines += ["", "## Single-video judges vs human single labels (both clips)", ""] + agree(single, lambda r: r[0])
    lines += ["", "By type:", ""] + agree(single, lambda r: r[1])
    lines += ["", "## Retention: which VLM decision matches human retention?",
              "Items where humans say FULL satisfies the requirement. Human retention = accelerated satisfies it and "
              "is not worse.", "", "### Single-video VLM (accelerated P(Yes) >= 0.5)", ""] + agree(kept_single, lambda r: r[1])
    lines += ["", "### Pairwise VLM (both orders >= 0.5)", ""] + agree(kept_pair, lambda r: r[1])
    lines += ["", "### Measured judges (provisional thresholds)", ""] + agree(measured, lambda r: r[0])
    lines += ["", "## Calibration of single-video scores against human labels",
              "AUC of the raw score (P(Yes), or optical flow in px; for still questions lower flow means yes) and the "
              "cut with the highest accuracy. The current cut is 0.5 for P(Yes) and `PROVISIONAL` for flow.", "",
              "| judge / expect | n | human yes | AUC | best cut | accuracy at best cut |", "| --- | --- | --- | --- | --- | --- |"]
    for key_, pts in sorted(calib.items()):
        auc, cut, acc = calibrate(pts, lower_is_yes=key_.endswith("still"))
        lines.append(f"| {key_} | {len(pts)} | {sum(y for _, y in pts) / len(pts):.2f} | {auc:.2f} | {cut:.3g} | {acc:.2f} |")
    lines += ["", "## Method-level retention on VLM questions: human vs judges", "",
              "| family | n | human | single VLM | pairwise VLM |", "| --- | --- | --- | --- | --- |"]
    for fam, d in sorted(by_family.items()):
        n = len(d["human"])
        lines.append(f"| {fam} | {n} | {sum(d['human']) / n:.2f} | {sum(d['single']) / n:.2f} | {sum(d['pair']) / n:.2f} |")
    report = "\n".join(lines) + "\n"
    (out / "analysis.md").write_text(report, encoding="utf-8")
    print(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--dyn-root", required=True)
    p.add_argument("--prompts", nargs="+", default=[str(ROOT / "prompts/univ_controlled_factor_v1.jsonl"),
                                                     str(ROOT / "prompts/univ_controlled_factor_dev2.jsonl")])
    p.add_argument("--out", required=True)
    p = sub.add_parser("kit")
    p.add_argument("--out", required=True)
    p.add_argument("--annotator", required=True, choices=("A", "B"))
    p.add_argument("--ffmpeg", default="ffmpeg")
    p = sub.add_parser("analyze")
    p.add_argument("--out", required=True)
    p.add_argument("--labels", nargs="+", required=True, help="exported label files, one per annotator")
    args = parser.parse_args(argv)
    {"sample": sample, "kit": kit, "analyze": analyze}[args.mode](args)


if __name__ == "__main__":
    main()
