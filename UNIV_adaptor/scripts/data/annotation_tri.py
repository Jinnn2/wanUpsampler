"""Human check of the three-way quality judge (QUALITY_V3.md, UNIV_adaptor/annotation/HUMAN_TRI_V1.md).

People see two clips of one prompt side by side (sides randomized, methods hidden) and answer, for an overall
question and three aspects, which looks better or that they are about the same. The pairs come from the
quality_tri seed-triangle pilot, whose judge answers are already known; the pair list is drawn here with a fixed
seed and committed before anyone labels (pre-registration).

Sets (40 pairs, plus 6 repeats with sides swapped for self-consistency):
- seed100 (12): FULL seed 100 against seed 42 or 2024, high motion. Is seed 100 really worse?
- null (4): FULL seed 2024 against seed 42, high motion.
- svg2 (12): SVG2 against FULL seed 42, high motion. Is SVG2 really worse?
- scaling (6): SCALING10 against FULL; expected about the same.
- bad (6): RCM1 or LR_BICUBIC against FULL; expected clearly worse (annotator check).

Usage:
  python annotation_tri.py select --tri-csv results/attention/quality_tri_seeds/tri_by_pair.csv \
      --out prompts/annotation_tri_v1                                   # local, once; commit pairs.json
  python annotation_tri.py kit --pairs prompts/annotation_tri_v1/pairs.json \
      --plan outputs/quality_tri_seeds/plan.json --out outputs/annotation_tri_v1 --annotator A    # server
  python annotation_tri.py analyze --pairs prompts/annotation_tri_v1/pairs.json \
      --labels labels_tri-A.json [labels_tri-B.json] --out outputs/annotation_tri_v1
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic  # noqa: E402
from UNIV_adaptor.scripts.data import requirements_v2 as rv2  # noqa: E402
from UNIV_adaptor.scripts.data.annotation_v2 import encode  # noqa: E402

SCHEMA = "univ_annotation_tri_v1"
SEED = 20261112
ASPECTS = ("sharp", "artifact", "aesthetic", "temporal", "acceptable")   # the judge's five aspects
NAMED = (9, 25, 147, 155, 163)   # high-motion prompts where the judge and the single-clip scores disagree most
SET_SIZE = {"seed100": 12, "null": 4, "svg2": 12, "scaling": 6, "bad": 6}
REPEATS = {"seed100": 4, "svg2": 2}
MIN_REPEAT_GAP = 10   # tasks between a pair and its swapped repeat
QUESTIONS = (
    ("overall", "Which clip looks better overall (sharpness, artifacts, temporal stability), or are they about "
     "the same?", "整体画质哪边更好？综合看清晰度、伪影和时间上的稳定性；差别不明显就选差不多。",
     "画质总判断：只看画面本身，不管是否符合文字描述"),
    ("sharp", "Which clip is sharper and clearer, without blur or loss of fine detail?",
     "哪边更清晰？没有模糊，细节没有糊掉。", "可暂停逐帧看，注意纹理、边缘、文字、毛发等细节"),
    ("artifact", "Which clip has fewer visual artifacts (distorted or melted shapes, smeared textures, noise, "
     "blocky patches, broken structures)?", "哪边伪影更少？（形状扭曲或融化、纹理涂抹、噪点、色块、结构破碎）",
     "可暂停逐帧看，人脸、手、物体边缘最容易出问题"),
    ("temporal", "Which clip is more temporally stable, without flicker, jitter, stutter, or shapes and textures "
     "that warp or boil over time?", "哪边在时间上更稳定？没有闪烁、抖动、卡顿，形状和纹理不随时间扭动或沸腾。",
     "正常速度播放看整体，再用 0.25× 看细节"),
)
QIDS = tuple(q[0] for q in QUESTIONS)
HUMAN = {"left", "same", "right"}
PROMPTS = (ROOT / "prompts/univ_controlled_factor_v1.jsonl", ROOT / "prompts/univ_controlled_factor_dev2.jsonl")
# Pilot prompts not in prompts/annotation_v1/zh_translation.json.
ZH_EXTRA = {
    9: "固定镜头下，一只超大的素面银色怀表像钟摆一样在空荡的深色背景前快速摆动。怀表表面光滑无标记，每秒划过一大段弧线，"
       "在恒定光线下始终清晰可见。画面中没有小工具或装饰纹理。",
    27: "固定的全景舞台视角：二十名舞者一动不动，身穿精细刺绣的深色服装、精致的蕾丝袖子、串珠头饰和许多细丝带。"
        "每位演员都保持定格姿势，密集的装饰细节在稳定的舞台灯光下保持清晰。",
    104: "一枚超大的素面黑色国际象棋王静止地立在空白的白色平面中央。镜头锁定，棋子侧面光滑无装饰，画面中没有其他东西。",
    114: "一片光滑的大橙色枫叶被狂风吹过素灰色的水泥地面，左右快速翻滚、滑动。画面中没有其他东西，镜头不动。",
    145: "锁定的俯视镜头拍摄一台复古弹珠机的台面：精细的插画图案、小小的弹射柱、细金属轨道、小星形翻转片和密集排列的灯片。"
         "机器关着，没有任何东西在动。",
    147: "几颗银色钢珠在细节丰富的复古弹珠台上快速弹来弹去，挡板啪地击打，弹射柱闪烁，钢珠滚过插画图案、细轨道和小灯片。"
         "俯视固定镜头。",
    155: "杂乱的工作间里，一辆架在支架上的自行车车轮飞速旋转，链条在齿轮上疾驰；细辐条变得模糊，细小的链节飞转，"
         "挂着的工具在固定镜头前摇晃。",
    163: "数百本书脊烫金的旧书从高高的图书馆书架上快速滚落，书页翻飞，书本相互碰撞、遮挡，掠过黄铜栏杆和华丽的木雕落下。"
         "固定镜头。",
}


def seeds_of(arm):
    """(x seed, y seed) of a tri_by_pair row; arms are seed 42 against FULL seed 42."""
    if arm == "FULL_SEED":
        return 100, 42
    if arm.startswith("FULL_S"):
        sx, sy = arm[len("FULL_S"):].split("_vs_S")
        return int(sx), int(sy)
    return 42, 42


def judge_of(r, aspects=ASPECTS):
    """The judge's answer on a tri_by_pair row: mean P(x better), P(y better), P(Same) over the aspects, argmax."""
    p = {k: sum(float(r[f"p{k}_{a}"]) for a in aspects) / len(aspects) for k in ("x", "y", "s")}
    out = max(("x", "y", "same"), key=lambda o: p["s" if o == "same" else o])
    return {"px": round(p["x"], 4), "py": round(p["y"], 4), "ps": round(p["s"], 4),
            "net": round(p["x"] - p["y"], 4), "out": out}


# ---------------------------------------------------------------- select

def select(args):
    with Path(args.tri_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by = {(r["arm"], int(r["prompt_id"])): r for r in rows}
    motion = {int(r["prompt_id"]): r["motion_level"] for r in rows}
    high = sorted(p for p, m in motion.items() if m == "high")
    low = sorted(p for p, m in motion.items() if m == "low")
    if any(64 <= p <= 79 for p in motion):
        raise ValueError("locked test prompts (64-79) in the pilot csv")
    if not set(NAMED) <= set(high):
        raise ValueError("a named prompt is not a high-motion pilot prompt")
    rng = random.Random(SEED)
    picks = []   # (set, row)

    # seed100: the five named prompts and 7 more high-motion prompts; the 4 left over give the null pairs.
    others = [p for p in high if p not in NAMED]
    rng.shuffle(others)
    s100 = sorted(list(NAMED) + others[:SET_SIZE["seed100"] - len(NAMED)])
    for i, p in enumerate(p for p in s100 if p not in NAMED):
        # Partner seed: alternately the one the judge separates from seed 100 more and less strongly.
        cand = [by[("FULL_SEED", p)], by[("FULL_S2024_vs_S100", p)]]
        cand.sort(key=lambda r: -abs(float(r["net"])))
        picks.append(("seed100", cand[i % 2]))
    # The named prompts' disagreement is seed 100 against seed 42.
    picks += [("seed100", by[("FULL_SEED", p)]) for p in NAMED]
    picks.sort(key=lambda sr: int(sr[1]["prompt_id"]))
    for p in sorted(others[SET_SIZE["seed100"] - len(NAMED):])[:SET_SIZE["null"]]:
        picks.append(("null", by[("FULL_S2024_vs_S42", p)]))
    for p in sorted(rng.sample(high, SET_SIZE["svg2"])):
        picks.append(("svg2", by[("SVG2", p)]))
    for p in sorted(rng.sample(high, 3) + rng.sample(low, 3)):
        picks.append(("scaling", by[("SCALING10", p)]))
    bad_high, bad_low = rng.sample(high, 3), rng.sample(low, 3)
    for arm, ps in (("RCM1", bad_high[:2] + bad_low[:1]), ("LR_BICUBIC", bad_high[2:] + bad_low[1:])):
        picks += [("bad", by[(arm, p)]) for p in sorted(ps)]
    if collections.Counter(s for s, _ in picks) != collections.Counter(SET_SIZE):
        raise AssertionError("set sizes")

    prompts = rv2.load_prompts([str(p) for p in PROMPTS])
    zh = json.loads((ROOT / "prompts/annotation_v1/zh_translation.json").read_text(encoding="utf-8"))["prompts"]
    pairs = []
    for i, (s, r) in enumerate(picks):
        p = int(r["prompt_id"])
        sx, sy = seeds_of(r["arm"])
        text = prompts[p]["prompt"]
        pairs.append({"pair": f"p{i + 1:02d}", "set": s, "kind": r["kind"], "arm": r["arm"], "prompt_id": p,
                      "motion_level": r["motion_level"], "x_seed": sx, "y_seed": sy,
                      "x_case": r["arm"] if r["kind"] == "arm" else "FULL_NATIVE_50", "y_case": "FULL_NATIVE_50",
                      "prompt": text, "prompt_zh": zh.get(text) or ZH_EXTRA[p],
                      "judge": {"overall": judge_of(r), "overall3": judge_of(r, ("sharp", "artifact", "temporal")),
                                **{a: judge_of(r, (a,)) for a in ASPECTS}}})
    for q in pairs:
        q["x_left"] = rng.random() < 0.5
    tasks = [{"pair": q["pair"], "x_left": q["x_left"], "repeat": False} for q in pairs]
    for s, n in REPEATS.items():
        for q in rng.sample([q for q in pairs if q["set"] == s], n):
            tasks.append({"pair": q["pair"], "x_left": not q["x_left"], "repeat": True})
    rng.shuffle(tasks)
    for i, t in enumerate(tasks):
        t["tid"] = f"t{i + 1:02d}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "pairs.json", {"schema": SCHEMA, "seed": SEED, "tri_csv": str(args.tri_csv),
                                           "tri_csv_sha256": sha256_file(args.tri_csv), "pairs": pairs,
                                           "tasks": tasks})
    print(f"{len(pairs)} pairs, {len(tasks)} tasks: {dict(collections.Counter(q['set'] for q in pairs))}")


def load_pairs(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("not a tri annotation pair list")
    value["sha256"] = sha256_file(path)
    return value


def order(value, annotator):
    """The annotator's task order: shuffled, with each repeat at least MIN_REPEAT_GAP tasks from its original."""
    rng = random.Random(f"{SEED}-{annotator}")
    tasks = list(value["tasks"])
    for _ in range(10000):
        rng.shuffle(tasks)
        at = collections.defaultdict(list)
        for i, t in enumerate(tasks):
            at[t["pair"]].append(i)
        if all(len(v) < 2 or abs(v[0] - v[1]) > MIN_REPEAT_GAP for v in at.values()):
            return tasks
    raise RuntimeError("no order with repeats far enough apart")


# ---------------------------------------------------------------- kit

def kit(args):
    value = load_pairs(args.pairs)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    clips = {(r["arm"], int(r["x"]["prompt_id"])): r for r in plan["pairs"] if r["kind"] != "identity"}
    pairs = {q["pair"]: q for q in value["pairs"]}
    kdir = Path(args.out) / f"kit_tri_{args.annotator}"
    (kdir / "videos").mkdir(parents=True, exist_ok=True)
    tasks, jobs = [], []
    for t in order(value, args.annotator):
        q = pairs[t["pair"]]
        r = clips[(q["arm"], q["prompt_id"])]
        for side, clip in (("L", r["x"] if t["x_left"] else r["y"]), ("R", r["y"] if t["x_left"] else r["x"])):
            dst = kdir / "videos" / f"{t['tid']}_{side}.mp4"
            if not dst.exists():
                jobs.append((clip, dst))
        tasks.append({"gid": t["tid"], "prompt": q["prompt"], "prompt_zh": q["prompt_zh"],
                      "questions": [{"id": i, "type": "tri", "question": en, "question_zh": zh, "hint": hint}
                                    for i, en, zh, hint in QUESTIONS]})

    def one(job):
        clip, dst = job
        if sha256_file(clip["video_path"]) != clip["video_sha256"]:
            raise ValueError(f"video changed since the judge run: {clip['video_path']}")
        tmp = dst.with_name(dst.stem + ".tmp.mp4")
        encode(clip["video_path"], tmp, args.ffmpeg)
        tmp.replace(dst)

    print(f"[tri kit {args.annotator}] encoding {len(jobs)} clips", flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(one, jobs))
    payload = {"annotator": args.annotator, "kind": "tri", "sample_sha256": value["sha256"], "tasks": tasks}
    (kdir / "tasks_tri.js").write_text("window.ANNOTATION = " + json.dumps(payload, ensure_ascii=False) + ";\n",
                                       encoding="utf-8")
    html = (ROOT / "UNIV_adaptor/annotation/annotate.html").read_text(encoding="utf-8")
    if '<script src="tasks.js"></script>' not in html:
        raise ValueError("annotate.html no longer loads tasks.js the expected way")
    (kdir / "annotate_tri.html").write_text(html.replace('<script src="tasks.js"></script>',
                                                         '<script src="tasks_tri.js"></script>'), encoding="utf-8")
    shutil.copy(ROOT / "UNIV_adaptor/annotation/TRI_README_zh.txt", kdir / "标注说明.txt")
    print(f"tri kit ready: {kdir} ({len(tasks)} tasks)")


# ---------------------------------------------------------------- analyze

def kappa3(pairs):
    """Cohen's kappa over the three outcomes."""
    if not pairs:
        return float("nan")
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in ("x", "y", "same")) / n / n
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def sign_test(k, n):
    """One-sided exact binomial P(X >= k), X ~ Bin(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else float("nan")


def fmt(v):
    return "nan" if isinstance(v, float) and math.isnan(v) else (f"{v:.2f}" if isinstance(v, float) else str(v))


def analyze(args):
    value = load_pairs(args.pairs)
    pairs = {q["pair"]: q for q in value["pairs"]}
    tasks = {t["tid"]: t for t in value["tasks"]}
    human = {}    # (annotator, pair, qid, repeat) -> "x" | "y" | "same"
    names = []
    for path in args.labels:
        lab = json.loads(Path(path).read_text(encoding="utf-8"))
        if lab.get("kind") != "tri" or lab.get("sample_sha256") != value["sha256"]:
            raise ValueError(f"{path} is not from a kit of this pair list")
        names.append(lab["annotator"])
        for tid, ans in lab["answers"].items():
            t = tasks[tid]
            for qid, a in ans.items():
                b = a.get("better")
                if qid not in QIDS or b not in HUMAN:
                    continue
                side = {"left": t["x_left"], "right": not t["x_left"]}
                human[(lab["annotator"], t["pair"], qid, t["repeat"])] = \
                    "same" if b == "same" else ("x" if side[b] else "y")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["annotator", "pair", "set", "arm", "prompt_id", "motion_level", "x_seed", "y_seed", "question",
              "repeat", "human", "judge_out", "judge_net", "judge_px", "judge_py", "judge_ps"]
    with (out / "human_tri_by_pair.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (who, pid, qid, rep), h in sorted(human.items()):
            q, j = pairs[pid], pairs[pid]["judge"][qid]
            writer.writerow({"annotator": who, "pair": pid, "set": q["set"], "arm": q["arm"],
                             "prompt_id": q["prompt_id"], "motion_level": q["motion_level"], "x_seed": q["x_seed"],
                             "y_seed": q["y_seed"], "question": qid, "repeat": int(rep), "human": h,
                             "judge_out": j["out"], "judge_net": j["net"], "judge_px": j["px"], "judge_py": j["py"],
                             "judge_ps": j["ps"]})

    def h1(who, pid, qid):
        return human.get((who, pid, qid, False))

    lines = ["# Human three-way quality check", "",
             f"Pair list sha256 {value['sha256'][:12]}; annotators: {', '.join(names)}. "
             "x = the arm, or the later FULL seed; y = FULL seed 42, or the earlier seed.", "",
             "## 1. Annotator check (bad set: RCM1 / LR_BICUBIC against FULL, overall question)", "",
             "| annotator | FULL better | same | arm better | passes (FULL better on >= 5 of 6) |",
             "| --- | --- | --- | --- | --- |"]
    valid = []
    for who in names:
        c = collections.Counter(h1(who, p, "overall") for p, q in pairs.items() if q["set"] == "bad")
        ok = c["y"] >= 5
        valid += [who] if ok else []
        lines.append(f"| {who} | {c['y']} | {c['same']} | {c['x']} | {'yes' if ok else 'NO (excluded below)'} |")
    lines += ["", "## 2. Self-consistency (repeats, sides swapped)", "", "| annotator | question | agree | n |",
              "| --- | --- | --- | --- |"]
    for who in names:
        for qid in QIDS:
            ps = [(h1(who, p, qid), human.get((who, p, qid, True))) for p in pairs]
            ps = [(a, b) for a, b in ps if a and b]
            lines.append(f"| {who} | {qid} | {fmt(sum(a == b for a, b in ps) / len(ps)) if ps else 'nan'} | {len(ps)} |")
    hard = [p for p, q in pairs.items() if q["set"] in ("seed100", "null", "svg2")]
    if len(valid) >= 2:
        a, b = valid[:2]
        lines += ["", f"## 3. Annotator agreement ({a} vs {b}, first presentation)", "",
                  "| question | pairs | agree | kappa (3-way) | agree, hard sets | kappa, hard sets |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for qid in QIDS:
            cells = []
            for subset in (list(pairs), hard):
                ps = [(h1(a, p, qid), h1(b, p, qid)) for p in subset if h1(a, p, qid) and h1(b, p, qid)]
                cells += [fmt(sum(x == y for x, y in ps) / len(ps)) if ps else "nan", fmt(kappa3(ps))]
            lines.append(f"| {qid} | {len(pairs)} | " + " | ".join(cells) + " |")

    def consensus(pid, qid):
        hs = [h1(w, pid, qid) for w in valid]
        hs = [h for h in hs if h]
        return hs[0] if hs and all(h == hs[0] for h in hs) else None

    raters = [(w, lambda p, q, w=w: h1(w, p, q)) for w in valid]
    if len(valid) >= 2:
        raters.append(("consensus", consensus))
    lines += ["", "## 4. Human vs judge (judge = argmax of mean P over aspects; overall uses all five aspects)", "",
              "Hard sets = seed100, null, svg2 (the primary agreement). Sign agreement: pairs where both the human "
              "and the judge pick a side.", "",
              "| rater | question | sets | n | 3-way agree | kappa | human same | judge same | sign agree (n) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, get in raters:
        for qid in QIDS + ("overall3",):
            hq = "overall" if qid == "overall3" else qid
            for label, subset in (("hard", hard), ("all", list(pairs))):
                ps = [(get(p, hq), pairs[p]["judge"][qid]["out"]) for p in subset if get(p, hq)]
                if not ps:
                    continue
                sg = [(a, b) for a, b in ps if "same" not in (a, b)]
                lines.append(f"| {name} | {qid} | {label} | {len(ps)} | {fmt(sum(a == b for a, b in ps) / len(ps))} | "
                             f"{fmt(kappa3(ps))} | {fmt(sum(a == 'same' for a, _ in ps) / len(ps))} | "
                             f"{fmt(sum(b == 'same' for _, b in ps) / len(ps))} | "
                             f"{fmt(sum(a == b for a, b in sg) / len(sg)) if sg else 'nan'} ({len(sg)}) |")

    lines += ["", "## 5. Directional questions (overall question)", "",
              "Seed 100: counted from seed 100's side (worse = the other seed looks better). SVG2 / SCALING10 / bad: "
              "counted from the arm's side. One-sided exact sign test over the pairs where a side is picked.", "",
              "| set | rater | worse | same | better | sign test p (worse) | judge worse / same / better |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for s in ("seed100", "null", "svg2", "scaling", "bad"):
        sub = [p for p, q in pairs.items() if q["set"] == s]

        def worse_better(p, o):
            if o == "same" or o is None:
                return o
            q = pairs[p]
            hundred = "x" if q["x_seed"] == 100 else "y"   # seed100 set: the side that is seed 100
            mine = hundred if s == "seed100" else "x"
            return "better" if o == mine else "worse"

        jc = collections.Counter(worse_better(p, pairs[p]["judge"]["overall"]["out"]) for p in sub)
        for name, get in raters:
            c = collections.Counter(worse_better(p, get(p, "overall")) for p in sub)
            n = c["worse"] + c["better"]
            lines.append(f"| {s} | {name} | {c['worse']} | {c['same']} | {c['better']} | "
                         f"{fmt(sign_test(c['worse'], n))} | {jc['worse']} / {jc['same']} / {jc['better']} |")
    lines += ["", "Decision rules: UNIV_adaptor/annotation/HUMAN_TRI_V1.md."]
    report = "\n".join(lines) + "\n"
    (out / "human_tri_analysis.md").write_text(report, encoding="utf-8")
    print(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("select")
    p.add_argument("--tri-csv", required=True, help="tri_by_pair.csv of the quality_tri seed-triangle pilot")
    p.add_argument("--out", required=True)
    p = sub.add_parser("kit")
    p.add_argument("--pairs", required=True)
    p.add_argument("--plan", required=True, help="plan.json of the quality_tri seed-triangle pilot (video paths)")
    p.add_argument("--out", required=True)
    p.add_argument("--annotator", required=True, choices=("A", "B"))
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--jobs", type=int, default=16)
    p = sub.add_parser("analyze")
    p.add_argument("--pairs", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"select": select, "kit": kit, "analyze": analyze}[args.mode](args)


if __name__ == "__main__":
    main()
