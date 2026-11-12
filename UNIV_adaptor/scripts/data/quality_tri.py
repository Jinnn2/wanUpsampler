"""Pilot: a three-way pairwise quality question (A better / B better / Same) in place of the binary
"is X at least as good as Y" of score_quality.score_pair.

Why. The binary question has no "equally good" answer. For two different clips of equal quality (FULL at two seeds)
the judge often says No in both directions, so the quality layer charges for "different" (quality_null: seed-null
quality_kept 0.71, 0.51 on high-motion prompts). Here the judge may say Same.

Per aspect the five QUALITY questions of score_quality are reused. Frame aspects use the same 5 frame pairs, video
aspects the same 32-frame clips. Each pair is asked in both presentation orders. The answer is read from the first
token: P(A), P(B), P(Same) renormalised over the three, plus the mass of those tokens in the full softmax (does the
model answer in format at all). Per aspect: p_x, p_y, p_same = means over orders (and frames), mapped back from
A/B to X/Y; net = p_x - p_y; outcome = argmax of (x better, y better, same).

Pairs (seed 42; --prompts per motion level, spread over prompt ids):
  identity  (FULL42, FULL42)    ceiling: should be Same, with no A/B position bias
  seed      (FULL100, FULL42)   null: equally good, different. Same or split, mean net ~ 0 on high and low motion
  arm       (ARM42, FULL42)     sensitivity: RCM1 / LR_BICUBIC clearly worse, SCALING10 ~ FULL

  python quality_tri.py plan   --quality-root Q --out OUT [--arms ...] [--prompts 16]
  python quality_tri.py pair   --out OUT --model M --shard-index I --shard-count N   (one GPU each)
  python quality_tri.py report --out OUT [--old-cache DIR ...]   (old binary answers, for comparison)
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import write_json_atomic  # noqa: E402
from UNIV_adaptor.scripts.data.score_controlled_factor_requirements import pick, shard  # noqa: E402

PLAN_SCHEMA = "univ_quality_tri_v1"
PROTOCOL = "quality_tri_v1"
FULL = "FULL_NATIVE_50"
DEFAULT_ARMS = ("SVG2", "SPARGE50", "SCALING10", "RCM1", "LR_BICUBIC", "F41_DUP")
CHOICES = ("A", "B", "Same")
TOKENS = {"A": ("A", " A"), "B": ("B", " B"), "Same": ("Same", " Same", "same")}
TRI = ("{A} and {B} come from two videos generated from the same description.\n"
       "Aspect to compare: {q}\n"
       "In this respect, which {kind} is better? Answer with exactly one word: A if {kind} A is better, "
       "B if {kind} B is better, or Same if they are about equally good.")


def sq():
    from UNIV_adaptor.scripts.data import score_quality
    return score_quality


def key(x, y):
    return f"{x['video_sha256']}__{y['video_sha256']}"


# ---------------------------------------------------------------- plan

def plan(args):
    quality_root = Path(args.quality_root)
    videos = json.loads((quality_root / "plan.json").read_text(encoding="utf-8"))["videos"]
    full, arm, motion = {}, {}, {}
    for v in videos:  # several sources share a file; one clip per (prompt, seed, case)
        p, s = str(v["prompt_id"]), str(v["base_seed"])
        motion[p] = v["motion_level"]
        if v["case_id"] == FULL:
            full.setdefault((p, s), v)
        elif s == "42" and v["case_id"] in args.arms:
            arm.setdefault((p, v["case_id"]), v)
    seeds = [str(s) for s in args.seeds]
    ok = sorted({p for p, _ in full if all((p, s) in full for s in seeds + ["42"])
                 and all((p, a) in arm for a in args.arms)}, key=int)
    prompts = []
    for m in ("high", "low"):  # evenly spaced over the sorted ids, so v1 and dev2 and most families are covered
        cand = [p for p in ok if motion[p] == m]
        idx = np.unique(np.linspace(0, len(cand) - 1, min(args.prompts, len(cand))).round().astype(int))
        prompts += [cand[i] for i in idx]
    pairs = []
    for p in prompts:
        f42 = full[(p, "42")]
        pairs.append(("identity", "FULL", f42, f42))
        for i, sy in enumerate(seeds):  # X = the later seed in --seeds, Y = the earlier one
            for sx in seeds[i + 1:]:
                pairs.append(("seed", seed_label(sx, sy), full[(p, sx)], full[(p, sy)]))
        pairs += [("arm", a, arm[(p, a)], f42) for a in args.arms]
    rows = [{"kind": k, "arm": a, "key": key(x, y), "x": x, "y": y} for k, a, x, y in pairs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "plan.json", {"schema": PLAN_SCHEMA, "protocol": PROTOCOL, "tri_prompt": TRI,
                                          "quality_root": str(quality_root.resolve()), "arms": list(args.arms),
                                          "prompts": prompts, "pairs": rows})
    print(f"planned {len(rows)} pairs on {len(prompts)} prompts "
          f"({collections.Counter(motion[p] for p in prompts)}) {dict(collections.Counter(r['kind'] for r in rows))}")


def seed_label(sx, sy):
    return "FULL_SEED" if (sx, sy) == ("100", "42") else f"FULL_S{sx}_vs_S{sy}"


def seed_pair(label):
    return ("100", "42") if label == "FULL_SEED" else tuple(label[len("FULL_S"):].split("_vs_S"))


def load(out):
    value = json.loads((Path(out) / "plan.json").read_text(encoding="utf-8"))
    if value.get("schema") != PLAN_SCHEMA or value.get("protocol") != PROTOCOL:
        raise ValueError("unsupported plan; re-plan into a new output root")
    return value


# ---------------------------------------------------------------- judge

def tri_judge(model, device):
    """score_dynamic.Judge with a three-way first-token readout."""
    m = sq()
    judge = m.Judge(model, device)
    ids = {c: sorted({judge.tokenizer.encode(w, add_special_tokens=False)[0] for w in TOKENS[c]}) for c in CHOICES}
    flat = [i for c in CHOICES for i in ids[c]]
    if len(flat) != len(set(flat)):
        raise ValueError(f"answer tokens overlap: {ids}")
    torch = judge.torch

    def p_tri(prompts, images=None, videos=None, metadata=None):
        kw = {"text": [judge._text(p) for p in prompts], "padding": True, "return_tensors": "pt"}
        if images:
            kw.update(images=images, size=judge.image_size)
        if videos:
            kw.update(videos=videos, video_metadata=metadata, do_sample_frames=False, size=judge.video_size,
                      cap_pixels_per_frame=True)
        inputs = judge.processor(**kw).to(judge.model.device)
        with torch.inference_mode():
            logits = judge.model(**inputs, logits_to_keep=1).logits[:, -1].float()
        logp = torch.log_softmax(logits, 1)
        g = torch.stack([torch.logsumexp(logp[:, ids[c]], 1) for c in CHOICES], 1)
        mass = g.exp().sum(1)
        p = torch.softmax(g, 1)
        return [{"A": a, "B": b, "Same": s, "mass": w} for (a, b, s), w in zip(p.cpu().tolist(), mass.cpu().tolist())]
    judge.p_tri = p_tri
    return judge


def tri_prompt(kind, q):
    noun = kind.capitalize()
    return [f"{noun} A:", kind, f"{noun} B:", kind, TRI.format(A=f"{noun} A", B=f"{kind} B", q=q, kind=kind)]


def score_tri(judge, x, y):
    """{aspect: {"x_first": [per-frame answers], "y_first": [...]}} with A/B as presented."""
    m = sq()
    (fx, fps_x), (fy, fps_y) = x, y
    fq = [a for a in m.QUALITY if a[1] == "frames"]
    vq = [a for a in m.QUALITY if a[1] == "video"]
    out = {a[0]: {"x_first": [], "y_first": []} for a in m.QUALITY}
    for i, j in zip(pick(len(fx), m.FRAME_COUNT), pick(len(fy), m.FRAME_COUNT)):
        xf = judge.p_tri([tri_prompt("image", a[2]) for a in fq], images=[fx[i], fy[j]] * len(fq))
        yf = judge.p_tri([tri_prompt("image", a[2]) for a in fq], images=[fy[j], fx[i]] * len(fq))
        for a, p, r in zip(fq, xf, yf):
            out[a[0]]["x_first"].append(p)
            out[a[0]]["y_first"].append(r)
    (cx, mx), (cy, my) = m.video_clip(fx, fps_x), m.video_clip(fy, fps_y)
    for a in vq:
        p, = judge.p_tri([tri_prompt("video", a[2])], videos=[cx, cy], metadata=[mx, my])
        r, = judge.p_tri([tri_prompt("video", a[2])], videos=[cy, cx], metadata=[my, mx])
        out[a[0]]["x_first"].append(p)
        out[a[0]]["y_first"].append(r)
    return out


def pair(args):
    value = load(args.out)
    cache = Path(args.out) / "pair"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = sorted({r["key"]: r for r in value["pairs"]}.items())
    todo = [j for j in shard(jobs, args.shard_index, args.shard_count) if not (cache / f"{j[0]}.json").exists()]
    print(f"[tri {args.shard_index}/{args.shard_count}] {len(todo)} to judge", flush=True)
    if not todo:
        return
    m = sq()
    judge = tri_judge(args.model, args.device)
    for i, (k, r) in enumerate(todo):
        x = m.verified(r["x"])
        y = x if r["x"]["video_sha256"] == r["y"]["video_sha256"] else m.verified(r["y"])
        write_json_atomic(cache / f"{k}.json", {"pair": k, "protocol": PROTOCOL, "model": str(args.model),
                                                "answers": score_tri(judge, x, y)})
        print(f"[tri {args.shard_index}/{args.shard_count}] {i + 1}/{len(todo)} {r['kind']} {r['arm']}", flush=True)


# ---------------------------------------------------------------- report

def resolve(ans):
    """Order-averaged P(X better), P(Y better), P(Same); A/B position bias; in-format mass."""
    xs = [(p["A"], p["B"], p["Same"]) for p in ans["x_first"]]   # X shown as A
    ys = [(p["B"], p["A"], p["Same"]) for p in ans["y_first"]]   # X shown as B
    px, py, ps = np.mean(xs + ys, axis=0)
    bias = float(np.mean([p["A"] - p["B"] for p in ans["x_first"] + ans["y_first"]]))
    mass = float(np.mean([p["mass"] for p in ans["x_first"] + ans["y_first"]]))
    return float(px), float(py), float(ps), bias, mass


def old_answers(dirs, k):
    for d in dirs:
        p = Path(d) / f"{k}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))["answers"]
    return None


def kept(a):
    return int(a["p_full_first"] >= 0.5 and a["p_accel_first"] >= 0.5)


def report(args):
    value = load(args.out)
    out = Path(args.out)
    aspects = sq().ASPECTS
    rows = []
    for r in value["pairs"]:
        path = out / "pair" / f"{r['key']}.json"
        if not path.exists():
            raise ValueError(f"missing judge call {r['kind']} {r['arm']} {r['key']}; run pair first")
        ans = json.loads(path.read_text(encoding="utf-8"))["answers"]
        v = r["x"] if r["kind"] == "arm" else r["y"]
        row = {"kind": r["kind"], "arm": r["arm"], "prompt_id": v["prompt_id"], "motion_level": v["motion_level"],
               "detail_level": v["detail_level"], "family_id": v["family_id"]}
        for a in aspects:
            px, py, ps, bias, mass = resolve(ans[a])
            row.update({f"px_{a}": round(px, 4), f"py_{a}": round(py, 4), f"ps_{a}": round(ps, 4),
                        f"net_{a}": round(px - py, 4), f"bias_{a}": round(bias, 4), f"mass_{a}": round(mass, 4),
                        f"out_{a}": ("x", "y", "same")[int(np.argmax([px, py, ps]))]})
        row["net"] = float(np.mean([row[f"net_{a}"] for a in aspects]))
        fwd = old_answers(args.old_cache, r["key"])                                   # old: X >= Y
        rev = old_answers(args.old_cache, f"{r['y']['video_sha256']}__{r['x']['video_sha256']}")  # old: Y >= X
        if fwd and r["kind"] != "identity":
            row["old_quality_kept"] = float(np.mean([kept(fwd[a]) for a in aspects]))
            if rev:
                row["old_not_worse"] = float(np.mean([not (not kept(fwd[a]) and kept(rev[a])) for a in aspects]))
        rows.append(row)
    fields = list(dict.fromkeys(k for x in rows for k in x))
    with (out / "tri_by_pair.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summarize(rows, aspects)


def summarize(rows, aspects):
    def mean(sub, k):
        v = [r[k] for r in sub if r.get(k) is not None]
        return float(np.mean(v)) if v else float("nan")

    def se(sub, k):
        v = [r[k] for r in sub if r.get(k) is not None]
        return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")

    def share(sub, o):
        return float(np.mean([[r[f"out_{a}"] == o for a in aspects] for r in sub])) if sub else float("nan")

    print("Three-way pairwise quality (X = arm or FULL seed 100, Y = FULL seed 42); net = P(X better) - P(Y better)")
    print("in-format token mass (min over pairs, per aspect): "
          + " ".join(f"{a} {min(r[f'mass_{a}'] for r in rows):.3f}" for a in aspects))
    ident = [r for r in rows if r["kind"] == "identity"]
    print(f"identity n={len(ident)}: outcome same {share(ident, 'same'):.2f}; P(Same) "
          + " ".join(f"{a[:5]} {mean(ident, f'ps_{a}'):.2f}" for a in aspects)
          + "; A-B position bias " + " ".join(f"{a[:5]} {mean(ident, f'bias_{a}'):+.2f}" for a in aspects))
    print("\ngroup                     n   net (se)        | x>y   y>x   same  | P(Same) | old kept  old not_worse | net per aspect")
    groups = []
    for kind, arm in dict.fromkeys((r["kind"], r["arm"]) for r in rows if r["kind"] != "identity"):
        sub = [r for r in rows if r["kind"] == kind and r["arm"] == arm]
        groups += [(arm, sub)] + [(f"  {m}-motion", [r for r in sub if r["motion_level"] == m]) for m in ("high", "low")]
    for name, sub in groups:
        print(f"{name:22} {len(sub):4}  {mean(sub, 'net'):+.3f} ({se(sub, 'net'):.3f}) | "
              f"{share(sub, 'x'):.2f}  {share(sub, 'y'):.2f}  {share(sub, 'same'):.2f}  | "
              f"{np.mean([mean(sub, f'ps_{a}') for a in aspects]):.2f}    | "
              f"{mean(sub, 'old_quality_kept'):8.3f}  {mean(sub, 'old_not_worse'):12.3f} | "
              + " ".join(f"{a[:5]} {mean(sub, f'net_{a}'):+.2f}" for a in aspects))
    seed_effect(rows)


def seed_effect(rows):
    """With three or more seeds: is there a seed main effect shared across prompts (the same seed is the same initial
    noise for every prompt), and are the per-prompt judgments transitive (net(c,a) ~ net(c,b) + net(b,a))?"""
    net = {}
    for r in rows:
        if r["kind"] == "seed":
            sx, sy = seed_pair(r["arm"])
            net[(r["prompt_id"], sx, sy)] = r["net"]
            net[(r["prompt_id"], sy, sx)] = -r["net"]
    seeds = sorted({k[1] for k in net}, key=int)
    if len(seeds) < 3:
        return
    motion = {r["prompt_id"]: r["motion_level"] for r in rows}
    print("\nSeed effect: per prompt, a seed's score is its mean net against the other seeds (sums to 0 per prompt).")
    print("If seeds were exchangeable, every mean is ~0. se over prompts; sign = share of prompts with score > 0.")
    rng = np.random.default_rng(0)
    for m in ("high", "low", "all"):
        ps = sorted({k[0] for k in net if m == "all" or motion[k[0]] == m}, key=int)
        ps = [p for p in ps if all((p, a, b) in net for a in seeds for b in seeds if a != b)]
        sc = {s: np.array([np.mean([net[(p, s, t)] for t in seeds if t != s]) for p in ps]) for s in seeds}
        print(f"  {m:4} n={len(ps):3}  " + "   ".join(
            f"s{s} {sc[s].mean():+.3f} ({sc[s].std(ddof=1) / np.sqrt(len(ps)):.3f}) pos {np.mean(sc[s] > 0):.2f}" for s in seeds))
        if len(seeds) == 3:
            a, b, c = seeds
            direct = np.array([net[(p, c, a)] for p in ps])
            via = np.array([net[(p, c, b)] + net[(p, b, a)] for p in ps])
            sign = np.sign([[net[(p, b, a)], net[(p, c, b)], net[(p, a, c)]] for p in ps])
            cyc = np.mean([abs(s.sum()) == 3 for s in sign])  # a<b<c<a or the reverse: an intransitive cycle
            # chance level for cycles with independent random signs is 2/8 = 0.25
            perm = [np.mean([abs(rng.choice([-1, 1], 3).sum()) == 3 for _ in ps]) for _ in range(2000)]
            print(f"       transitivity: corr(net s{c}>s{a}, via s{b}) {np.corrcoef(direct, via)[0, 1]:+.2f}; "
                  f"cycles {cyc:.2f} (random signs {np.mean(perm):.2f})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--quality-root", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    a.add_argument("--prompts", type=int, default=16, help="per motion level")
    a.add_argument("--seeds", nargs="+", default=["42", "100"], help="FULL seeds compared pairwise (later vs earlier)")
    a = sub.add_parser("pair")
    a.add_argument("--out", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--device", default="cuda")
    a.add_argument("--shard-index", type=int, default=0)
    a.add_argument("--shard-count", type=int, default=1)
    a = sub.add_parser("report")
    a.add_argument("--out", required=True)
    a.add_argument("--old-cache", nargs="*", default=[])
    args = p.parse_args(argv)
    {"plan": plan, "pair": pair, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
