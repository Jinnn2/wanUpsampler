"""Null controls for the pairwise quality layer: does `quality_kept` measure "worse than FULL" or only "different"?

score_quality.score_pair(judge, X, Y) asks, per aspect and in both presentation orders, whether X is at least as good
as Y. The quality layer only ever asks it one way, X = accelerated and Y = FULL, and calls an aspect kept when both
orders say Yes (P >= 0.5). If the judge also says No for two different but equally good videos, any recomposed
clip loses `kept` without being worse. Here every needed ordered pair (X, Y) is judged with the same function and
judge, and cached under the same key as the quality layer ("<X sha>__<Y sha>"), so the forward answers are reused:

  identity  (F, F): the same FULL clip on both sides. The ceiling; it should be kept.
  seed      (F_s, F_t) and (F_t, F_s): two FULL clips of one prompt, seeds 42 and 100, equally good by
            construction but different. Their kept rate is what "merely different" costs.
  reverse   (FULL, A) for each arm A in --arms, next to the cached (A, FULL).

Per pair and aspect: fwd = A >= FULL, rev = FULL >= A, both in both orders. The outcome is equal (fwd and rev),
worse (rev only), better (fwd only) or undecided (neither: the judge will not call either at least as good).
`not_worse` = 1 - worse share counts only judged losses; `quality_kept` (the layer's score) = fwd share.
Per-clip PSNR of A against FULL (all frames, 0-255) shows whether judged losses follow how much the clip changed.

  python quality_null.py plan   --quality-root Q --out OUT [--arms SVG2 SPARGE50 SCALING10 TEA008] [--identity 32]
  python quality_null.py pair   --out OUT --model M --shard-index I --shard-count N   (one GPU each)
  python quality_null.py psnr   --out OUT --shard-index I --shard-count N              (CPU)
  python quality_null.py report --out OUT
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import write_json_atomic  # noqa: E402
from UNIV_adaptor.scripts.data.score_controlled_factor_requirements import shard  # noqa: E402

PLAN_SCHEMA = "univ_quality_null_v1"
FULL = "FULL_NATIVE_50"
SEEDS = ("42", "100")
DEFAULT_ARMS = ("SVG2", "SPARGE50", "SCALING10", "TEA008")


def sq():
    from UNIV_adaptor.scripts.data import score_quality
    return score_quality


def key(x, y):
    return f"{x['video_sha256']}__{y['video_sha256']}"


def plan(args):
    quality_root = Path(args.quality_root)
    value = json.loads((quality_root / "plan.json").read_text(encoding="utf-8"))
    videos = value["videos"]
    full = {}
    for v in videos:  # one FULL clip per (prompt, seed); several sources share the same file
        if v["case_id"] == FULL:
            full.setdefault((str(v["prompt_id"]), str(v["base_seed"])), v)
    by_group = {(v["source"], v["group_id"]): v for v in videos if v["case_id"] == FULL}
    pairs = []  # (kind, arm, X, Y, meta)
    prompts = sorted({p for p, s in full if all((p, t) in full for t in SEEDS)}, key=int)
    for p in prompts[:args.identity]:
        f = full[(p, SEEDS[0])]
        pairs.append(("identity", "FULL", f, f))
    for p in prompts:
        a, b = full[(p, SEEDS[0])], full[(p, SEEDS[1])]
        pairs += [("seed", "FULL_SEED", b, a), ("seed", "FULL_SEED", a, b)]
    seen = set()
    for v in videos:
        if v["case_id"] in args.arms and v["video_sha256"] not in seen:
            seen.add(v["video_sha256"])
            f = by_group[(v["source"], v["group_id"])]
            pairs += [("forward", v["case_id"], v, f), ("reverse", v["case_id"], f, v)]
    rows = [{"kind": k, "arm": a, "key": key(x, y), "x": x, "y": y} for k, a, x, y in pairs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "plan.json", {"schema": PLAN_SCHEMA, "protocol": sq().PROTOCOL,
                                          "quality_root": str(quality_root.resolve()), "arms": list(args.arms),
                                          "seeds": list(SEEDS), "pairs": rows})
    counts = collections.Counter(r["kind"] for r in rows)
    print(f"planned {len(rows)} ordered pairs {dict(counts)}; {len({r['key'] for r in rows})} distinct judge calls")


def load(out):
    value = json.loads((Path(out) / "plan.json").read_text(encoding="utf-8"))
    if value.get("schema") != PLAN_SCHEMA or value.get("protocol") != sq().PROTOCOL:
        raise ValueError("unsupported null-control plan or quality protocol changed")
    return value


def pair(args):
    value = load(args.out)
    cache = Path(args.out) / "pair"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = sorted({r["key"]: r for r in value["pairs"]}.items())
    todo = [j for j in shard(jobs, args.shard_index, args.shard_count) if not (cache / f"{j[0]}.json").exists()]
    print(f"[null pair {args.shard_index}/{args.shard_count}] {len(todo)} to judge", flush=True)
    if not todo:
        return
    m = sq()
    judge = m.Judge(args.model, args.device)
    for i, (k, r) in enumerate(todo):
        x = m.verified(r["x"])
        y = x if r["x"]["video_sha256"] == r["y"]["video_sha256"] else m.verified(r["y"])
        write_json_atomic(cache / f"{k}.json", {"pair": k, "protocol": m.PROTOCOL, "model": str(args.model),
                                                "answers": m.score_pair(judge, x, y)})
        print(f"[null pair {args.shard_index}/{args.shard_count}] {i + 1}/{len(todo)} {r['kind']} {r['arm']}", flush=True)


def psnr_of(a, b):
    n = min(len(a), len(b))
    mse = float(((a[:n].astype(np.float32) - b[:n].astype(np.float32)) ** 2).mean())
    return 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def psnr(args):
    value = load(args.out)
    cache = Path(args.out) / "psnr"
    cache.mkdir(parents=True, exist_ok=True)
    m = sq()
    jobs = sorted({r["key"]: r for r in value["pairs"] if r["kind"] in ("forward", "seed", "identity")}.items())
    todo = [j for j in shard(jobs, args.shard_index, args.shard_count) if not (cache / f"{j[0]}.json").exists()]
    for i, (k, r) in enumerate(todo):
        x = m.decode(r["x"]["video_path"])[0]
        y = x if r["x"]["video_sha256"] == r["y"]["video_sha256"] else m.decode(r["y"]["video_path"])[0]
        write_json_atomic(cache / f"{k}.json", {"pair": k, "psnr": psnr_of(np.asarray(x), np.asarray(y))})
        if i % 20 == 0:
            print(f"[psnr {args.shard_index}/{args.shard_count}] {i + 1}/{len(todo)}", flush=True)


def kept(answer):
    return int(answer["p_full_first"] >= 0.5 and answer["p_accel_first"] >= 0.5)


def outcome(fwd, rev):
    return {(1, 1): "equal", (0, 1): "worse", (1, 0): "better", (0, 0): "undecided"}[(fwd, rev)]


def report(args):
    value = load(args.out)
    out = Path(args.out)
    aspects = sq().ASPECTS
    answers = {}
    for r in value["pairs"]:
        path = out / "pair" / f"{r['key']}.json"
        if not path.exists():
            raise ValueError(f"missing judge call {r['kind']} {r['arm']} {r['key']}; run pair first")
        answers[r["key"]] = json.loads(path.read_text(encoding="utf-8"))["answers"]
    psnr_v = {}
    for p in (out / "psnr").glob("*.json") if (out / "psnr").exists() else []:
        d = json.loads(p.read_text(encoding="utf-8"))
        psnr_v[d["pair"]] = d["psnr"]
    rows = []
    by_kind = collections.defaultdict(list)
    for r in value["pairs"]:
        by_kind[r["kind"]].append(r)
    # identity: one call, fwd only (X = Y)
    for r in by_kind["identity"]:
        rows.append({"kind": "identity", "arm": "FULL", **meta(r["x"]), "psnr": psnr_v.get(r["key"]),
                     **{f"fwd_{a}": kept(answers[r["key"]][a]) for a in aspects}})
    # seed: (B, A) is fwd for "B >= A", (A, B) is rev
    seed = by_kind["seed"]
    for fwd_r, rev_r in zip(seed[0::2], seed[1::2]):
        rows.append(row("seed", "FULL_SEED", fwd_r, answers[fwd_r["key"]], answers[rev_r["key"]], aspects,
                        psnr_v.get(fwd_r["key"]) or psnr_v.get(rev_r["key"])))
    rev_of = {(r["arm"], r["y"]["video_sha256"], r["x"]["video_sha256"]): r for r in by_kind["reverse"]}
    for r in by_kind["forward"]:
        rv = rev_of[(r["arm"], r["x"]["video_sha256"], r["y"]["video_sha256"])]
        rows.append(row("arm", r["arm"], r, answers[r["key"]], answers[rv["key"]], aspects, psnr_v.get(r["key"])))
    fields = sorted({k for x in rows for k in x}, key=lambda k: (k.startswith(("fwd_", "rev_", "out_")), k))
    with (out / "null_by_pair.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summarize(rows, aspects)


def meta(v):
    return {k: v[k] for k in ("prompt_id", "base_seed", "motion_level", "detail_level", "family_id", "source")}


def row(kind, arm, r, fwd_ans, rev_ans, aspects, p):
    x = {"kind": kind, "arm": arm, **meta(r["x"] if kind == "arm" else r["y"]), "psnr": p}
    for a in aspects:
        f, b = kept(fwd_ans[a]), kept(rev_ans[a])
        x[f"fwd_{a}"], x[f"rev_{a}"], x[f"out_{a}"] = f, b, outcome(f, b)
    x["quality_kept"] = float(np.mean([x[f"fwd_{a}"] for a in aspects]))
    x["not_worse"] = float(np.mean([x[f"out_{a}"] != "worse" for a in aspects]))
    return x


def summarize(rows, aspects):
    def share(sub, kind):
        return np.mean([[r[f"out_{a}"] == kind for a in aspects] for r in sub]) if sub else float("nan")

    def mean(sub, k):
        v = [r[k] for r in sub if r.get(k) is not None]
        return float(np.mean(v)) if v else float("nan")
    ident = [r for r in rows if r["kind"] == "identity"]
    print(f"identity (same clip both sides, n={len(ident)}): kept per aspect "
          + " ".join(f"{a} {np.mean([r[f'fwd_{a}'] for r in ident]):.2f}" for a in aspects))
    print("\ngroup                      n  psnr   quality_kept  not_worse | equal worse better undecided | kept per aspect")
    groups = [("seed null (FULL vs FULL)", [r for r in rows if r["kind"] == "seed"])]
    for arm in sorted({r["arm"] for r in rows if r["kind"] == "arm"}):
        sub = [r for r in rows if r["kind"] == "arm" and r["arm"] == arm]
        groups += [(arm, sub)] + [(f"  {arm} {m}-motion", [r for r in sub if r["motion_level"] == m]) for m in ("high", "low")]
    groups += [(f"  seed null {m}-motion", [r for r in rows if r["kind"] == "seed" and r["motion_level"] == m])
               for m in ("high", "low")]
    for name, sub in groups:
        print(f"{name:25} {len(sub):4} {mean(sub, 'psnr'):5.1f}  {mean(sub, 'quality_kept'):11.3f} {mean(sub, 'not_worse'):10.3f} | "
              f"{share(sub, 'equal'):5.2f} {share(sub, 'worse'):5.2f} {share(sub, 'better'):6.2f} {share(sub, 'undecided'):9.2f} | "
              + " ".join(f"{a[:5]} {np.mean([r[f'fwd_{a}'] for r in sub]):.2f}" for a in aspects))
    print("\nPSNR vs judged quality per arm (Spearman over clips): quality_kept, not_worse")
    for arm in sorted({r["arm"] for r in rows if r["kind"] == "arm"}):
        sub = [r for r in rows if r["kind"] == "arm" and r["arm"] == arm and r.get("psnr") is not None]
        if len(sub) > 10:
            print(f"   {arm:10} n={len(sub)}  {spearman([r['psnr'] for r in sub], [r['quality_kept'] for r in sub]):+.2f}  "
                  f"{spearman([r['psnr'] for r in sub], [r['not_worse'] for r in sub]):+.2f}")


def spearman(x, y):
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--quality-root", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    a.add_argument("--identity", type=int, default=32)
    a = sub.add_parser("pair")
    a.add_argument("--out", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--device", default="cuda")
    a.add_argument("--shard-index", type=int, default=0)
    a.add_argument("--shard-count", type=int, default=1)
    a = sub.add_parser("psnr")
    a.add_argument("--out", required=True)
    a.add_argument("--shard-index", type=int, default=0)
    a.add_argument("--shard-count", type=int, default=1)
    a = sub.add_parser("report")
    a.add_argument("--out", required=True)
    args = p.parse_args(argv)
    {"plan": plan, "pair": pair, "psnr": psnr, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
