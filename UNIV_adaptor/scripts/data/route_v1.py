"""Prompt -> method router v1: maximize speed subject to keeping semantics and visual quality, from prompt text alone.

A policy assigns each motion x detail cell a distribution over the accelerated methods (and FULL itself). Given a
floor tau, it minimizes mean wall-clock fraction (1 / speedup) subject to

    mean retention >= tau   and   mean quality_kept >= tau,

both measured relative to the same-seed FULL clip (dynamic and quality layers), so the floor means "keep tau of
FULL's semantics and of its visual quality" and no weighting between the two is needed. This is a small linear
program; the prompt-blind baseline is the same program with one distribution shared by all cells (it reduces to
the best mixture of at most three methods), so any gain is what knowing the prompt's cell buys.

At deployment the cell is not known: it is predicted from the prompt text. Classifiers:
- oracle: the designed cell labels (upper bound; not available for a new prompt)
- llm:    zero-shot Qwen3-VL text-only judgment with two fixed questions (mode `classify`; no training on labels)
- bow:    bag-of-words logistic regression trained inside each fold (supervised reference)
The policy is learned in the predicted-cell space, i.e. on the cells the deployed router would see.

Evaluation never touches the test split (it is absent from the fixed plan): family-grouped k-fold cross-validation
over dev + dev2, repeated with fresh fold draws, plus the pre-existing dev -> dev2 split. Confidence intervals
bootstrap families on the out-of-fold results.

Usage (see run_univ_route.sh):
  python route_v1.py table    --fixed-root F --dynamic D/dynamic_by_video.csv --quality Q/quality_by_video.csv --out OUT
  python route_v1.py classify --prompts P1 P2 --model M --out OUT       # GPU, a few minutes
  python route_v1.py evaluate --out OUT
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic  # noqa: E402

FULL = "FULL_NATIVE_50"
CELLS = ("hh", "hl", "lh", "ll")  # motion, detail
TAUS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60)
SEED = 20261026
DEFAULT_PROMPTS = [ROOT / "prompts/univ_controlled_factor_v1.jsonl", ROOT / "prompts/univ_controlled_factor_dev2.jsonl"]
# Fixed before any classification was run. The prompt is quoted verbatim; P(Yes) >= 0.5 means "high".
QUESTIONS = {
    "motion": ("Here is a text prompt for a short generated video:\n\"{prompt}\"\n"
               "Does the prompt ask for substantial motion - clearly moving subjects, fast action, or camera movement - "
               "rather than a mostly still scene? Answer with only Yes or No."),
    "detail": ("Here is a text prompt for a short generated video:\n\"{prompt}\"\n"
               "Does the prompt ask for dense fine spatial detail - many small objects, intricate textures, or "
               "high-frequency patterns filling the frame - rather than a few large, plain, smooth shapes? "
               "Answer with only Yes or No."),
}


def cell_of(motion, detail):
    return motion[0] + detail[0]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def flt(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------- table

def seconds_by_video(fixed_plan):
    """(source, stem) -> generation wall-clock seconds, from each source's own records."""
    from UNIV_adaptor.scripts.data.score_controlled_factor_requirements import published_rows

    out = {}
    for name, src in fixed_plan["sources"].items():
        if src["kind"] == "published":
            for r in published_rows(src["root"])[2]:
                out[(name, r["observation_id"])] = float(r["pipeline_seconds"])
        elif src["kind"] == "dvg":
            for v in fixed_plan["videos"]:
                if v["source"] == name:
                    receipt = json.loads((Path(src["root"]) / "receipts" / f"{v['stem']}.json").read_text(encoding="utf-8"))
                    out[(name, v["stem"])] = float(receipt["generate_seconds"])
        else:
            raise ValueError(f"no runtime records for source kind {src['kind']}")
    return out


def table(args):
    fixed_path = Path(args.fixed_root) / "plan.json"
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    seconds = seconds_by_video(fixed)
    dyn = {(r["source"], r["stem"]): r for r in read_csv(args.dynamic)}
    qual = {(r["source"], r["stem"]): r for r in read_csv(args.quality)}
    full_s = {(v["source"], v["group_id"]): seconds[(v["source"], v["stem"])] for v in fixed["videos"] if v["case_id"] == FULL}
    acc = collections.defaultdict(lambda: collections.defaultdict(list))  # (pid, arm) -> metric -> per-seed values
    meta = {}
    for v in fixed["videos"]:
        if v["case_id"] == FULL or v["split"] == "test":
            continue
        key = (v["source"], v["stem"])
        if key not in dyn or key not in qual:
            raise ValueError(f"no dynamic/quality score for {key}; finish those layers first")
        pid = int(v["prompt_id"])
        meta[pid] = {"family_id": v["family_id"], "split": v["split"], "cell": cell_of(v["motion_level"], v["detail_level"])}
        a = acc[(pid, v["case_id"])]
        a["tf"].append(seconds[key] / full_s[(v["source"], v["group_id"])])
        a["retention"].append(flt(dyn[key]["retention"]))  # NaN when FULL satisfies none of the requirements
        a["quality_kept"].append(flt(qual[key][args.quality_column]))
    rows = []
    for (pid, arm), a in sorted(acc.items()):
        r = {k: float(np.nanmean(x)) if np.isfinite(x).any() else float("nan") for k, x in a.items()}
        rows.append({"prompt_id": pid, **meta[pid], "arm": arm, **r, "n": len(a["tf"])})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "route_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json_atomic(out / "table_inputs.json", {"fixed_plan_sha256": sha256_file(fixed_path),
                                                  "dynamic_sha256": sha256_file(args.dynamic),
                                                  "quality_sha256": sha256_file(args.quality),
                                                  "quality_column": args.quality_column})
    print(f"wrote {out / 'route_table.csv'}: {len(meta)} prompts, {len({r['arm'] for r in rows})} arms")


class Data:
    """Prompt x arm matrices (FULL added as an arm with tf = retention = quality_kept = 1)."""

    def __init__(self, path):
        rows = read_csv(path)
        arms = sorted({r["arm"] for r in rows})
        by = collections.defaultdict(dict)
        info = {}
        for r in rows:
            pid = int(r["prompt_id"])
            info[pid] = {"family": r["family_id"], "split": r["split"], "cell": r["cell"]}
            by[pid][r["arm"]] = (flt(r["tf"]), flt(r["retention"]), flt(r["quality_kept"]))
        complete = [p for p in sorted(by) if all(a in by[p] and all(map(math.isfinite, by[p][a])) for a in arms)]
        dropped = len(by) - len(complete)
        missing = collections.Counter(a for p in by for a in arms if a not in by[p])
        if missing:
            print(f"NOTE: arms missing for some prompts (prompts dropped): {dict(missing)}")
        if dropped:
            print(f"NOTE: {dropped} prompts dropped for missing or undefined scores")
        self.arms = arms + [FULL]
        self.pids = complete
        self.info = info
        m = np.array([[by[p][a] for a in arms] for p in complete])  # (P, A, 3)
        m = np.concatenate([m, np.ones((len(complete), 1, 3))], axis=1)
        self.tf, self.ret, self.q = m[..., 0], m[..., 1], m[..., 2]
        self.cell = np.array([CELLS.index(info[p]["cell"]) for p in complete])
        self.family = np.array([info[p]["family"] for p in complete])
        self.split = np.array([info[p]["split"] for p in complete])


# ---------------------------------------------------------------- classify

def classify(args):
    from UNIV_adaptor.scripts.data import requirements_v2 as rv2
    from UNIV_adaptor.scripts.data.score_dynamic import Judge

    prompts = [p for p in rv2.load_prompts(args.prompts).values() if p["split"] != "test"]
    judge = Judge(args.model)
    result = {}
    for i in range(0, len(prompts), args.batch):
        chunk = prompts[i:i + args.batch]
        ps = {k: judge.p_yes([[q.format(prompt=p["prompt"])] for p in chunk]) for k, q in QUESTIONS.items()}
        for j, p in enumerate(chunk):
            result[str(p["prompt_id"])] = {f"p_{k}": round(ps[k][j], 5) for k in QUESTIONS}
        print(f"[classify] {min(i + args.batch, len(prompts))}/{len(prompts)}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "cells_llm.json", {"model": str(args.model), "questions": QUESTIONS, "prompts": result})
    truth = {p["prompt_id"]: (p["motion_level"], p["detail_level"]) for p in prompts}
    for k, idx in (("motion", 0), ("detail", 1)):
        acc = np.mean([(result[str(pid)][f"p_{k}"] >= 0.5) == (t[idx] == "high") for pid, t in truth.items()])
        print(f"{k} accuracy against the designed labels: {acc:.3f}")


def llm_cells(out, pids):
    value = json.loads((Path(out) / "cells_llm.json").read_text(encoding="utf-8"))
    if value["questions"] != QUESTIONS:
        raise ValueError("cells_llm.json was made with different questions; rerun classify")
    p = value["prompts"]
    return np.array([CELLS.index(("h" if p[str(i)]["p_motion"] >= 0.5 else "l") + ("h" if p[str(i)]["p_detail"] >= 0.5 else "l"))
                     for i in pids])


def tokens(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def bow_cells(texts, labels, train, test):
    """Two L2-regularized logistic regressions on binary unigram features, fit on `train` only."""
    vocab = sorted(set().union(*(tokens(texts[i]) for i in train)))
    index = {w: j for j, w in enumerate(vocab)}

    def feats(ids):
        x = np.zeros((len(ids), len(vocab) + 1))
        x[:, -1] = 1.0
        for r, i in enumerate(ids):
            for w in tokens(texts[i]):
                if w in index:
                    x[r, index[w]] = 1.0
        return x

    xtr, xte = feats(train), feats(test)
    out = np.zeros((len(test), 2), bool)
    for k in range(2):
        y = np.array([labels[i][k] for i in train], float)
        w = np.zeros(xtr.shape[1])
        for _ in range(400):
            p = 1 / (1 + np.exp(-xtr @ w))
            w -= 0.5 * (xtr.T @ (p - y) / len(y) + 1e-2 * np.r_[w[:-1], 0])
        out[:, k] = xte @ w >= 0
    return np.array([CELLS.index(("h" if m else "l") + ("h" if d else "l")) for m, d in out])


# ---------------------------------------------------------------- policy

def solve(tf, ret, q, cell, tau, blind=False):
    """Per-cell arm distributions x (4, A) minimizing mean tf with mean retention and quality_kept >= tau.

    tf/ret/q: (P, A) training values; cell: (P,) cell index used by the policy. With blind=True one distribution
    is shared by every cell. A cell without training prompts falls back to the blind policy.
    """
    from scipy.optimize import linprog

    a = tf.shape[1]
    w = np.array([(cell == c).mean() for c in range(len(CELLS))])
    if blind or (w == 0).any():
        res = linprog(tf.mean(0), A_ub=-np.stack([ret.mean(0), q.mean(0)]), b_ub=[-tau, -tau],
                      A_eq=np.ones((1, a)), b_eq=[1], bounds=(0, None), method="highs")
        if res.status != 0:
            raise RuntimeError(f"blind LP failed: {res.message}")
        x0 = np.clip(res.x, 0, None) / np.clip(res.x, 0, None).sum()
        if blind:
            return np.tile(x0, (len(CELLS), 1))
    mean = lambda m, c: m[cell == c].mean(0) if w[c] > 0 else np.zeros(a)  # noqa: E731
    cs = [c for c in range(len(CELLS)) if w[c] > 0]
    n = len(cs) * a
    obj = np.concatenate([w[c] * mean(tf, c) for c in cs])
    a_ub = -np.stack([np.concatenate([w[c] * mean(m, c) for c in cs]) for m in (ret, q)])
    a_eq = np.zeros((len(cs), n))
    for i in range(len(cs)):
        a_eq[i, i * a:(i + 1) * a] = 1
    res = linprog(obj, A_ub=a_ub, b_ub=[-tau, -tau], A_eq=a_eq, b_eq=np.ones(len(cs)), bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"routing LP failed: {res.message}")
    x = np.tile(x0, (len(CELLS), 1)) if (w == 0).any() else np.zeros((len(CELLS), a))
    for i, c in enumerate(cs):
        xi = np.clip(res.x[i * a:(i + 1) * a], 0, None)
        x[c] = xi / xi.sum()
    return x


def apply(x, cell, *ms):
    """Per-prompt expected values of each metric matrix under policy x and the given cells."""
    return [np.sum(x[cell] * m, 1) for m in ms]


# ---------------------------------------------------------------- evaluate

def folds(families, k, rng):
    fam = sorted(set(families))
    rng.shuffle(fam)
    group = {f: i % k for i, f in enumerate(fam)}
    return np.array([group[f] for f in families])


def predictor(name, data, out, texts):
    if name == "oracle":
        return lambda train, test: (data.cell[train], data.cell[test])
    if name == "llm":
        pred = llm_cells(out, data.pids)
        return lambda train, test: (pred[train], pred[test])
    if name == "bow":
        labels = [(CELLS[c][0] == "h", CELLS[c][1] == "h") for c in data.cell]
        return lambda train, test: (bow_cells(texts, labels, train, train), bow_cells(texts, labels, train, test))
    raise ValueError(name)


def run_protocol(data, predict, tau, splits):
    """splits: list of (train idx, test idx) covering each prompt at most once per repeat; returns per-prompt
    out-of-fold (router tf, ret, q, blind tf, ret, q) averaged over repeats, and the cell accuracy."""
    total = np.zeros((len(data.pids), 6))
    count = np.zeros(len(data.pids))
    hits = []
    for train, test in splits:
        ctr, cte = predict(train, test)
        hits += list(cte == data.cell[test])
        xr = solve(data.tf[train], data.ret[train], data.q[train], ctr, tau)
        xb = solve(data.tf[train], data.ret[train], data.q[train], ctr, tau, blind=True)
        ms = (data.tf[test], data.ret[test], data.q[test])
        total[test] += np.stack(apply(xr, cte, *ms) + apply(xb, cte, *ms), 1)
        count[test] += 1
    keep = count > 0
    return total[keep] / count[keep, None], data.family[keep], float(np.mean(hits))


def summarize(values, families, rng, n_boot=2000):
    fam = sorted(set(families))
    sums = np.stack([values[families == f].sum(0) for f in fam])  # (F, 6)
    sizes = np.array([(families == f).sum() for f in fam])

    def stats(s, n):
        m = s.sum(0) / n.sum()
        return np.array([1 / m[0], m[1], m[2], 1 / m[3], m[4], m[5], m[3] / m[0] - 1, m[1] - m[4], m[2] - m[5]])

    point = stats(sums, sizes)
    boot = np.array([stats(sums[i], sizes[i]) for i in (rng.integers(0, len(fam), len(fam)) for _ in range(n_boot))])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return point, lo, hi


def evaluate(args):
    rng = np.random.default_rng(SEED)
    out = Path(args.out)
    data = Data(out / "route_table.csv")
    from UNIV_adaptor.scripts.data import requirements_v2 as rv2

    prompts = rv2.load_prompts(args.prompts)
    texts = [prompts[p]["prompt"] for p in data.pids]
    classifiers = [c for c in args.classifiers if c != "llm" or (out / "cells_llm.json").exists()]
    if "llm" in args.classifiers and "llm" not in classifiers:
        print("NOTE: no cells_llm.json yet (run classify); skipping the llm classifier")
    idx = np.arange(len(data.pids))
    cv_splits = []
    for _ in range(args.repeats):
        f = folds(list(data.family), args.folds, rng)
        cv_splits += [(idx[f != k], idx[f == k]) for k in range(args.folds)]
    dev = np.isin(data.split, ["train", "validation"])
    protocols = {"cv": cv_splits, "dev->dev2": [(idx[dev], idx[~dev])]}
    names = ["router_speed", "router_ret", "router_q", "blind_speed", "blind_ret", "blind_q", "speed_gain", "d_ret", "d_q"]
    records = []
    for clf in classifiers:
        predict = predictor(clf, data, out, texts)
        for proto, splits in protocols.items():
            for tau in args.taus:
                values, fams, acc = run_protocol(data, predict, tau, splits)
                point, lo, hi = summarize(values, fams, rng)
                records.append({"classifier": clf, "protocol": proto, "tau": tau, "cell_accuracy": acc,
                                **{n: point[i] for i, n in enumerate(names)},
                                **{f"{n}_lo": lo[i] for i, n in enumerate(names[6:], 6)},
                                **{f"{n}_hi": hi[i] for i, n in enumerate(names[6:], 6)}})
                print(f"[{clf} {proto} tau={tau}] speed {point[0]:.2f}x vs {point[3]:.2f}x "
                      f"(gain {point[6]:+.1%} [{lo[6]:+.1%}, {hi[6]:+.1%}])", flush=True)
    with (out / "route_eval.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    # Deployment rules: learned on every non-test prompt.
    rules = {}
    for clf in classifiers:
        ctr, _ = predictor(clf, data, out, texts)(idx, idx[:0])
        rules[clf] = {str(tau): {CELLS[c]: {data.arms[a]: round(float(p), 3) for a, p in enumerate(row) if p > 1e-3}
                                 for c, row in enumerate(solve(data.tf, data.ret, data.q, ctr, tau))}
                      for tau in args.taus}
        rules[clf]["blind"] = {str(tau): {data.arms[a]: round(float(p), 3) for a, p in
                                          enumerate(solve(data.tf, data.ret, data.q, ctr, tau, blind=True)[0]) if p > 1e-3}
                               for tau in args.taus}
    write_json_atomic(out / "route_rules.json", rules)
    write_report(out, records, rules, data, args)


def write_report(out, records, rules, data, args):
    pct = lambda r, n: f"{r[n]:+.1%} [{r[n + '_lo']:+.1%}, {r[n + '_hi']:+.1%}]"  # noqa: E731
    pp = lambda r, n: f"{100 * r[n]:+.1f} [{100 * r[n + '_lo']:+.1f}, {100 * r[n + '_hi']:+.1f}]"  # noqa: E731
    lines = ["# Router v1: fastest policy that keeps tau of FULL's semantics and visual quality", "",
             f"{len(data.pids)} prompts (test split excluded), {len(data.arms)} arms including FULL. "
             f"CV: {args.folds}-fold by family x {args.repeats} repeats. Router and prompt-blind mixture both meet "
             "the floor on their training prompts; columns give out-of-fold results. speed gain = blind time / "
             "router time - 1; d_ret, d_q in percentage points (router - blind); 95% CI by family bootstrap. "
             "slack = min(ret, Q) - tau out of fold: the floor is what is promised, so a negative d_ret with "
             "slack >= 0 only means the router spent margin the blind mixture left unused.", ""]
    for proto in ("cv", "dev->dev2"):
        lines += [f"## {proto}", "", "| classifier | cell acc | tau | router speed | blind speed | speed gain | "
                  "router ret / Q | blind ret / Q | slack router / blind | d_ret | d_Q |", "| --- |" + " --- |" * 10]
        for r in records:
            if r["protocol"] == proto:
                slack = (min(r["router_ret"], r["router_q"]) - r["tau"], min(r["blind_ret"], r["blind_q"]) - r["tau"])
                lines.append(f"| {r['classifier']} | {r['cell_accuracy']:.2f} | {r['tau']:.2f} | {r['router_speed']:.2f}x | "
                             f"{r['blind_speed']:.2f}x | {pct(r, 'speed_gain')} | {r['router_ret']:.3f} / {r['router_q']:.3f} | "
                             f"{r['blind_ret']:.3f} / {r['blind_q']:.3f} | {slack[0]:+.3f} / {slack[1]:+.3f} | "
                             f"{pp(r, 'd_ret')} | {pp(r, 'd_q')} |")
        lines.append("")
    lines += ["## Deployment rules (all non-test prompts)", ""]
    for clf, by_tau in rules.items():
        lines.append(f"### {clf}")
        for tau in map(str, args.taus):
            cells = "; ".join(f"{c}: " + ", ".join(f"{a} {p:.2f}" for a, p in arms.items()) for c, arms in by_tau[tau].items())
            blind = ", ".join(f"{a} {p:.2f}" for a, p in by_tau["blind"][tau].items())
            lines.append(f"- tau {tau}: {cells}  |  blind: {blind}")
        lines.append("")
    text = "\n".join(lines)
    (out / "route_report.md").write_text(text, encoding="utf-8")
    print("\n" + text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("table")
    p.add_argument("--fixed-root", required=True)
    p.add_argument("--dynamic", required=True)
    p.add_argument("--quality", required=True)
    p.add_argument("--quality-column", default="quality_kept",
                   help="e.g. quality_kept_gate from quality_v2.py for the stricter variant")
    p.add_argument("--out", required=True)
    p = sub.add_parser("classify")
    p.add_argument("--prompts", nargs="+", default=[str(x) for x in DEFAULT_PROMPTS])
    p.add_argument("--model", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--out", required=True)
    p.add_argument("--prompts", nargs="+", default=[str(x) for x in DEFAULT_PROMPTS])
    p.add_argument("--classifiers", nargs="+", default=["oracle", "llm", "bow"])
    p.add_argument("--taus", nargs="+", type=float, default=list(TAUS))
    p.add_argument("--folds", type=int, default=8)
    p.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    {"table": table, "classify": classify, "evaluate": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
