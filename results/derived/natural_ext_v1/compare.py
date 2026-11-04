"""natural_ext v1 vs natural v1 (short): the descriptive pairs listed in UNIV_adaptor/NATURAL_EXT_V1.md.

python compare.py EMB_SHORT_DIR EMB_EXT_DIR   (each holds emb_t5/emb_qwen/emb_qwenmid.npz from the server's
route_*_embed_analysis.tgz; the rest is read from results/)
"""
import collections
import csv
import math
import sys

import numpy as np

R = "results/"
SHORT = R + "natural/natural_v1_results/"
EXT = R + "natural/natural_ext_v1_results/"
ARMS = ["F41_RIFE", "LR41_FLASH_RIFE", "LR_FLASH", "RCM4", "SCALING10", "STEP20", "TEA015"]
CELLS = ["hh", "hl", "lh", "ll"]


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def f(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


def per_seed(path_dyn, path_q, seed=None):
    """(prompt, arm) -> (retention, quality_kept), averaged over the seeds kept; prompt -> cell."""
    q = {(r["source"], r["stem"]): f(r["quality_kept"]) for r in rd(path_q)}
    acc = collections.defaultdict(lambda: [[], []]); cell = {}
    for r in rd(path_dyn):
        if r["case_id"] not in ARMS or (seed and r["base_seed"] != seed):
            continue
        a = acc[(int(r["prompt_id"]), r["case_id"])]
        a[0].append(f(r["retention"])); a[1].append(q.get((r["source"], r["stem"]), float("nan")))
        cell[int(r["prompt_id"])] = r["motion_level"][0] + r["detail_level"][0]
    out = {k: (np.nanmean(v[0]) if np.isfinite(v[0]).any() else np.nan, np.nanmean(v[1]) if np.isfinite(v[1]).any() else np.nan)
           for k, v in acc.items()}
    return out, cell


def matrix(vals, cell, k):
    P = sorted(p for p in cell if all((p, a) in vals and math.isfinite(vals[(p, a)][k]) for a in ARMS))
    return P, np.array([[vals[(p, a)][k] for a in ARMS] for p in P]), np.array([cell[p] for p in P])


def shares(X, C):
    g = X.mean(); tot = ((X - g) ** 2).sum(); am = X.mean(0); pm = X.mean(1)
    inter = sum((C == c).sum() * ((X[C == c].mean(0) - X[C == c].mean() - am + g) ** 2).sum() for c in set(C))
    resid = X - am[None] - pm[:, None] + g
    return {"arm": len(X) * ((am - g) ** 2).sum() / tot, "prompt": X.shape[1] * ((pm - g) ** 2).sum() / tot,
            "cell_x_arm": inter / tot, "other_prompt_x_arm": (resid ** 2).sum() / tot - inter / tot}


def full_flow(path, ids):
    fl = collections.defaultdict(list)
    for r in rd(path):
        if r["case_id"].startswith("FULL") and int(r["prompt_id"]) in ids:
            fl[(int(r["prompt_id"]), r["base_seed"])].append(f(r["flow_mag"]))
    return {k: float(np.mean(v)) for k, v in fl.items()}


def ridge_r2(X, y, lams=(10, 100, 1000)):
    best = -9
    for lam in lams:
        out = []
        for rep in range(10):
            fo = np.random.default_rng(rep).permutation(len(y)) % 8; pred = np.zeros_like(y)
            for k in range(8):
                tr, te = fo != k, fo == k; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
                w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y[tr] - y[tr].mean()))
                pred[te] = y[tr].mean() + B @ w
            out.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        best = max(best, float(np.mean(out)))
    return best


def emb(d, name, P):
    z = np.load(f"{d}/emb_{name}.npz"); ids = list(z["prompt_ids"]); X = z["x"].astype(float)[[ids.index(p) for p in P]]
    X /= np.linalg.norm(X, axis=1, keepdims=True); X -= X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :64] * S[:64]


def extremes(X, flow, rng):
    lo, hi = flow <= np.quantile(flow, .15), flow >= np.quantile(flow, .85)
    D = X - X.mean(1, keepdims=True)
    out = {}
    for j, a in enumerate(ARMS):
        bs = [D[rng.choice(np.where(hi)[0], hi.sum())][:, j].mean() - D[rng.choice(np.where(lo)[0], lo.sum())][:, j].mean()
              for _ in range(2000)]
        out[a] = (D[hi, j].mean() - D[lo, j].mean(), np.quantile(bs, .025), np.quantile(bs, .975))
    return out, float(np.median(flow[lo])), float(np.median(flow[hi]))


def main():
    emb_short, emb_ext = sys.argv[1], sys.argv[2]
    sets = {"short (seed 42)": (SHORT + "dynamic_natural/dynamic_by_video.csv", SHORT + "quality_natural_v2/quality_by_video.csv", None,
                                SHORT + "fixed_metrics_natural/fixed_by_video.csv", emb_short, "42"),
            "ext seed 42": (EXT + "dynamic_natural_ext/dynamic_by_video.csv", EXT + "quality_natural_ext_v2/quality_by_video.csv", "42",
                            EXT + "fixed_metrics_natural_ext/fixed_by_video.csv", emb_ext, "42"),
            "ext seed 100": (EXT + "dynamic_natural_ext/dynamic_by_video.csv", EXT + "quality_natural_ext_v2/quality_by_video.csv", "100",
                             EXT + "fixed_metrics_natural_ext/fixed_by_video.csv", emb_ext, "100"),
            "ext both seeds": (EXT + "dynamic_natural_ext/dynamic_by_video.csv", EXT + "quality_natural_ext_v2/quality_by_video.csv", None,
                               EXT + "fixed_metrics_natural_ext/fixed_by_video.csv", emb_ext, None)}
    dev = {(r["arm"], r["cell"]): float(r["min_ret_q"]) for r in rd(R + "derived/fig1_v1/fixed_arms_by_cell_dev2.csv")}
    rng = np.random.default_rng(0)
    for name, (dyn, qual, seed, fixed, ed, fseed) in sets.items():
        vals, cell = per_seed(dyn, qual, seed)
        print(f"\n===== {name}")
        P, Xm, C = matrix({k: (min(v), min(v)) for k, v in vals.items()}, cell, 0)
        print("prompts", len(P), "cells", {c: int((C == c).sum()) for c in CELLS})
        for k, lab in ((0, "retention"), (1, "quality_kept")):
            Pk, X, Ck = matrix(vals, cell, k)
            print(f"{lab:13} arm means " + " ".join(f"{a[:8]}={m:.2f}" for a, m in zip(ARMS, X.mean(0)))
                  + "  | shares " + " ".join(f"{s}={v:.3f}" for s, v in shares(X, Ck).items()))
        # sign agreement of min(ret, Q) cell x arm interaction with dev2
        M = {(a, c): np.mean([min(vals[(p, a)]) for p in P if cell[p] == c]) for a in ARMS for c in CELLS if (C == c).any()}
        cs = sorted({c for _, c in M})
        def inter(T):
            g = np.mean([T[(a, c)] for a in ARMS for c in cs]); ra = {a: np.mean([T[(a, c)] for c in cs]) for a in ARMS}
            cc = {c: np.mean([T[(a, c)] for a in ARMS]) for c in cs}
            return {(a, c): T[(a, c)] - ra[a] - cc[c] + g for a in ARMS for c in cs}
        IN, ID = inter(M), inter({k: dev[k] for k in M})
        agree = sum((IN[k] > 0) == (ID[k] > 0) for k in IN)
        print(f"cell x arm sign agreement with dev2: {agree}/{len(IN)}, r = {np.corrcoef(list(IN.values()), [ID[k] for k in IN])[0, 1]:+.2f}")
        # FULL motion and text -> motion
        fl = full_flow(fixed, set(cell))
        flow = np.array([np.mean([v for (p2, s), v in fl.items() if p2 == p and (fseed is None or s == fseed)]) for p in P])
        print("FULL flow median", round(float(np.median(flow)), 2), "p85", round(float(np.quantile(flow, .85)), 2),
              "by cell", {c: round(float(np.median(flow[C == c])), 2) for c in CELLS if (C == c).any()})
        y = np.log1p(flow)
        mv = np.array([[c[0] == "h"] for c in C], float)
        print("text -> log(1+flow) held-out R2: cell", round(ridge_r2(mv, y, (1,)), 3),
              " " + " ".join(f"{n}={ridge_r2(emb(ed, n, P), y):.3f}" for n in ("t5", "qwen", "qwenmid")))
        ex, lo, hi = extremes(Xm, flow, rng)
        print(f"top-15% vs bottom-15% flow (median {hi:.2f} vs {lo:.2f}), min(ret,Q) relative to prompt mean:")
        print("   " + "  ".join(f"{a[:8]} {d:+.3f}[{l:+.2f},{u:+.2f}]" for a, (d, l, u) in ex.items()))
    # seed-to-seed reliability on ext
    v42, cell = per_seed(EXT + "dynamic_natural_ext/dynamic_by_video.csv", EXT + "quality_natural_ext_v2/quality_by_video.csv", "42")
    v100, _ = per_seed(EXT + "dynamic_natural_ext/dynamic_by_video.csv", EXT + "quality_natural_ext_v2/quality_by_video.csv", "100")
    for k, lab in ((0, "retention"), (1, "quality_kept")):
        P1, X1, _ = matrix(v42, cell, k); P2, X2, _ = matrix(v100, cell, k)
        common = sorted(set(P1) & set(P2)); A = X1[[P1.index(p) for p in common]]; B = X2[[P2.index(p) for p in common]]
        res = lambda X: X - X.mean(0)[None] - X.mean(1)[:, None] + X.mean()
        print(f"\next seed 42 vs 100, {lab}: corr of prompt x arm residuals {np.corrcoef(res(A).ravel(), res(B).ravel())[0, 1]:.3f}, "
              f"of prompt means {np.corrcoef(A.mean(1), B.mean(1))[0, 1]:.3f} (n={len(common)})")


if __name__ == "__main__":
    main()
