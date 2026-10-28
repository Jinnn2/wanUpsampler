import csv, collections, math, statistics as st, sys
R = "results/"
import numpy as np
def rd(p): return list(csv.DictReader(open(p, encoding="utf-8")))
nat = rd(R + "natural/natural_v1_results/route_natural_embed/route_table.csv")
ctl = [r for r in rd(R + "route/route_v1_q2/route_table.csv") if r["split"] != "test"]
arms = sorted({r["arm"] for r in nat})
ctl = [r for r in ctl if r["arm"] in arms]
def f(x):
    try: return float(x)
    except: return float("nan")
def decomp(rows, key):
    # prompt x arm matrix over shared arms, complete prompts only
    M = collections.defaultdict(dict); cell = {}
    for r in rows:
        v = f(r[key]); M[r["prompt_id"]][r["arm"]] = v; cell[r["prompt_id"]] = r["cell"]
    P = [p for p in M if len(M[p]) == len(arms) and all(math.isfinite(M[p][a]) for a in arms)]
    X = np.array([[M[p][a] for a in arms] for p in P]); C = np.array([cell[p] for p in P])
    g = X.mean(); tot = ((X - g) ** 2).sum()
    arm = X.mean(0) - g
    prm = X.mean(1) - g
    # cell x arm interaction
    cm = {c: X[C == c].mean(0) for c in set(C)}
    cellmain = {c: X[C == c].mean() - g for c in cm}
    inter = np.array([cm[c] - X[C == c].mean() - X.mean(0) + g for c in C])
    resid = X - g - arm[None] - prm[:, None]
    ss = lambda a: float((a ** 2).sum()) if a.ndim == 2 else None
    return {"n": len(P), "arm": len(P) * (arm ** 2).sum() / tot, "prompt": len(arms) * (prm ** 2).sum() / tot,
            "cellxarm": (inter ** 2).sum() / tot, "promptxarm_rest": ((resid - inter) ** 2).sum() / tot,
            "arm_means": dict(zip(arms, X.mean(0).round(2)))}
for key in ("retention", "quality_kept"):
    for name, rows in (("controlled", ctl), ("natural", nat)):
        d = decomp(rows, key)
        print(f"{key:13} {name:10} n={d['n']:3}  arm {d['arm']:.2f}  prompt {d['prompt']:.2f}  cell x arm {d['cellxarm']:.3f}  other prompt x arm {d['promptxarm_rest']:.2f}")
        print("   arm means:", d["arm_means"])
# measured motion / detail of FULL clips per cell
def full_stats(path, cellfn):
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rd(path):
        if r["case_id"] != "FULL50" or r.get("split") == "test": continue
        c = cellfn(r)
        for k in ("flow_mag", "sharpness"):
            out[c][k].append(f(r[k]))
    return {c: {k: round(st.median(v), 2) for k, v in d.items()} | {"n": len(d["flow_mag"])} for c, d in sorted(out.items())}
cmap = {r["prompt_id"]: r["cell"] for r in nat}
print("---- FULL measured stats")
def full2(path, sel, cellfn):
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rd(path):
        if not sel(r): continue
        for k in ("flow_mag", "sharpness", "flow_local"):
            out[cellfn(r)][k].append(f(r[k]))
    return {c: {k: round(st.median(v), 2) for k, v in d.items()} | {"n": len(d["flow_mag"])} for c, d in sorted(out.items())}
print(full2(R + "layers/fixed_v1/fixed_by_video.csv", lambda r: r["case_id"].startswith("FULL") and r["split"] != "test", lambda r: r["motion_level"][0] + r["detail_level"][0]))
print(full2(R + "natural/natural_v1_results/fixed_metrics_natural/fixed_by_video.csv", lambda r: r["case_id"].startswith("FULL") and int(r["prompt_id"]) >= 1000, lambda r: cmap.get(r["prompt_id"], "?")))
print("---- controlled retention, single seed (seed 42) vs seed-mean, from dynamic_v1_all")
dyn = [r for r in rd(R + "layers/dynamic_v1_all/dynamic_by_video.csv") if r["split"] != "test" and r["case_id"] in arms]
print("arms there:", sorted({r["case_id"] for r in dyn}), "seeds:", sorted({r["base_seed"] for r in dyn}))
cellc = {r["prompt_id"]: r["motion_level"][0] + r["detail_level"][0] for r in dyn}
for seed in sorted({r["base_seed"] for r in dyn}):
    rows = [{"prompt_id": r["prompt_id"], "arm": r["case_id"], "cell": cellc[r["prompt_id"]], "retention": r["retention"]} for r in dyn if r["base_seed"] == seed]
    a2 = arms
    globals()["arms"] = sorted({r["arm"] for r in rows})
    d = decomp(rows, "retention"); globals()["arms"] = a2
    print(f"seed {seed}: n={d['n']} arm {d['arm']:.2f} prompt {d['prompt']:.2f} cell x arm {d['cellxarm']:.3f} other {d['promptxarm_rest']:.2f}")
# seed-to-seed reliability of prompt x arm residual
M = collections.defaultdict(dict)
for r in dyn: M[(r["prompt_id"], r["case_id"])][r["base_seed"]] = f(r["retention"])
seeds = sorted({r["base_seed"] for r in dyn})
A = sorted({r["case_id"] for r in dyn})
def resid(seed):
    P = sorted({p for p, a in M if all(math.isfinite(M[(p, b)].get(seed, float('nan'))) for b in A)})
    X = np.array([[M[(p, a)][seed] for a in A] for p in P]); g = X.mean()
    return P, X - X.mean(0)[None] - X.mean(1)[:, None] + g
P1, R1 = resid(seeds[0]); P2, R2 = resid(seeds[1])
common = sorted(set(P1) & set(P2)); i1 = [P1.index(p) for p in common]; i2 = [P2.index(p) for p in common]
print("seed-to-seed corr of prompt x arm residuals (controlled, retention):", round(np.corrcoef(R1[i1].ravel(), R2[i2].ravel())[0, 1], 3), "n prompts", len(common))
