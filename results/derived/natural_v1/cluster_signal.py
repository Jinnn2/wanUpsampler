"""Do unsupervised prompt-vector clusters carry a cluster x arm signal that the motion x detail cells miss?"""
import csv, collections, json, math, sys
import numpy as np
TABLE, EMB = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(TABLE, encoding="utf-8")))
arms = sorted({r["arm"] for r in rows})
M = collections.defaultdict(dict); cell = {}; prompt_text = {}
for r in rows:
    M[int(r["prompt_id"])][r["arm"]] = (float(r["retention"]), float(r["quality_kept"])); cell[int(r["prompt_id"])] = r["cell"]
P = sorted(p for p in M if len(M[p]) == len(arms) and all(math.isfinite(v) for a in arms for v in M[p][a]))
Y = {"retention": np.array([[M[p][a][0] for a in arms] for p in P]), "quality_kept": np.array([[M[p][a][1] for a in arms] for p in P])}
Y["min"] = np.minimum(Y["retention"], Y["quality_kept"])
rng = np.random.default_rng(0)

def kmeans(X, k, restarts=20, iters=100):
    best = None
    for _ in range(restarts):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C): break
            C = newC
        sse = ((X - C[lab]) ** 2).sum()
        if best is None or sse < best[0]: best = (sse, lab)
    return best[1]

def inter_share(Yk, lab):
    g = Yk.mean(); tot = ((Yk - g) ** 2).sum()
    am = Yk.mean(0); s = 0.0
    for j in np.unique(lab):
        m = lab == j
        s += m.sum() * ((Yk[m].mean(0) - Yk[m].mean() - am + g) ** 2).sum()
    return s / tot

def oof_gain(Yk, lab, folds=8, reps=10):
    """Held-out R^2 of prompt x arm scores: (prompt's own mean over arms) + cluster x arm profile from train folds,
    vs + global arm profile. Prompt's own row mean uses the held-out row (a router never needs it: it shifts every arm)."""
    gains = []
    for r in range(reps):
        f = np.random.default_rng(100 + r).permutation(len(Yk)) % folds
        num = den = 0.0
        for k in range(folds):
            tr, te = f != k, f == k
            dev_tr = Yk[tr] - Yk[tr].mean(1, keepdims=True)
            base = dev_tr.mean(0)
            pred = np.array([dev_tr[lab[tr] == lab[i]].mean(0) if (lab[tr] == lab[i]).sum() >= 3 else base for i in np.where(te)[0]])
            truth = Yk[te] - Yk[te].mean(1, keepdims=True)
            num += ((truth - base) ** 2).sum() - ((truth - pred) ** 2).sum(); den += ((truth - base) ** 2).sum()
        gains.append(num / den)
    return float(np.mean(gains))

def perm_p(Yk, lab, stat, n=300):
    obs = stat(Yk, lab)
    null = [stat(Yk, rng.permutation(lab)) for _ in range(n)]
    return obs, float(np.mean(np.array(null) >= obs)), float(np.mean(null))

FEATS = ["flow_mag","flow_camera","flow_local","flow_alternation","flicker","boil","sharpness","sharpness_temporal_std","saturation","colorfulness","luma_mean","warp_error"]
if len(sys.argv) > 4:
    fx = collections.defaultdict(list)
    for r in csv.DictReader(open(sys.argv[4], encoding="utf-8")):
        if r["case_id"].startswith("FULL") and int(r["prompt_id"]) in M:
            fx[int(r["prompt_id"])].append([float(r[k]) for k in FEATS])
    VX = np.array([np.mean(fx[p], 0) for p in P]); VX[:, :3] = np.log1p(VX[:, :3]); VX = (VX - VX.mean(0)) / VX.std(0)
cell_lab = np.array([["hh", "hl", "lh", "ll"].index(cell[p]) for p in P])
print(f"{len(P)} prompts, {len(arms)} arms\n")
print("label            k   metric        interaction share (null mean, p)     held-out R2 of profile (null mean, p)")
def report(name, lab):
    for key in ("retention", "quality_kept", "min"):
        s, ps, ns = perm_p(Y[key], lab, inter_share)
        g, pg, ng = perm_p(Y[key], lab, oof_gain, n=100)
        print(f"{name:16} {len(np.unique(lab)):2}  {key:12}  {s:.3f} ({ns:.3f}, p={ps:.2f})              {g:+.3f} ({ng:+.3f}, p={pg:.2f})")
report("predicted cell", cell_lab)
if len(sys.argv) > 4:
    for k in (2, 4, 8):
        report("kmeans FULL-video", kmeans(VX, k))
    for j, k in enumerate(FEATS[:3] + ["sharpness"]):
        col = VX[:, FEATS.index(k)]
        report(f"{k} quartile", np.searchsorted(np.quantile(col, [.25, .5, .75]), col))
    sys.exit()
out = {}
for emb in EMB.split(","):
    z = np.load(emb); ids = list(z["prompt_ids"]); X = z["x"].astype(np.float64)
    X = X[[ids.index(p) for p in P]]
    X = X / np.linalg.norm(X, axis=1, keepdims=True); X -= X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False); X = U[:, :32] * S[:32]
    name = emb.replace("\\", "/").split("/")[-1].replace("emb_", "").replace(".npz", "")
    for k in (2, 4, 8):
        lab = kmeans(X, k)
        out[(name, k)] = lab
        report(f"kmeans {name}", lab)
        agree = max(np.mean(lab == cell_lab), 0)
        # adjusted mutual information would be nicer; contingency is enough to see overlap with cells
    print()
json.dump({f"{n}_k{k}": {str(p): int(l) for p, l in zip(P, lab)} for (n, k), lab in out.items()},
          open(sys.argv[3], "w"), indent=0)
