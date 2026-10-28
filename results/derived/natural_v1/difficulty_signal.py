"""Can prompt features predict a prompt's overall difficulty (mean over arms), as opposed to which arm suits it?"""
import csv, collections, math, sys, json
import numpy as np
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
arms = sorted({r["arm"] for r in rows})
M = collections.defaultdict(dict); cell = {}
for r in rows:
    M[int(r["prompt_id"])][r["arm"]] = (float(r["retention"]), float(r["quality_kept"])); cell[int(r["prompt_id"])] = r["cell"]
P = sorted(p for p in M if len(M[p]) == len(arms) and all(math.isfinite(v) for a in arms for v in M[p][a]))
Y = {"retention": np.array([[M[p][a][0] for a in arms] for p in P]), "quality_kept": np.array([[M[p][a][1] for a in arms] for p in P])}
Y["min"] = np.minimum(Y["retention"], Y["quality_kept"])
def emb(path):
    z = np.load(path); ids = list(z["prompt_ids"]); X = z["x"].astype(float)[[ids.index(p) for p in P]]
    X = X / np.linalg.norm(X, axis=1, keepdims=True); X -= X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False); return U[:, :64] * S[:64]
def oof(X, y, lam, folds=8, reps=10):
    out = []
    for r in range(reps):
        f = np.random.default_rng(r).permutation(len(y)) % folds; pred = np.zeros_like(y)
        for k in range(folds):
            tr, te = f != k, f == k
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; A = (X[tr] - mu) / sd; B = (X[te] - mu) / sd
            w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y[tr] - y[tr].mean()))
            pred[te] = y[tr].mean() + B @ w
        out.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(out))
cells = np.array([[cell[p] == c for c in ("hh", "hl", "lh")] for p in P], float)
E = {"cell (one-hot)": cells}
for n in sys.argv[2].split(","):
    E[n.split("/")[-1][4:-4]] = emb(n)
print("held-out R2 of per-prompt difficulty (mean over 7 arms); best ridge lambda in {1,10,100,1000}")
for key in ("retention", "quality_kept", "min"):
    y = Y[key].mean(1)
    print(key, {name: round(max(oof(X, y, l) for l in (1, 10, 100, 1000)), 3) for name, X in E.items()})
