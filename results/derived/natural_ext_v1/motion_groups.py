"""Exploratory: route_table copies whose `cell` is a motion group, for route_embed evaluate (blind / cell / bow only)."""
import csv, sys, collections, numpy as np
sys.path.insert(0, "results/derived/natural_ext_v1")
import compare as cm
EXT, EMB, OUT = cm.EXT, sys.argv[1], sys.argv[2]
rows = cm.rd(EXT + "route_natural_ext_embed/route_table.csv")
P = sorted({int(r["prompt_id"]) for r in rows})
fl = cm.full_flow(EXT + "fixed_metrics_natural_ext/fixed_by_video.csv", set(P))
flow = np.array([np.mean([v for (p, s), v in fl.items() if p == p0]) for p0 in P]); y = np.log1p(flow)
X = cm.emb(EMB, "qwenmid", P)
pred = np.zeros_like(y); f = np.random.default_rng(7).permutation(len(y)) % 8
for k in range(8):
    tr, te = f != k, f == k; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
    w = np.linalg.solve(A.T @ A + 10 * np.eye(A.shape[1]), A.T @ (y[tr] - y[tr].mean())); pred[te] = y[tr].mean() + B @ w
print("OOF R2 of predicted log flow:", round(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), 3))
def groups(v, q):
    return dict(zip(P, np.searchsorted(np.quantile(v, np.linspace(0, 1, q + 1)[1:-1]), v)))
import os
for name, g in (("pred_motion4", groups(pred, 4)), ("oracle_motion4", groups(y, 4))):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "route_table.csv"), "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader()
        for r in rows: w.writerow({**r, "cell": ["ll", "lh", "hl", "hh"][g[int(r["prompt_id"])]]})
    print(name, collections.Counter(g.values()))
