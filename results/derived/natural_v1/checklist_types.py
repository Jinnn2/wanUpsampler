import csv, collections, sys, math
import numpy as np
ARMS = ["F41_RIFE", "LR41_FLASH_RIFE", "LR_FLASH", "RCM4", "SCALING10", "STEP20", "TEA015"]
CONTENT = {"entity", "attribute", "negative", "spatial", "action"}
def load(path, keep_types):
    # type-balanced retention per video over questions FULL passed, then mean over seeds per (prompt, arm)
    v = collections.defaultdict(lambda: collections.defaultdict(list)); cell = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["split"] == "test" or r["case_id"] not in ARMS or r["kept"] == "" or r["type"] not in keep_types: continue
        v[(r["prompt_id"], r["case_id"], r["base_seed"])][r["type"]].append(float(r["kept"]))
        cell[r["prompt_id"]] = r["motion_level"][0] + r["detail_level"][0]
    pa = collections.defaultdict(list)
    for (p, a, s), d in v.items():
        pa[(p, a)].append(np.mean([np.mean(x) for x in d.values()]))
    P = sorted({p for p, a in pa if all((p, b) in pa for b in ARMS)})
    X = np.array([[np.mean(pa[(p, a)]) for a in ARMS] for p in P]); C = np.array([cell[p] for p in P])
    return X, C
def share(X, C):
    g = X.mean(); tot = ((X - g) ** 2).sum(); am = X.mean(0)
    inter = sum((C == c).sum() * ((X[C == c].mean(0) - X[C == c].mean() - am + g) ** 2).sum() for c in set(C))
    return inter / tot, len(X)
ALL = {"entity","attribute","negative","spatial","action","camera","integrity","detail","static","lighting","motion","count","focus"}
for name, path in (("controlled", sys.argv[1]), ("natural", sys.argv[2])):
    for tname, ts in (("all types", ALL), ("content types only", CONTENT), ("motion/static/camera/detail only", {"motion","static","camera","detail"})):
        X, C = load(path, ts); s, n = share(X, C)
        print(f"{name:10} {tname:34} prompts {n:3}  cell x arm share {s:.3f}   RCM4 by cell " + str({c: round(X[C == c, 3].mean(), 2) for c in sorted(set(C))}))
