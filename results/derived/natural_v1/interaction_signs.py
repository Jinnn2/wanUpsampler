import csv, collections, math, sys
nat = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
dev = list(csv.DictReader(open(sys.argv[2], encoding="utf-8")))
CELLS = ["hh", "hl", "lh", "ll"]
acc = collections.defaultdict(lambda: [[], [], []])
bad = {r["prompt_id"] for r in nat if not all(math.isfinite(float(r[k])) for k in ("retention", "quality_kept"))}
for r in nat:
    if r["prompt_id"] in bad: continue
    a = acc[(r["arm"], r["cell"])]
    a[0].append(float(r["retention"])); a[1].append(float(r["quality_kept"])); a[2].append(float(r["tf"]))
arms = sorted({a for a, _ in acc})
n = {c: len({r["prompt_id"] for r in nat if r["cell"] == c and r["prompt_id"] not in bad}) for c in CELLS}
print("prompts per predicted cell:", n)
X = {k: min(sum(v[0]) / len(v[0]), sum(v[1]) / len(v[1])) for k, v in acc.items()}
D = {(r["arm"], r["cell"]): float(r["min_ret_q"]) for r in dev if r["arm"] in arms}
def inter(M):
    g = sum(M[(a, c)] for a in arms for c in CELLS) / (len(arms) * 4)
    ra = {a: sum(M[(a, c)] for c in CELLS) / 4 for a in arms}
    cc = {c: sum(M[(a, c)] for a in arms) / len(arms) for c in CELLS}
    return {(a, c): M[(a, c)] - ra[a] - cc[c] + g for a in arms for c in CELLS}
IN, ID = inter(X), inter(D)
agree = tot = 0; big_agree = big = 0
print(f"{'arm':16} " + "  ".join(f"{c}: nat  dev2 " for c in CELLS))
for a in arms:
    row = []
    for c in CELLS:
        s = (IN[(a, c)] > 0) == (ID[(a, c)] > 0); agree += s; tot += 1
        if abs(ID[(a, c)]) >= 0.05: big += 1; big_agree += s
        row.append(f"{X[(a,c)]:.2f}({IN[(a,c)]:+.2f}) {D[(a,c)]:.2f}({ID[(a,c)]:+.2f}){'' if s else '*'}")
    print(f"{a:16} " + "  ".join(row))
print(f"interaction sign agreement {agree}/{tot}; where |dev2 interaction|>=0.05: {big_agree}/{big}")
import statistics
xs=[IN[k] for k in IN]; ys=[ID[k] for k in IN]
print("pearson r of interactions:", round(statistics.correlation(xs, ys), 3))
