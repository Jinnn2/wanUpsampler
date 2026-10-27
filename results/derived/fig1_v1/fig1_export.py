"""Figure 1 data export (dev->dev2 scoreboard + fixed arms on dev2). Reuses route_v1's Data/solve unchanged."""
import csv, json, sys, collections
from pathlib import Path
import numpy as np
REPO, RQ2, PF, TS, OUT = map(Path, sys.argv[1:6])
DVG = Path(sys.argv[6]) if len(sys.argv) > 6 else None  # DVG study dev2 quality_by_video.csv (its own FULL)
sys.path.insert(0, str(REPO))
from UNIV_adaptor.scripts.data import route_v1 as rv

data = rv.Data(RQ2 / "route_table.csv")
A = data.arms
FULL = rv.FULL
# per-prompt, per-arm mean vbench5 over seeds; delta vs FULL50 of the same prompt
vb = collections.defaultdict(list)
for f in (PF, TS):
    for r in csv.DictReader(open(f, encoding="utf-8-sig")):
        arm = FULL if r["action_id"] == "FULL50" else r["action_id"]
        vb[(int(r["prompt_key"][2:]), arm, r["seed"])].append(float(r["vbench5"]))
vbm = collections.defaultdict(list)
for (p, a, s), v in vb.items():
    vbm[(p, a)].append(np.mean(v))  # FULL50 appears in both files with the same video
if DVG:  # DVG arms: delta vs the DVG study's own FULL clip of the same prompt and seed
    rows = list(csv.DictReader(open(DVG, encoding="utf-8-sig")))
    full_vb = {(r["prompt_id"], r["base_seed"]): float(r["vbench5"]) for r in rows if r["case_id"] == FULL}
    dvg_d = collections.defaultdict(list)
    for r in rows:
        if r["case_id"] != FULL:
            dvg_d[(int(r["prompt_id"]), r["case_id"])].append(float(r["vbench5"]) - full_vb[(r["prompt_id"], r["base_seed"])])
dv = np.full(data.tf.shape, np.nan)
for i, p in enumerate(data.pids):
    if (p, FULL) in vbm:
        base = np.mean(vbm[(p, FULL)])
        for j, a in enumerate(A):
            if (p, a) in vbm:
                dv[i, j] = np.mean(vbm[(p, a)]) - base
    if DVG:
        for j, a in enumerate(A):
            if (p, a) in dvg_d:
                dv[i, j] = np.mean(dvg_d[(p, a)])
dev = np.isin(data.split, ["train", "validation"])
te = np.where(~dev)[0]
assert np.isfinite(dv[te][:, [A.index(a) for a in A if not a.startswith("DVG")]]).all()
rng = np.random.default_rng(20261027)
fam = data.family[te]
F = sorted(set(fam))

def agg(cols):  # cols: dict name -> per-prompt vector over te
    def stat(idx):
        m = {k: v[idx].mean() for k, v in cols.items()}
        return {"speedup": 1 / m["tf"], "retention": m["ret"], "quality_kept": m["q"],
                "min_ret_q": min(m["ret"], m["q"]), "d_vbench5_pp": 100 * m["dv"]}
    point = stat(np.arange(len(te)))
    groups = [np.where(fam == f)[0] for f in F]
    boots = [stat(np.concatenate([groups[k] for k in rng.integers(0, len(F), len(F))])) for _ in range(2000)]
    return {k: {"point": point[k], "ci95": list(np.percentile([b[k] for b in boots], [2.5, 97.5]))} for k in point}

def policy_cols(x):
    c = data.cell[te]
    w = x[c]
    if w[:, [j for j, a in enumerate(A) if a.startswith("DVG")]].sum() > 1e-6:
        raise SystemExit("policy puts weight on DVG arms (no dev2 VBench in these tables)")
    w = w.copy(); dvz = np.nan_to_num(dv[te])
    return {"tf": (w * data.tf[te]).sum(1), "ret": (w * data.ret[te]).sum(1), "q": (w * data.q[te]).sum(1), "dv": (w * dvz).sum(1)}

rows, rules = [], {}
for arm in ("RCM4", "LR_FLASH", "LR_BICUBIC", "F41_RIFE", "SCALING10", "TEA015", "STEP20", "LR41_FLASH_RIFE"):
    j = A.index(arm)
    rows.append({"policy": arm, "tau": "", **agg({"tf": data.tf[te, j], "ret": data.ret[te, j], "q": data.q[te, j], "dv": dv[te, j]})})
for tau in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60):
    xr = rv.solve(data.tf[dev], data.ret[dev], data.q[dev], data.cell[dev], tau)
    xb = rv.solve(data.tf[dev], data.ret[dev], data.q[dev], data.cell[dev], tau, blind=True)
    rules[str(tau)] = {"router": {rv.CELLS[c]: {A[a]: round(float(p), 3) for a, p in enumerate(xr[c]) if p > 1e-3} for c in range(4)},
                       "blind": {A[a]: round(float(p), 3) for a, p in enumerate(xb[0]) if p > 1e-3}}
    for name, x in (("router", xr), ("blind", xb)):
        rows.append({"policy": name, "tau": tau, **agg(policy_cols(x))})
# per-cell fixed-arm profile on dev2 (for panel a)
cells = []
for arm in [a for a in A if a != FULL]:
    j = A.index(arm)
    for c, cn in enumerate(rv.CELLS):
        m = te[data.cell[te] == c]
        cells.append({"arm": arm, "cell": cn, "prompts": len(m), "speedup": 1 / data.tf[m, j].mean(),
                      "retention": data.ret[m, j].mean(), "quality_kept": data.q[m, j].mean(),
                      "min_ret_q": min(data.ret[m, j].mean(), data.q[m, j].mean()), "d_vbench5_pp": 100 * dv[m, j].mean()})
OUT.mkdir(parents=True, exist_ok=True)
flat = [{"policy": r["policy"], "tau": r["tau"], **{f"{k}{s}": (v["point"] if s == "" else v["ci95"][s == "_hi"])
         for k, v in r.items() if isinstance(v, dict) for s in ("", "_lo", "_hi")}} for r in rows]
with open(OUT / "scoreboard_dev2.csv", "w", newline="", encoding="utf-8") as h:
    w = csv.DictWriter(h, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
with open(OUT / "fixed_arms_by_cell_dev2.csv", "w", newline="", encoding="utf-8") as h:
    w = csv.DictWriter(h, fieldnames=list(cells[0])); w.writeheader(); w.writerows(cells)
json.dump({"n_dev2_prompts": int(len(te)), "n_dev_prompts": int(dev.sum()), "families_dev2": len(F), "policies_dev_fit": rules},
          open(OUT / "scoreboard_dev2_policies.json", "w"), indent=1)
for r in flat:
    print(f"{r['policy']:>16} {str(r['tau']):>5}  speed {r['speedup']:.2f}x  ret {r['retention']:.3f}  Q {r['quality_kept']:.3f}  min {r['min_ret_q']:.3f}  dVB {r['d_vbench5_pp']:+.2f}pp [{r['d_vbench5_pp_lo']:+.2f},{r['d_vbench5_pp_hi']:+.2f}]")
print(len(te), "dev2 prompts")
