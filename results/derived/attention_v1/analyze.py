"""Attention v1: the pre-registered tests of UNIV_adaptor/ATTENTION_V1.md (A1-A3) plus the descriptive tables.

python results/derived/attention_v1/analyze.py   (from the repo root; reads results/attention/attention_v1_results)
"""
import collections
import csv
import math
import statistics as st

import numpy as np

R = "results/attention/attention_v1_results/"
NEW = ["SVG2", "SPARGE50"]
FULL = "FULL_NATIVE_50"
B = 10_000


def rd(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def score(r):
    return min(float(r["retention"]), float(r["quality_kept"]))


def family_boot(rows, stat, level, rng):
    fams = sorted({r["family_id"] for r in rows})
    by = collections.defaultdict(list)
    for r in rows:
        by[r["family_id"]].append(r)
    point, draws = stat(rows), []
    for _ in range(B):
        sample = [r for f in rng.choice(fams, len(fams)) for r in by[f]]
        v = stat(sample)
        if v is not None:
            draws.append(v)
    a = (1 - level) / 2
    return point, float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))


def delta(axis):
    """low-cell mean minus high-cell mean of min(ret, Q); axis 0 = motion, 1 = detail (cell = motion + detail letter)."""
    def stat(rows):
        lo = [score(r) for r in rows if r["cell"][axis] == "l"]
        hi = [score(r) for r in rows if r["cell"][axis] == "h"]
        return (st.mean(lo) - st.mean(hi)) if lo and hi else None
    return stat


def sign_test(k, n):
    """two-sided exact binomial p for k of n under p = 0.5"""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    rng = np.random.default_rng(0)
    table = rd(R + "route_attn/route_table.csv")
    dropped = sorted({r["prompt_id"] for r in table if not math.isfinite(score(r))}, key=int)
    table = [r for r in table if r["prompt_id"] not in dropped]  # no checklist -> no retention (the router drops them too)
    print(f"dropped prompts without a dynamic score (all arms): {dropped}")
    arm_rows = {a: [r for r in table if r["arm"] == a] for a in sorted({r["arm"] for r in table})}
    print(f"route table: {len({r['prompt_id'] for r in table})} prompts, {len(arm_rows)} accelerated arms; "
          "score = min(retention, quality_kept), each the 3-seed mean, as the router uses it\n")

    print("== A1  SPARGE50: low-motion minus high-motion score > 0 (95% family-bootstrap CI)")
    p, lo, hi = family_boot(arm_rows["SPARGE50"], delta(0), 0.95, rng)
    print(f"   delta_motion = {p:+.3f} [{lo:+.3f}, {hi:+.3f}]  ->  {'HOLDS' if lo > 0 else 'does not hold'}")
    for s in ("v1", "dev2"):
        sub = [r for r in arm_rows["SPARGE50"] if (int(r["prompt_id"]) < 100) == (s == "v1")]
        p2, lo2, hi2 = family_boot(sub, delta(0), 0.95, rng)
        print(f"   {s:5} (descriptive) {p2:+.3f} [{lo2:+.3f}, {hi2:+.3f}]")

    print("\n== A2  SVG2: motion and detail deltas inside +-0.05 (90% CIs, equivalence)")
    ok = True
    for axis, name in ((0, "motion"), (1, "detail")):
        p, lo, hi = family_boot(arm_rows["SVG2"], delta(axis), 0.90, rng)
        inside = -0.05 < lo and hi < 0.05
        ok &= inside
        print(f"   delta_{name} = {p:+.3f} [{lo:+.3f}, {hi:+.3f}]  inside: {inside}")
    print(f"   ->  {'HOLDS' if ok else 'does not hold'}")

    print("\n== A3  SPARGE50 fails the v2 judder/flicker gates more often than SVG2 (paired clips, exact sign test)")
    q = rd(R + "quality_attn_v2/quality_by_video.csv")
    fail = {(r["prompt_id"], r["base_seed"], r["case_id"]): int(r["judder"] == "1" or r["flicker"] == "1")
            for r in q if r["case_id"] in NEW}
    pairs = {(p, s) for p, s, _ in fail}
    only_sp = sum(fail[(p, s, "SPARGE50")] and not fail[(p, s, "SVG2")] for p, s in pairs)
    only_svg = sum(fail[(p, s, "SVG2")] and not fail[(p, s, "SPARGE50")] for p, s in pairs)
    pval = sign_test(only_sp, only_sp + only_svg)
    print(f"   clips {len(pairs)}; fail rate SPARGE50 {st.mean(fail[(p, s, 'SPARGE50')] for p, s in pairs):.3f}, "
          f"SVG2 {st.mean(fail[(p, s, 'SVG2')] for p, s in pairs):.3f}; discordant: SPARGE50-only {only_sp}, SVG2-only {only_svg}")
    print(f"   p = {pval:.4f}  ->  {'HOLDS' if only_sp > only_svg and pval < 0.05 else 'does not hold'}")

    print("\n== Post hoc (not pre-registered): motion delta beyond the shared one = new arm minus the mean of the 15 old arms")
    old = [a for a in arm_rows if a not in NEW]

    def beyond(arm):
        def stat(rows):
            ds = {}
            for a in [arm, *old]:
                v = delta(0)([r for r in rows if r["arm"] == a])
                if v is None:
                    return None
                ds[a] = v
            return ds[arm] - st.mean(ds[a] for a in old)
        return stat
    for a in NEW:
        p, lo, hi = family_boot([r for r in table if r["arm"] in (a, *old)], beyond(a), 0.95, np.random.default_rng(1))
        print(f"   {a:9} {p:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\n== Descriptive: every arm (prompt means; speed = 1 / time fraction of FULL)")
    print(f"   {'arm':16} {'speed':>6} {'ret':>6} {'Q':>6} {'min':>6}   by cell min(ret,Q): hh hl lh ll")
    for a, rows in sorted(arm_rows.items(), key=lambda kv: st.mean(float(r["tf"]) for r in kv[1])):
        cells = {c: st.mean(score(r) for r in rows if r["cell"] == c) for c in ("hh", "hl", "lh", "ll")}
        mark = " <-" if a in NEW else ""
        print(f"   {a:16} {1 / st.mean(float(r['tf']) for r in rows):5.2f}x {st.mean(float(r['retention']) for r in rows):6.3f} "
              f"{st.mean(float(r['quality_kept']) for r in rows):6.3f} {st.mean(score(r) for r in rows):6.3f}   "
              + " ".join(f"{cells[c]:.2f}" for c in cells) + mark)

    print("\n== Descriptive: VBench5 vs the layered score for the new arms (per clip, against the same-seed FULL50)")
    for s in ("v1", "dev2"):
        vb = rd(R + f"attention_wan_{s}/metrics/quality_by_video.csv")
        full = {r["group_id"]: float(r["vbench5"]) for r in vb if r["action_id"] == "FULL50"}
        for a in NEW:
            d = [float(r["vbench5"]) - full[r["group_id"]] for r in vb if r["action_id"] == a]
            dims = {k: st.mean(float(r[k]) - float(next(x[k] for x in vb if x["action_id"] == "FULL50" and x["group_id"] == r["group_id"]))
                               for r in vb if r["action_id"] == a)
                    for k in ("subject_consistency", "motion_smoothness", "aesthetic_quality", "imaging_quality", "dynamic_degree")}
            print(f"   {s:5} {a:9} d_vbench5 {st.mean(d):+.4f} (clips worse than FULL: {sum(x < 0 for x in d)}/{len(d)})  "
                  + " ".join(f"{k.split('_')[0]} {v:+.4f}" for k, v in dims.items()))


if __name__ == "__main__":
    main()
