"""Read-only variance decomposition of acceleration loss: prompt/content versus seed.

Question: does an accelerator's quality change relative to matched FULL vary
systematically with prompt content (motion/detail cell, family, prompt), or is
the cross-prompt spread mostly seed noise? If the stable prompt component is
small, an average score cannot "hide" a category failure, because no stable
category failure exists.

Input is `quality_by_video.csv` from score_controlled_factor_dataset.py. Locked
test rows are dropped BEFORE any numeric parsing and are never reported.
All intervals are family-cluster bootstraps conditioned on the observed seeds;
they are exploratory (no multiplicity correction).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FULL = "FULL_NATIVE_50"
QUALITY = ("subject_consistency", "background_consistency", "motion_smoothness", "aesthetic_quality", "imaging_quality")
METRICS = ("vbench5", *QUALITY, "overall_consistency")
CELLS = (("low", "low"), ("low", "high"), ("high", "low"), ("high", "high"))  # (motion, detail)
TAUS = (0.01, 0.02, 0.03)
BOOT = int(__import__("os").environ.get("VARIANCE_BOOT", "5000"))


def load(path, metrics=METRICS):
    """Return (cube, meta): cube[action][metric] is a prompts x seeds delta array vs FULL."""
    rows, dropped = [], 0
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                dropped += 1
                continue
            rows.append(row)
    by_group = defaultdict(dict)
    for row in rows:
        by_group[row["group_id"]][row["case_id"]] = row
    prompts = sorted({int(r["prompt_id"]) for r in rows})
    seeds = sorted({int(r["base_seed"]) for r in rows})
    actions = sorted({r["case_id"] for r in rows} - {FULL})
    pi, si = {p: i for i, p in enumerate(prompts)}, {s: i for i, s in enumerate(seeds)}
    shape = (len(prompts), len(seeds))
    cube = {a: {m: np.full(shape, np.nan) for m in (*metrics, "dyn_full", "dyn_action")} for a in actions}
    meta = {}
    for cases in by_group.values():
        ref = cases[FULL]
        p, s = pi[int(ref["prompt_id"])], si[int(ref["base_seed"])]
        meta[p] = (ref["family_id"], ref["motion_level"], ref["detail_level"], ref["split"])
        for action in actions:
            cand = cases[action]
            for metric in metrics:
                cube[action][metric][p, s] = float(cand[metric]) - float(ref[metric])
            cube[action]["dyn_full"][p, s] = float(ref["dynamic_degree"])
            cube[action]["dyn_action"][p, s] = float(cand["dynamic_degree"])
    for action in actions:
        for metric in metrics:
            if np.isnan(cube[action][metric]).any():
                raise ValueError(f"incomplete prompt/seed coverage for {action}/{metric}")
    families = sorted({m[0] for m in meta.values()})
    info = {
        "prompts": prompts, "seeds": seeds, "actions": actions, "families": families,
        "family_of": np.array([families.index(meta[p][0]) for p in range(len(prompts))]),
        "motion": np.array([meta[p][1] for p in range(len(prompts))]),
        "detail": np.array([meta[p][2] for p in range(len(prompts))]),
        "dropped_test_rows": dropped,
    }
    return cube, info


def ranks(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


def components(y):
    """Method-of-moments split of a prompts x seeds matrix."""
    n_seed = y.shape[1]
    seed_var = float(np.mean(np.var(y, axis=1, ddof=1)))
    mean_var = float(np.var(y.mean(axis=1), ddof=1))
    prompt_var = mean_var - seed_var / n_seed
    pairs = [(a, b) for a in range(n_seed) for b in range(a + 1, n_seed)]
    rho = float(np.mean([np.corrcoef(ranks(y[:, a]), ranks(y[:, b]))[0, 1] for a, b in pairs]))
    return {
        "seed_var": seed_var,
        "prompt_var": prompt_var,
        "prompt_to_seed_ratio": prompt_var / seed_var if seed_var > 0 else float("inf"),
        "prompt_mean_reliability": prompt_var / mean_var if mean_var > 0 else float("nan"),
        "cross_seed_spearman": rho,
    }


def cell_r2(prompt_means, info, with_family):
    """Share of prompt-mean variance explained by motion/detail cells (and families)."""
    cols = [np.ones(len(prompt_means)), info["motion"] == "high", info["detail"] == "high",
            (info["motion"] == "high") & (info["detail"] == "high")]
    if with_family:
        cols += [info["family_of"] == f for f in range(1, len(info["families"]))]
    x = np.column_stack(cols).astype(np.float64)
    fit = x @ np.linalg.lstsq(x, prompt_means, rcond=None)[0]
    total = np.sum((prompt_means - prompt_means.mean()) ** 2)
    return float(1 - np.sum((prompt_means - fit) ** 2) / total) if total > 0 else float("nan")


def family_contrasts(prompt_means, info):
    """Within-family contrasts; every family holds the full 2x2 so family effects cancel."""
    out = {"detail": [], "motion": [], "interaction": []}
    for f in range(len(info["families"])):
        cell = {}
        for motion, detail in CELLS:
            mask = (info["family_of"] == f) & (info["motion"] == motion) & (info["detail"] == detail)
            cell[motion, detail] = float(prompt_means[mask].mean())
        out["detail"].append((cell["low", "high"] + cell["high", "high"] - cell["low", "low"] - cell["high", "low"]) / 2)
        out["motion"].append((cell["high", "low"] + cell["high", "high"] - cell["low", "low"] - cell["low", "high"]) / 2)
        out["interaction"].append(cell["high", "high"] - cell["high", "low"] - cell["low", "high"] + cell["low", "low"])
    return {k: np.asarray(v) for k, v in out.items()}


def family_bootstrap(y, info, rng):
    """Resample families; return bootstrap draws of ratio and contrasts."""
    n_fam = len(info["families"])
    members = [np.flatnonzero(info["family_of"] == f) for f in range(n_fam)]
    contrasts = family_contrasts(y.mean(axis=1), info)
    draws = defaultdict(list)
    for _ in range(BOOT):
        pick = rng.integers(0, n_fam, size=n_fam)
        idx = np.concatenate([members[f] for f in pick])
        draws["prompt_to_seed_ratio"].append(components(y[idx])["prompt_to_seed_ratio"])
        for name, values in contrasts.items():
            draws[name].append(values[pick].mean())
    return {k: [float(q) for q in np.quantile(v, [0.025, 0.975])] for k, v in draws.items()}


def analyze(cube, info, metrics=METRICS, seed=20261016):
    rng = np.random.default_rng(seed)
    variance, contrasts, tails, motion = [], [], [], []
    for action in info["actions"]:
        for metric in metrics:
            y = cube[action][metric]
            comp = components(y)
            ci = family_bootstrap(y, info, rng)
            prompt_means = y.mean(axis=1)
            variance.append({
                "action": action, "metric": metric, "mean_delta": float(y.mean()),
                **comp, "ratio_ci_low": ci["prompt_to_seed_ratio"][0], "ratio_ci_high": ci["prompt_to_seed_ratio"][1],
                "cell_r2": cell_r2(prompt_means, info, False), "cell_family_r2": cell_r2(prompt_means, info, True),
            })
            for name, values in family_contrasts(prompt_means, info).items():
                contrasts.append({
                    "action": action, "metric": metric, "contrast": name, "mean": float(values.mean()),
                    "ci_low": ci[name][0], "ci_high": ci[name][1],
                    "families_negative": int(np.sum(values < 0)), "families": len(values),
                })
        for metric in metrics:
            y = cube[action][metric]
            for motion_level, detail in CELLS:
                mask = (info["motion"] == motion_level) & (info["detail"] == detail)
                row = {"action": action, "metric": metric, "motion": motion_level, "detail": detail,
                       "videos": int(y[mask].size), "mean_delta": float(y[mask].mean())}
                for tau in TAUS:
                    row[f"p_drop_gt_{tau}"] = float(np.mean(y[mask] < -tau))
                tails.append(row)
        for motion_level in ("low", "high"):
            mask = info["motion"] == motion_level
            full_dyn, act_dyn = cube[action]["dyn_full"][mask], cube[action]["dyn_action"][mask]
            moving = full_dyn > 0.5
            motion.append({
                "action": action, "motion": motion_level,
                "full_dynamic_rate": float(full_dyn.mean()), "action_dynamic_rate": float(act_dyn.mean()),
                "retained_given_full_dynamic": float(act_dyn[moving].mean()) if moving.any() else float("nan"),
                "full_dynamic_videos": int(moving.sum()),
            })
    return {"variance": variance, "contrasts": contrasts, "tails": tails, "motion": motion}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def headline(result, metric="vbench5"):
    lines = [f"| action | mean d{metric} (x100) | prompt/seed var | 95% CI | cross-seed rho | cell R2 | detail (pp) | motion (pp) | worst cell (pp) |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in result["variance"]:
        if row["metric"] != metric:
            continue
        a = row["action"]
        c = {r["contrast"]: r for r in result["contrasts"] if r["action"] == a and r["metric"] == metric}
        worst = min((r for r in result["tails"] if r["action"] == a and r["metric"] == metric), key=lambda r: r["mean_delta"])
        lines.append(
            f"| {a} | {100 * row['mean_delta']:+.2f} | {row['prompt_to_seed_ratio']:.2f} | "
            f"[{row['ratio_ci_low']:.2f}, {row['ratio_ci_high']:.2f}] | {row['cross_seed_spearman']:.2f} | "
            f"{row['cell_r2']:.2f} | {100 * c['detail']['mean']:+.2f} [{100 * c['detail']['ci_low']:+.2f}, {100 * c['detail']['ci_high']:+.2f}] | "
            f"{100 * c['motion']['mean']:+.2f} [{100 * c['motion']['ci_low']:+.2f}, {100 * c['motion']['ci_high']:+.2f}] | "
            f"{100 * worst['mean_delta']:+.2f} ({worst['motion']} motion/{worst['detail']} detail) |"
        )
    return "\n".join(lines)


def run(quality_csv, out_dir, metrics=METRICS):
    cube, info = load(quality_csv, metrics)
    result = analyze(cube, info, metrics)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in result.items():
        write_csv(out / f"{name}.csv", rows)
    summary = {
        "source": str(Path(quality_csv).resolve()),
        "prompts": len(info["prompts"]), "families": len(info["families"]), "seeds": info["seeds"],
        "actions": info["actions"], "dropped_test_rows": info["dropped_test_rows"],
        "bootstrap": f"{BOOT} family-cluster resamples, conditioned on observed seeds",
        "headline": {metric: headline(result, metric) for metric in metrics},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return result, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("quality_csv", help="metrics/controlled_factor_vbench/quality_by_video.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    _, summary = run(args.quality_csv, args.out)
    print(f"{summary['prompts']} prompts / {summary['families']} families / seeds {summary['seeds']}; "
          f"dropped {summary['dropped_test_rows']} locked test rows")
    print(summary["headline"]["vbench5"])


if __name__ == "__main__":
    main()
