"""Read-only development audit of prompt versus seed decision value.

No test score is converted or included. Fixed policies and latency profiles are
fit on train; reported policy performance is validation-only. Cross-seed is a
two-seed selector evaluated on the third seed, NOT a population prompt oracle.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import canonical_sha256, sha256_file, write_json_atomic  # noqa: E402
from UNIV_adaptor.scripts.data.score_controlled_factor_dataset import (  # noqa: E402
    ACTIONS,
    ANALYSIS_SCHEMA,
    FULL,
    csv_write,
    validate_hashed,
)
from UNIV_adaptor.scripts.data.score_phase2_dataset import output_lock  # noqa: E402

NAMES = (FULL, *ACTIONS)
SETS = {"ST": (1, 2), "FST": (0, 1, 2), "FSTC": (0, 1, 2, 3)}
SEEDS = (42, 100, 2024)


def load_cube(path):
    """Validate complete matched prompt/seed/action coverage before aggregation."""
    metadata, observed, references = {}, {}, {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                continue
            if row["split"] not in {"train", "validation"}:
                raise ValueError("Unknown split")
            p, seed, action = int(row["prompt_id"]), int(row["base_seed"]), row["action_id"]
            if seed not in SEEDS or action not in ACTIONS:
                raise ValueError("Unexpected seed/action")
            if int(row["seed"]) != seed + p:
                raise ValueError("Actual seed does not match base_seed + prompt_id")
            identity = {k: row[k] for k in ("split", "family_id", "prompt", "prompt_sha256")}
            if canonical_sha256(row["prompt"]) != row["prompt_sha256"]:
                raise ValueError("Prompt hash mismatch")
            if p in metadata and identity != metadata[p]:
                raise ValueError("Prompt metadata changed across rows")
            metadata[p] = identity
            key = (p, seed, action)
            if key in observed:
                raise ValueError("Duplicate prompt/seed/action")
            q0, qa, delta, t0, ta, ratio = [float(row[k]) for k in (
                "full_vbench5", "action_vbench5", "delta_vbench5",
                "full_seconds", "action_seconds", "time_ratio_to_full",
            )]
            if not np.isfinite([q0, qa, delta, t0, ta, ratio]).all() or min(t0, ta) <= 0:
                raise ValueError("Non-finite scores or invalid times")
            if not np.isclose(qa-q0, delta, atol=1e-10, rtol=0) or not np.isclose(ta/t0, ratio):
                raise ValueError("Inconsistent relative quality/time")
            ref = (q0, t0)
            if (p, seed) in references and references[p, seed] != ref:
                raise ValueError("FULL reference differs across matched actions")
            references[p, seed] = ref
            observed[key] = (delta, ratio)
    ids = sorted(metadata)
    if not ids:
        raise ValueError("No development data")
    families = {}
    for p in ids:
        family, split = metadata[p]["family_id"], metadata[p]["split"]
        if family in families and families[family] != split:
            raise ValueError("Family leakage between train and validation")
        families[family] = split
    q, costs = np.zeros((len(ids), 3, 4)), np.ones((len(ids), 3, 4))
    for i, p in enumerate(ids):
        for s, seed in enumerate(SEEDS):
            for a, action in enumerate(ACTIONS, 1):
                if (p, seed, action) not in observed:
                    raise ValueError(f"Missing action/seed: {p}/{seed}/{action}")
                q[i, s, a], costs[i, s, a] = observed[p, seed, action]
    meta = [{"prompt_id": p, **metadata[p]} for p in ids]
    if {r["split"] for r in meta} != {"train", "validation"}:
        raise ValueError("Both train and validation required")
    return meta, q, costs


def oracle_values(u, fixed):
    """u[P,S,A]; tie-breaking uses declared action order."""
    p, s, _ = u.shape
    pi, si = np.arange(p)[:, None], np.arange(s)[None, :]
    mean_choice = np.argmax(u.mean(axis=1), axis=1)
    instance_choice = np.argmax(u, axis=2)
    cross_choice = np.empty((p, s), dtype=int)
    for held in range(s):
        # Explicitly exclude held-out scores from the selector.
        cross_choice[:, held] = np.argmax(u[:, np.arange(s) != held].mean(axis=1), axis=1)
    values = {
        "fixed_train": u[:, :, fixed],
        "prompt_oracle_in_sample": u[pi, si, mean_choice[:, None]],
        "instance_oracle_in_sample": u.max(axis=2),
        "cross_seed_selector": u[pi, si, cross_choice],
    }
    return values, mean_choice, instance_choice, cross_choice


def interval(values, meta, repetitions, seed):
    """Resample entire semantic families, retaining all prompts and seeds."""
    groups = sorted({r["family_id"] for r in meta})
    arrays = [np.asarray(values)[[i for i, r in enumerate(meta) if r["family_id"] == f]] for f in groups]
    sums = np.array([x.sum() for x in arrays])
    sizes = np.array([x.size for x in arrays])
    draws = np.random.default_rng(seed).integers(len(groups), size=(repetitions, len(groups)))
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def pair_diagnostics(d):
    """Random-intercept moment estimates; negative ICC retained as diagnostic."""
    p, s = d.shape
    within = float(np.var(d, axis=1, ddof=1).mean())
    between_ms = float(s * np.var(d.mean(axis=1), ddof=1)) if p > 1 else 0.0
    denominator = between_ms + (s-1)*within
    return {
        "mean_difference": float(d.mean()),
        "within_prompt_seed_variance": within,
        "prompt_mean_variance": float(np.var(d.mean(axis=1), ddof=1)) if p > 1 else 0.0,
        "prompt_variance_moment_unclipped": (between_ms-within)/s,
        "icc_unclipped": (between_ms-within)/denominator if denominator else None,
        "sign_flip_prompt_fraction": float(np.mean((d.min(axis=1) < 0) & (d.max(axis=1) > 0))),
    }


def run(args):
    scored = Path(args.scored_dir).resolve()
    analysis_path = scored / "analysis.json"
    source = scored / "relative_to_full.csv"
    if not analysis_path.is_file():
        raise FileNotFoundError(
            f"Controlled-factor analysis is missing: {analysis_path}; "
            "set CONTROLLED_FACTOR_ROOT or CONTROLLED_FACTOR_SCORED_DIR"
        )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    validate_hashed(analysis, ANALYSIS_SCHEMA, "analysis_sha256")
    if not source.is_file():
        raise FileNotFoundError(f"Need per-seed scores: {source}; prompt means cannot recover seed variation")
    meta, q, cost = load_cube(source)
    tr = np.array([r["split"] == "train" for r in meta])
    va = ~tr
    vm = [r for r in meta if r["split"] == "validation"]
    profile = cost[tr].mean(axis=(0, 1))
    summary, details, pairs, margins, held_seed_rows = [], [], [], [], []
    for label, subset in SETS.items():
        # ST is equal-latency quality-only; FST/FSTC use the declared lambdas.
        for lam in ([0.] if label == "ST" else args.lambdas):
            u = q[:, :, subset] - lam*(profile[list(subset)]-1)
            fixed = int(np.argmax(u[tr].mean(axis=(0, 1))))
            values, pm, ins, cross = oracle_values(u[va], fixed)
            base = values["fixed_train"]
            for name, value in values.items():
                delta = value-base
                for s, held_seed in enumerate(SEEDS):
                    held_seed_rows.append({
                        "action_set": label, "lambda": lam, "policy": name,
                        "held_base_seed": held_seed, "mean_utility": float(value[:, s].mean()),
                        "gain_over_fixed": float(delta[:, s].mean()),
                    })
                lo, hi = interval(delta.mean(axis=1), vm, args.bootstrap, args.seed)
                summary.append({
                    "action_set": label, "lambda": lam, "policy": name,
                    "fixed_action": NAMES[subset[fixed]], "validation_prompts": len(vm),
                    "families": len({r["family_id"] for r in vm}),
                    "mean_utility": float(value.mean()), "gain_over_fixed": float(delta.mean()),
                    "family_ci_low": lo, "family_ci_high": hi,
                    "instance_regret": float((values["instance_oracle_in_sample"]-value).mean()),
                    "loss_vs_fixed_gt_epsilon_rate": float(np.mean(delta < -args.epsilon)),
                })
            sorted_u = np.sort(u[va], axis=2)
            margin = sorted_u[:, :, -1]-sorted_u[:, :, -2]
            flip = np.any(ins != ins[:, :1], axis=1)
            for i, r in enumerate(vm):
                for s, seed in enumerate(SEEDS):
                    details.append({
                        "action_set": label, "lambda": lam, "prompt_id": r["prompt_id"],
                        "family_id": r["family_id"], "base_seed": seed,
                        "prompt_oracle_action": NAMES[subset[pm[i]]],
                        "instance_oracle_action": NAMES[subset[ins[i, s]]],
                        "cross_seed_action": NAMES[subset[cross[i, s]]],
                        **{name: float(v[i, s]) for name, v in values.items()},
                        "top1_top2_margin": float(margin[i, s]),
                        "cross_seed_regret": float(values["instance_oracle_in_sample"][i, s]-values["cross_seed_selector"][i, s]),
                    })
            for condition, mask in (("all", np.ones(len(vm), bool)), ("winner_changes", flip)):
                x = margin[mask].ravel()
                margins.append({
                    "action_set": label, "lambda": lam, "condition": condition,
                    "prompt_count": int(mask.sum()), "winner_changes_fraction": float(flip.mean()),
                    "median_margin": float(np.median(x)) if len(x) else None,
                    "p90_margin": float(np.quantile(x, .9)) if len(x) else None,
                    "margin_le_epsilon_fraction": float(np.mean(x <= args.epsilon)) if len(x) else None,
                })
    # Pairwise differences cancel FULL and describe what actually changes rankings.
    for split, mask in (("train", tr), ("validation", va)):
        for a, b in itertools.combinations(range(4), 2):
            pairs.append({"split": split, "action_a": NAMES[a], "action_b": NAMES[b],
                          **pair_diagnostics(q[mask, :, a]-q[mask, :, b])})
    learned = []
    for feature in ("t5", "tfidf"):
        directory = scored / f"prompt_prior_{feature}"
        if not (directory / "predictions.csv").is_file():
            continue
        request = json.loads((directory / "training_request.json").read_text(encoding="utf-8"))
        analysis = json.loads((scored / "analysis.json").read_text(encoding="utf-8"))
        if request["input_sha256"] != analysis["input_sha256"]:
            raise ValueError("Learned policy dataset mismatch")
        with (directory / "predictions.csv").open(encoding="utf-8", newline="") as handle:
            pr = {int(r["prompt_id"]): r for r in csv.DictReader(handle) if r["split"] == "validation_holdout"}
        prediction = np.zeros((len(vm), 4))
        for i, row in enumerate(vm):
            r = pr[row["prompt_id"]]
            if r["prompt"] != row["prompt"]:
                raise ValueError("Learned prediction prompt mismatch")
            for a, action in enumerate(ACTIONS, 1):
                prediction[i, a] = float(r[f"predicted_delta__{action.lower()}"])
                if not np.isclose(float(r[f"target_delta__{action.lower()}"]), q[va][i, :, a].mean()):
                    raise ValueError("Learned targets differ from per-seed data")
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite learned predictions")
        for label, subset in SETS.items():
            for lam in ([0.] if label == "ST" else args.lambdas):
                u = q[:, :, subset]-lam*(profile[list(subset)]-1)
                fixed = int(np.argmax(u[tr].mean(axis=(0, 1))))
                chosen = np.argmax(prediction[:, subset]-lam*(profile[list(subset)]-1), axis=1)
                v = u[va][np.arange(len(vm))[:, None], np.arange(3)[None, :], chosen[:, None]]
                delta = (v-u[va, :, fixed]).mean(axis=1)
                lo, hi = interval(delta, vm, args.bootstrap, args.seed)
                learned.append({"feature": feature, "action_set": label, "lambda": lam,
                                "mean_utility": float(v.mean()), "gain_over_fixed": float(delta.mean()),
                                "family_ci_low": lo, "family_ci_high": hi})
    out = Path(args.out_dir).resolve() if args.out_dir else scored / "seed_value_audit"
    out.mkdir(parents=True, exist_ok=True)
    request = {"analysis_sha256": analysis["analysis_sha256"],
               "source_csv_sha256": sha256_file(source), "script_sha256": sha256_file(__file__),
               "lambdas": args.lambdas, "epsilon": args.epsilon, "bootstrap": args.bootstrap, "seed": args.seed}
    with output_lock(out):
        previous = out / "audit_request.json"
        if previous.exists() and json.loads(previous.read_text(encoding="utf-8")) != request:
            raise ValueError("Audit inputs changed; choose a new --out-dir")
        write_json_atomic(previous, request)
        for name, rows in (("oracle_summary", summary), ("decisions_by_seed", details),
                           ("pair_variance", pairs), ("margin_summary", margins),
                           ("policy_by_held_seed", held_seed_rows), ("learned_summary", learned)):
            if rows:
                csv_write(out / f"{name}.csv", rows)
        timing_gap = abs(profile[1]-profile[2])/np.mean(profile[1:3])
        result = {**request, "test_scores_analyzed": False, "train_time_ratio_profile": profile.tolist(),
                  "st_train_latency_relative_gap": float(timing_gap), "st_matched_within_5pct": bool(timing_gap <= .05),
                  "oracle_summary": summary, "learned_summary": learned,
                  "policy_by_held_seed": held_seed_rows,
                  "caveats": ["In-sample oracles have selection optimism.",
                              "Cross-seed selector uses two observed seeds, not prompt text alone.",
                              "Family bootstrap is conditional on the three observed seeds; only four validation families.",
                              "Pair variance includes seed-by-configuration effects and metric limitations, not identified pure noise.",
                              "Utility uses train-calibrated costs, not per-instance measured costs; router overhead excluded."]}
        write_json_atomic(out / "audit.json", result)
        lines = ["# Prompt and seed decision-value audit", "", "Validation only; fixed action and costs fitted on train.", "",
                 "| Candidates | Lambda | Policy | Gain over fixed | Family CI |", "|---|---:|---|---:|---|"]
        for r in summary:
            lines.append(f"| {r['action_set']} | {r['lambda']:g} | {r['policy']} | {r['gain_over_fixed']:+.6f} | [{r['family_ci_low']:+.6f}, {r['family_ci_high']:+.6f}] |")
        lines.extend(["", *result["caveats"]])
        (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(out / "report.md")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scored-dir", required=True)
    p.add_argument("--out-dir")
    p.add_argument("--lambdas", nargs="+", type=float, default=[0., .01, .03, .05, .08, .1])
    p.add_argument("--epsilon", type=float, default=.001)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260922)
    args = p.parse_args()
    if args.bootstrap < 100 or not np.isfinite([args.epsilon, *args.lambdas]).all() or min(args.epsilon, *args.lambdas) < 0:
        p.error("Invalid bootstrap/lambda/epsilon")
    run(args)


if __name__ == "__main__":
    main()
