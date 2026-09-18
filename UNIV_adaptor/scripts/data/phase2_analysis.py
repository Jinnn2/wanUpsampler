"""CPU-only paired analyses for the native + five-action Phase 2 experiment."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from itertools import combinations
import json
import math
import random
import statistics as st

from UNIV_adaptor.data_protocol import write_json_atomic

NATIVE = "native_hr50"
QUALITY_DIMENSIONS = ("subject_consistency", "background_consistency", "motion_smoothness",
                      "aesthetic_quality", "imaging_quality")
DIAGNOSTICS = ("dynamic_degree", "overall_consistency")
DIMENSIONS = QUALITY_DIMENSIONS + DIAGNOSTICS


def csv_write(path, rows, fields=None):
    fields = list(rows[0]) if rows else fields
    if not fields:
        raise ValueError(f"no columns for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def mean(values):
    values = list(values)
    return st.fmean(values) if values else None


def bootstrap_ci(values, repetitions=2000):
    """Paired prompt bootstrap; descriptive uncertainty, not independent seed CI."""
    if len(values) < 2:
        return None, None
    rng = random.Random(20260918)
    samples = sorted(st.fmean(rng.choices(values, k=len(values))) for _ in range(repetitions))
    return samples[int(.025 * repetitions)], samples[int(.975 * repetitions)]


def group(rows, fields):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[f] for f in fields)].append(row)
    return result


def validate_rows(rows):
    if not rows:
        raise ValueError("no rows")
    stems = [r["stem"] for r in rows]
    if len(set(stems)) != len(stems):
        raise ValueError("duplicate video stem")
    actions = {r["action_id"] for r in rows}
    if NATIVE not in actions or len(actions) != 6:
        raise ValueError("expected native and five candidates")
    owners = {}
    seeds_by_split = defaultdict(set)
    for (split, pid), prompt_rows in group(rows, ("split", "prompt_id")).items():
        texts = {r["prompt"] for r in prompt_rows}
        if len(texts) != 1:
            raise ValueError("inconsistent prompt text")
        text = next(iter(texts))
        if text in owners and owners[text] != (split, pid):
            raise ValueError("prompt overlap across identities/splits")
        owners[text] = (split, pid)
        base_seeds = set()
        for _, seed_rows in group(prompt_rows, ("seed",)).items():
            if len(seed_rows) != 6 or {r["action_id"] for r in seed_rows} != actions:
                raise ValueError("incomplete prompt/seed action pairing")
            bases = {r["base_seed"] for r in seed_rows}
            if len(bases) != 1:
                raise ValueError("inconsistent base seed")
            base_seeds.update(bases)
        if len(base_seeds) != len({r["seed"] for r in prompt_rows}):
            raise ValueError("duplicate base seed")
        seeds_by_split[split].add(tuple(sorted(base_seeds)))
    if set(seeds_by_split) != {"train", "validation"} or any(len(s) != 1 for s in seeds_by_split.values()):
        raise ValueError("inconsistent split/seed coverage")
    if len(next(iter(seeds_by_split["validation"]))) < 2:
        raise ValueError("validation needs multiple seeds for stability")
    for r in rows:
        if not math.isfinite(r["pipeline_seconds"]) or r["pipeline_seconds"] <= 0:
            raise ValueError("invalid pipeline_seconds")


def runtime_report(rows, out):
    validate_rows(rows)
    paired = group(rows, ("split", "prompt_id", "seed"))
    times = []
    for records in paired.values():
        native = next(r["pipeline_seconds"] for r in records if r["action_id"] == NATIVE)
        for row in records:
            times.append({k: row[k] for k in ("split", "prompt_id", "seed", "action_id", "pipeline_seconds")} |
                         {"native_seconds": native, "native_speedup": native / row["pipeline_seconds"],
                          "native_cost_ratio": row["pipeline_seconds"] / native})
    csv_write(out / "runtime_by_video.csv", times)
    summary = []
    for (split, action), rs in sorted(group(times, ("split", "action_id")).items()):
        prompts = list(group(rs, ("prompt_id",)).values())
        summary.append(dict(split=split, action_id=action, videos=len(rs), prompts=len(prompts),
                            pipeline_seconds_mean=mean(mean(r["pipeline_seconds"] for r in p) for p in prompts),
                            pipeline_seconds_median=st.median(r["pipeline_seconds"] for r in rs),
                            paired_speedup_mean=mean(mean(r["native_speedup"] for r in p) for p in prompts),
                            paired_speedup_median=st.median(r["native_speedup"] for r in rs)))
    csv_write(out / "runtime_summary.csv", summary)
    catalog = {r["action_id"]: {k: r.get(k) for k in ("requested_action", "resolved_schedule", "transition", "proxy_compute_density")}
               for r in rows}
    write_json_atomic(out / "action_catalog.json", catalog)
    return summary


def enrich(rows, scores):
    validate_rows(rows)
    if set(scores) != {r["stem"] for r in rows}:
        raise ValueError("score coverage does not exactly match videos")
    result = []
    for row in rows:
        values = {d: float(scores[row["stem"]][d]) for d in DIMENSIONS}
        if any(not math.isfinite(v) or not (-1 if d == "overall_consistency" else 0) <= v <= 1
               for d, v in values.items()):
            raise ValueError("scores must be finite and in the VBench dimension range")
        result.append({**row, **values, "vbench5": mean(values[d] for d in QUALITY_DIMENSIONS)})
    for rs in group(result, ("split", "prompt_id", "seed")).values():
        native = next(r for r in rs if r["action_id"] == NATIVE)
        for r in rs:
            r["native_speedup"] = native["pipeline_seconds"] / r["pipeline_seconds"]
            r["native_cost_ratio"] = r["pipeline_seconds"] / native["pipeline_seconds"]
            for field in ("vbench5", *DIMENSIONS):
                r["delta_native_" + field] = r[field] - native[field]
    return result


def aggregate_prompts(rows):
    fields = ("pipeline_seconds", "vbench5", "native_speedup", "native_cost_ratio", *DIMENSIONS,
              *("delta_native_" + d for d in ("vbench5", *DIMENSIONS)))
    return [dict(split=k[0], prompt_id=k[1], prompt=rs[0]["prompt"], action_id=k[2], seeds=len(rs),
                 **{f: mean(r[f] for r in rs) for f in fields})
            for k, rs in sorted(group(rows, ("split", "prompt_id", "action_id")).items())]


def winner(rs):
    # Deterministic tie break: quality, then lower measured cost, then action id.
    return min(rs, key=lambda r: (-r["vbench5"], r["pipeline_seconds"], r["action_id"]))


def preference_reports(rows, prompts, out, epsilon):
    preferences, pairs, stability = [], [], []
    train = [r for r in prompts if r["split"] == "train" and r["action_id"] != NATIVE]
    train_global = winner([dict(action_id=k[0], vbench5=mean(r["vbench5"] for r in rs),
                                pipeline_seconds=mean(r["pipeline_seconds"] for r in rs))
                           for k, rs in group(train, ("action_id",)).items()])["action_id"]
    by_video = group(rows, ("split", "prompt_id"))
    for (split, pid), rs in sorted(group(prompts, ("split", "prompt_id")).items()):
        cs = [r for r in rs if r["action_id"] != NATIVE]
        ordered = sorted(cs, key=lambda r: (-r["vbench5"], r["pipeline_seconds"], r["action_id"]))
        preferences.append(dict(split=split, prompt_id=pid, prompt=rs[0]["prompt"],
                                best_action=ordered[0]["action_id"], margin_top2=ordered[0]["vbench5"]-ordered[1]["vbench5"],
                                top_tie_count=sum(ordered[0]["vbench5"]-r["vbench5"] <= epsilon for r in cs),
                                ranking=[r["action_id"] for r in ordered]))
        seed_rows = by_video[(split, pid)]
        lookup = {(r["seed"], r["action_id"]): r for r in seed_rows}
        seeds = sorted({r["seed"] for r in seed_rows})
        for a, b in combinations(sorted(cs, key=lambda r: r["action_id"]), 2):
            diffs = [lookup[(s, b["action_id"])]["vbench5"]-lookup[(s, a["action_id"])]["vbench5"] for s in seeds]
            signs = [1 if d > epsilon else -1 if d < -epsilon else 0 for d in diffs]
            pairs.append(dict(split=split, prompt_id=pid, action_a=a["action_id"], action_b=b["action_id"],
                              delta_vbench5=b["vbench5"]-a["vbench5"], delta_seconds=b["pipeline_seconds"]-a["pipeline_seconds"],
                              seeds=len(seeds), b_wins=sum(s == 1 for s in signs), a_wins=sum(s == -1 for s in signs),
                              ties=sum(s == 0 for s in signs), all_seed_same_strict_preference=len(seeds)>1 and len(set(signs))==1 and signs[0]!=0))
        if len(seeds) < 2:
            continue
        winners = []
        loso_gains, loso_regrets = [], []
        for seed in seeds:
            held = [lookup[(seed, r["action_id"])] for r in cs]
            winners.append(winner(held)["action_id"])
            other = [dict(action_id=r["action_id"], vbench5=mean(lookup[(s,r["action_id"])]["vbench5"] for s in seeds if s!=seed),
                          pipeline_seconds=mean(lookup[(s,r["action_id"])]["pipeline_seconds"] for s in seeds if s!=seed)) for r in cs]
            chosen = lookup[(seed, winner(other)["action_id"])]["vbench5"]
            loso_gains.append(chosen-lookup[(seed,train_global)]["vbench5"])
            loso_regrets.append(winner(held)["vbench5"]-chosen)
        stability.append(dict(split=split, prompt_id=pid, seeds=len(seeds), seed_winners=winners,
                              unanimous_winner=len(set(winners))==1,
                              pairwise_winner_agreement=mean(a==b for a,b in combinations(winners,2)),
                              train_global_action=train_global, loso_gain_vs_train_global=mean(loso_gains),
                              loso_regret_vs_seed_oracle=mean(loso_regrets)))
    csv_write(out / "prompt_preferences.csv", preferences)
    csv_write(out / "pairwise_preferences.csv", pairs)
    csv_write(out / "seed_stability.csv", stability)
    return stability


def budget_analysis(prompts, budgets=None):
    candidates = [r for r in prompts if r["action_id"] != NATIVE]
    train = [r for r in candidates if r["split"] == "train"]
    calibration = [dict(action_id=k[0], pipeline_seconds=mean(r["pipeline_seconds"] for r in rs),
                        vbench5=mean(r["vbench5"] for r in rs)) for k, rs in sorted(group(train, ("action_id",)).items())]
    budgets = sorted(set(budgets if budgets is not None else [r["pipeline_seconds"] for r in calibration]))
    summary, details = [], []
    native_train = mean(r["pipeline_seconds"] for r in prompts if r["split"]=="train" and r["action_id"]==NATIVE)
    for cap in budgets:
        eligible = [r for r in calibration if r["pipeline_seconds"] <= cap + 1e-9]
        fixed = winner(eligible)["action_id"] if eligible else None
        # Training eligibility freezes an action set before validation inspection.
        eligible_ids = {r["action_id"] for r in eligible}
        for split in ("train", "validation"):
            subset = [r for r in candidates if r["split"] == split]
            entries = []
            for (pid,), rs in sorted(group(subset, ("prompt_id",)).items()):
                fixed_row = next((r for r in rs if r["action_id"] == fixed), None)
                feasible = [r for r in rs if r["action_id"] in eligible_ids and r["pipeline_seconds"] <= cap + 1e-9]
                oracle = winner(feasible) if feasible else None
                fixed_ok = fixed_row is not None and fixed_row["pipeline_seconds"] <= cap + 1e-9
                paired = fixed_ok and oracle is not None
                row = dict(split=split, budget_seconds=cap, prompt_id=pid, fixed_action=fixed,
                           eligible_actions=sorted(eligible_ids), fixed_feasible=fixed_ok, oracle_feasible=oracle is not None,
                           oracle_action=oracle["action_id"] if oracle else None,
                           fixed_seconds=fixed_row["pipeline_seconds"] if fixed_row else None,
                           oracle_seconds=oracle["pipeline_seconds"] if oracle else None,
                           fixed_vbench5=fixed_row["vbench5"] if fixed_row else None,
                           oracle_vbench5=oracle["vbench5"] if oracle else None,
                           paired=paired, paired_gain=oracle["vbench5"]-fixed_row["vbench5"] if paired else None)
                entries.append(row)
            common = [r for r in entries if r["paired"]]
            gains = [r["paired_gain"] for r in common]
            low, high = bootstrap_ci(gains)
            summary.append(dict(split=split, budget_seconds=cap, budget_over_train_native=native_train and cap/native_train,
                                fixed_action=fixed, eligible_actions=sorted(eligible_ids), prompts=len(entries),
                                fixed_violation_fraction=mean(not r["fixed_feasible"] for r in entries) if fixed else None,
                                oracle_coverage=mean(r["oracle_feasible"] for r in entries),
                                paired_prompts=len(common), paired_coverage=len(common)/len(entries),
                                fixed_quality_on_paired=mean(r["fixed_vbench5"] for r in common),
                                oracle_quality_on_paired=mean(r["oracle_vbench5"] for r in common),
                                paired_gain=mean(gains), gain_ci_low=low, gain_ci_high=high,
                                fixed_seconds_on_paired=mean(r["fixed_seconds"] for r in common),
                                oracle_seconds_on_paired=mean(r["oracle_seconds"] for r in common),
                                oracle_action_counts=dict(Counter(r["oracle_action"] for r in common))))
            details.extend(entries)
    return calibration, summary, details


def report(rows, scores, out, *, budgets=None, epsilon=.001):
    enriched = enrich(rows, scores)
    prompts = aggregate_prompts(enriched)
    csv_write(out / "quality_by_video.csv", enriched)
    csv_write(out / "quality_by_prompt.csv", prompts)
    summary = []
    fields = ("pipeline_seconds", "vbench5", "native_speedup", "native_cost_ratio", *DIMENSIONS,
              *("delta_native_" + d for d in ("vbench5", *DIMENSIONS)))
    for (split, action), rs in sorted(group(prompts, ("split", "action_id")).items()):
        summary.append(dict(split=split, action_id=action, prompts=len(rs), **{f:mean(r[f] for r in rs) for f in fields}))
    csv_write(out / "quality_cost_summary.csv", summary)
    stability = preference_reports(enriched, prompts, out, epsilon)
    calibration, budget_summary, details = budget_analysis(prompts, budgets)
    csv_write(out / "train_calibration.csv", calibration)
    csv_write(out / "budget_oracle_summary.csv", budget_summary)
    csv_write(out / "budget_oracle_by_prompt.csv", details)
    write_json_atomic(out / "analysis_settings.json", {"quality_dimensions": list(QUALITY_DIMENSIONS),
                      "diagnostics": list(DIAGNOSTICS), "tie_epsilon": epsilon,
                      "budget_source": "explicit_seconds" if budgets is not None else "train_candidate_mean_seconds",
                      "budgets_seconds": sorted({r["budget_seconds"] for r in budget_summary}),
                      "cost_unit": "mean_pipeline_seconds_over_seeds_per_prompt", "bootstrap_seed": 20260918,
                      "bootstrap_repetitions": 2000})
    fmt = lambda x: "NA" if x is None else f"{x:.5f}"
    lines = ["# Phase 2 quality, cost and prompt preferences", "",
             "VBench5 is the arithmetic mean of five dimensions, NOT the official VBench total. Diagnostics are excluded.",
             "Seeds are averaged within prompt. Validation sample size is the number of prompts, not prompt/seed groups.",
             "Native HR50 is a paired generation baseline, not pixel ground truth; native_fidelity is not estimated.",
             "Pipeline latency is observed generation time, not a separately warmed/calibrated benchmark.",
             "P2 SKIP retains solver steps and reuses cache predictions; it is not independent timestep skipping.",
             "Action labels are identifiers, not measured B25/B30 tiers. HR switch steps and noise paths also differ.",
             "This evaluates Phase 2 alone. Do not pool Phase 1 validation with overlapping Phase 2 training prompts.", "",
             "| Split | Action | Prompts | Seconds | Paired speedup | VBench5 | Delta native |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['split']} | {r['action_id']} | {r['prompts']} | {r['pipeline_seconds']:.2f} | {r['native_speedup']:.3f} | {r['vbench5']:.5f} | {r['delta_native_vbench5']:.5f} |")
    lines += ["", "## Preference stability (unconstrained candidates)", "",
              f"Quality tie threshold: {epsilon}. Strict winners use deterministic tie-breaking; consult top_tie_count and pairwise ties.",
              "LOSO selects an action using the other seeds of the same prompt, then scores the held-out seed. It is not a prompt-only predictor."]
    for split in sorted({r["split"] for r in stability}):
        rs = [r for r in stability if r["split"]==split]
        lines += [f"- {split}: {len(rs)} prompts; unanimous winner {mean(r['unanimous_winner'] for r in rs):.1%}; "
                  f"seed-pair winner agreement {mean(r['pairwise_winner_agreement'] for r in rs):.1%}; "
                  f"LOSO quality gain vs train global action {mean(r['loso_gain_vs_train_global'] for r in rs):.5f}."]
    lines += ["", "## Empirical cost-constrained oracle", "",
              "Default budget knots and action costs are train prompt-mean measured seconds. Fixed action maximizes train quality among eligible actions.",
              "At each budget, eligibility is frozen from train costs. Each prompt must additionally satisfy its observed seed-mean latency cap.",
              "This is a cap on the mean across seeds, NOT a per-video hard latency guarantee. Native is excluded from the five-action selector.",
              "Oracle is a hindsight quality upper bound within that action set. Gains compare the SAME prompts where both fixed and oracle are feasible.",
              "Coverage/violations are mandatory: a conditional gain at low coverage is not a full-dataset improvement. NA means no feasible comparison.",
              "95% paired prompt bootstrap intervals describe this hindsight gap; they do not correct oracle selection bias or prove a learned controller gain.", "",
              "| Split | Cap seconds | Fixed action | Paired/total | Oracle coverage | Fixed violation | Gain | 95% interval |",
              "|---|---:|---|---:|---:|---:|---:|---|"]
    for r in budget_summary:
        lines.append(f"| {r['split']} | {r['budget_seconds']:.2f} | {r['fixed_action'] or 'NA'} | {r['paired_prompts']}/{r['prompts']} | {r['oracle_coverage']:.1%} | {fmt(r['fixed_violation_fraction'])} | {fmt(r['paired_gain'])} | {fmt(r['gain_ci_low'])}, {fmt(r['gain_ci_high'])} |")
    lines += ["", "Next decision: inspect gain magnitude, coverage, semantic diagnostics, and LOSO stability together before training a prompt allocator.",
              "These artifacts are analysis-ready. Raw generation records remain generated_unscored; no fabricated native_fidelity or formal training-ready schema is written."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
