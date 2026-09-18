"""Offline fixed-lambda prompt -> utility -> action development experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    write_json_atomic,
)
from UNIV_adaptor.scripts.data.phase2_analysis import (  # noqa: E402
    DIMENSIONS,
    NATIVE,
    bootstrap_ci,
    csv_write,
)
from UNIV_adaptor.scripts.data.score_phase2_dataset import output_lock  # noqa: E402
from UNIV_adaptor.scripts.router.phase2_gain_model import (  # noqa: E402
    normalize,
    text_features,
)
from UNIV_adaptor.scripts.router.phase2_utility_model import (  # noqa: E402
    choose_actions,
    cross_validate_utility,
    predict_utility_gains,
    utility_matrix,
)
from UNIV_adaptor.scripts.router.train_phase2_gain_prior import (  # noqa: E402
    ACTION_SETS,
    extract_t5,
    load_data,
    load_t5,
)


def train_latency_profile(rows, actions):
    """Build a train-only paired median action/native latency profile."""
    train = [row for row in rows if row["split"] == "train"]
    by_case = {}
    for row in train:
        key = (int(row["prompt_id"]), int(row["seed"]))
        by_case.setdefault(key, {})[row["action_id"]] = float(row["pipeline_seconds"])
    expected = {NATIVE, *actions}
    ratios = {action: [] for action in actions}
    seconds = {action: [] for action in actions}
    native_seconds = []
    for key, values in by_case.items():
        if not expected.issubset(values):
            raise ValueError(f"incomplete train latency pairing for {key}")
        native = values[NATIVE]
        native_seconds.append(native)
        for action in actions:
            seconds[action].append(values[action])
            ratios[action].append(values[action] / native)
    if not by_case:
        raise ValueError("no train latency pairs")
    normalized = np.asarray(
        [float(np.median(ratios[action])) for action in actions], dtype=np.float64
    )
    body = {
        "schema": "phase2_train_paired_median_latency_profile_v1",
        "source_split": "train",
        "pair_count": len(by_case),
        "native_median_seconds": float(np.median(native_seconds)),
        "action_median_seconds": {
            action: float(np.median(seconds[action])) for action in actions
        },
        "normalized_action_cost": {
            action: float(normalized[index]) for index, action in enumerate(actions)
        },
        "normalization": "median_of_paired_action_over_native_pipeline_seconds",
    }
    return normalized, {**body, "profile_sha256": canonical_sha256(body)}


def oracle_rows(samples, actions, normalized_cost, utility_lambda):
    rows = []
    for sample in samples:
        quality = np.asarray(sample["quality"], dtype=np.float64)[None, :]
        utility = utility_matrix(quality, normalized_cost, utility_lambda)[0]
        order = sorted(
            range(len(actions)),
            key=lambda index: (-utility[index], normalized_cost[index], index),
        )
        rows.append(
            {
                "split": sample["split"],
                "prompt_id": sample["prompt_id"],
                "prompt": sample["prompt"],
                "seed_count": len(sample["seed_seconds"][0]),
                "utility_lambda": utility_lambda,
                "action_ids": actions,
                "quality": quality[0].tolist(),
                "normalized_cost": normalized_cost.tolist(),
                "utility": utility.tolist(),
                "oracle_action": actions[order[0]],
                "runner_up_action": actions[order[1]],
                "oracle_margin": float(utility[order[0]] - utility[order[1]]),
            }
        )
    return rows


def lambda_diagnostics(samples, actions, normalized_cost, lambdas, active_lambda):
    """Describe train-only oracle geometry without using validation labels."""
    train = [sample for sample in samples if sample["split"] == "train"]
    quality = np.asarray([sample["quality"] for sample in train], dtype=np.float64)
    rows = []
    for utility_lambda in sorted(set(float(value) for value in lambdas)):
        utility = utility_matrix(quality, normalized_cost, utility_lambda)
        oracle = choose_actions(utility, normalized_cost)
        mean_utility = utility.mean(axis=0)
        fixed = min(
            range(len(actions)),
            key=lambda index: (-mean_utility[index], normalized_cost[index], index),
        )
        order = np.argsort(utility, axis=1)
        margins = (
            utility[np.arange(len(train)), order[:, -1]]
            - utility[np.arange(len(train)), order[:, -2]]
        )
        oracle_utility = utility[np.arange(len(train)), oracle]
        fixed_utility = utility[:, fixed]
        rows.append(
            {
                "utility_lambda": utility_lambda,
                "active_run_lambda": math.isclose(
                    utility_lambda, active_lambda, rel_tol=0.0, abs_tol=1e-12
                ),
                "prompts": len(train),
                "fixed_train_action": actions[fixed],
                "oracle_mean_vbench5": float(
                    np.mean(quality[np.arange(len(train)), oracle])
                ),
                "oracle_mean_normalized_cost": float(np.mean(normalized_cost[oracle])),
                "oracle_mean_utility": float(np.mean(oracle_utility)),
                "fixed_mean_vbench5": float(np.mean(quality[:, fixed])),
                "fixed_normalized_cost": float(normalized_cost[fixed]),
                "fixed_mean_utility": float(np.mean(fixed_utility)),
                "oracle_gap_over_fixed": float(np.mean(oracle_utility - fixed_utility)),
                "mean_oracle_margin": float(np.mean(margins)),
                "near_tie_rate_margin_le_0p001": float(np.mean(margins <= 0.001)),
                "oracle_action_fractions": {
                    action: float(np.mean(oracle == index))
                    for index, action in enumerate(actions)
                },
            }
        )
    return rows


def evaluate_policy(
    samples,
    predicted_utility,
    actions,
    normalized_cost,
    utility_lambda,
    fixed_action,
    split,
    rng_seed,
    harm_epsilon,
):
    quality = np.asarray([sample["quality"] for sample in samples], dtype=np.float64)
    seconds = np.asarray([sample["seconds"] for sample in samples], dtype=np.float64)
    dimensions = {
        dimension: np.asarray(
            [sample["dimensions"][dimension] for sample in samples], dtype=np.float64
        )
        for dimension in DIMENSIONS
    }
    true_utility = utility_matrix(quality, normalized_cost, utility_lambda)
    chosen = choose_actions(predicted_utility, normalized_cost)
    oracle = choose_actions(true_utility, normalized_cost)
    histogram = np.bincount(chosen, minlength=len(actions)) / len(chosen)
    eye = np.eye(len(actions))
    policies = {
        "fixed_train": np.tile(eye[fixed_action], (len(samples), 1)),
        "prompt_utility": eye[chosen],
        "shuffled_router_hist_expected": np.tile(histogram, (len(samples), 1)),
        "utility_oracle_hindsight": eye[oracle],
    }
    realized = {
        name: (weights * true_utility).sum(axis=1) for name, weights in policies.items()
    }
    rng = np.random.default_rng(rng_seed)
    observed = float(np.mean(realized["prompt_utility"]))
    null = [
        float(np.mean(true_utility[np.arange(len(samples)), rng.permutation(chosen)]))
        for _ in range(2000)
    ]
    permutation_p = (1 + sum(value >= observed - 1e-12 for value in null)) / (
        len(null) + 1
    )
    summary = []
    details = []
    for name, weights in policies.items():
        policy_quality = (weights * quality).sum(axis=1)
        policy_cost = (weights * normalized_cost[None, :]).sum(axis=1)
        policy_seconds = (weights * seconds).sum(axis=1)
        policy_utility = realized[name]
        fixed_gain = policy_utility - realized["fixed_train"]
        shuffle_gain = policy_utility - realized["shuffled_router_hist_expected"]
        regret = true_utility.max(axis=1) - policy_utility
        fixed_ci = bootstrap_ci(fixed_gain.tolist())
        shuffle_ci = bootstrap_ci(shuffle_gain.tolist())
        summary.append(
            {
                "split": split,
                "utility_lambda": utility_lambda,
                "policy": name,
                "prompts": len(samples),
                "mean_vbench5": float(np.mean(policy_quality)),
                "mean_normalized_cost": float(np.mean(policy_cost)),
                "mean_pipeline_seconds": float(np.mean(policy_seconds)),
                "mean_utility": float(np.mean(policy_utility)),
                "mean_policy_regret": float(np.mean(regret)),
                "gain_vs_fixed": float(np.mean(fixed_gain)),
                "gain_fixed_ci_low": fixed_ci[0],
                "gain_fixed_ci_high": fixed_ci[1],
                "gain_vs_histogram_shuffle": float(np.mean(shuffle_gain)),
                "gain_shuffle_ci_low": shuffle_ci[0],
                "gain_shuffle_ci_high": shuffle_ci[1],
                "material_harm_rate_vs_fixed": float(
                    np.mean(fixed_gain < -harm_epsilon)
                ),
                "oracle_exact_action_rate": float(
                    np.mean(weights[np.arange(len(samples)), oracle])
                ),
                "router_permutation_p": permutation_p
                if name == "prompt_utility"
                else None,
                "action_fractions": {
                    action: float(np.mean(weights[:, index]))
                    for index, action in enumerate(actions)
                },
                **{
                    dimension: float(np.mean((weights * values).sum(axis=1)))
                    for dimension, values in dimensions.items()
                },
            }
        )
        for index, sample in enumerate(samples):
            selected = int(np.argmax(weights[index]))
            details.append(
                {
                    "split": split,
                    "prompt_id": sample["prompt_id"],
                    "prompt": sample["prompt"],
                    "utility_lambda": utility_lambda,
                    "policy": name,
                    "action_weights": weights[index].tolist(),
                    "selected_action": actions[selected]
                    if np.isclose(weights[index, selected], 1.0)
                    else "expected_mixture",
                    "oracle_action": actions[oracle[index]],
                    "vbench5": float(policy_quality[index]),
                    "normalized_cost": float(policy_cost[index]),
                    "pipeline_seconds": float(policy_seconds[index]),
                    "utility": float(policy_utility[index]),
                    "policy_regret": float(regret[index]),
                    "gain_vs_fixed": float(fixed_gain[index]),
                }
            )
    return summary, details


def train(args, samples, rows, identity):
    actions = ACTION_SETS[args.action_set]
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    embeddings, feature_digest = (
        load_t5(Path(args.t5_dir).resolve(), samples)
        if args.features == "t5"
        else (None, None)
    )
    train_ids = np.asarray(
        [index for index, sample in enumerate(samples) if sample["split"] == "train"]
    )
    validation_ids = np.asarray(
        [
            index
            for index, sample in enumerate(samples)
            if sample["split"] == "validation"
        ]
    )
    texts = [sample["prompt"] for sample in samples]
    quality = np.asarray([sample["quality"] for sample in samples], dtype=np.float64)
    normalized_cost, latency_profile = train_latency_profile(rows, actions)
    utility = utility_matrix(quality, normalized_cost, args.utility_lambda)
    source_files = (
        Path(__file__),
        Path(__file__).with_name("phase2_utility_model.py"),
        Path(__file__).with_name("phase2_gain_model.py"),
        Path(__file__).with_name("train_phase2_gain_prior.py"),
    )
    request = {
        **identity,
        "actions": actions,
        "action_set": args.action_set,
        "utility_lambda": args.utility_lambda,
        "time_normalization": latency_profile["normalization"],
        "latency_profile_sha256": latency_profile["profile_sha256"],
        "features": args.features,
        "t5_manifest_sha256": feature_digest,
        "folds": args.folds,
        "seed": args.seed,
        "alphas": args.alphas,
        "max_features": args.max_features,
        "harm_epsilon": args.harm_epsilon,
        "diagnostic_lambdas": args.diagnostic_lambdas,
        "source_sha256": {path.name: sha256_file(path) for path in source_files},
    }
    with output_lock(out):
        request_path = out / "training_request.json"
        if request_path.exists() and json.loads(request_path.read_text()) != request:
            raise ValueError("training request changed; use a new --out-dir")
        write_json_atomic(request_path, request)
        write_json_atomic(out / "latency_profile.json", latency_profile)
        started = time.perf_counter()
        model, text_state, oof, folds, cv = cross_validate_utility(
            [texts[index] for index in train_ids],
            utility[train_ids],
            normalized_cost,
            embeddings=embeddings[train_ids] if embeddings is not None else None,
            folds=args.folds,
            seed=args.seed,
            alphas=args.alphas,
            max_features=args.max_features,
        )
        if text_state is None:
            validation_features = normalize(embeddings[validation_ids])
        else:
            validation_features = text_features(
                [texts[index] for index in validation_ids], text_state
            )
        validation_prediction = predict_utility_gains(validation_features, model)
        duration = time.perf_counter() - started
        train_mean_utility = utility[train_ids].mean(axis=0)
        fixed_action = min(
            range(len(actions)),
            key=lambda index: (
                -train_mean_utility[index],
                normalized_cost[index],
                index,
            ),
        )
        np.savez_compressed(out / "model.npz", **model)
        selected_alpha = next(row["alpha"] for row in cv if row["selected"])
        model_meta = {
            "schema": "phase2_fixed_lambda_prompt_utility_ridge_v1",
            "actions": actions,
            "reference_action": actions[0],
            "utility_lambda": args.utility_lambda,
            "normalized_action_cost": normalized_cost.tolist(),
            "latency_profile_sha256": latency_profile["profile_sha256"],
            "features": args.features,
            "text_state": text_state,
            "selected_alpha": selected_alpha,
            "fixed_train_action": actions[fixed_action],
            "training_seconds": duration,
            "model_sha256": sha256_file(out / "model.npz"),
            "request_sha256": canonical_sha256(request),
        }
        write_json_atomic(out / "model.json", model_meta)
        csv_write(out / "cross_validation.csv", cv)
        csv_write(
            out / "train_lambda_diagnostics.csv",
            lambda_diagnostics(
                samples,
                actions,
                normalized_cost,
                args.diagnostic_lambdas,
                args.utility_lambda,
            ),
        )
        csv_write(
            out / "oracle_labels.csv",
            oracle_rows(samples, actions, normalized_cost, args.utility_lambda),
        )

        prediction_rows = []
        summary = []
        details = []
        for ids, prediction, split in (
            (train_ids, oof, "train_oof_selection"),
            (validation_ids, validation_prediction, "validation_development"),
        ):
            true = utility[ids]
            chosen = choose_actions(prediction, normalized_cost)
            oracle = choose_actions(true, normalized_cost)
            for position, sample_id in enumerate(ids):
                prediction_rows.append(
                    {
                        "split": split,
                        "prompt_id": samples[sample_id]["prompt_id"],
                        "prompt": texts[sample_id],
                        "fold": int(folds[position])
                        if split == "train_oof_selection"
                        else None,
                        "predicted_utility_gain": prediction[position].tolist(),
                        "true_utility": true[position].tolist(),
                        "chosen_action": actions[chosen[position]],
                        "oracle_action": actions[oracle[position]],
                        "policy_regret": float(
                            true[position, oracle[position]]
                            - true[position, chosen[position]]
                        ),
                    }
                )
            split_summary, split_details = evaluate_policy(
                [samples[index] for index in ids],
                prediction,
                actions,
                normalized_cost,
                args.utility_lambda,
                fixed_action,
                split,
                args.seed,
                args.harm_epsilon,
            )
            summary.extend(split_summary)
            details.extend(split_details)
        csv_write(out / "utility_predictions.csv", prediction_rows)
        csv_write(out / "policy_summary.csv", summary)
        csv_write(out / "policy_by_prompt.csv", details)
        body = {
            "schema": "phase2_fixed_lambda_utility_selection_v1",
            "evaluation_role": "development_existing_videos",
            "factorial_claim": False,
            "test_accessed": False,
            "actions": actions,
            "utility_lambda": args.utility_lambda,
            "utility_definition": "vbench5 - lambda * paired_train_median_latency_ratio",
            "latency_profile": latency_profile,
            "model": model_meta,
            "results": summary,
        }
        write_json_atomic(
            out / "policy_summary.json",
            {**body, "summary_sha256": canonical_sha256(body)},
        )
        lines = [
            "# Phase 2 fixed-lambda prompt utility prior",
            "",
            f"Development-only reuse of existing videos. Lambda: {args.utility_lambda:.6f}; features: {args.features}; actions: {actions}.",
            f"Train-only paired median latency profile; fixed train action: {actions[fixed_action]}; selected alpha: {selected_alpha} (null = mean-only).",
            "This tests prompt-conditioned utility selection among existing presets. It does not test a 2^3 factorial action space or independent binary operation heads.",
            "Validation was previously inspected and remains validation_development. No test records are read.",
            "",
            "| Split | Policy | VBench5 | Norm. cost | Utility | Regret | Gain vs fixed | Gain vs shuffled | Harm |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary:
            if row["split"] == "validation_development":
                lines.append(
                    f"| {row['split']} | {row['policy']} | {row['mean_vbench5']:.5f} | "
                    f"{row['mean_normalized_cost']:.5f} | {row['mean_utility']:.5f} | "
                    f"{row['mean_policy_regret']:.5f} | {row['gain_vs_fixed']:.5f} | "
                    f"{row['gain_vs_histogram_shuffle']:.5f} | "
                    f"{row['material_harm_rate_vs_fixed']:.1%} |"
                )
        lines += [
            "",
            "Decision gate: continue to a newly generated 2^3 dataset only if prompt_utility improves utility over both fixed_train and the matched action-histogram control with acceptable harm and non-degenerate action use.",
        ]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            f"Selected alpha={selected_alpha}; fit {duration:.2f}s; report: {out / 'report.md'}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "embed", "train"))
    parser.add_argument("--quality-dir", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--action-set", choices=tuple(ACTION_SETS), default="three")
    parser.add_argument("--utility-lambda", type=float, required=True)
    parser.add_argument("--features", choices=("tfidf", "t5"), default="tfidf")
    parser.add_argument("--t5-dir")
    parser.add_argument(
        "--model-root", default="/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--max-features", type=int, default=4096)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1, 10, 100])
    parser.add_argument(
        "--diagnostic-lambdas",
        nargs="+",
        type=float,
        default=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    )
    parser.add_argument("--harm-epsilon", type=float, default=0.001)
    args = parser.parse_args()
    if (
        args.folds < 2
        or args.max_features < 1
        or not math.isfinite(args.utility_lambda)
        or args.utility_lambda < 0
        or not math.isfinite(args.harm_epsilon)
        or args.harm_epsilon < 0
        or any(not math.isfinite(alpha) or alpha <= 0 for alpha in args.alphas)
        or not args.diagnostic_lambdas
        or any(
            not math.isfinite(value) or value < 0 for value in args.diagnostic_lambdas
        )
    ):
        parser.error("invalid folds, lambda, harm epsilon, max features, or alphas")
    args.diagnostic_lambdas = sorted(
        set([*args.diagnostic_lambdas, args.utility_lambda])
    )
    quality_dir = Path(args.quality_dir).resolve()
    lambda_tag = f"{args.utility_lambda:.6g}".replace(".", "p")
    args.out_dir = args.out_dir or str(
        quality_dir
        / f"utility_prior_{args.features}_{args.action_set}_lambda_{lambda_tag}"
    )
    args.t5_dir = args.t5_dir or str(quality_dir / "t5_phase2")
    samples, rows, identity = load_data(quality_dir, ACTION_SETS[args.action_set])
    print(
        f"Verified {len(rows)} scored videos; "
        f"{sum(sample['split'] == 'train' for sample in samples)} train prompts, "
        f"{sum(sample['split'] == 'validation' for sample in samples)} validation prompts",
        flush=True,
    )
    if args.mode == "check":
        _, profile = train_latency_profile(rows, ACTION_SETS[args.action_set])
        print(json.dumps(profile, ensure_ascii=False, indent=2), flush=True)
    elif args.mode == "embed":
        extract_t5(args, samples)
    else:
        train(args, samples, rows, identity)


if __name__ == "__main__":
    main()
