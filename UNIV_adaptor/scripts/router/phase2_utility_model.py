"""Fixed-lambda prompt utility regression with train-only cross-validation."""

from __future__ import annotations

import numpy as np

from UNIV_adaptor.scripts.router.phase2_gain_model import (
    fit_ridge,
    fit_text,
    normalize,
    predict,
    text_features,
)


def utility_matrix(quality, normalized_cost, utility_lambda):
    quality = np.asarray(quality, dtype=np.float64)
    cost = np.asarray(normalized_cost, dtype=np.float64)
    if quality.ndim != 2 or cost.shape != (quality.shape[1],):
        raise ValueError("quality must be [prompt, action] and cost must be [action]")
    if not np.isfinite(quality).all() or not np.isfinite(cost).all():
        raise ValueError("quality and cost must be finite")
    if not np.isfinite(utility_lambda) or utility_lambda < 0:
        raise ValueError("utility_lambda must be finite and non-negative")
    return quality - float(utility_lambda) * cost[None, :]


def choose_actions(scores, normalized_cost):
    """Choose maximum score; exact ties prefer the lower-cost action, then index."""
    scores = np.asarray(scores, dtype=np.float64)
    cost = np.asarray(normalized_cost, dtype=np.float64)
    if scores.ndim != 2 or cost.shape != (scores.shape[1],):
        raise ValueError("scores must be [prompt, action] and cost must be [action]")
    if not np.isfinite(scores).all() or not np.isfinite(cost).all():
        raise ValueError("scores and cost must be finite")
    choices = []
    for row in scores:
        best = max(row)
        tied = np.flatnonzero(np.isclose(row, best, rtol=0.0, atol=1e-12))
        choices.append(min(tied, key=lambda index: (cost[index], index)))
    return np.asarray(choices, dtype=np.int64)


def policy_regret(predicted_utility, true_utility, normalized_cost):
    choices = choose_actions(predicted_utility, normalized_cost)
    realized = true_utility[np.arange(len(choices)), choices]
    return float(np.mean(true_utility.max(axis=1) - realized))


def cross_validate_utility(
    texts,
    utility,
    normalized_cost,
    *,
    embeddings=None,
    folds=5,
    seed=20260918,
    alphas=(0.1, 1.0, 10.0, 100.0),
    max_features=4096,
):
    """Select ridge strength by OOF policy regret, with utility MSE as tie-break."""
    utility = np.asarray(utility, dtype=np.float64)
    if utility.ndim != 2 or utility.shape[0] != len(texts) or utility.shape[1] < 2:
        raise ValueError("utility must cover every text and at least two actions")
    if len(texts) < folds or folds < 2:
        raise ValueError("need at least one training prompt per fold and folds >= 2")
    if embeddings is not None and len(embeddings) != len(texts):
        raise ValueError("embedding count differs from text count")

    # A shared reference removes prompt-wide quality offsets that cannot affect choice.
    target = utility[:, 1:] - utility[:, [0]]
    order = np.random.default_rng(seed).permutation(len(texts))
    fold_ids = np.empty(len(texts), dtype=np.int64)
    for fold, held in enumerate(np.array_split(order, folds)):
        fold_ids[held] = fold
    if any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("alphas must be finite and positive")
    candidates = [None, *alphas]
    oof = [np.zeros_like(target) for _ in candidates]
    for fold in range(folds):
        train = np.where(fold_ids != fold)[0]
        held = np.where(fold_ids == fold)[0]
        if embeddings is None:
            state = fit_text([texts[index] for index in train], max_features)
            x = text_features(texts, state)
        else:
            x = normalize(embeddings)
        for position, alpha in enumerate(candidates):
            model = fit_ridge(x[train], target[train], alpha)
            oof[position][held] = predict(x[held], model)

    true_relative = np.column_stack([np.zeros(len(target)), target])
    rows = []
    for position, (alpha, prediction) in enumerate(zip(candidates, oof, strict=True)):
        predicted_relative = np.column_stack([np.zeros(len(prediction)), prediction])
        choices = choose_actions(predicted_relative, normalized_cost)
        oracle = choose_actions(true_relative, normalized_cost)
        rows.append(
            {
                "alpha": alpha,
                "mean_policy_regret": policy_regret(
                    predicted_relative, true_relative, normalized_cost
                ),
                "mean_squared_utility_gain_error": float(
                    np.mean((prediction - target) ** 2)
                ),
                "oracle_exact_action_rate": float(np.mean(choices == oracle)),
                "selected": False,
                "candidate_position": position,
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["mean_policy_regret"],
            row["mean_squared_utility_gain_error"],
            row["candidate_position"],
        ),
    )
    selected["selected"] = True
    choice = int(selected["candidate_position"])

    state = fit_text(texts, max_features) if embeddings is None else None
    x = text_features(texts, state) if embeddings is None else normalize(embeddings)
    model = fit_ridge(x, target, candidates[choice])
    selected_oof = np.column_stack([np.zeros(len(target)), oof[choice]])
    for row in rows:
        row.pop("candidate_position")
    return model, state, selected_oof, fold_ids, rows


def predict_utility_gains(features, model):
    prediction = predict(features, model)
    return np.column_stack([np.zeros(len(prediction)), prediction])
