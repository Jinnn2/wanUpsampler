"""Small prompt gain regressors; NumPy only, with fold-local text fitting."""
from __future__ import annotations

from collections import Counter
import re
import numpy as np


def terms(text):
    words = re.findall(r"\b\w+\b", text.lower())
    return words + [a + " " + b for a, b in zip(words, words[1:])]


def fit_text(texts, max_features=4096):
    df = Counter(t for text in texts for t in set(terms(text)))
    vocab = sorted(df, key=lambda t: (-df[t], t))[:max_features]
    return {"vocabulary": vocab, "idf": [float(np.log((1+len(texts))/(1+df[t]))+1) for t in vocab]}


def text_features(texts, state):
    index = {t: i for i, t in enumerate(state["vocabulary"])}
    x = np.zeros((len(texts), len(index)), dtype=np.float64)
    for i, text in enumerate(texts):
        for t, count in Counter(terms(text)).items():
            if t in index:
                x[i, index[t]] = 1 + np.log(count)
    return normalize(x * np.asarray(state["idf"]))


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("features must be a finite matrix")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def fit_ridge(x, y, alpha):
    """Centered dual ridge, with an unpenalized intercept and a mean-only control."""
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc = x-xm
    if alpha is None:
        weights = np.zeros((x.shape[1], y.shape[1]))
    else:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("ridge alpha must be finite and positive")
        dual = np.linalg.solve(xc @ xc.T + alpha*np.eye(len(x)), y-ym)
        weights = xc.T @ dual
    return {"weights": weights, "x_mean": xm, "y_mean": ym}


def predict(x, model):
    return (x-model["x_mean"]) @ model["weights"] + model["y_mean"]


def cross_validate(texts, y, *, embeddings=None, folds=5, seed=20260918,
                   alphas=(.1, 1., 10., 100.), max_features=4096):
    if len(texts) < folds or folds < 2:
        raise ValueError("need at least one training prompt per fold and folds >= 2")
    indices = np.random.default_rng(seed).permutation(len(texts))
    fold_ids = np.empty(len(texts), dtype=int)
    for fold, held in enumerate(np.array_split(indices, folds)):
        fold_ids[held] = fold
    candidates = [None, *alphas]  # ties prefer the mean-only baseline
    oof = [np.zeros_like(y) for _ in candidates]
    for fold in range(folds):
        train, held = np.where(fold_ids != fold)[0], np.where(fold_ids == fold)[0]
        if embeddings is None:
            state = fit_text([texts[i] for i in train], max_features)
            x = text_features(texts, state)
        else:
            x = normalize(embeddings)
        for j, alpha in enumerate(candidates):
            oof[j][held] = predict(x[held], fit_ridge(x[train], y[train], alpha))
    losses = [float(np.mean((pred-y)**2)) for pred in oof]
    choice = int(np.argmin(losses))
    state = fit_text(texts, max_features) if embeddings is None else None
    x = text_features(texts, state) if embeddings is None else normalize(embeddings)
    model = fit_ridge(x, y, candidates[choice])
    return model, state, oof[choice], fold_ids, [
        {"alpha": alpha, "mean_squared_gain_error": loss, "selected": j==choice}
        for j, (alpha, loss) in enumerate(zip(candidates, losses))]


def select_actions(predicted_gains, calibrated_costs, budget):
    eligible = np.flatnonzero(np.asarray(calibrated_costs) <= budget + 1e-9)
    if len(eligible) == 0:
        return None, eligible
    # Tie-break uses train costs, never evaluation latency or labels.
    eligible = np.asarray(sorted(eligible, key=lambda j: (calibrated_costs[j], j)))
    return eligible[np.argmax(predicted_gains[:, eligible], axis=1)], eligible


def train_mixture(train_quality, train_cost, eligible, target_cost):
    """Optimal prompt-independent distribution under an average calibrated cost cap."""
    candidates = []
    n = len(train_quality)
    for a in eligible:
        if train_cost[a] <= target_cost + 1e-9:
            w = np.eye(n)[a]
            candidates.append(w)
        for b in eligible:
            if train_cost[a] < target_cost < train_cost[b]:
                weight_b = (target_cost-train_cost[a])/(train_cost[b]-train_cost[a])
                w = np.zeros(n)
                w[a], w[b] = 1-weight_b, weight_b
                candidates.append(w)
    if not candidates:
        raise ValueError("no feasible train mixture")
    return max(candidates, key=lambda w: (float(w@train_quality), -float(w@train_cost)))
