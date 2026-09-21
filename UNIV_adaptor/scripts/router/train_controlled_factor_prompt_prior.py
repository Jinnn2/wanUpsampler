"""Train a factorized prompt prior for FULL/S/T/C controlled-factor data.

The model predicts three continuous, seed-averaged quality deltas relative to
FULL.  Lambda is applied only at policy evaluation time, so one frozen model
supports multiple quality-versus-latency preferences.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    write_json_atomic,
)
from UNIV_adaptor.scripts.data.score_controlled_factor_dataset import (  # noqa: E402
    ACTIONS,
    ANALYSIS_SCHEMA,
    FULL,
)
from UNIV_adaptor.scripts.data.score_phase2_dataset import output_lock  # noqa: E402
from UNIV_adaptor.scripts.router.phase2_gain_model import (  # noqa: E402
    fit_ridge,
    fit_text,
    normalize,
    predict,
    text_features,
)
from UNIV_adaptor.scripts.router.train_phase2_gain_prior import load_t5  # noqa: E402


MODEL_SCHEMA = "univ_controlled_factor_prompt_prior_v1"
CONFIRM_SCHEMA = "univ_controlled_factor_test_confirmation_v1"


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def target_column(action: str) -> str:
    return f"mean_delta_vbench5__{action.lower()}"


def time_column(action: str) -> str:
    return f"mean_time_ratio__{action.lower()}"


def load_samples(
    scored_dir: Path, *, include_test_targets: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    analysis_path = scored_dir / "analysis.json"
    inputs_path = scored_dir / "prompt_inputs.csv"
    targets_path = scored_dir / "prompt_targets_train_validation.csv"
    test_targets_path = scored_dir / "prompt_targets_test.csv"
    analysis = read_json(analysis_path)
    if analysis.get("schema") != ANALYSIS_SCHEMA:
        raise ValueError("unsupported controlled-factor analysis")
    body = {key: value for key, value in analysis.items() if key not in {"schema", "analysis_sha256"}}
    if canonical_sha256(body) != analysis.get("analysis_sha256"):
        raise ValueError("controlled-factor analysis hash mismatch")
    inputs = read_csv(inputs_path)
    target_rows = read_csv(targets_path)
    if include_test_targets:
        target_rows.extend(read_csv(test_targets_path))
    if len(inputs) != 80 or len(target_rows) != (80 if include_test_targets else 64):
        raise ValueError("controlled prompt input/target coverage mismatch")
    targets_by_id = {int(row["prompt_id"]): row for row in target_rows}
    samples = []
    for expected_id, input_row in enumerate(inputs):
        row = targets_by_id.get(expected_id)
        prompt_id = int(input_row["prompt_id"])
        if prompt_id != expected_id:
            raise ValueError("prompt input ids must be contiguous and ordered")
        for key, value in input_row.items():
            if row is not None and row.get(key) != value:
                raise ValueError(f"prompt input/target identity mismatch at {expected_id}: {key}")
        target = None
        time_ratio = None
        if row is not None:
            target = np.asarray([float(row[target_column(action)]) for action in ACTIONS], dtype=np.float64)
            time_ratio = np.asarray([float(row[time_column(action)]) for action in ACTIONS], dtype=np.float64)
            if not np.isfinite(target).all() or not np.isfinite(time_ratio).all() or np.min(time_ratio) <= 0:
                raise ValueError(f"non-finite prompt target at prompt {prompt_id}")
        samples.append(
            {
                "prompt_id": prompt_id,
                "family_id": input_row["family_id"],
                "split": input_row["split"],
                "motion_level": input_row["motion_level"],
                "detail_level": input_row["detail_level"],
                "factor_cell": input_row["factor_cell"],
                "prompt": input_row["prompt"],
                "prompt_sha256": input_row["prompt_sha256"],
                "target": target,
                "time_ratio": time_ratio,
            }
        )
    expected = {"train": 48, "validation": 16, "test": 16}
    observed = {split: sum(row["split"] == split for row in samples) for split in expected}
    if observed != expected:
        raise ValueError(f"controlled target split counts mismatch: {observed}")
    identity = {
        "analysis_file_sha256": sha256_file(analysis_path),
        "analysis_sha256": analysis["analysis_sha256"],
        "score_payload_sha256": analysis["score_payload_sha256"],
        "input_sha256": analysis["input_sha256"],
        "prompt_inputs_file_sha256": sha256_file(inputs_path),
        "prompt_targets_train_validation_file_sha256": sha256_file(targets_path),
    }
    if include_test_targets:
        identity["prompt_targets_test_file_sha256"] = sha256_file(test_targets_path)
    return samples, identity


def lightx2v_env(repo: str | Path) -> dict[str, str]:
    resolved = Path(repo).resolve()
    if not (resolved / "lightx2v").is_dir():
        raise FileNotFoundError(f"LightX2V package not found: {resolved / 'lightx2v'}")
    environment = dict(os.environ)
    roots = [str(resolved), str(ROOT)]
    if environment.get("PYTHONPATH"):
        roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def embed(args: argparse.Namespace, samples: list[dict[str, Any]], identity: dict[str, Any]) -> None:
    from changing_resolution_uni.scripts.data.extract_prompt_t5_embeddings import (
        directory_file_inventory,
    )

    directory = Path(args.t5_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    model = Path(args.model_root).resolve()
    extractor = ROOT / "changing_resolution_uni/scripts/data/extract_prompt_t5_embeddings.py"
    texts = [sample["prompt"] for sample in samples]
    request = {
        "source_input_sha256": identity["input_sha256"],
        "prompts": texts,
        "checkpoint_sha256": sha256_file(model / "models_t5_umt5-xxl-enc-bf16.pth"),
        "tokenizer_files": directory_file_inventory(model / "google/umt5-xxl"),
        "extractor_sha256": sha256_file(extractor),
        "precision": "bf16",
        "max_seq_len": 512,
    }
    request_path = directory / "controlled_factor_embedding_request.json"
    with output_lock(directory):
        if request_path.is_file() and read_json(request_path) != request:
            raise ValueError("T5 extraction request changed; use a new --t5-dir")
        if not request_path.is_file() and list(directory.glob("prompt_*.npz")):
            raise ValueError("existing unbound T5 cache; use a new --t5-dir")
        if not request_path.is_file():
            write_json_atomic(request_path, request)
        if (directory / "t5_manifest.json").is_file():
            load_t5(directory, samples)
            print(f"Reusing verified T5 features: {directory}")
            return
        prompts_path = directory / "prompts.json"
        write_json_atomic(prompts_path, texts)
        subprocess.run(
            [
                sys.executable,
                str(extractor),
                "--prompts_file",
                str(prompts_path),
                "--out_dir",
                str(directory),
                "--model_path",
                str(model),
                "--required_backend",
                "wan_native",
                "--precision",
                "bf16",
                "--device",
                args.device,
            ],
            env=lightx2v_env(args.lightx2v_repo),
            check=True,
        )
        load_t5(directory, samples)


def family_folds(samples: list[dict[str, Any]], indices: np.ndarray, folds: int, seed: int) -> np.ndarray:
    families = sorted({samples[int(index)]["family_id"] for index in indices})
    if len(families) < folds:
        raise ValueError("not enough train families for family-disjoint CV")
    shuffled = np.random.default_rng(seed).permutation(families)
    assignment: dict[str, int] = {}
    for fold, chunk in enumerate(np.array_split(shuffled, folds)):
        assignment.update({str(family): fold for family in chunk})
    return np.asarray([assignment[samples[int(index)]["family_id"]] for index in indices])


def feature_matrix(
    samples: list[dict[str, Any]],
    indices: np.ndarray,
    *,
    features: str,
    embeddings: np.ndarray | None,
    text_state: dict[str, Any] | None,
) -> np.ndarray:
    if features == "t5":
        if embeddings is None:
            raise ValueError("T5 embeddings are required")
        return normalize(embeddings[indices])
    if text_state is None:
        raise ValueError("TF-IDF state is required")
    return text_features([samples[int(index)]["prompt"] for index in indices], text_state)


def select_alpha(
    samples: list[dict[str, Any]],
    train_indices: np.ndarray,
    *,
    features: str,
    embeddings: np.ndarray | None,
    folds: int,
    seed: int,
    alphas: list[float],
    max_features: int,
) -> tuple[float | None, np.ndarray, list[dict[str, Any]]]:
    fold_ids = family_folds(samples, train_indices, folds, seed)
    targets = np.stack([samples[int(index)]["target"] for index in train_indices])
    candidates: list[float | None] = [None, *alphas]
    predictions = [np.zeros_like(targets) for _ in candidates]
    for fold in range(folds):
        fit_positions = np.where(fold_ids != fold)[0]
        held_positions = np.where(fold_ids == fold)[0]
        fit_indices = train_indices[fit_positions]
        held_indices = train_indices[held_positions]
        state = None
        if features == "tfidf":
            state = fit_text(
                [samples[int(index)]["prompt"] for index in fit_indices],
                max_features,
            )
        x_fit = feature_matrix(
            samples, fit_indices, features=features, embeddings=embeddings, text_state=state
        )
        x_held = feature_matrix(
            samples, held_indices, features=features, embeddings=embeddings, text_state=state
        )
        y_fit = targets[fit_positions]
        for position, alpha in enumerate(candidates):
            predictions[position][held_positions] = predict(
                x_held, fit_ridge(x_fit, y_fit, alpha)
            )
    losses = [float(np.mean((prediction - targets) ** 2)) for prediction in predictions]
    choice = int(np.argmin(losses))
    diagnostics = [
        {
            "alpha": "mean_only" if alpha is None else alpha,
            "family_disjoint_oof_mse": loss,
            "selected": index == choice,
        }
        for index, (alpha, loss) in enumerate(zip(candidates, losses))
    ]
    return candidates[choice], predictions[choice], diagnostics


def regression_metrics(
    samples: list[dict[str, Any]], indices: np.ndarray, prediction: np.ndarray, split: str
) -> list[dict[str, Any]]:
    target = np.stack([samples[int(index)]["target"] for index in indices])
    rows = []
    for action_index, action in enumerate(ACTIONS):
        y = target[:, action_index]
        p = prediction[:, action_index]
        denominator = float(np.sum((y - y.mean()) ** 2))
        correlation = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else 0.0
        rows.append(
            {
                "split": split,
                "action_id": action,
                "prompts": len(indices),
                "mse": float(np.mean((p - y) ** 2)),
                "mae": float(np.mean(np.abs(p - y))),
                "r2": float(1.0 - np.sum((p - y) ** 2) / denominator) if denominator > 0 else 0.0,
                "pearson": correlation,
            }
        )
    return rows


def bootstrap_family_gain(
    samples: list[dict[str, Any]], indices: np.ndarray, gains: np.ndarray, seed: int, repetitions: int
) -> tuple[float, float]:
    by_family: dict[str, list[float]] = {}
    for position, index in enumerate(indices):
        by_family.setdefault(samples[int(index)]["family_id"], []).append(float(gains[position]))
    values = np.asarray([np.mean(items) for items in by_family.values()], dtype=np.float64)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(boot, [0.025, 0.975]))


def policy_metrics(
    samples: list[dict[str, Any]],
    indices: np.ndarray,
    prediction: np.ndarray,
    *,
    split: str,
    train_mean_target: np.ndarray,
    calibrated_time_ratios: np.ndarray,
    lambdas: list[float],
    seed: int,
    repetitions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actual_delta = np.column_stack(
        [np.zeros(len(indices)), np.stack([samples[int(index)]["target"] for index in indices])]
    )
    predicted_delta = np.column_stack([np.zeros(len(indices)), prediction])
    train_delta = np.concatenate([[0.0], train_mean_target])
    rows = []
    details = []
    for lambda_value in lambdas:
        speed_reward = -lambda_value * (calibrated_time_ratios - 1.0)
        actual_utility = actual_delta + speed_reward
        predicted_utility = predicted_delta + speed_reward
        chosen = np.argmax(predicted_utility, axis=1)
        fixed = int(np.argmax(train_delta + speed_reward))
        oracle = np.argmax(actual_utility, axis=1)
        positions = np.arange(len(indices))
        policy_value = actual_utility[positions, chosen]
        fixed_value = actual_utility[:, fixed]
        oracle_value = actual_utility[positions, oracle]
        gain = policy_value - fixed_value
        ci_low, ci_high = bootstrap_family_gain(
            samples, indices, gain, seed + int(round(lambda_value * 10000)), repetitions
        )
        rows.append(
            {
                "split": split,
                "lambda": lambda_value,
                "prompts": len(indices),
                "policy_mean_utility": float(policy_value.mean()),
                "fixed_action": (FULL, *ACTIONS)[fixed],
                "fixed_mean_utility": float(fixed_value.mean()),
                "gain_over_fixed": float(gain.mean()),
                "gain_over_fixed_family_bootstrap_ci_low": ci_low,
                "gain_over_fixed_family_bootstrap_ci_high": ci_high,
                "mean_oracle_regret": float((oracle_value - policy_value).mean()),
                "expected_action_accuracy": float(np.mean(chosen == oracle)),
                "harm_below_full_rate": float(np.mean(policy_value < -0.001)),
            }
        )
        for position, index in enumerate(indices):
            details.append(
                {
                    "split": split,
                    "lambda": lambda_value,
                    "prompt_id": samples[int(index)]["prompt_id"],
                    "family_id": samples[int(index)]["family_id"],
                    "motion_level": samples[int(index)]["motion_level"],
                    "detail_level": samples[int(index)]["detail_level"],
                    "chosen_action": (FULL, *ACTIONS)[int(chosen[position])],
                    "oracle_action": (FULL, *ACTIONS)[int(oracle[position])],
                    "policy_utility": float(policy_value[position]),
                    "fixed_utility": float(fixed_value[position]),
                    "oracle_utility": float(oracle_value[position]),
                }
            )
    return rows, details


def fit_final_model(
    samples: list[dict[str, Any]],
    train_indices: np.ndarray,
    *,
    features: str,
    embeddings: np.ndarray | None,
    alpha: float | None,
    max_features: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    text_state = None
    if features == "tfidf":
        text_state = fit_text(
            [samples[int(index)]["prompt"] for index in train_indices], max_features
        )
    x = feature_matrix(
        samples, train_indices, features=features, embeddings=embeddings, text_state=text_state
    )
    y = np.stack([samples[int(index)]["target"] for index in train_indices])
    return fit_ridge(x, y, alpha), text_state


def predictions_for(
    samples: list[dict[str, Any]],
    indices: np.ndarray,
    *,
    features: str,
    embeddings: np.ndarray | None,
    model: dict[str, np.ndarray],
    text_state: dict[str, Any] | None,
) -> np.ndarray:
    x = feature_matrix(
        samples, indices, features=features, embeddings=embeddings, text_state=text_state
    )
    return predict(x, model)


def prediction_rows(
    samples: list[dict[str, Any]], indices: np.ndarray, prediction: np.ndarray, split: str
) -> list[dict[str, Any]]:
    rows = []
    for position, index in enumerate(indices):
        sample = samples[int(index)]
        row = {
            "split": split,
            "prompt_id": sample["prompt_id"],
            "family_id": sample["family_id"],
            "motion_level": sample["motion_level"],
            "detail_level": sample["detail_level"],
            "prompt": sample["prompt"],
        }
        for action_index, action in enumerate(ACTIONS):
            row[f"target_delta__{action.lower()}"] = sample["target"][action_index]
            row[f"predicted_delta__{action.lower()}"] = prediction[position, action_index]
        rows.append(row)
    return rows


def model_features(args: argparse.Namespace, samples: list[dict[str, Any]]) -> tuple[np.ndarray | None, str | None]:
    if args.features == "tfidf":
        return None, None
    embeddings, digest = load_t5(Path(args.t5_dir).resolve(), samples)
    return embeddings, digest


def train(args: argparse.Namespace) -> None:
    scored_dir = Path(args.scored_dir).resolve()
    samples, identity = load_samples(scored_dir, include_test_targets=False)
    embeddings, t5_digest = model_features(args, samples)
    train_indices = np.asarray([index for index, row in enumerate(samples) if row["split"] == "train"])
    validation_indices = np.asarray([index for index, row in enumerate(samples) if row["split"] == "validation"])
    selected_alpha, train_oof, cv_rows = select_alpha(
        samples,
        train_indices,
        features=args.features,
        embeddings=embeddings,
        folds=args.folds,
        seed=args.seed,
        alphas=args.alphas,
        max_features=args.max_features,
    )
    model, text_state = fit_final_model(
        samples,
        train_indices,
        features=args.features,
        embeddings=embeddings,
        alpha=selected_alpha,
        max_features=args.max_features,
    )
    validation_prediction = predictions_for(
        samples,
        validation_indices,
        features=args.features,
        embeddings=embeddings,
        model=model,
        text_state=text_state,
    )
    train_target = np.stack([samples[int(index)]["target"] for index in train_indices])
    train_mean_target = train_target.mean(axis=0)
    calibrated_time_ratios = np.concatenate(
        [
            [1.0],
            np.stack([samples[int(index)]["time_ratio"] for index in train_indices]).mean(axis=0),
        ]
    )
    regression = [
        *regression_metrics(samples, train_indices, train_oof, "train_family_oof"),
        *regression_metrics(samples, validation_indices, validation_prediction, "validation_holdout"),
    ]
    policy = []
    policy_details = []
    for indices, prediction, split in (
        (train_indices, train_oof, "train_family_oof"),
        (validation_indices, validation_prediction, "validation_holdout"),
    ):
        summary, details = policy_metrics(
            samples,
            indices,
            prediction,
            split=split,
            train_mean_target=train_mean_target,
            calibrated_time_ratios=calibrated_time_ratios,
            lambdas=args.lambdas,
            seed=args.seed,
            repetitions=args.bootstrap_repetitions,
        )
        policy.extend(summary)
        policy_details.extend(details)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    request = {
        "schema": MODEL_SCHEMA,
        **identity,
        "features": args.features,
        "t5_manifest_sha256": t5_digest,
        "actions": list(ACTIONS),
        "reference_action": FULL,
        "target_definition": "three_seed_mean_action_minus_full_vbench5",
        "selected_alpha": selected_alpha,
        "folds": args.folds,
        "fold_unit": "semantic_family",
        "seed": args.seed,
        "max_features": args.max_features,
        "candidate_alphas": args.alphas,
        "lambdas": args.lambdas,
        "calibrated_train_mean_time_ratios": calibrated_time_ratios.tolist(),
        "test_accessed": False,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    with output_lock(out):
        request_path = out / "training_request.json"
        if request_path.is_file() and read_json(request_path) != request:
            raise RuntimeError("training request changed; use a new --out-dir")
        if not request_path.is_file():
            write_json_atomic(request_path, request)
        np.savez_compressed(
            out / "model.npz",
            weights=model["weights"],
            x_mean=model["x_mean"],
            y_mean=model["y_mean"],
        )
        if text_state is not None:
            write_json_atomic(out / "tfidf_state.json", text_state)
        csv_write(out / "cv_selection.csv", cv_rows)
        csv_write(out / "regression_metrics.csv", regression)
        csv_write(
            out / "predictions.csv",
            [
                *prediction_rows(samples, train_indices, train_oof, "train_family_oof"),
                *prediction_rows(samples, validation_indices, validation_prediction, "validation_holdout"),
            ],
        )
        csv_write(out / "policy_summary.csv", policy)
        csv_write(out / "policy_by_prompt.csv", policy_details)
        body = {
            **request,
            "model_file_sha256": sha256_file(out / "model.npz"),
            "tfidf_state_file_sha256": sha256_file(out / "tfidf_state.json") if text_state is not None else None,
            "regression_metrics": regression,
            "policy_summary": policy,
        }
        write_json_atomic(
            out / "validation_summary.json",
            {**body, "summary_sha256": canonical_sha256(body)},
        )
        validation_lambda = min(args.lambdas, key=lambda value: abs(value - 0.05))
        selected = next(
            row for row in policy
            if row["split"] == "validation_holdout" and row["lambda"] == validation_lambda
        )
        lines = [
            "# Controlled-factor prompt prior", "",
            f"Features: `{args.features}`; selected alpha: `{selected_alpha}`.",
            "Model selection used train semantic-family CV. The report contains train OOF and validation only; test was not accessed.", "",
            f"At lambda={validation_lambda:.4f}, validation gain over the train-selected fixed action is {selected['gain_over_fixed']:.6f} ",
            f"with family-bootstrap 95% CI [{selected['gain_over_fixed_family_bootstrap_ci_low']:.6f}, {selected['gain_over_fixed_family_bootstrap_ci_high']:.6f}].",
            "", "Run the explicit confirmation mode only after freezing this directory.",
        ]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote controlled prompt prior: {out / 'report.md'}")


def load_frozen_model(out: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any] | None]:
    summary = read_json(out / "validation_summary.json")
    body = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if canonical_sha256(body) != summary.get("summary_sha256"):
        raise ValueError("validation summary hash mismatch")
    if sha256_file(out / "model.npz") != summary["model_file_sha256"]:
        raise ValueError("frozen model hash mismatch")
    with np.load(out / "model.npz", allow_pickle=False) as data:
        model = {key: np.asarray(data[key], dtype=np.float64) for key in ("weights", "x_mean", "y_mean")}
    state = None
    if summary["features"] == "tfidf":
        if sha256_file(out / "tfidf_state.json") != summary["tfidf_state_file_sha256"]:
            raise ValueError("frozen TF-IDF state hash mismatch")
        state = read_json(out / "tfidf_state.json")
    return summary, model, state


def confirm(args: argparse.Namespace) -> None:
    if not args.confirm_test_access:
        raise RuntimeError("locked test requires --confirm-test-access (shell wrapper uses CONFIRM_TEST_ACCESS=1)")
    out = Path(args.out_dir).resolve()
    if (out / "test_access_guard.json").exists():
        raise RuntimeError("test was already accessed for this frozen output directory")
    summary, model, text_state = load_frozen_model(out)
    samples, identity = load_samples(
        Path(args.scored_dir).resolve(), include_test_targets=True
    )
    for key in (
        "analysis_sha256",
        "score_payload_sha256",
        "input_sha256",
        "prompt_inputs_file_sha256",
        "prompt_targets_train_validation_file_sha256",
    ):
        if summary[key] != identity[key]:
            raise ValueError(f"frozen model/source mismatch: {key}")
    if args.features != summary["features"]:
        raise ValueError("confirmation features differ from frozen training request")
    embeddings, t5_digest = model_features(args, samples)
    if t5_digest != summary["t5_manifest_sha256"]:
        raise ValueError("confirmation T5 manifest differs from frozen training")
    test_indices = np.asarray([index for index, row in enumerate(samples) if row["split"] == "test"])
    prediction = predictions_for(
        samples,
        test_indices,
        features=args.features,
        embeddings=embeddings,
        model=model,
        text_state=text_state,
    )
    train_indices = np.asarray([index for index, row in enumerate(samples) if row["split"] == "train"])
    train_mean_target = np.stack([samples[int(index)]["target"] for index in train_indices]).mean(axis=0)
    calibrated = np.asarray(summary["calibrated_train_mean_time_ratios"], dtype=np.float64)
    regression = regression_metrics(samples, test_indices, prediction, "locked_test")
    policy, details = policy_metrics(
        samples,
        test_indices,
        prediction,
        split="locked_test",
        train_mean_target=train_mean_target,
        calibrated_time_ratios=calibrated,
        lambdas=[float(value) for value in summary["lambdas"]],
        seed=int(summary["seed"]),
        repetitions=args.bootstrap_repetitions,
    )
    guard_body = {
        "frozen_validation_summary_sha256": summary["summary_sha256"],
        "model_file_sha256": summary["model_file_sha256"],
        "source_input_sha256": summary["input_sha256"],
        "confirmed_at_protocol_date": "2026-09-22",
    }
    confirmation_body = {
        "frozen_validation_summary_sha256": summary["summary_sha256"],
        "test_prompt_count": len(test_indices),
        "regression_metrics": regression,
        "policy_summary": policy,
    }
    with output_lock(out):
        csv_write(out / "test_predictions.csv", prediction_rows(samples, test_indices, prediction, "locked_test"))
        csv_write(out / "test_regression_metrics.csv", regression)
        csv_write(out / "test_policy_summary.csv", policy)
        csv_write(out / "test_policy_by_prompt.csv", details)
        write_json_atomic(
            out / "test_confirmation.json",
            {
                "schema": CONFIRM_SCHEMA,
                "confirmation_sha256": canonical_sha256(confirmation_body),
                **confirmation_body,
            },
        )
        # Write the one-time guard last so an interrupted export remains retryable.
        write_json_atomic(out / "test_access_guard.json", guard_body)
    print(f"Locked test confirmed once: {out / 'test_confirmation.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "embed", "train", "confirm"))
    parser.add_argument("--scored-dir", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--features", choices=("tfidf", "t5"), default="t5")
    parser.add_argument("--t5-dir")
    parser.add_argument("--model-root", default="/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--lightx2v-repo", default="/mnt/afs_2/houze/LightX2V")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--max-features", type=int, default=4096)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.01, 0.03, 0.05, 0.08, 0.1])
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--confirm-test-access", action="store_true")
    args = parser.parse_args()
    if (
        args.folds < 2
        or args.max_features < 1
        or args.bootstrap_repetitions < 100
        or any(not math.isfinite(value) or value <= 0 for value in args.alphas)
        or any(not math.isfinite(value) or value < 0 for value in args.lambdas)
    ):
        parser.error("invalid CV, ridge, lambda, or bootstrap arguments")
    scored = Path(args.scored_dir).resolve()
    args.t5_dir = args.t5_dir or str(scored / "t5_controlled_factor")
    args.out_dir = args.out_dir or str(scored / f"prompt_prior_{args.features}")
    return args


def main() -> None:
    args = parse_args()
    samples, identity = load_samples(
        Path(args.scored_dir).resolve(),
        include_test_targets=args.mode == "confirm",
    )
    print(
        f"Verified controlled targets: {len(samples)} prompts; input={identity['input_sha256'][:12]}",
        flush=True,
    )
    if args.mode == "check":
        return
    if args.mode == "embed":
        embed(args, samples, identity)
    elif args.mode == "train":
        train(args)
    else:
        confirm(args)


if __name__ == "__main__":
    main()
