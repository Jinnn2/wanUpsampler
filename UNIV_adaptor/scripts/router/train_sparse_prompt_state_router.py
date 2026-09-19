"""Development audit for prompt priors plus realized-video state proxies.

This script deliberately separates two questions:

1. Can prompt text predict which sparse action is useful?
2. Does a seed-specific visual observation add signal beyond that prompt prior?

The ``video_proxy`` state is extracted only from the common REFERENCE video of
each prompt/seed group.  It is a post-hoc capacity proxy, not a deployable early
latent: the report and manifests keep that limitation explicit.  If it passes,
the same trainer can consume a future content-bound early-state NPZ without
regenerating or rescoring the action videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
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
from UNIV_adaptor.scripts.data.phase2_analysis import csv_write  # noqa: E402
from UNIV_adaptor.scripts.data.score_phase2_dataset import output_lock  # noqa: E402
from UNIV_adaptor.scripts.router.phase2_gain_model import (  # noqa: E402
    fit_text,
    normalize,
    text_features,
)
from UNIV_adaptor.scripts.router.train_phase2_gain_prior import (  # noqa: E402
    load_t5,
)


SCORED_SCHEMA = "univ_sparse_action_scored_dataset_v1"
STATE_SCHEMA = "univ_sparse_reference_video_proxy_v1"
MODEL_SCHEMA = "univ_sparse_prompt_state_router_development_v1"
AXES = (
    "spatial_aggressive",
    "temporal_aggressive",
    "lr_compute_aggressive",
)
FAMILIES = (
    "action_main",
    "prompt_only",
    "state_only",
    "prompt_state",
    "prompt_state_shuffled",
)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_scored_dir(
    scored_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    scored = read_json(scored_dir / "scored_dataset.json")
    if scored.get("schema") != SCORED_SCHEMA:
        raise ValueError(f"unsupported scored dataset: {scored.get('schema')}")
    body = {
        key: value
        for key, value in scored.items()
        if key not in {"schema", "dataset_sha256"}
    }
    if canonical_sha256(body) != scored.get("dataset_sha256"):
        raise ValueError("scored dataset hash mismatch")
    for name in ("quality_by_video.csv", "relative_quality_pairs.csv"):
        path = scored_dir / name
        expected = scored["output_files"][name]["file_sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"scored CSV hash mismatch: {path}")
    inputs = read_json(scored_dir / "evaluation_inputs.json")
    input_body = {"identity": inputs["identity"], "rows": inputs["rows"]}
    if canonical_sha256(input_body) != inputs.get("input_sha256"):
        raise ValueError("evaluation input hash mismatch")
    if inputs["input_sha256"] != scored["input_sha256"]:
        raise ValueError("evaluation/scored input identity mismatch")
    pairs = read_csv(scored_dir / "relative_quality_pairs.csv")
    if len(inputs["rows"]) != int(scored["video_count"]):
        raise ValueError("evaluation input video count mismatch")
    if len(pairs) != int(scored["relative_pair_count"]):
        raise ValueError("relative pair count mismatch")
    references = [row for row in inputs["rows"] if row["action_id"] == "REFERENCE"]
    if len(references) != int(scored["group_count"]):
        raise ValueError("REFERENCE coverage differs from group_count")
    if len({row["group_id"] for row in references}) != len(references):
        raise ValueError("duplicate REFERENCE group")
    pair_groups = {row["group_id"] for row in pairs}
    if pair_groups != {row["group_id"] for row in references}:
        raise ValueError("relative pairs and REFERENCE groups differ")
    return scored, references, pairs


def proxy_features(frames: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Compute fixed cheap appearance/motion statistics from gray frames."""

    x = np.asarray(frames, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 2 or min(x.shape[1:]) < 2:
        raise ValueError("proxy frames must have shape [T,H,W], T/H/W >= 2")
    if not np.isfinite(x).all():
        raise ValueError("proxy frames contain non-finite values")
    if x.max() > 1.0 or x.min() < 0.0:
        x = x / 255.0
    if x.min() < 0.0 or x.max() > 1.0:
        raise ValueError("proxy frames must be in [0,1] or [0,255]")
    frame_mean = x.mean(axis=(1, 2))
    frame_std = x.std(axis=(1, 2))
    dx = np.abs(np.diff(x, axis=2))
    dy = np.abs(np.diff(x, axis=1))
    dt = np.abs(np.diff(x, axis=0))
    motion = dt.mean(axis=(1, 2))
    # Change in spatial edges is a crude motion/structure interaction.
    edge_x = np.diff(x, axis=2)
    edge_y = np.diff(x, axis=1)
    edge_motion = 0.5 * (
        np.abs(np.diff(edge_x, axis=0)).mean() + np.abs(np.diff(edge_y, axis=0)).mean()
    )
    names = [
        "luma_mean",
        "luma_std",
        "luma_p10",
        "luma_p50",
        "luma_p90",
        "frame_mean_std",
        "frame_mean_range",
        "frame_contrast_mean",
        "frame_contrast_std",
        "spatial_dx_abs_mean",
        "spatial_dx_abs_std",
        "spatial_dy_abs_mean",
        "spatial_dy_abs_std",
        "temporal_abs_mean",
        "temporal_abs_std",
        "temporal_abs_rms",
        "temporal_abs_p90",
        "motion_frame_mean",
        "motion_frame_std",
        "motion_frame_max",
        "edge_motion_abs_mean",
    ]
    values = np.asarray(
        [
            x.mean(),
            x.std(),
            *np.quantile(x, [0.1, 0.5, 0.9]),
            frame_mean.std(),
            np.ptp(frame_mean),
            frame_std.mean(),
            frame_std.std(),
            dx.mean(),
            dx.std(),
            dy.mean(),
            dy.std(),
            dt.mean(),
            dt.std(),
            np.sqrt(np.mean(dt**2)),
            np.quantile(dt, 0.9),
            motion.mean(),
            motion.std(),
            motion.max(),
            edge_motion,
        ],
        dtype=np.float64,
    )
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise RuntimeError("invalid proxy feature vector")
    return values, names


def decode_proxy_frames(
    video: Path, *, ffmpeg: str, size: int, fps: float, frames: int
) -> np.ndarray:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={fps:.8g},scale={size}:{size}:flags=area,format=gray",
        "-frames:v",
        str(frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    pixels = np.frombuffer(result.stdout, dtype=np.uint8)
    frame_pixels = size * size
    if pixels.size % frame_pixels or pixels.size < 2 * frame_pixels:
        raise RuntimeError(
            f"ffmpeg returned {pixels.size} bytes for {video}; expected >=2 frames"
        )
    return pixels.reshape(-1, size, size)


def extract_video_proxy(args: argparse.Namespace) -> None:
    scored_dir = Path(args.scored_dir).resolve()
    scored, references, _ = validate_scored_dir(scored_dir)
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        raise RuntimeError(f"ffmpeg not found: {args.ffmpeg}")
    out_dir = Path(args.state_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "schema": STATE_SCHEMA,
        "source_input_sha256": scored["input_sha256"],
        "source_score_payload_sha256": scored["score_payload_sha256"],
        "observation_role": "posthoc_reference_video_capacity_proxy",
        "deployable_early_state": False,
        "reference_only": True,
        "size": args.proxy_size,
        "fps": args.proxy_fps,
        "max_frames": args.proxy_frames,
        "verify_video_hashes": not args.skip_video_hash,
        "group_ids": [row["group_id"] for row in references],
        "video_sha256": [row["video_sha256"] for row in references],
    }
    request_path = out_dir / "state_request.json"
    if request_path.is_file() and read_json(request_path) != request:
        raise RuntimeError("state extraction request changed; use a new --state-dir")
    if not request_path.is_file():
        write_json_atomic(request_path, request)
    manifest_path = out_dir / "state_manifest.json"
    feature_path = out_dir / "state_features.npz"
    if manifest_path.is_file():
        load_state_features(
            out_dir,
            expected_input_sha256=scored["input_sha256"],
            expected_groups=set(request["group_ids"]),
        )
        print(f"Reusing verified REFERENCE video proxies: {feature_path}")
        return
    if feature_path.exists():
        raise RuntimeError(
            "unbound state feature file exists without a manifest; use a new --state-dir"
        )
    matrix = []
    names: list[str] | None = None
    for index, row in enumerate(references, start=1):
        video = Path(row["video_path"]).resolve()
        if not video.is_file():
            raise FileNotFoundError(f"missing REFERENCE video: {video}")
        if not args.skip_video_hash and sha256_file(video) != row["video_sha256"]:
            raise ValueError(f"REFERENCE video hash mismatch: {video}")
        values, current_names = proxy_features(
            decode_proxy_frames(
                video,
                ffmpeg=ffmpeg,
                size=args.proxy_size,
                fps=args.proxy_fps,
                frames=args.proxy_frames,
            )
        )
        if names is None:
            names = current_names
        elif names != current_names:
            raise RuntimeError("proxy feature schema changed during extraction")
        matrix.append(values)
        if index % 25 == 0 or index == len(references):
            print(f"[proxy] {index}/{len(references)}", flush=True)
    np.savez_compressed(
        feature_path,
        group_ids=np.asarray(request["group_ids"]),
        features=np.stack(matrix),
        feature_names=np.asarray(names),
    )
    body = {
        **{key: value for key, value in request.items() if key != "schema"},
        "request_sha256": canonical_sha256(request),
        "request_file_sha256": sha256_file(request_path),
        "feature_file": str(feature_path),
        "feature_file_sha256": sha256_file(feature_path),
        "feature_count": len(matrix),
        "feature_dim": len(names or []),
        "feature_names": names,
        "ffmpeg_path": ffmpeg,
    }
    manifest = {
        "schema": STATE_SCHEMA,
        "manifest_sha256": canonical_sha256(body),
        **body,
    }
    write_json_atomic(manifest_path, manifest)
    print(f"Extracted {len(matrix)} REFERENCE video proxies: {feature_path}")


def load_state_features(
    state_dir: Path, *, expected_input_sha256: str, expected_groups: set[str]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = read_json(state_dir / "state_manifest.json")
    if manifest.get("schema") != STATE_SCHEMA:
        raise ValueError(f"unsupported state manifest: {manifest.get('schema')}")
    body = {
        key: value
        for key, value in manifest.items()
        if key not in {"schema", "manifest_sha256"}
    }
    if canonical_sha256(body) != manifest.get("manifest_sha256"):
        raise ValueError("state manifest hash mismatch")
    if manifest["source_input_sha256"] != expected_input_sha256:
        raise ValueError("state/scored input identity mismatch")
    path = state_dir / Path(manifest["feature_file"]).name
    if sha256_file(path) != manifest["feature_file_sha256"]:
        raise ValueError("state feature file hash mismatch")
    with np.load(path, allow_pickle=False) as payload:
        group_ids = [str(value) for value in payload["group_ids"]]
        features = np.asarray(payload["features"], dtype=np.float64)
        names = [str(value) for value in payload["feature_names"]]
    if (
        features.shape != (len(group_ids), len(names))
        or not np.isfinite(features).all()
        or len(group_ids) != len(set(group_ids))
        or set(group_ids) != expected_groups
    ):
        raise ValueError("invalid state feature coverage or matrix")
    return dict(zip(group_ids, features)), manifest


def axis_levels(row: dict[str, str]) -> np.ndarray:
    values = []
    for axis in AXES:
        value = int(row[f"level_{axis}"])
        if value not in (0, 1):
            raise ValueError(f"invalid binary action level for {axis}")
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def action_main_basis(levels: np.ndarray) -> np.ndarray:
    """Quadratic action surface: O(d^2), without a 2^d combination basis."""

    values = np.asarray(levels, dtype=np.float64).reshape(-1)
    pairwise = [
        values[left] * values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]
    return np.asarray([1.0, *values, *pairwise], dtype=np.float64)


def action_interaction_basis(levels: np.ndarray) -> np.ndarray:
    """Low-order scalable basis used for context-action interactions."""

    signed = 2.0 * np.asarray(levels, dtype=np.float64) - 1.0
    return np.concatenate(([1.0], signed))


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-10] = 1.0
    return mean, scale


def interaction_features(context: np.ndarray, levels: np.ndarray) -> np.ndarray:
    basis = np.stack([action_interaction_basis(row) for row in levels])
    return np.einsum("ni,nj->nij", basis, context).reshape(len(context), -1)


def fit_partial_ridge(
    main: np.ndarray,
    context: np.ndarray | None,
    target: np.ndarray,
    alpha: float | None,
) -> dict[str, np.ndarray | None]:
    """Fit unpenalized action means plus penalized action-context residuals."""

    x = np.asarray(main, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1, 1)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("main/target shape mismatch")
    if context is None or alpha is None:
        beta = None
        main_coef = np.linalg.pinv(x) @ y
    else:
        z = np.asarray(context, dtype=np.float64)
        if z.ndim != 2 or len(z) != len(x):
            raise ValueError("context shape mismatch")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("ridge alpha must be positive")
        projection = x @ np.linalg.pinv(x)
        residualizer = np.eye(len(x)) - projection
        zr = residualizer @ z
        yr = residualizer @ y
        dual = np.linalg.solve(zr @ zr.T + alpha * np.eye(len(x)), yr)
        beta = zr.T @ dual
        main_coef = np.linalg.pinv(x) @ (y - z @ beta)
    return {"main_coef": main_coef, "context_coef": beta}


def predict_partial_ridge(
    model: dict[str, np.ndarray | None],
    main: np.ndarray,
    context: np.ndarray | None,
) -> np.ndarray:
    value = np.asarray(main) @ np.asarray(model["main_coef"])
    if model["context_coef"] is not None:
        if context is None:
            raise ValueError("model requires context features")
        value = value + np.asarray(context) @ np.asarray(model["context_coef"])
    return value[:, 0]


def shuffled_state_map(
    rows: list[dict[str, Any]], state: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    by_prompt: dict[str, list[str]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_key"], []).append(row["group_id"])
    result = dict(state)
    for groups in by_prompt.values():
        ordered = sorted(set(groups))
        if len(ordered) < 2:
            continue
        for index, group in enumerate(ordered):
            result[group] = state[ordered[(index + 1) % len(ordered)]]
    return result


def prompt_embedding_map(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    t5_dir: Path | None,
    train_prompts: set[str],
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    if mode == "tfidf":
        return None, None
    unique = {}
    for row in rows:
        unique[row["prompt_key"]] = {
            "prompt": row["prompt"],
            "prompt_key": row["prompt_key"],
        }
    samples = [unique[key] for key in sorted(unique)]
    matrix, digest = load_t5(t5_dir, samples)
    mapping = {
        sample["prompt_key"]: matrix[index] for index, sample in enumerate(samples)
    }
    # Scaling is fitted later on training rows only. The argument documents that
    # feature learning itself never consults held-out labels.
    if not train_prompts.issubset(mapping):
        raise ValueError("T5 train prompt coverage mismatch")
    return mapping, digest


def build_context(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    fit_indices: np.ndarray,
    *,
    family: str,
    prompt_mode: str,
    prompt_vectors: dict[str, np.ndarray] | None,
    state: dict[str, np.ndarray],
    shuffled_state: dict[str, np.ndarray],
    max_features: int,
) -> np.ndarray | None:
    if family == "action_main":
        return None
    use_prompt = family in {"prompt_only", "prompt_state", "prompt_state_shuffled"}
    use_state = family in {"state_only", "prompt_state", "prompt_state_shuffled"}
    parts = []
    if use_prompt:
        if prompt_mode == "tfidf":
            fit_texts = list(
                {
                    rows[index]["prompt_key"]: rows[index]["prompt"]
                    for index in fit_indices
                }.values()
            )
            text_state = fit_text(fit_texts, max_features=max_features)
            part = text_features(
                [rows[index]["prompt"] for index in indices], text_state
            )
        else:
            raw = np.stack(
                [prompt_vectors[rows[index]["prompt_key"]] for index in indices]
            )
            fit_raw = np.stack(
                [prompt_vectors[rows[index]["prompt_key"]] for index in fit_indices]
            )
            mean, scale = standardize_fit(fit_raw)
            part = normalize((raw - mean) / scale)
        parts.append(part)
    if use_state:
        source = shuffled_state if family == "prompt_state_shuffled" else state
        raw = np.stack([source[rows[index]["group_id"]] for index in indices])
        fit_raw = np.stack([source[rows[index]["group_id"]] for index in fit_indices])
        mean, scale = standardize_fit(fit_raw)
        parts.append((raw - mean) / scale)
    if not parts:
        return None
    context = np.concatenate(parts, axis=1)
    levels = np.stack([rows[index]["levels"] for index in indices])
    return interaction_features(context, levels)


def prompt_folds(
    rows: list[dict[str, Any]], indices: np.ndarray, folds: int, seed: int
) -> np.ndarray:
    prompts = sorted({rows[index]["prompt_key"] for index in indices})
    if len(prompts) < folds:
        raise ValueError("not enough train prompts for prompt-disjoint folds")
    shuffled = np.random.default_rng(seed).permutation(prompts)
    assignment = {}
    for fold, chunk in enumerate(np.array_split(shuffled, folds)):
        assignment.update({str(prompt): fold for prompt in chunk})
    return np.asarray([assignment[rows[index]["prompt_key"]] for index in indices])


def select_and_predict(
    rows: list[dict[str, Any]],
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    family: str,
    prompt_mode: str,
    prompt_vectors: dict[str, np.ndarray] | None,
    state: dict[str, np.ndarray],
    shuffled_state: dict[str, np.ndarray],
    folds: int,
    seed: int,
    max_features: int,
    alphas: list[float],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], float | None]:
    fold_ids = prompt_folds(rows, train_indices, folds, seed)
    candidates: list[float | None] = (
        [None] if family == "action_main" else [None, *alphas]
    )
    oof_by_candidate = [
        np.zeros(len(train_indices), dtype=np.float64) for _ in candidates
    ]
    for fold in range(folds):
        fit_local = np.flatnonzero(fold_ids != fold)
        held_local = np.flatnonzero(fold_ids == fold)
        fit_ids = train_indices[fit_local]
        held_ids = train_indices[held_local]
        main_fit = np.stack([rows[index]["main"] for index in fit_ids])
        main_held = np.stack([rows[index]["main"] for index in held_ids])
        target_fit = np.asarray([rows[index]["target"] for index in fit_ids])
        context_fit = build_context(
            rows,
            fit_ids,
            fit_ids,
            family=family,
            prompt_mode=prompt_mode,
            prompt_vectors=prompt_vectors,
            state=state,
            shuffled_state=shuffled_state,
            max_features=max_features,
        )
        context_held = build_context(
            rows,
            held_ids,
            fit_ids,
            family=family,
            prompt_mode=prompt_mode,
            prompt_vectors=prompt_vectors,
            state=state,
            shuffled_state=shuffled_state,
            max_features=max_features,
        )
        for candidate_index, alpha in enumerate(candidates):
            model = fit_partial_ridge(main_fit, context_fit, target_fit, alpha)
            oof_by_candidate[candidate_index][held_local] = predict_partial_ridge(
                model, main_held, context_held
            )
    truth = np.asarray([rows[index]["target"] for index in train_indices])
    losses = [
        float(np.mean((prediction - truth) ** 2)) for prediction in oof_by_candidate
    ]
    selected_index = min(
        range(len(candidates)), key=lambda index: (losses[index], index)
    )
    selected_alpha = candidates[selected_index]
    full_main = np.stack([rows[index]["main"] for index in train_indices])
    eval_main = np.stack([rows[index]["main"] for index in evaluation_indices])
    full_context = build_context(
        rows,
        train_indices,
        train_indices,
        family=family,
        prompt_mode=prompt_mode,
        prompt_vectors=prompt_vectors,
        state=state,
        shuffled_state=shuffled_state,
        max_features=max_features,
    )
    eval_context = build_context(
        rows,
        evaluation_indices,
        train_indices,
        family=family,
        prompt_mode=prompt_mode,
        prompt_vectors=prompt_vectors,
        state=state,
        shuffled_state=shuffled_state,
        max_features=max_features,
    )
    model = fit_partial_ridge(full_main, full_context, truth, selected_alpha)
    evaluation_prediction = predict_partial_ridge(model, eval_main, eval_context)
    diagnostics = [
        {
            "family": family,
            "alpha": alpha,
            "selected": index == selected_index,
            "train_prompt_oof_mse": losses[index],
        }
        for index, alpha in enumerate(candidates)
    ]
    return (
        oof_by_candidate[selected_index],
        evaluation_prediction,
        diagnostics,
        selected_alpha,
    )


def prompt_cluster_ci(
    values: dict[str, list[float]], *, seed: int, repetitions: int
) -> tuple[float, float]:
    prompt_values = np.asarray(
        [np.mean(items) for items in values.values()], dtype=np.float64
    )
    if not len(prompt_values):
        raise ValueError("cannot bootstrap empty prompt clusters")
    if len(prompt_values) == 1:
        return float(prompt_values[0]), float(prompt_values[0])
    rng = np.random.default_rng(seed)
    draws = np.mean(
        prompt_values[
            rng.integers(0, len(prompt_values), size=(repetitions, len(prompt_values)))
        ],
        axis=1,
    )
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def evaluate_policies(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    split: str,
    utility_lambda: float,
    observation_cost_ratio: float,
    harm_epsilon: float,
    bootstrap_repetitions: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    local_position = {
        int(row_index): position for position, row_index in enumerate(indices)
    }
    groups: dict[str, list[int]] = {}
    for row_index in indices:
        groups.setdefault(rows[int(row_index)]["group_id"], []).append(int(row_index))
    summary = []
    details = []
    policies = ["reference", *predictions]
    realized_by_policy: dict[str, dict[str, float]] = {name: {} for name in policies}
    prompt_by_group = {
        group: rows[group_rows[0]]["prompt_key"] for group, group_rows in groups.items()
    }
    for group, group_rows in sorted(groups.items()):
        truth = np.asarray([rows[index]["target"] for index in group_rows])
        action_ids = [rows[index]["action_id"] for index in group_rows]
        oracle_slot = int(np.argmax(np.concatenate(([0.0], truth))))
        oracle_action = "REFERENCE" if oracle_slot == 0 else action_ids[oracle_slot - 1]
        for policy in policies:
            if policy == "reference":
                selected_slot = 0
            else:
                predicted = np.asarray(
                    [predictions[policy][local_position[index]] for index in group_rows]
                )
                selected_slot = int(np.argmax(np.concatenate(([0.0], predicted))))
            selected_action = (
                "REFERENCE" if selected_slot == 0 else action_ids[selected_slot - 1]
            )
            realized = 0.0 if selected_slot == 0 else float(truth[selected_slot - 1])
            uses_state = policy in {
                "state_only",
                "prompt_state",
                "prompt_state_shuffled",
            }
            observation_penalty = (
                utility_lambda * observation_cost_ratio if uses_state else 0.0
            )
            net_realized = realized - observation_penalty
            oracle = float(max(0.0, truth.max()))
            realized_by_policy[policy][group] = net_realized
            details.append(
                {
                    "split": split,
                    "group_id": group,
                    "prompt_key": prompt_by_group[group],
                    "base_seed": rows[group_rows[0]]["base_seed"],
                    "policy": policy,
                    "selected_action": selected_action,
                    "oracle_action": oracle_action,
                    "gross_delta_utility": realized,
                    "observation_penalty": observation_penalty,
                    "net_delta_utility": net_realized,
                    "sampled_oracle_delta_utility": oracle,
                    "policy_regret": oracle - net_realized,
                    "material_harm": net_realized < -harm_epsilon,
                }
            )
    action_main = realized_by_policy["action_main"]
    for policy in policies:
        values = realized_by_policy[policy]
        prompt_values: dict[str, list[float]] = {}
        prompt_delta_main: dict[str, list[float]] = {}
        policy_rows = [row for row in details if row["policy"] == policy]
        for row in policy_rows:
            prompt_values.setdefault(row["prompt_key"], []).append(
                row["net_delta_utility"]
            )
            prompt_delta_main.setdefault(row["prompt_key"], []).append(
                row["net_delta_utility"] - action_main[row["group_id"]]
            )
        ci = prompt_cluster_ci(
            prompt_values, seed=seed, repetitions=bootstrap_repetitions
        )
        delta_ci = prompt_cluster_ci(
            prompt_delta_main, seed=seed + 1, repetitions=bootstrap_repetitions
        )
        summary.append(
            {
                "split": split,
                "policy": policy,
                "prompts": len(prompt_values),
                "groups": len(values),
                "mean_net_delta_utility": float(np.mean(list(values.values()))),
                "utility_ci_low": ci[0],
                "utility_ci_high": ci[1],
                "gain_vs_action_main": float(
                    np.mean([values[group] - action_main[group] for group in values])
                ),
                "gain_vs_action_main_ci_low": delta_ci[0],
                "gain_vs_action_main_ci_high": delta_ci[1],
                "mean_sampled_regret": float(
                    np.mean([row["policy_regret"] for row in policy_rows])
                ),
                "material_harm_rate": float(
                    np.mean([row["material_harm"] for row in policy_rows])
                ),
                "nonreference_rate": float(
                    np.mean(
                        [row["selected_action"] != "REFERENCE" for row in policy_rows]
                    )
                ),
                "exact_sampled_oracle_rate": float(
                    np.mean(
                        [
                            row["selected_action"] == row["oracle_action"]
                            for row in policy_rows
                        ]
                    )
                ),
                "observation_cost_ratio": observation_cost_ratio
                if policy in {"state_only", "prompt_state", "prompt_state_shuffled"}
                else 0.0,
            }
        )
    return summary, details


def prepare_rows(
    pairs: list[dict[str, str]], utility_lambda: float
) -> list[dict[str, Any]]:
    rows = []
    for raw in pairs:
        if raw["cohort"] not in {"existing_train", "fresh_development"}:
            raise ValueError(f"unsupported cohort: {raw['cohort']}")
        levels = axis_levels(raw)
        delta_quality = float(raw["delta_vbench5"])
        time_ratio = float(raw["time_ratio_to_reference"])
        target = delta_quality - utility_lambda * (time_ratio - 1.0)
        if not all(
            math.isfinite(value) for value in (delta_quality, time_ratio, target)
        ):
            raise ValueError("non-finite sparse target")
        rows.append(
            {
                "observation_id": raw["observation_id"],
                "group_id": raw["group_id"],
                "prompt_key": raw["prompt_key"],
                "cohort": raw["cohort"],
                "prompt": raw["prompt"],
                "base_seed": int(raw["base_seed"]),
                "action_id": raw["action_id"],
                "levels": levels,
                "main": action_main_basis(levels),
                "delta_vbench5": delta_quality,
                "time_ratio_to_reference": time_ratio,
                "target": target,
            }
        )
    return rows


def lightx2v_python_env(lightx2v_repo: str | Path) -> dict[str, str]:
    """Build an explicit import environment for the Wan-native T5 backend."""

    repo = Path(lightx2v_repo).resolve()
    package = repo / "lightx2v"
    if not package.is_dir():
        raise FileNotFoundError(
            f"LightX2V Python package not found: {package}; set --lightx2v-repo"
        )
    environment = dict(os.environ)
    python_roots = [str(repo), str(ROOT)]
    if environment.get("PYTHONPATH"):
        python_roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_roots)
    return environment


def embed_prompts(args: argparse.Namespace) -> None:
    from changing_resolution_uni.scripts.data.extract_prompt_t5_embeddings import (
        directory_file_inventory,
    )

    scored_dir = Path(args.scored_dir).resolve()
    scored, _, pairs = validate_scored_dir(scored_dir)
    unique = {row["prompt_key"]: row["prompt"] for row in pairs}
    texts = [unique[key] for key in sorted(unique)]
    directory = Path(args.t5_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    model = Path(args.model_root).resolve()
    extractor = (
        ROOT / "changing_resolution_uni/scripts/data/extract_prompt_t5_embeddings.py"
    )
    request = {
        "source_input_sha256": scored["input_sha256"],
        "prompts": texts,
        "checkpoint_sha256": sha256_file(model / "models_t5_umt5-xxl-enc-bf16.pth"),
        "tokenizer_files": directory_file_inventory(model / "google/umt5-xxl"),
        "extractor_sha256": sha256_file(extractor),
        "precision": "bf16",
        "max_seq_len": 512,
    }
    request_path = directory / "sparse_prompt_state_embedding_request.json"
    with output_lock(directory):
        if request_path.is_file() and read_json(request_path) != request:
            raise ValueError("T5 extraction request changed; use a new --t5-dir")
        if not request_path.is_file() and list(directory.glob("prompt_*.npz")):
            raise ValueError("existing unbound T5 cache; use a new --t5-dir")
        if not request_path.is_file():
            write_json_atomic(request_path, request)
        samples = [{"prompt": text} for text in texts]
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
            env=lightx2v_python_env(args.lightx2v_repo),
            check=True,
        )
        load_t5(directory, samples)


def train(args: argparse.Namespace) -> None:
    scored_dir = Path(args.scored_dir).resolve()
    scored, _, pairs = validate_scored_dir(scored_dir)
    rows = prepare_rows(pairs, args.utility_lambda)
    expected_groups = {row["group_id"] for row in rows}
    state, state_manifest = load_state_features(
        Path(args.state_dir).resolve(),
        expected_input_sha256=scored["input_sha256"],
        expected_groups=expected_groups,
    )
    shuffled = shuffled_state_map(rows, state)
    train_ids = np.asarray(
        [index for index, row in enumerate(rows) if row["cohort"] == "existing_train"]
    )
    fresh_ids = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["cohort"] == "fresh_development"
        ]
    )
    train_prompts = {rows[index]["prompt_key"] for index in train_ids}
    prompt_vectors, t5_digest = prompt_embedding_map(
        rows,
        mode=args.prompt_features,
        t5_dir=Path(args.t5_dir).resolve() if args.t5_dir else None,
        train_prompts=train_prompts,
    )
    all_predictions: dict[str, dict[str, np.ndarray]] = {"train": {}, "fresh": {}}
    cv_rows = []
    selected = {}
    for family in FAMILIES:
        oof, fresh, diagnostics, alpha = select_and_predict(
            rows,
            train_ids,
            fresh_ids,
            family=family,
            prompt_mode=args.prompt_features,
            prompt_vectors=prompt_vectors,
            state=state,
            shuffled_state=shuffled,
            folds=args.folds,
            seed=args.seed,
            max_features=args.max_features,
            alphas=args.alphas,
        )
        all_predictions["train"][family] = oof
        all_predictions["fresh"][family] = fresh
        cv_rows.extend(diagnostics)
        selected[family] = alpha
    summary = []
    details = []
    for indices, split_key, split_name in (
        (train_ids, "train", "existing_train_prompt_oof_selection"),
        (fresh_ids, "fresh", "fresh_development_holdout"),
    ):
        split_summary, split_details = evaluate_policies(
            rows,
            indices,
            all_predictions[split_key],
            split=split_name,
            utility_lambda=args.utility_lambda,
            observation_cost_ratio=args.observation_cost_ratio,
            harm_epsilon=args.harm_epsilon,
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        )
        summary.extend(split_summary)
        details.extend(split_details)
    prediction_rows = []
    for indices, split_key, split_name in (
        (train_ids, "train", "existing_train_prompt_oof_selection"),
        (fresh_ids, "fresh", "fresh_development_holdout"),
    ):
        for position, row_index in enumerate(indices):
            row = rows[int(row_index)]
            prediction_rows.append(
                {
                    "split": split_name,
                    "observation_id": row["observation_id"],
                    "group_id": row["group_id"],
                    "prompt_key": row["prompt_key"],
                    "base_seed": row["base_seed"],
                    "action_id": row["action_id"],
                    "target_delta_utility": row["target"],
                    **{
                        f"predicted_{family}": all_predictions[split_key][family][
                            position
                        ]
                        for family in FAMILIES
                    },
                }
            )
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    request = {
        "schema": MODEL_SCHEMA,
        "source_input_sha256": scored["input_sha256"],
        "source_scored_dataset_sha256": scored["dataset_sha256"],
        "state_manifest_sha256": state_manifest["manifest_sha256"],
        "state_role": state_manifest["observation_role"],
        "deployable_early_state": state_manifest["deployable_early_state"],
        "prompt_features": args.prompt_features,
        "t5_manifest_sha256": t5_digest,
        "utility_lambda": args.utility_lambda,
        "utility_definition": "delta_vbench5 - lambda * (time_ratio_to_reference - 1)",
        "observation_cost_ratio": args.observation_cost_ratio,
        "folds": args.folds,
        "seed": args.seed,
        "max_features": args.max_features,
        "alphas": args.alphas,
        "harm_epsilon": args.harm_epsilon,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "action_axes": list(AXES),
        "context_interaction_basis": "constant_plus_signed_axis_main_effects",
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    with output_lock(out):
        request_path = out / "training_request.json"
        if request_path.is_file() and read_json(request_path) != request:
            raise RuntimeError("training request changed; use a new --out-dir")
        if not request_path.is_file():
            write_json_atomic(request_path, request)
        csv_write(out / "model_cv.csv", cv_rows)
        csv_write(out / "predictions.csv", prediction_rows)
        csv_write(out / "policy_by_group.csv", details)
        csv_write(out / "policy_summary.csv", summary)
        body = {
            **request,
            "selected_alpha": selected,
            "evaluation_role": "development_existing_videos_with_posthoc_state_proxy",
            "test_accessed": False,
            "candidate_scope": "REFERENCE plus each group's three sampled probes",
            "results": summary,
        }
        write_json_atomic(
            out / "policy_summary.json",
            {**body, "summary_sha256": canonical_sha256(body)},
        )
        fresh = {
            row["policy"]: row
            for row in summary
            if row["split"] == "fresh_development_holdout"
        }
        lines = [
            "# Sparse prompt + state routing development audit",
            "",
            f"Lambda: {args.utility_lambda:.6f}; prompt features: {args.prompt_features}; state: `{state_manifest['observation_role']}`.",
            "",
            "The state comes from the completed common REFERENCE video. It is a post-hoc capacity proxy, not deployable early-latent evidence. Candidate videos and VBench labels are fully reused; no action is evaluated outside each prompt's three sampled probes plus REFERENCE.",
            "",
            "The action model uses main/pairwise action terms and low-order axis-context interactions. Adding an operation grows the parameterization polynomially; it does not require collecting all 2^d combinations per prompt.",
            "",
            "| Fresh policy | Net dUtility | 95% CI | Regret | Gain vs action-main | Harm | Nonref | Exact oracle |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for policy in ("reference", *FAMILIES):
            row = fresh[policy]
            lines.append(
                f"| {policy} | {row['mean_net_delta_utility']:.6f} | "
                f"[{row['utility_ci_low']:.6f}, {row['utility_ci_high']:.6f}] | "
                f"{row['mean_sampled_regret']:.6f} | {row['gain_vs_action_main']:.6f} | "
                f"{row['material_harm_rate']:.1%} | {row['nonreference_rate']:.1%} | "
                f"{row['exact_sampled_oracle_rate']:.1%} |"
            )
        fusion = fresh["prompt_state"]
        prompt = fresh["prompt_only"]
        shuffled_control = fresh["prompt_state_shuffled"]
        lines += [
            "",
            "Primary gate for collecting true early latents:",
            "",
            "1. prompt_state improves fresh utility/regret over prompt_only;",
            "2. prompt_state also beats prompt_state_shuffled, showing seed-specific rather than prompt-identity signal;",
            "3. the gain survives the declared observation-cost penalty and does not materially increase harm.",
            "",
            f"Observed point estimates: fusion minus prompt-only = {fusion['mean_net_delta_utility'] - prompt['mean_net_delta_utility']:.6f}; fusion minus shuffled-state = {fusion['mean_net_delta_utility'] - shuffled_control['mean_net_delta_utility']:.6f}.",
            "",
            "A pass justifies replacing the proxy with a content-bound early latent or low-resolution preview for the same 267 prompt-seed groups. It does not by itself establish an online policy or a paper claim.",
        ]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote prompt/state audit: {out / 'report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "extract-proxy", "embed", "train"))
    parser.add_argument("--scored-dir", required=True)
    parser.add_argument("--state-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--prompt-features", choices=("tfidf", "t5"), default="tfidf")
    parser.add_argument("--t5-dir")
    parser.add_argument(
        "--model-root", default="/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B"
    )
    parser.add_argument("--lightx2v-repo", default="/mnt/afs_2/houze/LightX2V")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--proxy-size", type=int, default=64)
    parser.add_argument("--proxy-fps", type=float, default=4.0)
    parser.add_argument("--proxy-frames", type=int, default=16)
    parser.add_argument("--skip-video-hash", action="store_true")
    parser.add_argument("--utility-lambda", type=float, default=0.05)
    parser.add_argument("--observation-cost-ratio", type=float, default=0.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--max-features", type=int, default=4096)
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0]
    )
    parser.add_argument("--harm-epsilon", type=float, default=0.001)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    finite_nonnegative = (
        args.utility_lambda,
        args.observation_cost_ratio,
        args.harm_epsilon,
    )
    if (
        args.proxy_size < 8
        or args.proxy_frames < 2
        or not math.isfinite(args.proxy_fps)
        or args.proxy_fps <= 0
        or args.folds < 2
        or args.max_features < 1
        or args.bootstrap_repetitions < 100
        or any(not math.isfinite(value) or value < 0 for value in finite_nonnegative)
        or any(not math.isfinite(alpha) or alpha <= 0 for alpha in args.alphas)
    ):
        parser.error("invalid proxy, CV, utility, bootstrap, or ridge arguments")
    scored = Path(args.scored_dir).resolve()
    args.state_dir = args.state_dir or str(scored / "reference_video_proxy")
    lambda_tag = f"{args.utility_lambda:.6g}".replace(".", "p")
    args.out_dir = args.out_dir or str(
        scored / f"prompt_state_{args.prompt_features}_lambda_{lambda_tag}"
    )
    args.t5_dir = args.t5_dir or str(scored / "t5_sparse_prompt_state")
    return args


def main() -> None:
    args = parse_args()
    scored, references, pairs = validate_scored_dir(Path(args.scored_dir).resolve())
    print(
        f"Verified sparse scores: {len(references)} groups, {len(pairs)} relative pairs, "
        f"input={scored['input_sha256'][:12]}",
        flush=True,
    )
    if args.mode == "check":
        return
    if args.mode == "extract-proxy":
        extract_video_proxy(args)
    elif args.mode == "embed":
        embed_prompts(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
