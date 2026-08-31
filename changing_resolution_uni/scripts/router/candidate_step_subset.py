"""Validated candidate-step subsetting shared by router training entrypoints."""

from __future__ import annotations

from typing import Any

import numpy as np


def resolve_candidate_subset(
    source_steps: np.ndarray, requested_steps: list[int] | None
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_steps, dtype=np.int64)
    if source.ndim != 1 or source.size < 2 or np.any(np.diff(source) <= 0):
        raise ValueError(
            "Manifest candidate steps must be a strictly increasing vector"
        )
    if requested_steps is None:
        return np.arange(source.size, dtype=np.int64), source.copy()
    requested = np.asarray(requested_steps, dtype=np.int64)
    if (
        requested.ndim != 1
        or requested.size < 2
        or np.unique(requested).size != requested.size
        or np.any(np.diff(requested) <= 0)
    ):
        raise ValueError(
            "--candidate-steps must be unique, strictly increasing, and contain at least two steps"
        )
    source_index = {int(step): index for index, step in enumerate(source)}
    missing = [int(step) for step in requested if int(step) not in source_index]
    if missing:
        raise ValueError(
            f"Requested candidate steps are absent from the dataset: {missing}"
        )
    if int(requested[-1]) != int(source[-1]):
        raise ValueError(
            f"Candidate subset must retain forced final step {int(source[-1])}"
        )
    indices = np.asarray(
        [source_index[int(step)] for step in requested], dtype=np.int64
    )
    return indices, requested


def subset_trajectory_candidates(
    trajectories: list[dict[str, Any]], indices: np.ndarray
) -> None:
    candidate_fields = (
        "features",
        "sigmas",
        "qualities",
        "costs",
        "latencies",
        "dimensions",
    )
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size < 2:
        raise ValueError("Candidate indices must be a vector with at least two items")
    for trajectory in trajectories:
        source_count = int(np.asarray(trajectory["qualities"]).shape[0])
        if np.any(selected < 0) or np.any(selected >= source_count):
            raise ValueError("Candidate subset index is outside a trajectory")
        for field in candidate_fields:
            value = np.asarray(trajectory[field])
            if value.shape[0] != source_count:
                raise ValueError(
                    f"Trajectory field {field} does not share the candidate axis"
                )
            trajectory[field] = value[selected].copy()
