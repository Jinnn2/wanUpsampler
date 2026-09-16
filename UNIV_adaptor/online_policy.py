from __future__ import annotations

import math
import random
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


SCHEMA = "univ_online_policy_existence_v1"
QUALITY_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
)
DIAGNOSTIC_DIMENSIONS = ("dynamic_degree", "overall_consistency")
CASE_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


def suffix_compute_steps(
    *, decision_step: int, reference_nfe: int, remaining_full_compute: int
) -> tuple[int, ...]:
    """Choose exact, endpoint-preserving full-compute positions after a decision.

    ``decision_step`` is the number of solver updates already completed. The
    returned indices are absolute zero-based positions in the reference grid.
    The first remaining position is recomputed when at least two suffix
    computations are available; the final position is always recomputed.
    """

    if not 1 <= decision_step < reference_nfe:
        raise ValueError("decision_step must be in [1, reference_nfe)")
    remaining = reference_nfe - decision_step
    if not 1 <= remaining_full_compute <= remaining:
        raise ValueError(
            "remaining_full_compute must be in [1, reference_nfe - decision_step]"
        )
    if remaining_full_compute == 1:
        return (reference_nfe - 1,)
    if remaining_full_compute == remaining:
        return tuple(range(decision_step, reference_nfe))
    local = tuple(
        int(round(i * (remaining - 1) / (remaining_full_compute - 1)))
        for i in range(remaining_full_compute)
    )
    if len(set(local)) != remaining_full_compute:
        raise RuntimeError(f"suffix top-k construction produced duplicates: {local}")
    return tuple(decision_step + index for index in local)


def full_compute_steps(
    *, decision_step: int, reference_nfe: int, remaining_full_compute: int
) -> tuple[int, ...]:
    return tuple(range(decision_step)) + suffix_compute_steps(
        decision_step=decision_step,
        reference_nfe=reference_nfe,
        remaining_full_compute=remaining_full_compute,
    )


def validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if spec.get("schema") != SCHEMA:
        raise ValueError(f"spec.schema must equal {SCHEMA!r}")
    normalized = dict(spec)
    reference_nfe = int(normalized.get("reference_nfe", 0))
    decision_step = int(normalized.get("decision_step", 0))
    if reference_nfe != 50:
        raise ValueError(
            "the first policy-existence experiment requires reference_nfe=50"
        )
    if not 2 <= decision_step <= reference_nfe - 2:
        raise ValueError(
            "decision_step must leave at least two prefix and suffix positions"
        )
    normalized["reference_nfe"] = reference_nfe
    normalized["decision_step"] = decision_step
    initial_group = str(normalized.get("initial_group", "")).strip()
    reference_case = str(normalized.get("reference_case", "")).strip()
    if not initial_group or not reference_case:
        raise ValueError("initial_group and reference_case are required")
    normalized["initial_group"] = initial_group
    normalized["reference_case"] = reference_case

    cases = normalized.get("cases")
    if not isinstance(cases, list) or len(cases) != 8:
        raise ValueError("the 8-GPU pilot must define exactly eight cases")
    names: list[str] = []
    normalized_cases: list[dict[str, Any]] = []
    for raw in cases:
        if not isinstance(raw, Mapping):
            raise ValueError("each case must be an object")
        case = dict(raw)
        name = str(case.get("name", "")).strip()
        group = str(case.get("initial_group", "")).strip()
        role = str(case.get("role", "")).strip()
        if not name or not group or role not in {"continue", "restart"}:
            raise ValueError(
                "each case needs name, initial_group, and continue/restart role"
            )
        if CASE_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(
                f"case name must contain only lowercase letters, digits, _ or -: {name}"
            )
        action = case.get("initial_action")
        if not isinstance(action, Mapping):
            raise ValueError(f"case {name} has no initial_action")
        spatial = float(action.get("spatial_ratio", 0.0))
        temporal = float(action.get("temporal_ratio", 0.0))
        if not 0.5 <= spatial <= 1.0 or not 0.0 < temporal <= 1.0:
            raise ValueError(f"case {name} has an invalid initial action")
        case["initial_action"] = {
            "spatial_ratio": spatial,
            "temporal_ratio": temporal,
        }
        case["remaining_lr_full_compute"] = int(
            case.get("remaining_lr_full_compute", 0)
        )
        suffix_compute_steps(
            decision_step=decision_step,
            reference_nfe=reference_nfe,
            remaining_full_compute=case["remaining_lr_full_compute"],
        )
        case["hr_refine_sigma"] = float(case.get("hr_refine_sigma", 0.0))
        case["hr_steps"] = int(case.get("hr_steps", -1))
        if case["hr_steps"] == 0:
            if case["hr_refine_sigma"] != 0.0:
                raise ValueError(f"case {name}: HR0 requires sigma=0")
        elif not (case["hr_steps"] > 0 and 0.0 < case["hr_refine_sigma"] < 1.0):
            raise ValueError(
                f"case {name}: HR refinement needs steps>0 and sigma in (0,1)"
            )
        case["name"] = name
        case["initial_group"] = group
        case["role"] = role
        names.append(name)
        normalized_cases.append(case)
    if len(names) != len(set(names)):
        raise ValueError("case names must be unique")
    if reference_case not in names:
        raise ValueError("reference_case is absent from cases")
    continues = [case for case in normalized_cases if case["role"] == "continue"]
    restarts = [case for case in normalized_cases if case["role"] == "restart"]
    if len(continues) != 6 or len(restarts) != 2:
        raise ValueError(
            "the pilot requires six continuation and two restart candidates"
        )
    if {case["initial_group"] for case in continues} != {initial_group}:
        raise ValueError("all continuation cases must share initial_group")
    continuation_actions = {
        tuple(sorted(case["initial_action"].items())) for case in continues
    }
    if len(continuation_actions) != 1:
        raise ValueError("all continuation cases must share one initial action")
    if any(case["initial_group"] == initial_group for case in restarts):
        raise ValueError("restart cases must use an alternative initial_group")
    if len({case["initial_group"] for case in restarts}) != 2:
        raise ValueError("restart cases must use two distinct initial groups")
    if len({tuple(sorted(case["initial_action"].items())) for case in restarts}) != 2:
        raise ValueError("restart cases must use two distinct initial actions")
    reference = next(
        case for case in normalized_cases if case["name"] == reference_case
    )
    if reference["role"] != "continue":
        raise ValueError("reference_case must be a continuation case")
    restart_suffixes = {
        (
            case["remaining_lr_full_compute"],
            case["hr_steps"],
            case["hr_refine_sigma"],
        )
        for case in restarts
    }
    if len(restart_suffixes) != 1:
        raise ValueError("restart cases must share one matched LR/HR suffix")
    matched_suffix = next(iter(restart_suffixes))
    matched_continues = [
        case
        for case in continues
        if (
            case["remaining_lr_full_compute"],
            case["hr_steps"],
            case["hr_refine_sigma"],
        )
        == matched_suffix
    ]
    if len(matched_continues) != 1:
        raise ValueError("exactly one continuation must match the restart LR/HR suffix")
    continuation_pairs = {
        (
            case["remaining_lr_full_compute"],
            case["hr_steps"],
            case["hr_refine_sigma"],
        )
        for case in continues
    }
    lr_levels = {case["remaining_lr_full_compute"] for case in continues}
    hr_levels = {(case["hr_steps"], case["hr_refine_sigma"]) for case in continues}
    if len(lr_levels) != 3 or len(hr_levels) != 2 or len(continuation_pairs) != 6:
        raise ValueError("continuation cases must form a complete 3 x 2 LR/HR matrix")
    base_action = continues[0]["initial_action"]
    base_density = base_action["spatial_ratio"] ** 2 * base_action["temporal_ratio"]
    for case in restarts:
        action = case["initial_action"]
        density_ratio = (
            action["spatial_ratio"] ** 2 * action["temporal_ratio"] / base_density
        )
        if not 0.95 <= density_ratio <= 1.05:
            raise ValueError(
                "restart initial actions must stay within 5% of balanced token density"
            )
    normalized["cases"] = normalized_cases

    lambdas = tuple(float(value) for value in normalized.get("lambda_values", ()))
    if not lambdas or any(not math.isfinite(value) or value < 0 for value in lambdas):
        raise ValueError("lambda_values must contain finite non-negative values")
    if tuple(sorted(set(lambdas))) != lambdas:
        raise ValueError("lambda_values must be unique and increasing")
    normalized["lambda_values"] = list(lambdas)
    normalized["primary_lambda"] = float(normalized.get("primary_lambda", -1))
    if normalized["primary_lambda"] not in lambdas:
        raise ValueError("primary_lambda must be present in lambda_values")
    normalized["near_tie_utility"] = float(normalized.get("near_tie_utility", 0.0))
    if normalized["near_tie_utility"] < 0:
        raise ValueError("near_tie_utility must be non-negative")
    return normalized


def mean_quality(scores: Mapping[str, float]) -> float:
    values = [float(scores[key]) for key in QUALITY_DIMENSIONS]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("quality dimensions must be finite values in [0,1]")
    return statistics.mean(values)


def bootstrap_mean_ci(
    values: Sequence[float], *, repetitions: int = 2000, seed: int = 20260907
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(
        statistics.mean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(repetitions)
    )
    lo = draws[int(0.025 * (repetitions - 1))]
    hi = draws[int(0.975 * (repetitions - 1))]
    return float(lo), float(hi)


def bootstrap_oracle_gain_ci(
    rows: Sequence[Mapping[str, float]],
    actions: Sequence[str],
    *,
    repetitions: int = 2000,
    seed: int = 20260907,
) -> tuple[float, float]:
    """Cluster-bootstrap oracle gain while reselecting the fixed action."""

    if not rows:
        raise ValueError("cannot bootstrap an empty sample")
    if len(actions) < 2 or len(set(actions)) != len(actions):
        raise ValueError("oracle bootstrap requires at least two unique actions")
    for row in rows:
        if any(action not in row for action in actions):
            raise ValueError("oracle bootstrap row is missing an action")
    rng = random.Random(seed)
    n = len(rows)
    draws = []
    for _ in range(repetitions):
        sampled = [rows[rng.randrange(n)] for _ in range(n)]
        oracle = statistics.mean(
            max(row[action] for action in actions) for row in sampled
        )
        fixed = max(
            statistics.mean(row[action] for row in sampled) for action in actions
        )
        draws.append(oracle - fixed)
    draws.sort()
    lo = draws[int(0.025 * (repetitions - 1))]
    hi = draws[int(0.975 * (repetitions - 1))]
    return float(lo), float(hi)


def action_counts(selected: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in selected:
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))
