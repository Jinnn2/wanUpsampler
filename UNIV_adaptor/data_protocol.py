from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import UniversalAction
from .schedule import resolve_schedule
from .transition import TRANSITION_BASELINES


PROTOCOL_SCHEMA = "univ_sparse_data_protocol_v1"
PLAN_SCHEMA = "univ_sparse_collection_plan_v1"
PROBE_PLAN_SCHEMA = "univ_probe_selection_plan_v1"
RECORD_SCHEMA = "univ_sparse_trajectory_record_v1"
QUALITY_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "native_fidelity",
)
COLLECTION_MODES = frozenset({"sparse_train", "dense_oracle"})


@dataclass(frozen=True)
class CandidateAction:
    action: UniversalAction
    transition: str
    proxy_density: float

    @property
    def key(self) -> str:
        return action_key(self.action, self.transition)

    def as_dict(
        self,
        *,
        reference_nfe: int,
        target_latent_shape: tuple[int, int, int, int],
    ) -> dict[str, Any]:
        schedule = resolve_schedule(
            self.action,
            reference_nfe=reference_nfe,
            target_latent_shape=target_latent_shape,
        )
        return {
            "action_key": self.key,
            "requested_action": {
                "spatial_ratio": self.action.spatial_ratio,
                "temporal_ratio": self.action.temporal_ratio,
                "lr_nfe_ratio": self.action.lr_nfe_ratio,
                "switch_ratio": self.action.switch_ratio,
            },
            "transition": self.transition,
            "proxy_compute_density": self.proxy_density,
            "resolved_schedule": schedule.as_dict(),
        }


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_key(action: UniversalAction, transition: str) -> str:
    return (
        f"rs{action.spatial_ratio:.6f}_rt{action.temporal_ratio:.6f}_"
        f"rn{action.lr_nfe_ratio:.6f}_sw{action.switch_ratio:.6f}_"
        f"tr{transition}"
    )


def proxy_compute_density(action: UniversalAction) -> float:
    """Cheap planning proxy, not a substitute for measured latency.

    Full DiT work in the LR prefix is scaled by the approximate token ratio;
    the native HR suffix remains full cost. Cache updates and transition costs
    are deliberately excluded and must be supplied by the measured CostProfile.
    """

    action.validate()
    low_token_ratio = (
        action.spatial_ratio**2
        * action.temporal_ratio
        * action.lr_nfe_ratio
    )
    return action.switch_ratio * low_token_ratio + (1.0 - action.switch_ratio)


def enumerate_action_pool(protocol: Mapping[str, Any]) -> list[CandidateAction]:
    validated = validate_protocol(protocol, require_selected_probe=False)
    action_space = validated["action_space"]
    transitions = validated["transitions"]
    pool: list[CandidateAction] = []
    for transition in transitions:
        for spatial in action_space["spatial_ratios"]:
            for temporal in action_space["temporal_ratios"]:
                for nfe in action_space["lr_nfe_ratios"]:
                    for switch in action_space["switch_ratios"]:
                        action = UniversalAction(spatial, temporal, nfe, switch)
                        action.validate()
                        pool.append(
                            CandidateAction(
                                action=action,
                                transition=transition,
                                proxy_density=proxy_compute_density(action),
                            )
                        )
    keys = [candidate.key for candidate in pool]
    if len(keys) != len(set(keys)):
        raise ValueError("action_space produces duplicate action keys")
    return pool


def budget_feasible_pool(
    pool: Sequence[CandidateAction],
    *,
    target_density: float,
    tolerance: float,
) -> list[CandidateAction]:
    selected = [
        candidate
        for candidate in pool
        if abs(candidate.proxy_density - target_density) <= tolerance
    ]
    if selected:
        return selected
    nearest_error = min(
        abs(candidate.proxy_density - target_density) for candidate in pool
    )
    return [
        candidate
        for candidate in pool
        if math.isclose(
            abs(candidate.proxy_density - target_density),
            nearest_error,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]


def diverse_candidates(
    pool: Sequence[CandidateAction],
    *,
    count: int,
    seed: int,
) -> list[CandidateAction]:
    """Select a deterministic, spread-out subset without running generators."""

    if count < 1:
        raise ValueError("candidate count must be positive")
    if count >= len(pool):
        return list(pool)
    rng = random.Random(seed)
    remaining = list(pool)
    first = remaining.pop(rng.randrange(len(remaining)))
    selected = [first]
    while len(selected) < count:
        best_distance = -1.0
        best: list[CandidateAction] = []
        for candidate in remaining:
            distance = min(
                _action_distance(candidate.action, chosen.action)
                for chosen in selected
            )
            if distance > best_distance + 1e-12:
                best_distance = distance
                best = [candidate]
            elif math.isclose(distance, best_distance, abs_tol=1e-12):
                best.append(candidate)
        choice = best[rng.randrange(len(best))]
        selected.append(choice)
        remaining.remove(choice)
    return selected


def build_collection_plan(
    protocol: Mapping[str, Any],
    prompts: Sequence[str],
) -> dict[str, Any]:
    validated = validate_protocol(protocol, require_selected_probe=True)
    expected_prompts = sum(split["prompt_count"] for split in validated["splits"])
    if len(prompts) != expected_prompts:
        raise ValueError(
            f"protocol requires exactly {expected_prompts} prompts, got {len(prompts)}"
        )
    normalized_prompts = [str(prompt).strip() for prompt in prompts]
    if any(not prompt for prompt in normalized_prompts):
        raise ValueError("prompts must be non-empty")

    pool = enumerate_action_pool(validated)
    budgets = validated["budgets"]
    target_shape = tuple(validated["target_latent_shape"])
    reference_nfe = validated["reference_nfe"]
    assignments: list[dict[str, Any]] = []
    cursor = 0
    for split in validated["splits"]:
        for local_index in range(split["prompt_count"]):
            prompt_id = cursor + local_index
            prompt = normalized_prompts[prompt_id]
            budget_index = _stable_index(
                f"{validated['split_seed']}:{split['name']}:{prompt_id}:{prompt}",
                len(budgets["target_densities"]),
            )
            target_density = budgets["target_densities"][budget_index]
            feasible = budget_feasible_pool(
                pool,
                target_density=target_density,
                tolerance=budgets["proxy_tolerance"],
            )
            for base_seed in split["base_seeds"]:
                actual_seed = int(base_seed) + prompt_id
                trajectory_key = (
                    f"{split['name']}_p{prompt_id:06d}_s{actual_seed}_"
                    f"b{budget_index}"
                )
                candidate_slots = _candidate_slots(
                    split,
                    feasible,
                    trajectory_key=trajectory_key,
                    reference_nfe=reference_nfe,
                    target_latent_shape=target_shape,
                )
                assignments.append(
                    {
                        "trajectory_key": trajectory_key,
                        "split": split["name"],
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "prompt_sha256": canonical_sha256(prompt),
                        "base_seed": int(base_seed),
                        "seed": actual_seed,
                        "budget_id": f"density_{target_density:.3f}",
                        "target_proxy_density": target_density,
                        "candidate_slots": candidate_slots,
                    }
                )
        cursor += split["prompt_count"]

    immutable_protocol = json.loads(json.dumps(validated))
    protocol_hash = canonical_sha256(immutable_protocol)
    plan_body = {
        "protocol_schema": PROTOCOL_SCHEMA,
        "protocol_sha256": protocol_hash,
        "protocol": immutable_protocol,
        "prompt_count": len(normalized_prompts),
        "prompts_sha256": canonical_sha256(normalized_prompts),
        "action_pool_size": len(pool),
        "assignments": assignments,
    }
    return {
        "schema": PLAN_SCHEMA,
        "plan_sha256": canonical_sha256(plan_body),
        **plan_body,
    }


def build_probe_selection_plan(
    protocol: Mapping[str, Any],
    prompts: Sequence[str],
) -> dict[str, Any]:
    validated = validate_protocol(protocol, require_selected_probe=False)
    selection = validated["probe_selection"]
    prompt_count = selection["prompt_count"]
    prompt_offset = selection["prompt_offset"]
    selected_prompts = [str(value).strip() for value in prompts][
        prompt_offset : prompt_offset + prompt_count
    ]
    if len(selected_prompts) != prompt_count or any(not value for value in selected_prompts):
        raise ValueError(
            "prompt file does not cover the configured probe-selection slice"
        )

    pool = enumerate_action_pool(validated)
    budgets = validated["budgets"]
    reference_nfe = validated["reference_nfe"]
    target_shape = tuple(validated["target_latent_shape"])
    probe_candidates = validated["common_probe"]["candidates"]
    assignments: list[dict[str, Any]] = []
    for local_index, prompt in enumerate(selected_prompts):
        prompt_id = prompt_offset + local_index
        budget_index = _stable_index(
            f"probe:{validated['split_seed']}:{prompt_id}:{prompt}",
            len(budgets["target_densities"]),
        )
        target_density = budgets["target_densities"][budget_index]
        feasible = budget_feasible_pool(
            pool,
            target_density=target_density,
            tolerance=budgets["proxy_tolerance"],
        )
        for base_seed in selection["base_seeds"]:
            actual_seed = int(base_seed) + prompt_id
            trajectory_key = f"probe_p{prompt_id:06d}_s{actual_seed}_b{budget_index}"
            downstream = diverse_candidates(
                feasible,
                count=selection["downstream_candidate_count"],
                seed=int.from_bytes(
                    hashlib.sha256(trajectory_key.encode("utf-8")).digest()[:8],
                    "big",
                ),
            )
            downstream_payload = [
                candidate.as_dict(
                    reference_nfe=reference_nfe,
                    target_latent_shape=target_shape,
                )
                for candidate in downstream
            ]
            assignments.append(
                {
                    "trajectory_key": trajectory_key,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "prompt_sha256": canonical_sha256(prompt),
                    "base_seed": int(base_seed),
                    "seed": actual_seed,
                    "budget_id": f"density_{target_density:.3f}",
                    "target_proxy_density": target_density,
                    "native_teacher_required": True,
                    "probe_branches": [
                        {
                            "probe": probe,
                            "downstream_candidates": downstream_payload,
                        }
                        for probe in probe_candidates
                    ],
                }
            )

    immutable_protocol = json.loads(json.dumps(validated))
    protocol_hash = canonical_sha256(immutable_protocol)
    plan_body = {
        "protocol_schema": PROTOCOL_SCHEMA,
        "protocol_sha256": protocol_hash,
        "protocol": immutable_protocol,
        "prompt_count": prompt_count,
        "prompts_sha256": canonical_sha256(selected_prompts),
        "probe_candidate_count": len(probe_candidates),
        "downstream_candidate_count": selection["downstream_candidate_count"],
        "action_pool_size": len(pool),
        "assignments": assignments,
    }
    return {
        "schema": PROBE_PLAN_SCHEMA,
        "plan_sha256": canonical_sha256(plan_body),
        **plan_body,
    }


def validate_protocol(
    protocol: Mapping[str, Any],
    *,
    require_selected_probe: bool,
) -> dict[str, Any]:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"protocol.schema must be {PROTOCOL_SCHEMA!r}")
    normalized = json.loads(json.dumps(protocol))
    reference_nfe = int(normalized.get("reference_nfe", 0))
    if reference_nfe != 50:
        raise ValueError("the first UNIV data protocol requires reference_nfe=50")
    shape = normalized.get("target_latent_shape")
    if not isinstance(shape, list) or len(shape) != 4 or min(map(int, shape)) <= 0:
        raise ValueError("target_latent_shape must contain four positive integers")
    normalized["target_latent_shape"] = [int(value) for value in shape]

    transitions = normalized.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("transitions must be a non-empty list")
    if len(transitions) != len(set(transitions)):
        raise ValueError("transitions must be unique")
    invalid_transitions = set(transitions) - set(TRANSITION_BASELINES)
    if invalid_transitions:
        raise ValueError(f"unsupported transitions: {sorted(invalid_transitions)}")

    action_space = normalized.get("action_space")
    if not isinstance(action_space, dict):
        raise ValueError("action_space must be an object")
    fields = (
        "spatial_ratios",
        "temporal_ratios",
        "lr_nfe_ratios",
        "switch_ratios",
    )
    for field in fields:
        values = action_space.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"action_space.{field} must be a non-empty list")
        numbers = [float(value) for value in values]
        if len(numbers) != len(set(numbers)):
            raise ValueError(f"action_space.{field} must contain unique values")
        action_space[field] = numbers
    for spatial in action_space["spatial_ratios"]:
        for temporal in action_space["temporal_ratios"]:
            for nfe in action_space["lr_nfe_ratios"]:
                for switch in action_space["switch_ratios"]:
                    UniversalAction(spatial, temporal, nfe, switch).validate()

    budgets = normalized.get("budgets")
    if not isinstance(budgets, dict):
        raise ValueError("budgets must be an object")
    densities = [float(value) for value in budgets.get("target_densities", [])]
    if not densities or any(not 0.0 < value <= 1.0 for value in densities):
        raise ValueError("budgets.target_densities must be in (0, 1]")
    if len(densities) != len(set(densities)):
        raise ValueError("budgets.target_densities must be unique")
    tolerance = float(budgets.get("proxy_tolerance", 0.0))
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("budgets.proxy_tolerance must be in [0, 1)")
    budgets["target_densities"] = densities
    budgets["proxy_tolerance"] = tolerance

    probe = normalized.get("common_probe")
    if not isinstance(probe, dict):
        raise ValueError("common_probe must be an object")
    selected = probe.get("selected")
    if require_selected_probe and not isinstance(selected, dict):
        raise ValueError(
            "common_probe.selected is required before controller data planning"
        )
    if isinstance(selected, dict):
        _validate_probe(selected, reference_nfe=reference_nfe)
    candidates = probe.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("common_probe.candidates must be a non-empty list")
    ids = []
    for candidate in candidates:
        _validate_probe(candidate, reference_nfe=reference_nfe)
        ids.append(candidate["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("common probe candidate ids must be unique")

    splits = normalized.get("splits")
    if not isinstance(splits, list) or not splits:
        raise ValueError("splits must be a non-empty list")
    split_names = []
    for split in splits:
        if not isinstance(split, dict):
            raise ValueError("each split must be an object")
        name = str(split.get("name", "")).strip()
        if not name:
            raise ValueError("split.name must be non-empty")
        split_names.append(name)
        split["name"] = name
        split["prompt_count"] = int(split.get("prompt_count", 0))
        if split["prompt_count"] < 1:
            raise ValueError(f"split {name} prompt_count must be positive")
        seeds = [int(value) for value in split.get("base_seeds", [])]
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError(f"split {name} base_seeds must be non-empty and unique")
        split["base_seeds"] = seeds
        mode = str(split.get("collection_mode", ""))
        if mode not in COLLECTION_MODES:
            raise ValueError(
                f"split {name} collection_mode must be one of {sorted(COLLECTION_MODES)}"
            )
        split["collection_mode"] = mode
        split["candidate_count"] = int(split.get("candidate_count", 0))
        minimum = 2 if mode == "sparse_train" else 3
        if split["candidate_count"] < minimum:
            raise ValueError(
                f"split {name} candidate_count must be at least {minimum} for {mode}"
            )
    if len(split_names) != len(set(split_names)):
        raise ValueError("split names must be unique")

    probe_selection = normalized.get("probe_selection")
    if not isinstance(probe_selection, dict):
        raise ValueError("probe_selection must be an object")
    for field in ("prompt_offset", "prompt_count", "downstream_candidate_count"):
        probe_selection[field] = int(probe_selection.get(field, -1))
    if probe_selection["prompt_offset"] < 0:
        raise ValueError("probe_selection.prompt_offset must be non-negative")
    if probe_selection["prompt_count"] < 1:
        raise ValueError("probe_selection.prompt_count must be positive")
    if probe_selection["downstream_candidate_count"] < 2:
        raise ValueError(
            "probe_selection.downstream_candidate_count must be at least two"
        )
    probe_seeds = [int(value) for value in probe_selection.get("base_seeds", [])]
    if not probe_seeds or len(probe_seeds) != len(set(probe_seeds)):
        raise ValueError("probe_selection.base_seeds must be non-empty and unique")
    probe_selection["base_seeds"] = probe_seeds
    normalized["split_seed"] = int(normalized.get("split_seed", 0))
    return normalized


def validate_collection_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") not in {PLAN_SCHEMA, PROBE_PLAN_SCHEMA}:
        raise ValueError(
            f"plan.schema must be {PLAN_SCHEMA!r} or {PROBE_PLAN_SCHEMA!r}"
        )
    body = {key: value for key, value in plan.items() if key not in {"schema", "plan_sha256"}}
    if canonical_sha256(body) != plan.get("plan_sha256"):
        raise ValueError("collection plan hash mismatch")
    protocol = plan.get("protocol")
    validate_protocol(
        protocol,
        require_selected_probe=plan.get("schema") == PLAN_SCHEMA,
    )
    if canonical_sha256(protocol) != plan.get("protocol_sha256"):
        raise ValueError("collection plan protocol hash mismatch")
    assignments = plan.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("collection plan has no assignments")
    keys = [row.get("trajectory_key") for row in assignments]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("trajectory keys must be non-empty and unique")


def validate_trajectory_record(
    record: Mapping[str, Any],
    *,
    expected_plan_sha256: str | None = None,
) -> None:
    if record.get("schema") != RECORD_SCHEMA:
        raise ValueError(f"record.schema must be {RECORD_SCHEMA!r}")
    if expected_plan_sha256 and record.get("plan_sha256") != expected_plan_sha256:
        raise ValueError("trajectory record belongs to a different collection plan")
    for field in ("trajectory_key", "split", "prompt", "common_probe"):
        if not record.get(field):
            raise ValueError(f"trajectory record requires {field}")
    int(record["prompt_id"])
    int(record["seed"])
    common_probe = record["common_probe"]
    if not isinstance(common_probe, dict):
        raise ValueError("trajectory record common_probe must be an object")
    for field in ("id", "boundary_step", "boundary_sigma", "feature_path"):
        if field not in common_probe or common_probe[field] in (None, ""):
            raise ValueError(f"trajectory record common_probe requires {field}")
    if int(common_probe["boundary_step"]) <= 0:
        raise ValueError("common_probe.boundary_step must be positive")
    if not math.isfinite(float(common_probe["boundary_sigma"])):
        raise ValueError("common_probe.boundary_sigma must be finite")
    teacher = record.get("native_teacher")
    if not isinstance(teacher, dict):
        raise ValueError("trajectory record requires native_teacher")
    _validate_artifact(teacher, "native_teacher")
    _validate_quality_and_cost(teacher, "native_teacher")
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("trajectory record requires at least two candidates")
    keys: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidates[{index}] must be an object")
        keys.append(str(candidate.get("action_key", "")))
        requested = candidate.get("requested_action")
        if not isinstance(requested, dict):
            raise ValueError(f"candidates[{index}] requires requested_action")
        action = UniversalAction(
            float(requested["spatial_ratio"]),
            float(requested["temporal_ratio"]),
            float(requested["lr_nfe_ratio"]),
            float(requested["switch_ratio"]),
        )
        action.validate()
        transition = str(candidate.get("transition", ""))
        if transition not in TRANSITION_BASELINES:
            raise ValueError(f"candidates[{index}] has unsupported transition")
        if action_key(action, transition) != candidate.get("action_key"):
            raise ValueError(f"candidates[{index}] action_key mismatch")
        if not isinstance(candidate.get("resolved_schedule"), dict):
            raise ValueError(f"candidates[{index}] requires resolved_schedule")
        selection = candidate.get("selection")
        if not isinstance(selection, dict) or not str(selection.get("source", "")).strip():
            raise ValueError(f"candidates[{index}] requires selection source")
        propensity = float(selection.get("propensity", 0.0))
        if not 0.0 < propensity <= 1.0:
            raise ValueError(
                f"candidates[{index}].selection.propensity must be in (0, 1]"
            )
        _validate_artifact(candidate, f"candidates[{index}]")
        _validate_quality_and_cost(candidate, f"candidates[{index}]")
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("candidate action keys must be non-empty and unique")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("trajectory record requires provenance")
    for field in ("code_commit", "model_sha256", "config_sha256"):
        if not str(provenance.get(field, "")).strip():
            raise ValueError(f"trajectory provenance requires {field}")


def _candidate_slots(
    split: Mapping[str, Any],
    feasible: Sequence[CandidateAction],
    *,
    trajectory_key: str,
    reference_nfe: int,
    target_latent_shape: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    count = int(split["candidate_count"])
    selection_seed = int.from_bytes(
        hashlib.sha256(trajectory_key.encode("utf-8")).digest()[:8], "big"
    )
    if split["collection_mode"] == "sparse_train":
        exploration_count = count - 1
        selected = diverse_candidates(
            feasible,
            count=exploration_count,
            seed=selection_seed,
        )
        slots: list[dict[str, Any]] = [
            {
                "slot": 0,
                "selector": "dvg_runtime",
                "selection_status": "deferred_until_common_probe",
                "propensity_required": True,
            }
        ]
        slots.extend(
            {
                "slot": index + 1,
                "selector": "deterministic_space_filling_exploration",
                "selection_status": "resolved",
                "propensity_required": True,
                "candidate": candidate.as_dict(
                    reference_nfe=reference_nfe,
                    target_latent_shape=target_latent_shape,
                ),
            }
            for index, candidate in enumerate(selected)
        )
        return slots

    selected = diverse_candidates(feasible, count=count, seed=selection_seed)
    return [
        {
            "slot": index,
            "selector": "deterministic_dense_oracle_subset",
            "selection_status": "resolved",
            "propensity_required": False,
            "candidate": candidate.as_dict(
                reference_nfe=reference_nfe,
                target_latent_shape=target_latent_shape,
            ),
        }
        for index, candidate in enumerate(selected)
    ]


def _action_distance(left: UniversalAction, right: UniversalAction) -> float:
    values_left = (
        left.spatial_ratio,
        left.temporal_ratio,
        left.lr_nfe_ratio,
        left.switch_ratio,
    )
    values_right = (
        right.spatial_ratio,
        right.temporal_ratio,
        right.lr_nfe_ratio,
        right.switch_ratio,
    )
    ranges = (0.5, 1.0, 1.0, 0.4)
    return math.sqrt(
        sum(
            ((left_value - right_value) / scale) ** 2
            for left_value, right_value, scale in zip(
                values_left, values_right, ranges
            )
        )
    )


def _stable_index(value: str, size: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def _validate_probe(probe: Mapping[str, Any], *, reference_nfe: int) -> None:
    if not isinstance(probe, Mapping):
        raise ValueError("common probe entries must be objects")
    if not str(probe.get("id", "")).strip():
        raise ValueError("common probe entry requires id")
    stop_step = int(probe.get("stop_step", 0))
    full_compute = int(probe.get("full_compute_steps", 0))
    if not 1 <= full_compute <= stop_step < reference_nfe * 0.6:
        raise ValueError(
            "common probe must satisfy 1 <= full_compute_steps <= stop_step < 30"
        )
    UniversalAction(
        float(probe["spatial_ratio"]),
        float(probe["temporal_ratio"]),
        1.0,
        0.6,
    ).validate()


def _validate_artifact(value: Mapping[str, Any], field: str) -> None:
    output = str(value.get("video_path", "")).strip()
    digest = str(value.get("video_sha256", "")).strip().lower()
    if not output:
        raise ValueError(f"{field}.video_path must be non-empty")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field}.video_sha256 must be a lowercase SHA256 digest")


def _validate_quality_and_cost(value: Mapping[str, Any], field: str) -> None:
    quality = value.get("quality")
    if not isinstance(quality, dict):
        raise ValueError(f"{field} requires quality")
    for name in QUALITY_DIMENSIONS:
        number = float(quality[name])
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{field}.quality.{name} must be finite and in [0, 1]")
    cost = value.get("cost")
    if not isinstance(cost, dict) or float(cost.get("warm_pipeline_seconds", 0)) <= 0:
        raise ValueError(f"{field} requires positive measured cost")


def load_prompts(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def write_json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
