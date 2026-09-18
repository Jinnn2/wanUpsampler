from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import UniversalAction
from .data_protocol import action_key, canonical_sha256, proxy_compute_density
from .schedule import resolve_schedule
from .transition import TRANSITION_BASELINES


PROTOCOL_SCHEMA = "univ_sparse_action_protocol_v1"
PLAN_SCHEMA = "univ_sparse_action_plan_v1"
RECORD_SCHEMA = "univ_sparse_action_record_v1"
ACTION_FIELDS = (
    "spatial_ratio",
    "temporal_ratio",
    "lr_nfe_ratio",
    "switch_ratio",
)


def validate_sparse_protocol(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"protocol.schema must be {PROTOCOL_SCHEMA!r}")
    protocol = json.loads(json.dumps(value))
    if protocol.get("controller_factorization") != "prompt_x_action_relative_quality":
        raise ValueError(
            "controller_factorization must be 'prompt_x_action_relative_quality'"
        )
    if protocol.get("selection_scope") != "development_train_only":
        raise ValueError("selection_scope must be development_train_only")
    if protocol.get("lambda_binding") != "none_score_quality_and_time_separately":
        raise ValueError("sparse data collection must not be bound to one lambda")

    reference_nfe = int(protocol.get("reference_nfe", 0))
    if reference_nfe != 50:
        raise ValueError("reference_nfe must be 50")
    protocol["reference_nfe"] = reference_nfe
    shape = protocol.get("target_latent_shape")
    if not isinstance(shape, list) or len(shape) != 4 or min(map(int, shape)) <= 0:
        raise ValueError("target_latent_shape must contain four positive integers")
    protocol["target_latent_shape"] = [int(item) for item in shape]
    transition = str(protocol.get("transition", ""))
    if transition not in TRANSITION_BASELINES:
        raise ValueError(f"unsupported transition: {transition!r}")
    protocol["transition"] = transition

    dimensions = protocol.get("action_dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("action_dimensions must be a non-empty list")
    dimension_ids: list[str] = []
    dimension_fields: list[str] = []
    for dimension in dimensions:
        if not isinstance(dimension, dict):
            raise ValueError("each action dimension must be an object")
        dimension_id = str(dimension.get("id", "")).strip()
        field = str(dimension.get("field", "")).strip()
        if not dimension_id or field not in ACTION_FIELDS:
            raise ValueError("action dimensions require an id and supported field")
        levels = dimension.get("levels")
        if not isinstance(levels, list) or len(levels) != 2:
            raise ValueError(f"dimension {dimension_id} must have exactly two levels")
        levels = [float(item) for item in levels]
        if any(not math.isfinite(item) for item in levels) or levels[0] == levels[1]:
            raise ValueError(
                f"dimension {dimension_id} levels must be finite and distinct"
            )
        dimension["id"] = dimension_id
        dimension["field"] = field
        dimension["levels"] = levels
        dimension_ids.append(dimension_id)
        dimension_fields.append(field)
    if len(dimension_ids) != len(set(dimension_ids)):
        raise ValueError("action dimension ids must be unique")
    if len(dimension_fields) != len(set(dimension_fields)):
        raise ValueError("action dimension fields must be unique")

    fixed = protocol.get("fixed_action_fields")
    if not isinstance(fixed, dict):
        raise ValueError("fixed_action_fields must be an object")
    if set(fixed) | set(dimension_fields) != set(ACTION_FIELDS) or set(fixed) & set(
        dimension_fields
    ):
        raise ValueError(
            "fixed and variable action fields must cover each action field once"
        )
    protocol["fixed_action_fields"] = {
        key: float(value) for key, value in fixed.items()
    }

    reference = _action_dict(protocol.get("reference_action"))
    protocol["reference_action"] = reference
    reference_source_id = str(protocol.get("reference_source_action_id", "")).strip()
    if not reference_source_id:
        raise ValueError("reference_source_action_id must be non-empty")
    protocol["reference_source_action_id"] = reference_source_id

    library = protocol.get("probe_library")
    if not isinstance(library, list) or len(library) < 4:
        raise ValueError("probe_library must contain at least four explicit rows")
    library_ids: list[str] = []
    level_vectors: list[tuple[int, ...]] = []
    action_keys: list[str] = []
    for row in library:
        if not isinstance(row, dict):
            raise ValueError("each probe library row must be an object")
        probe_id = str(row.get("id", "")).strip()
        levels = row.get("levels")
        if (
            not probe_id
            or not isinstance(levels, dict)
            or set(levels) != set(dimension_ids)
        ):
            raise ValueError("probe rows require an id and one level per dimension")
        normalized_levels = {key: int(levels[key]) for key in dimension_ids}
        if any(level not in (0, 1) for level in normalized_levels.values()):
            raise ValueError("probe levels must be binary")
        action = action_for_levels(protocol, normalized_levels)
        row["id"] = probe_id
        row["levels"] = normalized_levels
        library_ids.append(probe_id)
        level_vectors.append(tuple(normalized_levels[key] for key in dimension_ids))
        action_keys.append(action_key(action, transition))
    if len(library_ids) != len(set(library_ids)):
        raise ValueError("probe ids must be unique")
    if len(level_vectors) != len(set(level_vectors)):
        raise ValueError("probe level vectors must be unique")
    if len(action_keys) != len(set(action_keys)):
        raise ValueError("probe rows must resolve to distinct runtime actions")

    offsets = [int(item) for item in protocol.get("assignment_offsets", [])]
    if not offsets or len(offsets) != len(set(offsets)):
        raise ValueError("assignment_offsets must be non-empty and unique")
    if any(item < 0 or item >= len(library) for item in offsets):
        raise ValueError("assignment_offsets must index probe_library")
    protocol["assignment_offsets"] = offsets
    probes_per_prompt = int(protocol.get("probes_per_prompt", 0))
    if probes_per_prompt != len(offsets):
        raise ValueError("probes_per_prompt must equal assignment_offsets length")
    protocol["probes_per_prompt"] = probes_per_prompt

    for field in ("existing_train_prompt_count", "fresh_development_prompt_count"):
        count = int(protocol.get(field, 0))
        if count < 1:
            raise ValueError(f"{field} must be positive")
        protocol[field] = count
    seeds = [int(item) for item in protocol.get("base_seeds", [])]
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("base_seeds must contain at least two unique seeds")
    protocol["base_seeds"] = seeds
    reuse_seed = int(protocol.get("reuse_reference_base_seed", -1))
    if reuse_seed not in seeds:
        raise ValueError("reuse_reference_base_seed must be in base_seeds")
    protocol["reuse_reference_base_seed"] = reuse_seed
    fresh_seed_offset = int(protocol.get("fresh_seed_offset", 0))
    if fresh_seed_offset < 1:
        raise ValueError("fresh_seed_offset must be positive")
    protocol["fresh_seed_offset"] = fresh_seed_offset
    selection_salt = str(protocol.get("selection_salt", "")).strip()
    design_salt = str(protocol.get("design_salt", "")).strip()
    if not selection_salt or not design_salt:
        raise ValueError("selection_salt and design_salt must be non-empty")
    protocol["selection_salt"] = selection_salt
    protocol["design_salt"] = design_salt
    return protocol


def action_for_levels(
    protocol: Mapping[str, Any], levels: Mapping[str, int]
) -> UniversalAction:
    values = {
        key: float(value) for key, value in protocol["fixed_action_fields"].items()
    }
    for dimension in protocol["action_dimensions"]:
        level = int(levels[dimension["id"]])
        if level not in (0, 1):
            raise ValueError("action levels must be binary")
        values[dimension["field"]] = float(dimension["levels"][level])
    action = UniversalAction(**values)
    action.validate()
    return action


def action_catalog(protocol_value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    protocol = validate_sparse_protocol(protocol_value)
    transition = protocol["transition"]
    shape = tuple(protocol["target_latent_shape"])

    def payload(
        action_id: str,
        action: UniversalAction,
        *,
        role: str,
        levels: Mapping[str, int] | None,
    ) -> dict[str, Any]:
        return {
            "action_id": action_id,
            "role": role,
            "levels": dict(levels) if levels is not None else None,
            "active_mask": {
                dimension["id"]: levels is not None
                for dimension in protocol["action_dimensions"]
            },
            "requested_action": _action_dict(action),
            "transition": transition,
            "action_key": action_key(action, transition),
            "proxy_compute_density": proxy_compute_density(action),
            "resolved_schedule": resolve_schedule(
                action,
                reference_nfe=protocol["reference_nfe"],
                target_latent_shape=shape,
            ).as_dict(),
        }

    result = {
        "REFERENCE": payload(
            "REFERENCE",
            UniversalAction(**protocol["reference_action"]),
            role="reference",
            levels=None,
        )
    }
    for row in protocol["probe_library"]:
        result[row["id"]] = payload(
            row["id"],
            action_for_levels(protocol, row["levels"]),
            role="probe",
            levels=row["levels"],
        )
    return result


def assign_probe_ids(
    protocol_value: Mapping[str, Any], prompt_keys: Sequence[str]
) -> dict[str, list[str]]:
    protocol = validate_sparse_protocol(protocol_value)
    if len(prompt_keys) != len(set(prompt_keys)):
        raise ValueError("prompt keys must be unique")
    library = [row["id"] for row in protocol["probe_library"]]
    ordered = sorted(
        prompt_keys,
        key=lambda key: canonical_sha256([protocol["design_salt"], key]),
    )
    assignments: dict[str, list[str]] = {}
    for index, prompt_key in enumerate(ordered):
        assignments[prompt_key] = [
            library[(index + offset) % len(library)]
            for offset in protocol["assignment_offsets"]
        ]
    return assignments


def design_balance(
    protocol_value: Mapping[str, Any], assignments: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    protocol = validate_sparse_protocol(protocol_value)
    library = {row["id"]: row["levels"] for row in protocol["probe_library"]}
    action_counts: Counter[str] = Counter()
    marginal: dict[str, Counter[int]] = {
        dimension["id"]: Counter() for dimension in protocol["action_dimensions"]
    }
    pairwise: dict[str, Counter[str]] = {}
    dimension_ids = [dimension["id"] for dimension in protocol["action_dimensions"]]
    for probe_ids in assignments.values():
        if len(probe_ids) != protocol["probes_per_prompt"] or len(probe_ids) != len(
            set(probe_ids)
        ):
            raise ValueError(
                "each prompt must receive distinct probes_per_prompt actions"
            )
        for probe_id in probe_ids:
            if probe_id not in library:
                raise ValueError(f"unknown probe id: {probe_id}")
            action_counts[probe_id] += 1
            levels = library[probe_id]
            for dimension_id in dimension_ids:
                marginal[dimension_id][int(levels[dimension_id])] += 1
            for left_index, left in enumerate(dimension_ids):
                for right in dimension_ids[left_index + 1 :]:
                    key = f"{left}__{right}"
                    pairwise.setdefault(key, Counter())[
                        f"{levels[left]}{levels[right]}"
                    ] += 1
    return {
        "prompt_count": len(assignments),
        "probe_observation_groups": sum(map(len, assignments.values())),
        "action_prompt_counts": dict(sorted(action_counts.items())),
        "dimension_marginal_counts": {
            key: {str(level): counts[level] for level in (0, 1)}
            for key, counts in marginal.items()
        },
        "pairwise_level_counts": {
            key: {state: counts[state] for state in ("00", "01", "10", "11")}
            for key, counts in pairwise.items()
        },
    }


def expected_counts(protocol_value: Mapping[str, Any]) -> dict[str, int]:
    protocol = validate_sparse_protocol(protocol_value)
    existing = protocol["existing_train_prompt_count"]
    fresh = protocol["fresh_development_prompt_count"]
    seeds = len(protocol["base_seeds"])
    probes = protocol["probes_per_prompt"]
    reused = existing
    generated_existing = existing * (seeds * probes + seeds - 1)
    generated_fresh = fresh * seeds * (probes + 1)
    return {
        "existing_prompts": existing,
        "fresh_prompts": fresh,
        "generated_videos_upper_bound": generated_existing + generated_fresh,
        "reused_videos_lower_bound": reused,
        "scored_videos": generated_existing + generated_fresh + reused,
        "prompt_seed_groups": (existing + fresh) * seeds,
    }


def validate_sparse_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"plan.schema must be {PLAN_SCHEMA!r}")
    body = {
        key: value
        for key, value in plan.items()
        if key not in {"schema", "plan_sha256"}
    }
    if canonical_sha256(body) != plan.get("plan_sha256"):
        raise ValueError("sparse action plan hash mismatch")
    protocol = validate_sparse_protocol(plan.get("protocol", {}))
    if canonical_sha256(protocol) != plan.get("protocol_sha256"):
        raise ValueError("sparse action protocol hash mismatch")
    catalog = action_catalog(protocol)
    if plan.get("action_catalog") != catalog:
        raise ValueError("action catalog does not match protocol")
    groups = plan.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("sparse action plan has no groups")
    group_ids = [group.get("group_id") for group in groups]
    if any(not group_id for group_id in group_ids) or len(group_ids) != len(
        set(group_ids)
    ):
        raise ValueError("group ids must be non-empty and unique")
    prompt_probes: dict[str, tuple[str, ...]] = {}
    observation_ids: list[str] = []
    generated = reused = 0
    prompts_by_cohort: dict[str, set[str]] = {
        "existing_train": set(),
        "fresh_development": set(),
    }
    prompt_groups: dict[str, set[int]] = {}
    prompt_identity: dict[str, tuple[Any, ...]] = {}
    for group in groups:
        if group.get("cohort") not in {"existing_train", "fresh_development"}:
            raise ValueError("unsupported sparse plan cohort")
        prompt_key = str(group.get("prompt_key", ""))
        if not prompt_key:
            raise ValueError("sparse plan groups require prompt_key")
        prompts_by_cohort[group["cohort"]].add(prompt_key)
        identity = (
            group.get("cohort"),
            group.get("prompt"),
            group.get("prompt_sha256"),
            group.get("seed_offset"),
            group.get("source_prompt_id"),
        )
        previous_identity = prompt_identity.setdefault(prompt_key, identity)
        if previous_identity != identity:
            raise ValueError("prompt identity changed across sparse groups")
        base_seed = int(group.get("base_seed"))
        actual_seed = int(group.get("seed"))
        if actual_seed != base_seed + int(group.get("seed_offset")):
            raise ValueError("group seed must equal base_seed plus seed_offset")
        prompt_groups.setdefault(prompt_key, set()).add(base_seed)
        if (
            group["cohort"] == "existing_train"
            and group.get("source_prompt_id") is None
        ):
            raise ValueError("existing train groups require source_prompt_id")
        if (
            group["cohort"] == "fresh_development"
            and group.get("source_prompt_id") is not None
        ):
            raise ValueError("fresh development groups cannot have source_prompt_id")
        actions = group.get("actions")
        if (
            not isinstance(actions, list)
            or len(actions) != protocol["probes_per_prompt"] + 1
        ):
            raise ValueError(
                "each prompt-seed group requires reference plus sparse probes"
            )
        ids = [row.get("action_id") for row in actions]
        if ids[0] != "REFERENCE" or len(ids) != len(set(ids)):
            raise ValueError(
                "group actions must begin with one reference and be unique"
            )
        probes = tuple(ids[1:])
        previous = prompt_probes.setdefault(prompt_key, probes)
        if previous != probes:
            raise ValueError("probe assignment must stay fixed across seeds")
        for row in actions:
            if row.get("action_id") not in catalog:
                raise ValueError("group references an unknown action")
            observation_ids.append(str(row.get("observation_id", "")))
            mode = row.get("artifact_mode")
            may_reuse = (
                group["cohort"] == "existing_train"
                and base_seed == protocol["reuse_reference_base_seed"]
            )
            if mode == "generate":
                generated += 1
            elif mode == "reuse":
                if not may_reuse:
                    raise ValueError(
                        "only the declared existing-train base seed may reuse artifacts"
                    )
                reused += 1
                artifact = row.get("artifact")
                if not isinstance(artifact, dict) or not artifact.get("video_sha256"):
                    raise ValueError(
                        "reused observations require immutable artifact identity"
                    )
            else:
                raise ValueError("artifact_mode must be generate or reuse")
    if any(not item for item in observation_ids) or len(observation_ids) != len(
        set(observation_ids)
    ):
        raise ValueError("observation ids must be non-empty and unique")
    nominal = expected_counts(protocol)
    if (
        len(prompts_by_cohort["existing_train"])
        != protocol["existing_train_prompt_count"]
        or len(prompts_by_cohort["fresh_development"])
        != protocol["fresh_development_prompt_count"]
    ):
        raise ValueError("plan prompt counts do not match protocol")
    expected_seeds = set(protocol["base_seeds"])
    if any(seeds != expected_seeds for seeds in prompt_groups.values()):
        raise ValueError(
            "each prompt must contain every declared base seed exactly once"
        )
    if len(groups) != nominal["prompt_seed_groups"]:
        raise ValueError("plan prompt-seed group count does not match protocol")
    if generated + reused != nominal["scored_videos"]:
        raise ValueError("plan scored video count does not match protocol")
    counts = plan.get("counts")
    expected_actual_counts = {
        "existing_prompts": protocol["existing_train_prompt_count"],
        "fresh_prompts": protocol["fresh_development_prompt_count"],
        "generated_videos": generated,
        "reused_videos": reused,
        "scored_videos": generated + reused,
        "prompt_seed_groups": len(groups),
        "generated_videos_upper_bound": nominal["generated_videos_upper_bound"],
    }
    if counts != expected_actual_counts:
        raise ValueError("plan counts do not match actual sparse observations")
    expected_balance = design_balance(protocol, prompt_probes)
    if plan.get("design_balance") != expected_balance:
        raise ValueError("plan design balance mismatch")
    return json.loads(json.dumps(plan))


def validate_sparse_record(
    record: Mapping[str, Any], *, expected_plan_sha256: str | None = None
) -> dict[str, Any]:
    if record.get("schema") != RECORD_SCHEMA:
        raise ValueError(f"record.schema must be {RECORD_SCHEMA!r}")
    if expected_plan_sha256 and record.get("plan_sha256") != expected_plan_sha256:
        raise ValueError("record belongs to another sparse action plan")
    for field in (
        "group_id",
        "prompt_key",
        "prompt",
        "cohort",
        "actions",
        "provenance",
    ):
        if not record.get(field):
            raise ValueError(f"record requires {field}")
    actions = record["actions"]
    if not isinstance(actions, list) or not actions:
        raise ValueError("record actions must be a non-empty list")
    for row in actions:
        artifact = row.get("artifact")
        if not isinstance(artifact, dict):
            raise ValueError("record action requires artifact")
        path = str(artifact.get("video_path", "")).strip()
        digest = str(artifact.get("video_sha256", "")).strip().lower()
        if (
            not path
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("record artifacts require path and lowercase SHA256")
        seconds = float(artifact.get("cost", {}).get("pipeline_seconds", 0.0))
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("record artifacts require positive pipeline_seconds")
    return json.loads(json.dumps(record))


def write_json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _action_dict(raw: Any) -> dict[str, float]:
    if isinstance(raw, UniversalAction):
        values = {field: getattr(raw, field) for field in ACTION_FIELDS}
    elif isinstance(raw, Mapping):
        if set(raw) != set(ACTION_FIELDS):
            raise ValueError(f"action must contain exactly {list(ACTION_FIELDS)}")
        values = {field: float(raw[field]) for field in ACTION_FIELDS}
    else:
        raise ValueError("action must be an object")
    action = UniversalAction(**values)
    action.validate()
    return {field: float(getattr(action, field)) for field in ACTION_FIELDS}
