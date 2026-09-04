from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import UniversalAction
from .schedule import resolve_schedule
from .transition import TRANSITION_BASELINES


PROTOCOL_SCHEMA = "univ_prompt_budget_data_protocol_v2"
PLAN_SCHEMA = "univ_prompt_budget_collection_plan_v2"
RECORD_SCHEMA = "univ_prompt_budget_trajectory_record_v2"
QUALITY_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "native_fidelity",
)


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
            "requested_action": _action_dict(self.action),
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
    """Return the planning-only DiT density of a concrete budget preset."""

    action.validate()
    low_token_ratio = (
        action.spatial_ratio**2
        * action.temporal_ratio
        * action.lr_nfe_ratio
    )
    return action.switch_ratio * low_token_ratio + (1.0 - action.switch_ratio)


def enumerate_action_pool(protocol: Mapping[str, Any]) -> list[CandidateAction]:
    """Enumerate the calibration pool; collection uses only budget_presets."""

    validated = validate_protocol(protocol)
    action_space = validated["calibration_action_space"]
    transition = validated["transition"]
    pool: list[CandidateAction] = []
    for spatial in action_space["spatial_ratios"]:
        for temporal in action_space["temporal_ratios"]:
            for nfe in action_space["lr_nfe_ratios"]:
                for switch in action_space["switch_ratios"]:
                    action = UniversalAction(spatial, temporal, nfe, switch)
                    pool.append(
                        CandidateAction(
                            action=action,
                            transition=transition,
                            proxy_density=proxy_compute_density(action),
                        )
                    )
    keys = [candidate.key for candidate in pool]
    if len(keys) != len(set(keys)):
        raise ValueError("calibration_action_space produces duplicate action keys")
    return pool


def build_collection_plan(
    protocol: Mapping[str, Any],
    prompts: Sequence[str],
) -> dict[str, Any]:
    validated = validate_protocol(protocol)
    expected_prompts = sum(split["prompt_count"] for split in validated["splits"])
    if len(prompts) != expected_prompts:
        raise ValueError(
            f"protocol requires exactly {expected_prompts} prompts, got {len(prompts)}"
        )
    normalized_prompts = [str(prompt).strip() for prompt in prompts]
    if any(not prompt for prompt in normalized_prompts):
        raise ValueError("prompts must be non-empty")
    if len(set(normalized_prompts)) != len(normalized_prompts):
        raise ValueError("prompts must be unique before prompt-disjoint splitting")

    reference_nfe = validated["reference_nfe"]
    target_shape = tuple(validated["target_latent_shape"])
    transition = validated["transition"]
    budget_candidates = []
    for preset in validated["budget_presets"]:
        action = _action_from_mapping(preset["action"])
        candidate = CandidateAction(
            action=action,
            transition=transition,
            proxy_density=proxy_compute_density(action),
        ).as_dict(
            reference_nfe=reference_nfe,
            target_latent_shape=target_shape,
        )
        budget_candidates.append(
            {
                "budget_id": preset["id"],
                "target_cost_ratio": preset["target_cost_ratio"],
                "allocation_source": preset["allocation_source"],
                **candidate,
            }
        )

    assignments: list[dict[str, Any]] = []
    cursor = 0
    for split in validated["splits"]:
        for local_index in range(split["prompt_count"]):
            prompt_id = cursor + local_index
            prompt = normalized_prompts[prompt_id]
            for base_seed in split["base_seeds"]:
                actual_seed = int(base_seed) + prompt_id
                assignments.append(
                    {
                        "trajectory_key": (
                            f"{split['name']}_p{prompt_id:06d}_s{actual_seed}"
                        ),
                        "split": split["name"],
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "prompt_sha256": canonical_sha256(prompt),
                        "base_seed": int(base_seed),
                        "seed": actual_seed,
                        "native_teacher_required": True,
                        "budget_candidates": json.loads(json.dumps(budget_candidates)),
                    }
                )
        cursor += split["prompt_count"]

    immutable_protocol = json.loads(json.dumps(validated))
    body = {
        "protocol_schema": PROTOCOL_SCHEMA,
        "protocol_sha256": canonical_sha256(immutable_protocol),
        "protocol": immutable_protocol,
        "prompt_count": len(normalized_prompts),
        "prompts_sha256": canonical_sha256(normalized_prompts),
        "budget_count": len(budget_candidates),
        "assignments": assignments,
    }
    return {"schema": PLAN_SCHEMA, "plan_sha256": canonical_sha256(body), **body}


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"protocol.schema must be {PROTOCOL_SCHEMA!r}")
    normalized = json.loads(json.dumps(protocol))
    if normalized.get("controller_factorization") != "prompt_x_budget_quality_curve":
        raise ValueError(
            "controller_factorization must be 'prompt_x_budget_quality_curve'"
        )
    if normalized.get("observation_mode") != "prompt_only":
        raise ValueError(
            "this executable protocol requires observation_mode='prompt_only'; "
            "common-probe latent collection needs a branchable runner"
        )
    if normalized.get("trajectory_origin") != "independent_step0":
        raise ValueError("trajectory_origin must be 'independent_step0'")
    preset_status = str(normalized.get("preset_status", ""))
    if preset_status not in {
        "frozen_for_pilot_cost_calibration",
        "frozen_after_measured_cost",
    }:
        raise ValueError(
            "preset_status must be frozen_for_pilot_cost_calibration or "
            "frozen_after_measured_cost"
        )
    normalized["preset_status"] = preset_status

    reference_nfe = int(normalized.get("reference_nfe", 0))
    if reference_nfe != 50:
        raise ValueError("the first prompt-budget protocol requires reference_nfe=50")
    normalized["reference_nfe"] = reference_nfe
    shape = normalized.get("target_latent_shape")
    if not isinstance(shape, list) or len(shape) != 4 or min(map(int, shape)) <= 0:
        raise ValueError("target_latent_shape must contain four positive integers")
    normalized["target_latent_shape"] = [int(value) for value in shape]

    transition = str(normalized.get("transition", ""))
    if transition not in TRANSITION_BASELINES:
        raise ValueError(f"unsupported transition: {transition!r}")
    normalized["transition"] = transition

    calibration = normalized.get("calibration_action_space")
    if not isinstance(calibration, dict):
        raise ValueError("calibration_action_space must be an object")
    fields = (
        "spatial_ratios",
        "temporal_ratios",
        "lr_nfe_ratios",
        "switch_ratios",
    )
    for field in fields:
        values = calibration.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"calibration_action_space.{field} must be non-empty")
        numbers = [float(value) for value in values]
        if len(numbers) != len(set(numbers)):
            raise ValueError(f"calibration_action_space.{field} must be unique")
        calibration[field] = numbers
    if any(not 0.8 <= value <= 1.0 for value in calibration["switch_ratios"]):
        raise ValueError("all switch candidates must be in [0.8, 1.0]")
    for spatial in calibration["spatial_ratios"]:
        for temporal in calibration["temporal_ratios"]:
            for nfe in calibration["lr_nfe_ratios"]:
                for switch in calibration["switch_ratios"]:
                    UniversalAction(spatial, temporal, nfe, switch).validate()

    presets = normalized.get("budget_presets")
    if not isinstance(presets, list) or len(presets) != 5:
        raise ValueError("budget_presets must contain exactly five entries")
    ids: list[str] = []
    targets: list[float] = []
    keys: list[str] = []
    proxy_densities: list[float] = []
    for preset in presets:
        if not isinstance(preset, dict):
            raise ValueError("each budget preset must be an object")
        budget_id = str(preset.get("id", "")).strip()
        if not budget_id:
            raise ValueError("each budget preset requires an id")
        ids.append(budget_id)
        target = float(preset.get("target_cost_ratio", 0.0))
        if not 0.0 < target < 1.0:
            raise ValueError("budget target_cost_ratio must be in (0, 1)")
        preset["target_cost_ratio"] = target
        targets.append(target)
        source = str(preset.get("allocation_source", "")).strip()
        if not source:
            raise ValueError("each budget preset requires allocation_source")
        preset["allocation_source"] = source
        action = _action_from_mapping(preset.get("action"))
        if not 0.8 <= action.switch_ratio <= 1.0:
            raise ValueError("budget preset switch_ratio must be in [0.8, 1.0]")
        preset["action"] = _action_dict(action)
        keys.append(action_key(action, transition))
        proxy_densities.append(proxy_compute_density(action))
    if len(ids) != len(set(ids)):
        raise ValueError("budget preset ids must be unique")
    if targets != sorted(targets) or len(targets) != len(set(targets)):
        raise ValueError("budget target_cost_ratio values must be unique and increasing")
    if len(keys) != len(set(keys)):
        raise ValueError("budget presets must resolve to five distinct actions")
    if proxy_densities != sorted(proxy_densities):
        raise ValueError("budget preset proxy compute densities must be increasing")

    splits = normalized.get("splits")
    if not isinstance(splits, list) or not splits:
        raise ValueError("splits must be a non-empty list")
    split_names: list[str] = []
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
        if split.get("collection_mode") != "full_budget_curve":
            raise ValueError(f"split {name} collection_mode must be full_budget_curve")
    if len(split_names) != len(set(split_names)):
        raise ValueError("split names must be unique")
    if set(split_names) != {"train", "validation", "test"}:
        raise ValueError("splits must be exactly train, validation, and test")

    quality = normalized.get("quality_dimensions")
    if quality != list(QUALITY_DIMENSIONS):
        raise ValueError(f"quality_dimensions must equal {list(QUALITY_DIMENSIONS)}")
    return normalized


def validate_collection_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"plan.schema must be {PLAN_SCHEMA!r}")
    body = {
        key: value
        for key, value in plan.items()
        if key not in {"schema", "plan_sha256"}
    }
    if canonical_sha256(body) != plan.get("plan_sha256"):
        raise ValueError("collection plan hash mismatch")
    protocol = validate_protocol(plan.get("protocol", {}))
    if canonical_sha256(protocol) != plan.get("protocol_sha256"):
        raise ValueError("collection plan protocol hash mismatch")
    assignments = plan.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("collection plan has no assignments")
    keys = [row.get("trajectory_key") for row in assignments]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("trajectory keys must be non-empty and unique")
    budget_ids = [preset["id"] for preset in protocol["budget_presets"]]
    for row in assignments:
        candidates = row.get("budget_candidates")
        if not isinstance(candidates, list):
            raise ValueError("each assignment requires budget_candidates")
        if [candidate.get("budget_id") for candidate in candidates] != budget_ids:
            raise ValueError("every assignment must contain the five ordered budgets")


def validate_trajectory_record(
    record: Mapping[str, Any],
    *,
    expected_plan_sha256: str | None = None,
    require_scores: bool = True,
) -> None:
    if record.get("schema") != RECORD_SCHEMA:
        raise ValueError(f"record.schema must be {RECORD_SCHEMA!r}")
    if expected_plan_sha256 and record.get("plan_sha256") != expected_plan_sha256:
        raise ValueError("trajectory record belongs to a different collection plan")
    for field in ("trajectory_key", "split", "prompt", "native_teacher"):
        if not record.get(field):
            raise ValueError(f"trajectory record requires {field}")
    int(record["prompt_id"])
    int(record["seed"])
    _validate_artifact(record["native_teacher"], "native_teacher", require_scores)
    candidates = record.get("budget_candidates")
    if not isinstance(candidates, list) or len(candidates) != 5:
        raise ValueError("trajectory record requires exactly five budget candidates")
    ids: list[str] = []
    for index, candidate in enumerate(candidates):
        ids.append(str(candidate.get("budget_id", "")))
        _validate_artifact(candidate, f"budget_candidates[{index}]", require_scores)
        action = _action_from_mapping(candidate.get("requested_action"))
        transition = str(candidate.get("transition", ""))
        if transition not in TRANSITION_BASELINES:
            raise ValueError(f"budget_candidates[{index}] has unsupported transition")
        if action_key(action, transition) != candidate.get("action_key"):
            raise ValueError(f"budget_candidates[{index}] action_key mismatch")
        if not isinstance(candidate.get("resolved_schedule"), dict):
            raise ValueError(f"budget_candidates[{index}] requires resolved_schedule")
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("budget ids must be non-empty and unique")
    if not isinstance(record.get("provenance"), dict):
        raise ValueError("trajectory record requires provenance")


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


def _action_from_mapping(raw: Any) -> UniversalAction:
    if not isinstance(raw, Mapping):
        raise ValueError("action must be an object")
    fields = ("spatial_ratio", "temporal_ratio", "lr_nfe_ratio", "switch_ratio")
    missing = [field for field in fields if field not in raw]
    if missing:
        raise ValueError(f"action is missing required keys: {missing}")
    action = UniversalAction(**{field: float(raw[field]) for field in fields})
    action.validate()
    return action


def _action_dict(action: UniversalAction) -> dict[str, float]:
    return {
        "spatial_ratio": action.spatial_ratio,
        "temporal_ratio": action.temporal_ratio,
        "lr_nfe_ratio": action.lr_nfe_ratio,
        "switch_ratio": action.switch_ratio,
    }


def _validate_artifact(value: Any, field: str, require_scores: bool) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    output = str(value.get("video_path", "")).strip()
    digest = str(value.get("video_sha256", "")).strip().lower()
    if not output:
        raise ValueError(f"{field}.video_path must be non-empty")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field}.video_sha256 must be a lowercase SHA256 digest")
    cost = value.get("cost")
    if not isinstance(cost, Mapping) or float(cost.get("pipeline_seconds", 0.0)) <= 0:
        raise ValueError(f"{field} requires positive measured pipeline_seconds")
    if not require_scores:
        return
    quality = value.get("quality")
    if not isinstance(quality, Mapping):
        raise ValueError(f"{field} requires quality")
    for name in QUALITY_DIMENSIONS:
        number = float(quality[name])
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{field}.quality.{name} must be finite and in [0, 1]")
