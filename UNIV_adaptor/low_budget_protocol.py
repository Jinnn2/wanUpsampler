from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .core import UniversalAction
from .data_protocol import canonical_sha256
from .hr_refinement import (
    direct_hr_sigmas,
    quantize_float32_timesteps,
    resample_hr_sigmas,
    wan_reference_sigmas,
)
from .schedule import resolve_schedule
from .transition import DVG_LATENT_ANCHOR


PROTOCOL_SCHEMA = "univ_low_budget_extension_protocol_v1"
PLAN_SCHEMA = "univ_low_budget_extension_plan_v1"
RECORD_SCHEMA = "univ_low_budget_extension_record_v1"
COMBINED_RECORD_SCHEMA = "univ_prompt_budget_trajectory_record_v3"
EXPECTED_DISPLAY_BUDGETS = ("B10", "B15", "B20", "B30")


def proxy_compute_density(action: Mapping[str, Any], *, reference_nfe: int) -> float:
    return (
        float(action["spatial_ratio"]) ** 2
        * float(action["temporal_ratio"])
        * int(action["true_lr_steps"])
        / reference_nfe
        + int(action["hr_steps"]) / reference_nfe
    )


def action_key(action: Mapping[str, Any]) -> str:
    body = {
        "spatial_ratio": float(action["spatial_ratio"]),
        "temporal_ratio": float(action["temporal_ratio"]),
        "true_lr_steps": int(action["true_lr_steps"]),
        "transition": str(action["transition"]),
        "renoise_sigma": float(action["renoise_sigma"]),
        "hr_steps": int(action["hr_steps"]),
    }
    return f"univ_action_v3_{canonical_sha256(body)[:16]}"


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"protocol.schema must be {PROTOCOL_SCHEMA!r}")
    value = json.loads(json.dumps(protocol))
    if value.get("base_protocol_schema") != "univ_prompt_budget_data_protocol_v2":
        raise ValueError("base_protocol_schema must name the immutable v2 dataset")
    if value.get("controller_factorization") != "prompt_x_action_quality_curve":
        raise ValueError(
            "controller_factorization must be prompt_x_action_quality_curve"
        )
    if value.get("observation_mode") != "prompt_plus_endpoint":
        raise ValueError("observation_mode must be prompt_plus_endpoint")
    if value.get("trajectory_origin") != "independent_step0":
        raise ValueError("trajectory_origin must be independent_step0")
    reference_nfe = int(value.get("reference_nfe", 0))
    if reference_nfe != 50:
        raise ValueError("low-budget extension requires reference_nfe=50")
    value["reference_nfe"] = reference_nfe
    sample_shift = float(value.get("sample_shift", 0.0))
    if not math.isfinite(sample_shift) or sample_shift <= 0:
        raise ValueError("sample_shift must be finite and positive")
    value["sample_shift"] = sample_shift
    shape = value.get("target_latent_shape")
    if not isinstance(shape, list) or len(shape) != 4 or min(map(int, shape)) <= 0:
        raise ValueError("target_latent_shape must contain four positive integers")
    value["target_latent_shape"] = [int(item) for item in shape]

    endpoint = value.get("endpoint_state")
    if not isinstance(endpoint, dict) or endpoint.get("enabled") is not True:
        raise ValueError("endpoint_state.enabled must be true")
    if endpoint.get("archive_dtype") not in {"fp16", "bf16", "fp32"}:
        raise ValueError("endpoint_state.archive_dtype must be fp16, bf16, or fp32")
    if endpoint.get("tensors") != ["clean_lr", "clean_hr", "hr_noise"]:
        raise ValueError(
            "endpoint_state.tensors must preserve clean_lr, clean_hr, and hr_noise"
        )

    presets = value.get("budget_presets")
    if not isinstance(presets, list) or len(presets) != 4:
        raise ValueError("budget_presets must contain exactly four low-budget actions")
    display = [str(item.get("display_budget", "")) for item in presets]
    if tuple(display) != EXPECTED_DISPLAY_BUDGETS:
        raise ValueError(
            f"display budgets must be ordered as {EXPECTED_DISPLAY_BUDGETS}"
        )
    artifact_ids: list[str] = []
    targets: list[float] = []
    keys: list[str] = []
    proxies: list[float] = []
    for preset in presets:
        artifact_id = str(preset.get("artifact_id", "")).strip()
        if not artifact_id.startswith("LB"):
            raise ValueError("low-budget artifact_id must use an LB prefix")
        artifact_ids.append(artifact_id)
        target = float(preset.get("target_cost_ratio", 0.0))
        if not 0.0 < target < 1.0:
            raise ValueError("target_cost_ratio must be in (0, 1)")
        preset["target_cost_ratio"] = target
        targets.append(target)
        if not str(preset.get("allocation_source", "")).strip():
            raise ValueError("each preset requires allocation_source")
        action = preset.get("action")
        if not isinstance(action, dict):
            raise ValueError("each preset requires an action object")
        normalized_action = {
            "spatial_ratio": float(action.get("spatial_ratio", 0.0)),
            "temporal_ratio": float(action.get("temporal_ratio", 0.0)),
            "true_lr_steps": int(action.get("true_lr_steps", 0)),
            "transition": str(action.get("transition", "")),
            "renoise_sigma": float(action.get("renoise_sigma", 0.0)),
            "hr_steps": int(action.get("hr_steps", 0)),
        }
        UniversalAction(
            spatial_ratio=normalized_action["spatial_ratio"],
            temporal_ratio=normalized_action["temporal_ratio"],
            lr_nfe_ratio=1.0,
            switch_ratio=1.0,
        ).validate()
        if not 1 <= normalized_action["true_lr_steps"] <= reference_nfe:
            raise ValueError("true_lr_steps must be in [1, reference_nfe]")
        if normalized_action["transition"] != DVG_LATENT_ANCHOR:
            raise ValueError("low-budget extension requires dvg_latent_anchor")
        if normalized_action["hr_steps"] < 1:
            raise ValueError("low-budget actions require at least one HR step")
        if not 0.0 < normalized_action["renoise_sigma"] < 1.0:
            raise ValueError("renoise_sigma must be in (0, 1)")
        preset["action"] = normalized_action
        key = action_key(normalized_action)
        proxy = proxy_compute_density(normalized_action, reference_nfe=reference_nfe)
        if not math.isclose(proxy, target, abs_tol=0.006):
            raise ValueError(
                f"{artifact_id} proxy {proxy:.6f} misses target {target:.6f}"
            )
        keys.append(key)
        proxies.append(proxy)
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("artifact ids must be unique")
    if len(keys) != len(set(keys)):
        raise ValueError("low-budget actions must be distinct")
    if targets != sorted(targets) or proxies != sorted(proxies):
        raise ValueError("target and proxy costs must be strictly increasing")

    splits = value.get("splits")
    if not isinstance(splits, list) or [row.get("name") for row in splits] != [
        "train",
        "validation",
        "test",
    ]:
        raise ValueError("splits must be ordered train, validation, test")
    for split in splits:
        split["prompt_count"] = int(split.get("prompt_count", 0))
        split["base_seeds"] = [int(seed) for seed in split.get("base_seeds", [])]
        if split["prompt_count"] < 1 or not split["base_seeds"]:
            raise ValueError("every split requires prompts and base seeds")
        if len(split["base_seeds"]) != len(set(split["base_seeds"])):
            raise ValueError("base seeds must be unique within each split")
        if split.get("collection_mode") != "low_budget_extension":
            raise ValueError("collection_mode must be low_budget_extension")
    return value


def build_plan(protocol: Mapping[str, Any], prompts: Sequence[str]) -> dict[str, Any]:
    value = validate_protocol(protocol)
    expected = sum(split["prompt_count"] for split in value["splits"])
    normalized_prompts = [str(prompt).strip() for prompt in prompts]
    if len(normalized_prompts) != expected or any(
        not prompt for prompt in normalized_prompts
    ):
        raise ValueError(f"protocol requires exactly {expected} non-empty prompts")
    if len(normalized_prompts) != len(set(normalized_prompts)):
        raise ValueError("prompts must be unique")

    reference_sigmas = wan_reference_sigmas(
        reference_nfe=value["reference_nfe"], sample_shift=float(value["sample_shift"])
    )
    candidates = []
    for preset in value["budget_presets"]:
        action = preset["action"]
        base_action = UniversalAction(
            action["spatial_ratio"], action["temporal_ratio"], 1.0, 1.0
        )
        schedule = resolve_schedule(
            base_action,
            reference_nfe=value["reference_nfe"],
            target_latent_shape=tuple(value["target_latent_shape"]),
        )
        lr_steps = action["true_lr_steps"]
        lr_sigmas = resample_hr_sigmas(
            reference_sigmas, boundary_step=0, hr_steps=lr_steps
        )
        hr_sigmas = direct_hr_sigmas(
            start_sigma=action["renoise_sigma"], hr_steps=action["hr_steps"]
        )
        candidates.append(
            {
                "artifact_id": preset["artifact_id"],
                "display_budget": preset["display_budget"],
                "target_cost_ratio": preset["target_cost_ratio"],
                "allocation_source": preset["allocation_source"],
                "action_key": action_key(action),
                "execution_action": json.loads(json.dumps(action)),
                "proxy_compute_density": proxy_compute_density(
                    action, reference_nfe=value["reference_nfe"]
                ),
                "resolved_schedule": schedule.as_dict(),
                "planned_lr_schedule": {
                    "grid_policy": "linear_interpolation_in_reference_index",
                    "lr_steps": lr_steps,
                    "sigmas": list(lr_sigmas),
                    "model_timesteps": list(
                        quantize_float32_timesteps(
                            lr_sigmas[:-1], num_train_timesteps=1000
                        )
                    ),
                },
                "planned_hr_schedule": {
                    "grid_policy": "direct_sigma_linear",
                    "hr_steps": action["hr_steps"],
                    "sigmas": list(hr_sigmas),
                    "model_timesteps": list(
                        quantize_float32_timesteps(
                            hr_sigmas[:-1], num_train_timesteps=1000
                        )
                    ),
                },
            }
        )

    assignments = []
    cursor = 0
    for split in value["splits"]:
        for local_index in range(split["prompt_count"]):
            prompt_id = cursor + local_index
            prompt = normalized_prompts[prompt_id]
            for base_seed in split["base_seeds"]:
                seed = base_seed + prompt_id
                assignments.append(
                    {
                        "trajectory_key": f"{split['name']}_p{prompt_id:06d}_s{seed}",
                        "split": split["name"],
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "prompt_sha256": canonical_sha256(prompt),
                        "base_seed": base_seed,
                        "seed": seed,
                        "low_budget_candidates": json.loads(json.dumps(candidates)),
                    }
                )
        cursor += split["prompt_count"]
    body = {
        "protocol": value,
        "protocol_sha256": canonical_sha256(value),
        "prompts_sha256": canonical_sha256(normalized_prompts),
        "assignments": assignments,
    }
    return {"schema": PLAN_SCHEMA, "plan_sha256": canonical_sha256(body), **body}


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"plan.schema must be {PLAN_SCHEMA!r}")
    body = {
        key: value
        for key, value in plan.items()
        if key not in {"schema", "plan_sha256"}
    }
    if canonical_sha256(body) != plan.get("plan_sha256"):
        raise ValueError("low-budget plan hash mismatch")
    protocol = validate_protocol(plan.get("protocol", {}))
    if canonical_sha256(protocol) != plan.get("protocol_sha256"):
        raise ValueError("low-budget protocol hash mismatch")
    return json.loads(json.dumps(plan))
