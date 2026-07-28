"""Transition-local TTD LoRA activation from paper Sec. 3.3."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def collect_registered_lora_branches(*roots: Any) -> list[Any]:
    """Collect unique LightX2V leaf weights that already own LoRA tensors."""

    branches: list[Any] = []
    seen: set[int] = set()

    def visit(obj: Any) -> None:
        if obj is None:
            return
        object_id = id(obj)
        if object_id in seen:
            return
        seen.add(object_id)

        if (
            hasattr(obj, "has_lora_branch")
            and hasattr(obj, "lora_down")
            and hasattr(obj, "lora_up")
        ):
            branches.append(obj)

        for container_name in ("_modules", "_parameters"):
            container = getattr(obj, container_name, None)
            if isinstance(container, dict):
                for child in container.values():
                    visit(child)

    for root in roots:
        visit(root)
    return branches


def set_registered_lora_strength(branches: Iterable[Any], strength: float) -> int:
    """Toggle registered LoRA branches without reloading or copying their tensors.

    LightX2V dispatches the LoRA matrix multiplications based on
    ``has_lora_branch`` rather than the numeric strength.  Leaving that flag on
    at strength zero still performs both LoRA matmuls.  The tensors stay
    resident here; only the runtime branch flag and scalar strength change.
    """

    value = float(strength)
    if not math.isfinite(value):
        raise ValueError(f"LoRA strength must be finite, got {strength!r}")
    active = value != 0.0
    count = 0
    for branch in branches:
        if not hasattr(branch, "lora_down") or not hasattr(branch, "lora_up"):
            raise RuntimeError("A registered LoRA branch lost its resident tensors")
        branch.lora_strength = value
        branch.has_lora_branch = active
        count += 1
    return count
