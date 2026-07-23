from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


def iter_lora_branches(root: Any) -> Iterator[Any]:
    """Yield each registered LightX2V LoRA branch exactly once."""

    stack = [root]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if getattr(current, "has_lora_branch", False):
            yield current
        stack.extend(getattr(current, "_modules", {}).values())
        stack.extend(getattr(current, "_parameters", {}).values())
        for name in ("pre_weight", "transformer_weights", "post_weight"):
            child = getattr(current, name, None)
            if child is not None:
                stack.append(child)


def set_registered_lora_strength(root: Any, strength: float) -> int:
    """Change only the scale of already registered LoRA tensors."""

    count = 0
    for branch in iter_lora_branches(root):
        branch.lora_strength = float(strength)
        count += 1
    return count


def checkpoint_model_state(checkpoint: Any) -> Any:
    """Return the model state from a loaded training or raw state checkpoint."""

    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint
