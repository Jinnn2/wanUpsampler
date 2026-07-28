from __future__ import annotations

import importlib
import sys
import types
from typing import Any


LEGACY_FUNCTIONAL_TENSOR = "torchvision.transforms.functional_tensor"


def install_functional_tensor_shim(functional_module: Any | None = None) -> bool:
    """Provide the one legacy torchvision symbol imported by BasicSR 1.4.x."""

    if LEGACY_FUNCTIONAL_TENSOR in sys.modules:
        return False
    if functional_module is None:
        try:
            importlib.import_module(LEGACY_FUNCTIONAL_TENSOR)
            return False
        except ModuleNotFoundError as exc:
            if exc.name != LEGACY_FUNCTIONAL_TENSOR:
                raise
        from torchvision.transforms import functional as functional_module

    shim = types.ModuleType(LEGACY_FUNCTIONAL_TENSOR)
    shim.rgb_to_grayscale = functional_module.rgb_to_grayscale
    sys.modules[LEGACY_FUNCTIONAL_TENSOR] = shim
    return True
