from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace

from changing_resolution_distill.realesrgan_compat import (
    LEGACY_FUNCTIONAL_TENSOR,
    install_functional_tensor_shim,
)


class RealESRGANCompatibilityTest(unittest.TestCase):
    def test_installs_basic_sr_legacy_torchvision_symbol_once(self) -> None:
        sentinel = object()
        previous = sys.modules.pop(LEGACY_FUNCTIONAL_TENSOR, None)
        try:
            installed = install_functional_tensor_shim(
                SimpleNamespace(rgb_to_grayscale=sentinel)
            )
            self.assertTrue(installed)
            shim = sys.modules[LEGACY_FUNCTIONAL_TENSOR]
            self.assertIs(shim.rgb_to_grayscale, sentinel)
            self.assertFalse(
                install_functional_tensor_shim(
                    SimpleNamespace(rgb_to_grayscale=object())
                )
            )
        finally:
            sys.modules.pop(LEGACY_FUNCTIONAL_TENSOR, None)
            if previous is not None:
                sys.modules[LEGACY_FUNCTIONAL_TENSOR] = previous


if __name__ == "__main__":
    unittest.main()
