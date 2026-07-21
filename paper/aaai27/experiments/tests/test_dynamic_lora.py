from __future__ import annotations

import math
import unittest

from changing_resolution.dynamic_lora import (
    collect_registered_lora_branches,
    set_registered_lora_strength,
)


class FakeContainer:
    def __init__(self, *, modules=None, parameters=None):
        self._modules = modules or {}
        self._parameters = parameters or {}


class FakeLoRABranch(FakeContainer):
    def __init__(self):
        super().__init__()
        self.lora_down = object()
        self.lora_up = object()
        self.lora_strength = 0.0
        self.has_lora_branch = True


class DynamicLoRATest(unittest.TestCase):
    def test_collects_unique_registered_leaves(self):
        first = FakeLoRABranch()
        second = FakeLoRABranch()
        unrelated = FakeContainer()
        unrelated.has_lora_branch = True
        nested = FakeContainer(modules={"first": first, "again": first, "other": unrelated})
        root = FakeContainer(modules={"nested": nested}, parameters={"second": second})

        branches = collect_registered_lora_branches(root)

        self.assertEqual({id(branch) for branch in branches}, {id(first), id(second)})

    def test_zero_strength_bypasses_matmul_without_dropping_tensors(self):
        branch = FakeLoRABranch()

        count = set_registered_lora_strength([branch], 0.0)

        self.assertEqual(count, 1)
        self.assertFalse(branch.has_lora_branch)
        self.assertEqual(branch.lora_strength, 0.0)
        self.assertTrue(hasattr(branch, "lora_down"))
        self.assertTrue(hasattr(branch, "lora_up"))

    def test_nonzero_strength_reactivates_resident_branch(self):
        branch = FakeLoRABranch()
        set_registered_lora_strength([branch], 0.0)

        set_registered_lora_strength([branch], 0.75)

        self.assertTrue(branch.has_lora_branch)
        self.assertEqual(branch.lora_strength, 0.75)

    def test_rejects_nonfinite_strength(self):
        branch = FakeLoRABranch()
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    set_registered_lora_strength([branch], value)


if __name__ == "__main__":
    unittest.main()
