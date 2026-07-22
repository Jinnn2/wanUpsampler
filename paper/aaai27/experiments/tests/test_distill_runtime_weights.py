from __future__ import annotations

import ast
import unittest
from pathlib import Path

from changing_resolution_distill.runtime_weights import (
    checkpoint_model_state,
    iter_lora_branches,
    set_registered_lora_strength,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
BRIDGE = REPO_ROOT / "changing_resolution_distill/lightx2v_distill_bridge.py"


class Node:
    def __init__(self, *, branch: bool = False) -> None:
        self.has_lora_branch = branch
        self.lora_strength = -1.0
        self._modules: dict[str, Node] = {}
        self._parameters: dict[str, Node] = {}


class DistillRuntimeWeightsTest(unittest.TestCase):
    def test_strength_update_reuses_registered_branches(self) -> None:
        root = Node()
        first = Node(branch=True)
        second = Node(branch=True)
        root._modules["first"] = first
        root._parameters["same_first"] = first
        root.pre_weight = second

        self.assertEqual(len(list(iter_lora_branches(root))), 2)
        self.assertEqual(set_registered_lora_strength(root, 0.75), 2)
        self.assertEqual(first.lora_strength, 0.75)
        self.assertEqual(second.lora_strength, 0.75)

    def test_checkpoint_state_is_reused_from_loaded_payload(self) -> None:
        state = {"weight": object()}
        self.assertIs(checkpoint_model_state({"model": state, "ema": {}}), state)
        self.assertIs(checkpoint_model_state(state), state)

    def test_bridge_has_no_lora_reload_or_second_stage2_load(self) -> None:
        tree = ast.parse(BRIDGE.read_text(encoding="utf-8"))
        model_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "WanDistillModelLastStepLoRA"
        )
        update = next(
            node
            for node in model_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "_update_lora"
        )
        update_calls = [
            node.func.attr
            for node in ast.walk(update)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertNotIn("_load_lora_file", update_calls)

        runner_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "WanDistillCleanResizerBridgeRunner"
        )
        loader = next(
            node
            for node in runner_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "load_clean_resizer"
        )
        torch_loads = [
            node
            for node in ast.walk(loader)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "load"
        ]
        self.assertEqual(len(torch_loads), 1)
        self.assertFalse(
            any(
                isinstance(node, ast.Name) and node.id == "load_checkpoint"
                for node in ast.walk(loader)
            )
        )


if __name__ == "__main__":
    unittest.main()
