"""Backend-policy and shape regression tests, without claiming CUDA coverage."""
import ast
from contextlib import nullcontext
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest

from UNIV_adaptor.scripts.data import published_wan21_worker as worker


class ShapeTensor:
    """Shape-only facade sufficient to execute the unmodified Wan wrapper."""
    def __init__(self, shape, dtype="bf16"):
        self.shape, self.dtype = tuple(shape), dtype
        self.device = SimpleNamespace(type="cuda")
    def size(self, axis): return self.shape[axis]
    def to(self, dtype=None, **kwargs):
        return ShapeTensor(self.shape, dtype if dtype in ("bf16", "fp16", "int32") else self.dtype)
    def flatten(self, first, last):
        return ShapeTensor(self.shape[:first] + (math.prod(self.shape[first:last + 1]),) + self.shape[last + 1:], self.dtype)
    def new_zeros(self, sizes): return ShapeTensor(sizes, self.dtype)
    def cumsum(self, axis, dtype): return ShapeTensor(self.shape, dtype)
    def type(self, dtype): return ShapeTensor(self.shape, dtype)
    def __getitem__(self, index): return ShapeTensor(self.shape[1:], self.dtype)
    def unflatten(self, axis, sizes):
        if math.prod(sizes) != self.shape[axis]:
            raise RuntimeError(f"unflatten: sizes {sizes} don't match dimension {self.shape[axis]}")
        return ShapeTensor(self.shape[:axis] + tuple(sizes) + self.shape[axis + 1:], self.dtype)


def wrapper_module(relative):
    path = worker.ROOT / "UNIV_adaptor/external" / relative
    if not path.exists():
        raise unittest.SkipTest("Pinned external snapshot not available for shape-only wrapper test")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "flash_attention")
    module = ModuleType("shape_attention")
    module.torch = SimpleNamespace(
        bfloat16="bf16", float16="fp16", int32="int32",
        tensor=lambda values, dtype: ShapeTensor((len(values),), dtype),
        cat=lambda tensors: ShapeTensor((sum(t.shape[0] for t in tensors),) + tensors[0].shape[1:], tensors[0].dtype))
    module.flash_attn = SimpleNamespace(__file__="flash_attn/__init__.py", flash_attn_varlen_func=lambda **kw: kw["q"])
    # The interface returns Tensor, not the tuple expected by the pinned FA3 branch.
    module.flash_attn_interface = SimpleNamespace(__file__="flash_attn_interface.py", flash_attn_varlen_func=lambda **kw: kw["q"])
    module.FLASH_ATTN_2_AVAILABLE = module.FLASH_ATTN_3_AVAILABLE = True
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


class BackendTests(unittest.TestCase):
    def test_native_and_jenga_tensor_return_failure_is_reproduced_and_avoided(self):
        for relative in ("wan21/wan/modules/attention.py", "jenga/wan/modules/attention.py"):
            with self.subTest(wrapper=relative):
                module = wrapper_module(relative)
                q, k, v = ShapeTensor((1, 17, 12, 128)), ShapeTensor((1, 23, 12, 128)), ShapeTensor((1, 23, 12, 128))
                original = module.flash_attention
                with self.assertRaisesRegex(RuntimeError, "dimension 12"):
                    original(q, k, v)
                selected, metadata = worker.configure_attention_backend(module)
                self.assertIs(selected.flash_attention, original)  # no forward rewrite
                self.assertFalse(selected.FLASH_ATTN_3_AVAILABLE)
                self.assertEqual(metadata["dense_backend"], "flash_attention_2")
                self.assertTrue(metadata["fa3_detected_before_lock"])
                self.assertEqual(original(q, k, v).shape, (1, 17, 12, 128))

    def test_missing_fa2_does_not_silently_fallback(self):
        module = SimpleNamespace(FLASH_ATTN_2_AVAILABLE=False, FLASH_ATTN_3_AVAILABLE=True)
        with self.assertRaisesRegex(RuntimeError, "No SDPA/FA3 fallback"):
            worker.configure_attention_backend(module)

    def test_smoke_test_accepts_correct_layout_and_rejects_wrong_layout(self):
        fake_torch = SimpleNamespace(no_grad=nullcontext, bfloat16="bf16", int32="int32",
            randn=lambda *shape, **kw: ShapeTensor(shape, kw["dtype"]),
            tensor=lambda values, **kw: ShapeTensor((len(values),), kw["dtype"]),
            cuda=SimpleNamespace(synchronize=lambda: None),
            isfinite=lambda _: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)))
        module = SimpleNamespace(flash_attention=lambda **kw: kw["q"])
        self.assertTrue(worker.attention_smoke_test(module, fake_torch)["passed"])
        module.flash_attention = lambda **kw: kw["q"][0]
        with self.assertRaisesRegex(RuntimeError, "expected"):
            worker.attention_smoke_test(module, fake_torch)
        module.flash_attention = lambda **kw: ShapeTensor(kw["q"].shape, "fp16")
        with self.assertRaises(RuntimeError):
            worker.attention_smoke_test(module, fake_torch)

    def test_smoke_test_rejects_nonfinite_output(self):
        fake_torch = SimpleNamespace(no_grad=nullcontext, bfloat16="bf16", int32="int32",
            randn=lambda *shape, **kw: ShapeTensor(shape, kw["dtype"]),
            tensor=lambda values, **kw: ShapeTensor((len(values),), kw["dtype"]),
            cuda=SimpleNamespace(synchronize=lambda: None),
            isfinite=lambda _: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: False)))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            worker.attention_smoke_test(SimpleNamespace(flash_attention=lambda **kw: kw["q"]), fake_torch)


if __name__ == "__main__":
    unittest.main()
