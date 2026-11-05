import sys
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
import attention_wan as aw  # noqa: E402


def fake_model_module(log):
    def flash_attention(q, k, v, q_lens=None, k_lens=None, window_size=(-1, -1), **kw):
        log.append(("dense", q.shape[1], k.shape[1]))
        return q.clone()
    return types.SimpleNamespace(flash_attention=flash_attention)


def one_video(mod, L, steps=aw.STEPS, S=2048, text=512):
    q = torch.zeros(1, S, 2, 8)
    c = torch.zeros(1, text, 2, 8)
    for _ in range(steps):
        for _branch in range(2):          # cond, uncond
            for _layer in range(L):
                mod.flash_attention(q=q, k=q, v=q, k_lens=torch.tensor([S]), window_size=(-1, -1))
                mod.flash_attention(q, c, c, k_lens=torch.tensor([text]))  # cross-attention stays dense


class PatchTest(unittest.TestCase):
    def test_dense_counts_and_passes_cross_attention(self):
        log = []
        mod = fake_model_module(log)
        patch = aw.SelfAttentionPatch(mod, 3)
        patch.start("DENSE")
        one_video(mod, 3)
        self.assertEqual(patch.finish(), {"self_attention_calls": aw.STEPS * 2 * 3, "sparse_calls": 0, "mean_kept_fraction": None})
        self.assertEqual(sum(1 for e in log if e[2] == 512), aw.STEPS * 2 * 3)

    def test_svg2_dense_warmup_steps_and_layer_order(self):
        log, seen = [], []
        mod = fake_model_module(log)
        patch = aw.SelfAttentionPatch(mod, 3)

        class Fake:
            def __init__(self, layer):
                self.layer = layer

            def __call__(self, q, k, v, *a):
                seen.append(self.layer)
                return q, 0.25
        patch.kern = patch.flashinfer = object()  # skip the real imports
        patch.start("SVG2")
        patch.layers = {i: Fake(i) for i in range(3)}
        one_video(mod, 3)
        stats = patch.finish()
        dense_steps = int(aw.SVG2_OFFICIAL["first_times_fp"] * aw.STEPS)
        self.assertEqual(stats["sparse_calls"], (aw.STEPS - dense_steps) * 2 * 3)
        self.assertEqual(seen[:6], [0, 1, 2, 0, 1, 2])
        self.assertAlmostEqual(stats["mean_kept_fraction"], 0.25)

    def test_wrong_call_count_fails(self):
        mod = fake_model_module([])
        patch = aw.SelfAttentionPatch(mod, 3)
        patch.start("DENSE")
        one_video(mod, 2)
        with self.assertRaises(RuntimeError):
            patch.finish()

    def test_start_resets_state_per_video(self):
        mod = fake_model_module([])
        patch = aw.SelfAttentionPatch(mod, 2)
        patch.start("DENSE")
        one_video(mod, 2)
        patch.finish()
        patch.layers = {0: "stale"}
        patch.start("DENSE")
        self.assertEqual((patch.calls, patch.layers), (0, {}))


if __name__ == "__main__":
    unittest.main()
