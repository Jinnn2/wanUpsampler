import importlib.util
import json
import math
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2  # noqa: F401
except ImportError:  # pragma: no cover
    cv2 = None

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("score_fixed_metrics", REPO / "UNIV_adaptor/scripts/data/score_fixed_metrics.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)


def moving_square(n=33, hold=False, flicker=0.0):
    rng = np.random.default_rng(0)
    texture = rng.integers(0, 255, (96, 160, 3), dtype=np.uint8)
    frames = []
    for t in range(n):
        k = (t // 2) * 2 if hold else t
        f = texture.copy()
        f[30:60, 10 + 3 * k:40 + 3 * k] = (220, 40, 40)
        f = np.clip(f.astype(np.float32) + (flicker if t % 2 else 0.0), 0, 255).astype(np.uint8)
        frames.append(f)
    return np.stack(frames)


@unittest.skipIf(cv2 is None, "needs opencv")
class FrameStatsTest(unittest.TestCase):
    def test_frame_holds_are_detected(self):
        smooth = fm.frame_stats(moving_square())
        held = fm.frame_stats(moving_square(hold=True))
        self.assertLess(smooth["pixel_hold_fraction"], 0.05)
        self.assertGreater(held["pixel_hold_fraction"], 0.4)

    def test_flicker_is_detected(self):
        self.assertLess(fm.frame_stats(moving_square())["flicker"], 0.5)
        self.assertGreater(fm.frame_stats(moving_square(flicker=8.0))["flicker"], 5.0)


class PairedTest(unittest.TestCase):
    def test_more_and_less_motion_have_opposite_log_ratios(self):
        base = {k: 1.0 for k in fm.RAW}
        base["hue_deg"] = 350.0
        more = {**base, "flow_mag": 4.0, "hue_deg": 10.0}
        less = {**base, "flow_mag": 0.25}
        self.assertGreater(fm.paired(more, base)["logratio_flow_mag"], 1.0)
        self.assertLess(fm.paired(less, base)["logratio_flow_mag"], -1.0)
        self.assertAlmostEqual(fm.paired(more, base)["hue_shift_deg"], 20.0)
        self.assertTrue(math.isclose(fm.paired(base, base)["d_flicker"], 0.0))


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "needs torch")
class FlowSeriesTest(unittest.TestCase):
    def test_uniform_flow_is_camera_motion(self):
        def pan(a, b):  # stands in for RAFT: every pixel moves 3 px right
            flow = torch.zeros(a.shape[0], 2, *a.shape[2:])
            flow[:, 0] = 3.0
            return [flow]

        s = fm.flow_series(moving_square(n=9), pan, torch.device("cpu"))
        self.assertAlmostEqual(s["flow_camera"], 3.0, places=4)
        self.assertLess(s["flow_local"], 1e-4)
        self.assertEqual(len(s["flow_mag_series"]), 8)
        # No near-static pixels under a constant pan: boil is undefined and must still be valid JSON.
        self.assertIsNone(s["boil"])
        json.dumps(s, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
