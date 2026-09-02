from __future__ import annotations

import unittest

from UNIV_adaptor.rgb_super_resolution import cover_scale


class AdaptiveScaleTest(unittest.TestCase):
    def test_half_resolution_uses_at_most_x2(self):
        self.assertAlmostEqual(cover_scale(368, 624, 720, 1248), 2.0)

    def test_two_thirds_resolution_uses_x1p5(self):
        self.assertAlmostEqual(cover_scale(480, 832, 720, 1248), 1.5)

    def test_aspect_mismatch_uses_covering_axis(self):
        self.assertAlmostEqual(cover_scale(368, 640, 720, 1248), 720 / 368)


if __name__ == "__main__":
    unittest.main()
