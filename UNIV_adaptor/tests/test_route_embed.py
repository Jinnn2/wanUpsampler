import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from UNIV_adaptor.scripts.data import route_embed as re_  # noqa: E402

HAS_SCIPY = importlib.util.find_spec("scipy") is not None


class PredictTest(unittest.TestCase):
    def test_ridge_recovers_linear_map(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(80, 30))
        y = x @ rng.normal(size=(30, 4)) + 0.01 * rng.normal(size=(80, 4))
        pred = re_.ridge_gcv(x[:60], y[:60], x[60:])
        self.assertLess(np.abs(pred - y[60:]).mean(), 0.1 * np.abs(y[60:]).mean())

    def test_knn_averages_nearest(self):
        x = np.array([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.1, 0.9]])
        y = np.array([[1.0], [1.0], [5.0], [5.0]])
        self.assertEqual(re_.knn(x, y, np.array([[1.0, 0.05]]), k=2)[0, 0], 1.0)


@unittest.skipUnless(HAS_SCIPY, "scipy not installed")
class RouteTest(unittest.TestCase):
    def test_per_prompt_choice(self):
        # Arms: fast-good-on-prompt-0, fast-good-on-prompt-1, FULL. Floor 1 forces each prompt onto its own arm.
        pred = np.array([[[0.2, 0.3, 1.0], [1.0, 0.3, 1.0], [1.0, 0.3, 1.0]],
                         [[0.2, 0.3, 1.0], [0.3, 1.0, 1.0], [0.3, 1.0, 1.0]]])
        x = re_.route(pred, 1.0)
        np.testing.assert_allclose(x, [[1, 0, 0], [0, 1, 0]], atol=1e-6)

    def test_infeasible_goes_full(self):
        pred = np.full((2, 3, 2), 0.5)
        np.testing.assert_allclose(re_.route(pred, 0.9), [[0, 1], [0, 1]])


if __name__ == "__main__":
    unittest.main()
