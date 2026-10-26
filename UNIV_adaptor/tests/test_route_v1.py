import contextlib
import csv
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from UNIV_adaptor.scripts.data import route_v1 as rt  # noqa: E402

HAS_SCIPY = importlib.util.find_spec("scipy") is not None


def toy():
    """Two arms + FULL; arm 0 is fast and fine on cell 0 only, arm 1 is fast and fine on cell 1 only."""
    cell = np.array([0, 0, 1, 1])
    tf = np.array([[0.2, 0.25, 1.0]] * 4)
    good = np.array([[1.0, 0.3, 1.0], [1.0, 0.3, 1.0], [0.3, 1.0, 1.0], [0.3, 1.0, 1.0]])
    return tf, good, good.copy(), cell


@unittest.skipUnless(HAS_SCIPY, "scipy not installed")
class SolveTest(unittest.TestCase):
    def test_router_uses_the_cell(self):
        tf, ret, q, cell = toy()
        # A floor of 1 leaves no margin to trade between cells, so each cell must take its own fast arm.
        x = rt.solve(tf, ret, q, cell, 1.0)
        self.assertAlmostEqual(x[0, 0], 1.0, 6)
        self.assertAlmostEqual(x[1, 1], 1.0, 6)
        blind = rt.solve(tf, ret, q, cell, 1.0, blind=True)
        (t_r,), (t_b,) = rt.apply(x, cell, tf)[0].mean(keepdims=True), rt.apply(blind, cell, tf)[0].mean(keepdims=True)
        self.assertLess(t_r, t_b)  # blind must fall back to FULL often to meet the floor
        self.assertTrue(np.allclose(blind[0], blind[1]))
        r_b, q_b = (m.mean() for m in rt.apply(blind, cell, ret, q))
        self.assertGreaterEqual(min(r_b, q_b), 1.0 - 1e-6)

    def test_unreachable_floor_means_full(self):
        tf, ret, q, cell = toy()
        ret[:, :2] = np.minimum(ret[:, :2], 0.99)
        x = rt.solve(tf, ret, q, cell, 1.0)
        self.assertTrue(np.allclose(x[:2, -1], 1.0))

    def test_empty_cell_falls_back_to_blind(self):
        tf, ret, q, cell = toy()
        x = rt.solve(tf, ret, q, cell, 0.9)
        self.assertTrue(np.allclose(x[2], rt.solve(tf, ret, q, cell, 0.9, blind=True)[0]))


class BowTest(unittest.TestCase):
    def test_learns_keywords(self):
        texts = ["locked camera still fine detail", "locked camera still plain", "running fast fine detail",
                 "running fast plain"] * 3
        labels = [(False, True), (False, False), (True, True), (True, False)] * 3
        pred = rt.bow_cells(texts, labels, list(range(8)), list(range(8, 12)))
        self.assertEqual([rt.CELLS[c] for c in pred], ["lh", "ll", "hh", "hl"])


class TableTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_table_joins_and_averages_seeds(self):
        videos, secs, dyn, qual = [], {}, [], []
        for seed in (42, 100):
            for case, s, ret, qk in ((rt.FULL, 100.0, "", ""), ("FAST", 25.0, 0.8, 0.5 if seed == 42 else 0.7)):
                stem = f"p1_s{seed}_{case}"
                videos.append({"source": "src", "stem": stem, "group_id": f"g{seed}", "prompt_id": 1, "family_id": "f",
                               "split": "train", "motion_level": "high", "detail_level": "low", "base_seed": seed,
                               "case_id": case})
                secs[("src", stem)] = s
                dyn.append({"source": "src", "stem": stem, "retention": ret})
                qual.append({"source": "src", "stem": stem, "quality_kept": qk})
        (self.tmp / "fixed").mkdir()
        (self.tmp / "fixed" / "plan.json").write_text(json.dumps({"sources": {}, "videos": videos}), encoding="utf-8")
        for name, rows in (("dyn.csv", dyn), ("q.csv", qual)):
            with (self.tmp / name).open("w", encoding="utf-8", newline="") as h:
                w = csv.DictWriter(h, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        with mock.patch.object(rt, "seconds_by_video", return_value=secs), contextlib.redirect_stdout(io.StringIO()):
            rt.table(Namespace(fixed_root=str(self.tmp / "fixed"), dynamic=str(self.tmp / "dyn.csv"),
                               quality=str(self.tmp / "q.csv"), out=str(self.tmp / "out")))
        rows = rt.read_csv(self.tmp / "out" / "route_table.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["arm"], rows[0]["cell"], rows[0]["n"]), ("FAST", "hl", "2"))
        self.assertAlmostEqual(float(rows[0]["tf"]), 0.25)
        self.assertAlmostEqual(float(rows[0]["quality_kept"]), 0.6)
        data = rt.Data(self.tmp / "out" / "route_table.csv")
        self.assertEqual(data.arms, ["FAST", rt.FULL])
        self.assertTrue(np.allclose(data.tf[0], [0.25, 1.0]))


if __name__ == "__main__":
    unittest.main()
