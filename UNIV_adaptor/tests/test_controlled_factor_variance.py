"""Scientific checks: content-driven loss is separated from seed noise."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from UNIV_adaptor.scripts.data.analyze_controlled_factor_variance import FULL, METRICS, run

SEEDS = (42, 100, 2024)
CELLS = (("low", "low"), ("low", "high"), ("high", "low"), ("high", "high"))


def write_quality(path, families=16):
    """SPATIAL loses 0.04 on high detail with tiny seed noise; CACHE is pure seed noise."""
    rng = np.random.default_rng(0)
    rows, prompt_id = [], 0
    for family in range(families):
        split = "test" if family >= families - 2 else "train"
        for motion, detail in CELLS:
            for seed in SEEDS:
                group = f"p{prompt_id}_s{seed}"
                base = {m: 0.8 + rng.normal(0, 0.01) for m in METRICS}
                deltas = {
                    FULL: 0.0,
                    "SPATIAL": (-0.04 if detail == "high" else 0.0) + rng.normal(0, 0.002),
                    "CACHE": rng.normal(0, 0.01),
                }
                for case, delta in deltas.items():
                    row = {"group_id": group, "prompt_id": prompt_id, "family_id": f"f{family}", "split": split,
                           "motion_level": motion, "detail_level": detail, "base_seed": seed, "case_id": case,
                           "dynamic_degree": 0.0 if (case == "SPATIAL" and motion == "high") else float(motion == "high")}
                    row.update({m: base[m] + delta for m in METRICS})
                    rows.append(row)
            prompt_id += 1
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class VarianceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.csv = Path(self.tmp.name) / "quality_by_video.csv"
        write_quality(self.csv)
        self.result, self.summary = run(self.csv, Path(self.tmp.name) / "out")

    def row(self, table, **keys):
        return next(r for r in self.result[table] if all(r[k] == v for k, v in keys.items()))

    def test_locked_test_rows_never_enter(self):
        self.assertEqual(self.summary["families"], 14)
        self.assertEqual(self.summary["dropped_test_rows"], 2 * 4 * 3 * 3)

    def test_content_loss_dominates_seed_noise(self):
        spatial = self.row("variance", action="SPATIAL", metric="vbench5")
        cache = self.row("variance", action="CACHE", metric="vbench5")
        self.assertGreater(spatial["prompt_to_seed_ratio"], 20)
        self.assertGreater(spatial["cell_r2"], 0.9)
        self.assertLess(cache["prompt_to_seed_ratio"], 1)
        self.assertGreater(spatial["ratio_ci_low"], cache["ratio_ci_high"])

    def test_within_family_detail_contrast(self):
        detail = self.row("contrasts", action="SPATIAL", metric="vbench5", contrast="detail")
        self.assertAlmostEqual(detail["mean"], -0.04, delta=0.003)
        self.assertLess(detail["ci_high"], 0)
        self.assertEqual(detail["families_negative"], 14)

    def test_worst_cell_and_motion_retention(self):
        tail = self.row("tails", action="SPATIAL", metric="vbench5", motion="high", detail="high")
        self.assertEqual(tail["p_drop_gt_0.02"], 1.0)
        motion = self.row("motion", action="SPATIAL", motion="high")
        self.assertEqual(motion["retained_given_full_dynamic"], 0.0)


if __name__ == "__main__":
    unittest.main()
