from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "integrate_result_snapshots.py"
SPEC = importlib.util.spec_from_file_location("integrate_result_snapshots", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class IntegrateResultSnapshotsTest(unittest.TestCase):
    def test_missing_sentinel_is_not_data(self) -> None:
        self.assertFalse(MODULE.csv_has_data(b"status\nMISSING\n"))
        self.assertTrue(MODULE.csv_has_data(b"metric,value\nl1,0.1\n"))

    def test_paired_summary_orients_lower_is_better(self) -> None:
        result = MODULE.paired_summary(
            [(2.0, 1.0), (4.0, 3.0)],
            metric="l1",
            better="lower",
            bootstrap_samples=100,
        )
        self.assertEqual(result["wins"], 2)
        self.assertEqual(result["losses"], 0)
        self.assertAlmostEqual(result["oriented_improvement_mean"], 1.0)
        self.assertAlmostEqual(result["two_sided_sign_test_p"], 0.5)

    def test_factorial_effects_use_distill_case_names(self) -> None:
        rows = []
        for case, value in (
            ("step3_base_interp", 0.80),
            ("step3_base_stage2", 0.84),
            ("step3_lora_stage2", 0.85),
        ):
            row = {"case": case, "quality5_mean": value}
            row.update({metric: value for metric in MODULE.QUALITY5})
            rows.append(row)
        effects = MODULE.derive_factorial_effects(rows)
        self.assertEqual(len(effects), 3)
        self.assertAlmostEqual(effects[-1]["delta_quality5_b_minus_a"], 0.05)

    def test_quality5_per_video_normalizes_imaging_percent(self) -> None:
        numeric = {}
        for dimension in MODULE.QUALITY5:
            value = 50.0 if dimension == "imaging_quality" else 0.5
            numeric[f"run.{dimension}.1.0.video_results"] = value
        payload = {"cases": {"case": {"numeric_metrics": numeric}}}
        self.assertEqual(MODULE.quality5_per_video(payload, "case"), {0: 0.5})


if __name__ == "__main__":
    unittest.main()
