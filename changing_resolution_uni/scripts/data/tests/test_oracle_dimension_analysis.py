from __future__ import annotations

import unittest

from changing_resolution_uni.scripts.data.analyze_oracle_dimensions import analyze
from changing_resolution_uni.scripts.data.oracle_record_schema import (
    FORMAL_STEPS,
    QUALITY5_DIMENSIONS,
)


def make_normalized_record(prompt_id: int, seed: int) -> dict:
    candidates = []
    for index, step in enumerate(FORMAL_STEPS):
        value = 0.80 + index * 0.001
        candidates.append(
            {
                "step": step,
                "dimensions": {name: value for name in QUALITY5_DIMENSIONS},
                "diagnostics": {},
            }
        )
    return {
        "prompt_id": prompt_id,
        "seed": seed,
        "native_dimensions": {name: 0.82 for name in QUALITY5_DIMENSIONS},
        "native_diagnostics": {},
        "candidates": candidates,
    }


class OracleDimensionAnalysisTest(unittest.TestCase):
    def test_reports_late_step_metric_advantage(self) -> None:
        records = {
            0: [make_normalized_record(0, 42), make_normalized_record(0, 100)],
            1: [make_normalized_record(1, 43), make_normalized_record(1, 101)],
        }
        manifest = {"diagnostic_dimensions": []}
        step_rows, metric_rows = analyze(
            records,
            manifest,
            tie_tolerance=1e-6,
            flat_tolerance=1e-3,
        )
        subject = next(
            row for row in metric_rows if row["metric"] == "subject_consistency"
        )
        self.assertGreater(subject["step50_minus_step30_mean"], 0.0)
        self.assertEqual(subject["tie_fraction"], 0.0)
        step50 = next(
            row
            for row in step_rows
            if row["metric"] == "subject_consistency" and row["step"] == 50
        )
        self.assertEqual(step50["unique_winner_count"], 2)


if __name__ == "__main__":
    unittest.main()
