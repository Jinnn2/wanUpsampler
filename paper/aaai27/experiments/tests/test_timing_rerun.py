from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from paper.aaai27.experiments.benchmark_quality_efficiency import (
    desired_repeats,
    find_raw_row,
    run_signature,
)
from paper.aaai27.experiments.rerun_optimized_taa_timing import (
    TARGET_CASES,
    filter_spec,
    infer_source_gpu,
    replace_cases,
)


class ResumableBenchmarkTest(unittest.TestCase):
    def test_desired_repeats_separates_warmup_and_measured_indices(self):
        self.assertEqual(
            desired_repeats(1, 3),
            [("warmup", 0), ("measured", 0), ("measured", 1), ("measured", 2)],
        )

    def test_find_raw_row_detects_duplicates(self):
        rows = [
            {"case": "talh45", "phase": "measured", "repeat": "0"},
            {"case": "talh45", "phase": "measured", "repeat": 0},
        ]
        with self.assertRaises(RuntimeError):
            find_raw_row(rows, "talh45", "measured", 0)

    def test_signature_changes_with_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / "spec.json"
            spec.write_text('{"cases": []}', encoding="utf-8")
            base = Namespace(gpu=0, warmup=1, repeats=5, workdir=directory)
            changed = Namespace(gpu=1, warmup=1, repeats=5, workdir=directory)
            self.assertNotEqual(run_signature(spec, base), run_signature(spec, changed))


class OptimizedTAARerunTest(unittest.TestCase):
    def test_filters_only_target_cases_in_source_order(self):
        spec = {
            "schema_version": 2,
            "cases": [
                {"name": "full_hr50"},
                {"name": "talh45"},
                {"name": "talh40"},
            ],
        }
        filtered = filter_spec(spec)
        self.assertEqual([row["name"] for row in filtered["cases"]], ["talh45", "talh40"])

    def test_replaces_summary_rows_without_reordering_other_cases(self):
        old = [
            {"case": "full_hr50", "value": "old-native"},
            {"case": "talh40", "value": "old-40"},
            {"case": "talh45", "value": "old-45"},
            {"case": "ralu_nt45", "value": "old-ralu"},
        ]
        new = [
            {"case": "talh40", "value": "new-40"},
            {"case": "talh45", "value": "new-45"},
        ]
        merged = replace_cases(old, new)
        self.assertEqual(
            [(row["case"], row["value"]) for row in merged],
            [
                ("full_hr50", "old-native"),
                ("talh40", "new-40"),
                ("talh45", "new-45"),
                ("ralu_nt45", "old-ralu"),
            ],
        )

    def test_replaces_all_target_raw_repeats_once(self):
        old = [
            {"case": "talh40", "repeat": "0", "value": "old"},
            {"case": "talh40", "repeat": "1", "value": "old"},
            {"case": "talh45", "repeat": "0", "value": "old"},
            {"case": "other", "repeat": "0", "value": "keep"},
        ]
        new = [
            {"case": case, "repeat": str(repeat), "value": "new"}
            for case in TARGET_CASES
            for repeat in range(2)
        ]
        merged = replace_cases(old, new)
        target_rows = [row for row in merged if row["case"] in TARGET_CASES]
        self.assertEqual(len(target_rows), 4)
        self.assertTrue(all(row["value"] == "new" for row in target_rows))
        self.assertIn({"case": "other", "repeat": "0", "value": "keep"}, merged)

    def test_infers_single_source_gpu(self):
        self.assertEqual(infer_source_gpu([{"gpu": "0"}, {"gpu": "0"}]), 0)
        with self.assertRaises(SystemExit):
            infer_source_gpu([{"gpu": "0"}, {"gpu": "1"}])


if __name__ == "__main__":
    unittest.main()
