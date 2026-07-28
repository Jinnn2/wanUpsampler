from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper.aaai27.experiments.prepare_quality_efficiency import QUALITY_DIMENSIONS, vbench_case_scores


class QualityEfficiencySpecTest(unittest.TestCase):
    def test_extracts_only_aggregate_vbench_dimension_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vbench_v1_custom.json"
            numeric = {}
            for index, dimension in enumerate(QUALITY_DIMENSIONS):
                numeric[f"results_eval_results.{dimension}.0"] = 0.5 + index / 10
                numeric[f"results_eval_results.{dimension}.1.0.video_results"] = 0.1
            path.write_text(
                json.dumps({"cases": {"step3_lora_stage2": {"numeric_metrics": numeric}}}),
                encoding="utf-8",
            )

            scores = vbench_case_scores(path, "step3_lora_stage2")

            self.assertEqual(list(scores), QUALITY_DIMENSIONS)
            self.assertAlmostEqual(scores["subject_consistency"], 0.5)
            self.assertAlmostEqual(scores["imaging_quality"], 0.9)


if __name__ == "__main__":
    unittest.main()
