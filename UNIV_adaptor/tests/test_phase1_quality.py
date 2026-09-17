import tempfile
import unittest
from pathlib import Path
from UNIV_adaptor.scripts.data.score_phase1_dataset import report, QUALITY_DIMENSIONS, DIAGNOSTICS
from UNIV_adaptor.scripts.data.analyze_phase1_runtime import summarize

class Phase1QualityTests(unittest.TestCase):
    def test_missing_stage_time_is_not_zero(self):
        summary, _ = summarize([{"split":"train","trajectory_key":"x", "budget_candidates":[{"artifact_id":"a","cost":{"pipeline_seconds":10}}]}])
        self.assertIsNone(summary[0]["vae_seconds_mean"])
        self.assertIsNone(summary[0]["dit_seconds_mean"])

    def test_report_pairs_and_seed_aggregation(self):
        ids = ["P1_B15_BASE","P1_B20_BASE","P1_B25_SPATIAL","P1_B30_SPATIAL","P1_B35_HR","P1_B40_SPATIAL"]
        rows, scores = [], {}
        for seed in (1,2,3):
            for i, action in enumerate(ids):
                stem = f"{seed}_{action}"
                rows.append(dict(split="validation",prompt_id=0,seed=seed,action_id=action,stem=stem,pipeline_seconds=10+i))
                scores[stem] = {d: 0.5+i*0.01 for d in (*QUALITY_DIMENSIONS,*DIAGNOSTICS)}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            report(rows,scores,out)
            import csv
            with (out/"paired_gains.csv").open() as f:
                pairs = list(csv.DictReader(f))
            self.assertEqual(len(pairs),4)
            self.assertAlmostEqual(float(pairs[-1]["delta_vbench5"]),0.01)
            self.assertIn("No native-HR", (out/"report.md").read_text())
            scores.pop(next(iter(scores)))
            with self.assertRaisesRegex(ValueError,"coverage"):
                report(rows,scores,out)
