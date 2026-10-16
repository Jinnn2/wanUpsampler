"""Requirement scoring: detail statistics order, receipt merge, and retention accounting."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from UNIV_adaptor.data_protocol import sha256_file
from UNIV_adaptor.scripts.data import score_controlled_factor_requirements as req

FULL = "FULL_NATIVE_50"
CASES = (FULL, "SPATIAL", "CACHE")
CELLS = (("low", "low"), ("low", "high"), ("high", "low"), ("high", "high"))
SEEDS = (42, 100, 2024)


class DetailStatsTest(unittest.TestCase):
    def test_blur_lowers_every_detail_statistic(self):
        import cv2

        rng = np.random.default_rng(0)
        sharp = (rng.random((5, 240, 320, 3)) * 255).astype(np.uint8)
        blurred = np.stack([cv2.GaussianBlur(f, (0, 0), 3) for f in sharp])
        a, b = req.detail_stats(sharp), req.detail_stats(blurred)
        for key in ("log_laplacian_var", "hf_share", "edge_density"):
            self.assertGreater(a[key], b[key], key)


class ReportTest(unittest.TestCase):
    """SPATIAL loses the detail requirement on high-detail prompts; CACHE keeps everything."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.out = root / "req"
        for d in ("signals", "vlm"):
            (self.out / d).mkdir(parents=True)
        checklist = root / "checklist.jsonl"
        rng = np.random.default_rng(1)
        videos, quality, prompt_id, lines = [], [], 0, []
        for family in range(4):
            for motion, detail in CELLS:
                reqs = [{"id": "r0", "type": "subject", "question": "s?"},
                        {"id": "r1", "type": "detail" if detail == "high" else "plain", "question": "d?"}]
                lines.append(json.dumps({"prompt_id": prompt_id, "requirements": reqs}))
                for seed in SEEDS:
                    group = f"p{prompt_id}_s{seed}"
                    for case in CASES:
                        stem = f"{group}__{case}"
                        lost = case == "SPATIAL" and detail == "high"
                        videos.append({"stem": stem, "group_id": group, "prompt_id": prompt_id,
                                       "family_id": f"f{family}", "split": "train", "motion_level": motion,
                                       "detail_level": detail, "base_seed": seed, "case_id": case,
                                       "video_path": "", "video_sha256": stem})
                        sig = {k: float(rng.normal()) for k in req.SIGNAL_METRICS}
                        if lost:
                            sig["log_laplacian_var"] -= 2.0
                        (self.out / "signals" / f"{stem}.json").write_text(
                            json.dumps({"video_sha256": stem, **sig}), encoding="utf-8")
                        answers = [{**reqs[0], "p_yes": 0.9}, {**reqs[1], "p_yes": 0.1 if lost else 0.9}]
                        (self.out / "vlm" / f"{stem}.json").write_text(json.dumps(
                            {"video_sha256": stem, "checklist_sha256": "CHK", "answers": answers}), encoding="utf-8")
                        row = {"observation_id": stem, "group_id": group, "prompt_id": prompt_id,
                               "family_id": f"f{family}", "split": "train", "motion_level": motion,
                               "detail_level": detail, "base_seed": seed, "case_id": case, "dynamic_degree": 1.0}
                        row.update({m: 0.8 + rng.normal(0, 0.01) for m in ("vbench5", "imaging_quality")})
                        quality.append(row)
                prompt_id += 1
        checklist.write_text("\n".join(lines) + "\n", encoding="utf-8")
        plan = {"schema": req.PLAN_SCHEMA, "checklist_path": str(checklist),
                "checklist_sha256": sha256_file(checklist), "videos": videos}
        for path in (self.out / "vlm").iterdir():
            value = json.loads(path.read_text(encoding="utf-8"))
            value["checklist_sha256"] = plan["checklist_sha256"]
            path.write_text(json.dumps(value), encoding="utf-8")
        (self.out / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        self.quality = root / "quality_by_video.csv"
        with self.quality.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(quality[0]))
            writer.writeheader()
            writer.writerows(quality)

    def run_report(self):
        import argparse
        import os

        os.environ["VARIANCE_BOOT"] = "200"
        req.report(argparse.Namespace(out=str(self.out), quality_csv=str(self.quality)))
        with (self.out / "analysis" / "requirement_retention.csv").open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_retention_isolates_lost_detail(self):
        rows = self.run_report()
        detail = [r for r in rows if r["action"] == "SPATIAL" and r["type"] == "detail"]
        self.assertTrue(detail)
        self.assertTrue(all(float(r["retention_rate"]) == 0.0 for r in detail))
        kept = [r for r in rows if r["action"] == "CACHE"]
        self.assertTrue(all(float(r["retention_rate"]) == 1.0 for r in kept))

    def test_detail_signal_contrast_reaches_variance_analysis(self):
        self.run_report()
        with (self.out / "analysis" / "contrasts.csv").open(encoding="utf-8") as handle:
            row = next(r for r in csv.DictReader(handle)
                       if r["action"] == "SPATIAL" and r["metric"] == "log_laplacian_var" and r["contrast"] == "detail")
        self.assertLess(float(row["mean"]), -1.0)


if __name__ == "__main__":
    unittest.main()
