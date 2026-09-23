from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import unittest

from UNIV_adaptor.data_protocol import canonical_sha256, sha256_file, write_json_atomic
from UNIV_adaptor.hy15_protocol import (
    DEFAULT_PROTOCOL, density, immutable_json, make_plan, read, record_paths,
    refresh_indices, validate_protocol, verify_plan, verify_record,
)


class ProtocolTests(unittest.TestCase):
    def test_complete_balanced_plan(self):
        plan = make_plan()
        verify_plan(plan)
        self.assertEqual(len(plan["prompts"]), 16)
        self.assertEqual(len(plan["jobs"]), 512)
        self.assertEqual(len({j["group"] for j in plan["jobs"]}), 64)
        self.assertFalse(any(r["split"] == "test" for r in plan["prompts"]))
        for j in plan["jobs"]:
            self.assertEqual(j["seed"], j["base_seed"] + j["prompt"]["prompt_id"])
        for lane in range(8):
            self.assertEqual(sum((i // 8) % 8 == lane for i in range(512)), 64)

    def test_density_counts_repair_and_discretization(self):
        p = read(DEFAULT_PROTOCOL)
        cases = {c["id"]: c for c in p["cases"]}
        self.assertEqual(density(cases["S_B025"], p)["main_proxy"], 0.25)
        self.assertAlmostEqual(density(cases["S_B025"], p)["total_proxy_excluding_rgb"], 0.33)
        self.assertAlmostEqual(density(cases["T_B025"], p)["main_proxy"], 8 / 31)
        self.assertEqual(density(cases["C_B025"], p)["main_proxy"], 0.26)
        self.assertEqual(density(cases["FULL50"], p)["total_proxy_excluding_rgb"], 1)

    def test_refresh_schedule_and_invalid_protocol(self):
        for n in (13, 25, 50):
            indices = refresh_indices(50, n)
            self.assertEqual(len(set(indices)), n)
            self.assertEqual((indices[0], indices[-1]), (0, 49))
        p = read(DEFAULT_PROTOCOL)
        p["refine_sigmas"][0] = 0.3
        with self.assertRaises(ValueError):
            validate_protocol(p)
        p = read(DEFAULT_PROTOCOL)
        p["cases"][3]["height"] = 192
        with self.assertRaises(ValueError):
            validate_protocol(p)

    def test_plan_and_artifact_tamper(self):
        plan = make_plan()
        broken = copy.deepcopy(plan)
        broken["jobs"][0]["seed"] += 1
        with self.assertRaises(ValueError):
            verify_plan(broken)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            j = plan["jobs"][0]
            video, record = record_paths(root, j)
            video.parent.mkdir()
            video.write_bytes(b"test video")
            write_json_atomic(record, {"plan_sha256": plan["plan_sha256"], "job": j,
                "environment_sha256": "env", "video_sha256": sha256_file(video),
                "timing_seconds": {"candidate_total": 1}})
            verify_record(root, j, plan, "env")
            video.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                verify_record(root, j, plan, "env")
            path = root / "locked.json"
            immutable_json(path, {"a": 1})
            immutable_json(path, {"a": 1})
            with self.assertRaises(ValueError):
                immutable_json(path, {"a": 2})

    def test_paired_report(self):
        from UNIV_adaptor.scripts.data.score_hy15_endpoint_prior import DIMENSIONS, report
        rows, scores = [], {}
        for case in read(DEFAULT_PROTOCOL)["cases"]:
            r = {"stem": case["id"], "case": case["id"], "group": "p0_s42", "prompt_id": 0,
                 "base_seed": 42, "family_id": "test", "motion": "low", "detail": "high", "seconds": 100}
            rows.append(r)
            scores[case["id"]] = {d: 0.8 for d in DIMENSIONS}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report(rows, scores, root)
            self.assertTrue((root / "relative_to_full.csv").exists())
            self.assertTrue((root / "prompt_mean_targets.csv").exists())
            self.assertIn("NOT official", (root / "report.md").read_text())

    def test_partial_snapshot_requires_complete_group(self):
        from UNIV_adaptor.scripts.data.score_hy15_endpoint_prior import collect, coverage_report
        plan = make_plan()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json_atomic(root / "plan.json", plan)
            write_json_atomic(root / "environment.json", {"run": "synthetic"})
            env = canonical_sha256({"run": "synthetic"})
            for job in plan["jobs"][:9]:
                video, record = record_paths(root, job)
                video.parent.mkdir(exist_ok=True)
                record.parent.mkdir(exist_ok=True)
                video.write_bytes(job["id"].encode())
                write_json_atomic(record, {"schema": "hy15_endpoint_record_v1", "job": job,
                    "plan_sha256": plan["plan_sha256"], "environment_sha256": env,
                    "video_sha256": sha256_file(video),
                    "timing_seconds": {"candidate_total": 10, "main": 7, "transition": 1, "refine": 2}})
            progress = coverage_report(root)
            self.assertEqual((progress["completed_videos"], progress["complete_groups"]), (9, 1))
            rows, digest, snapshot = collect(root, partial=True)
            self.assertEqual(len(rows), 8)
            self.assertEqual(snapshot["complete_groups"], 1)
            self.assertEqual(len(snapshot["records"]), 8)
            self.assertEqual(digest, canonical_sha256(snapshot))
            video, _ = record_paths(root, plan["jobs"][0])
            video.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                collect(root, partial=True)


if __name__ == "__main__":
    unittest.main()
