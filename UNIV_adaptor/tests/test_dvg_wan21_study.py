"""DVG study planning, receipt hand-off to requirement scoring, and repro summary."""
import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from UNIV_adaptor.data_protocol import sha256_file
from UNIV_adaptor.scripts.data import dvg_wan21_study as study
from UNIV_adaptor.scripts.data import score_controlled_factor_requirements as req

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "prompts" / "univ_controlled_factor_v1.jsonl"
CHECKLIST = ROOT / "prompts" / "univ_controlled_factor_v1_requirements.jsonl"


class DvgStudyTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.vbench = self.root / "all_dimension.txt"
        self.vbench.write_text("\n".join(f"vbench prompt {i}" for i in range(946)) + "\n", encoding="utf-8")
        self.out = self.root / "study"
        study.plan(argparse.Namespace(out=str(self.out), prompts=str(PROMPTS), vbench_prompts=str(self.vbench),
                                      vbench_ext_prompts=None, repro_count=48, sets="main,repro"))
        self.value = study.load_plan(self.out)

    def test_plan_excludes_test_and_covers_arms(self):
        main = [j for j in self.value["jobs"] if j["set"] == "main"]
        repro = [j for j in self.value["jobs"] if j["set"] == "repro"]
        self.assertEqual(len(main), 64 * 3 * 3)
        self.assertEqual(len(repro), 48 * 2)
        self.assertNotIn("test", {j["split"] for j in main})
        self.assertEqual(len({j["stem"] for j in self.value["jobs"]}), len(self.value["jobs"]))
        dvg = [j for j in main if j["case_id"] == "DVG_D050"]
        self.assertTrue(all(j["budget"] == 0.5 for j in dvg))
        self.assertTrue(all(j["budget"] is None for j in main if j["case_id"] == study.FULL))

    def test_extended_prompts_align_with_short_indices(self):
        ext = self.root / "ext.txt"
        ext.write_text("\n".join(f"long rewrite {i}" for i in range(946)) + "\n", encoding="utf-8")
        out = self.root / "ext_study"
        args = argparse.Namespace(out=str(out), prompts=str(PROMPTS), vbench_prompts=str(self.vbench),
                                  vbench_ext_prompts=str(ext), repro_count=48, sets="repro_ext")
        study.plan(args)
        jobs = study.load_plan(out)["jobs"]
        short = {j["prompt_id"]: j["prompt"] for j in self.value["jobs"] if j["set"] == "repro"}
        self.assertEqual({j["set"] for j in jobs}, {"repro_ext"})
        self.assertEqual(len(jobs), 96)
        self.assertEqual({j["prompt_id"] for j in jobs}, set(short))
        for j in jobs:
            self.assertEqual(j["prompt"], f"long rewrite {j['prompt_id']}")
            self.assertEqual(j["short_prompt"], short[j["prompt_id"]])
        ext.write_text("too short\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            study.plan(args)

    def test_shards_mix_arms(self):
        for shard in range(4):
            jobs = [j for i, j in enumerate(self.value["jobs"]) if i % 4 == shard]
            self.assertEqual({j["case_id"] for j in jobs}, set(study.ARMS))

    def write_receipts(self, subset):
        (self.out / "receipts").mkdir(exist_ok=True)
        (self.out / "videos").mkdir(exist_ok=True)
        for j in self.value["jobs"]:
            if j["set"] != subset:
                continue
            video = self.out / "videos" / f"{j['stem']}.mp4"
            video.write_bytes(j["stem"].encode())
            receipt = {**j, "generate_seconds": 100.0 if j["case_id"] == study.FULL else 40.0, "dvg": None,
                       "video_path": str(video), "video_sha256": sha256_file(video)}
            (self.out / "receipts" / f"{j['stem']}.json").write_text(json.dumps(receipt), encoding="utf-8")

    def test_requirement_plan_reads_main_receipts(self):
        self.write_receipts("main")
        self.write_receipts("repro")
        req_out = self.root / "req"
        req.plan(argparse.Namespace(dvg_study=str(self.out), root=None, checklist=str(CHECKLIST), out=str(req_out)))
        value = req.load_plan(req_out)
        self.assertEqual(len(value["videos"]), 576)
        self.assertEqual({v["case_id"] for v in value["videos"]}, set(study.ARMS))

    def test_repro_summary_reports_speedup(self):
        rows = []
        for i in range(10):
            for arm, shift in ((study.FULL, 0.0), ("DVG_D050", -0.01)):
                row = {"group_id": f"g{i}", "case_id": arm, "pipeline_seconds": 100.0 if arm == study.FULL else 40.0}
                for m in ("vbench5", *study.QUALITY, *study.DIAGNOSTIC):
                    row[m] = 0.8 + shift
                rows.append(row)
        with (self.out / "repro_by_video.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        study.repro_summary(self.out)
        text = (self.out / "repro_summary.md").read_text(encoding="utf-8")
        self.assertIn("2.50x", text)
        self.assertIn("| vbench5 | 0.8000 | 0.7900 | -0.0100 |", text)


if __name__ == "__main__":
    unittest.main()
