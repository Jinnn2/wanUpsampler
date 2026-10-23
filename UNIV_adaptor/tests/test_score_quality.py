import contextlib
import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from UNIV_adaptor.data_protocol import sha256_file  # noqa: E402
from UNIV_adaptor.scripts.data import score_quality as sq  # noqa: E402


class QualityReportTest(unittest.TestCase):
    """plan -> (cached judge answers) -> report, without a model."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        fixed = self.tmp / "fixed"
        (fixed / "cache").mkdir(parents=True)
        videos = []
        for case, sharp in ((sq.FULL, 5.0), ("LR_BICUBIC", 3.5), ("SCALING10", 5.0)):
            clip = self.tmp / f"{case}.mp4"
            clip.write_bytes(case.encode())
            sha = sha256_file(clip)
            videos.append({"source": "src", "stem": f"p1_{case}", "group_id": "p1_s42", "prompt_id": 1, "family_id": 0,
                           "split": "train", "motion_level": "high", "detail_level": "low", "base_seed": 42,
                           "case_id": case, "video_path": str(clip), "video_sha256": sha})
            (fixed / "cache" / f"{sha}.json").write_text(json.dumps({"sharpness": sharp}))
        test = dict(videos[0], split="test", stem="p70_FULL", group_id="p70_s42", video_sha256="x")
        (fixed / "plan.json").write_text(json.dumps({"videos": videos + [test]}))
        self.out = self.tmp / "q"
        with contextlib.redirect_stdout(io.StringIO()):
            sq.plan(Namespace(fixed_root=str(fixed), out=str(self.out)))
        self.videos = {v["case_id"]: v for v in videos}

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def write(self, kind, name, answers, **extra):
        (self.out / kind).mkdir(exist_ok=True)
        (self.out / kind / f"{name}.json").write_text(json.dumps({"protocol": sq.PROTOCOL, "answers": answers, **extra}))

    def test_report(self):
        value = sq.load_plan(self.out)
        self.assertEqual(len(value["videos"]), 3)  # the test-split row never enters
        good = {a: {"p_yes": 0.9} for a in sq.ASPECTS}
        bad = {a: {"p_yes": 0.2} for a in sq.ASPECTS}
        full, lr, sc = (self.videos[c]["video_sha256"] for c in (sq.FULL, "LR_BICUBIC", "SCALING10"))
        self.write("vlm", full, good)
        self.write("vlm", lr, bad)
        self.write("vlm", sc, good)
        self.write("pair", f"{lr}__{full}", {a: {"p_full_first": 0.1, "p_accel_first": 0.6} for a in sq.ASPECTS})
        self.write("pair", f"{sc}__{full}", {a: {"p_full_first": 0.7, "p_accel_first": 0.8} for a in sq.ASPECTS})
        self.assertEqual([k for k, _ in sq.pair_jobs(value)], sorted([f"{lr}__{full}", f"{sc}__{full}"]))
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            sq.report(Namespace(out=str(self.out)))
        with (self.out / "quality_by_video.csv").open(encoding="utf-8") as handle:
            rows = {r["case_id"]: r for r in csv.DictReader(handle)}
        self.assertEqual(rows["LR_BICUBIC"]["quality_kept"], "0.0")  # one order below 0.5: not kept
        self.assertEqual(rows["SCALING10"]["quality_kept"], "1.0")
        self.assertEqual((rows["LR_BICUBIC"]["acceptable"], rows["LR_BICUBIC"]["full_acceptable"]), ("0", "1"))
        self.assertAlmostEqual(float(rows["LR_BICUBIC"]["d_sharpness"]), -1.5)
        self.assertEqual(rows[sq.FULL]["quality_kept"], "")
        self.assertIn("LR_BICUBIC", printed.getvalue())

    def test_changed_aspects_refuse_the_plan(self):
        plan = json.loads((self.out / "plan.json").read_text(encoding="utf-8"))
        plan["aspects"][0][2] = "Is it sharp?"
        (self.out / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaises(ValueError):
            sq.load_plan(self.out)


if __name__ == "__main__":
    unittest.main()
