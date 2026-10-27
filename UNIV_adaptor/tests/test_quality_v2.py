import contextlib
import csv
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from UNIV_adaptor.scripts.data import quality_v2 as q2  # noqa: E402


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class QualityV2Test(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_flags_and_scores(self):
        # 100 FULL clips with flow_alternation 0..0.99 and flicker 0..0.99: both q99 thresholds are ~0.98.
        fixed = [{"source": "s", "stem": f"f{i}", "case_id": q2.FULL, "flow_alternation": i / 100, "flicker": i / 100,
                  "d_flow_alternation": "", "d_flicker": ""} for i in range(100)]
        accel = {"ok": (0.5, 0.5, 0.0, 0.0), "judder": (1.9, 0.5, 1.0, 0.0), "flicker": (0.5, 3.0, 0.0, 2.0),
                 "high_but_like_full": (1.9, 0.5, -0.1, 0.0)}
        quality = [{"source": "s", "stem": "f0", "case_id": q2.FULL, "quality_kept": "",
                    **{f"kept_{a}": "" for a in q2.ASPECTS}}]
        for name, (alt, fl, d_alt, d_fl) in accel.items():
            fixed.append({"source": "s", "stem": name, "case_id": "X", "flow_alternation": alt, "flicker": fl,
                          "d_flow_alternation": d_alt, "d_flicker": d_fl})
            quality.append({"source": "s", "stem": name, "case_id": "X", "quality_kept": 1.0,
                            **{f"kept_{a}": 1 for a in q2.ASPECTS}})
        write(self.tmp / "fixed.csv", fixed)
        write(self.tmp / "q.csv", quality)
        out = self.tmp / "v2" / "quality_by_video.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            q2.main(["--quality", str(self.tmp / "q.csv"), "--fixed", str(self.tmp / "fixed.csv"), "--out", str(out)])
        rows = {r["stem"]: r for r in q2.read_csv(out)}
        self.assertEqual((rows["ok"]["judder"], rows["ok"]["flicker"], float(rows["ok"]["quality_kept"])), ("0", "0", 1.0))
        self.assertEqual((rows["judder"]["judder"], float(rows["judder"]["quality_kept"])), ("1", 0.8))
        self.assertEqual(float(rows["judder"]["quality_kept_gate"]), 0.0)
        self.assertEqual(rows["flicker"]["flicker"], "1")
        self.assertEqual(rows["high_but_like_full"]["judder"], "0")  # no worse than its own FULL
        self.assertEqual(rows["f0"]["quality_kept"], "")
        self.assertTrue((out.parent / "quality_v2_thresholds.json").exists())


if __name__ == "__main__":
    unittest.main()
