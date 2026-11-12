import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from UNIV_adaptor.scripts.data import quality_tri as qt  # noqa: E402

QUALITY = (("sharp", "frames", "q1"), ("artifact", "frames", "q2"), ("aesthetic", "frames", "q3"),
           ("temporal", "video", "q4"), ("acceptable", "video", "q5"))


def fake_sq():
    m = types.SimpleNamespace(QUALITY=QUALITY, ASPECTS=tuple(a[0] for a in QUALITY), FRAME_COUNT=5)
    m.verified = lambda v: (np.full((8, 2, 2, 3), int(v["video_sha256"][1:]), dtype=np.uint8), 16.0)
    m.video_clip = lambda frames, fps: (frames, {"fps": fps})
    return m


def quality(img):  # "quality" of a clip is its pixel value; a clip id encodes it
    return int(np.asarray(img).flat[0])


class FakeJudge:
    """Prefers the higher-valued clip; Same when values are within 5. Always a slight A bias."""
    def p_tri(self, prompts, images=None, videos=None, metadata=None):
        vis = images if images else videos
        out = []
        for k in range(len(prompts)):
            a, b = quality(vis[2 * k]), quality(vis[2 * k + 1])
            p = (0.1, 0.1, 0.8) if abs(a - b) <= 5 else ((0.8, 0.1, 0.1) if a > b else (0.1, 0.8, 0.1))
            out.append({"A": p[0] + 0.02, "B": p[1] - 0.02, "Same": p[2], "mass": 0.99})
        return out


def video(prompt, seed, case, motion, value):
    return {"source": "s", "group_id": f"g{prompt}_{seed}", "prompt_id": prompt, "base_seed": seed, "case_id": case,
            "motion_level": motion, "detail_level": "low", "family_id": f"f{prompt}", "video_path": "unused",
            "video_sha256": f"v{value}"}


class TriTest(unittest.TestCase):
    def test_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            videos = []
            for p, m in ((0, "high"), (1, "high"), (2, "low")):
                base = 100 + 20 * p
                videos += [video(p, 42, "FULL_NATIVE_50", m, base), video(p, 100, "FULL_NATIVE_50", m, base + 1),
                           video(p, 42, "GOOD", m, base + 2), video(p, 42, "BAD", m, base - 15)]
            videos.append(video(3, 42, "FULL_NATIVE_50", "low", 200))  # no seed 100: not planned
            (tmp / "q").mkdir()
            (tmp / "q" / "plan.json").write_text(json.dumps({"videos": videos}))
            out = tmp / "tri"
            old = tmp / "old"
            old.mkdir()
            # old binary: BAD >= FULL is No in both orders, FULL >= BAD Yes -> worse
            yes, no = {"p_full_first": .9, "p_accel_first": .9}, {"p_full_first": .1, "p_accel_first": .1}
            (old / "v105__v120.json").write_text(json.dumps({"answers": {a[0]: no for a in QUALITY}}))
            (old / "v120__v105.json").write_text(json.dumps({"answers": {a[0]: yes for a in QUALITY}}))
            with mock.patch.object(qt, "sq", fake_sq), mock.patch.object(qt, "tri_judge", lambda *a: FakeJudge()):
                qt.plan(types.SimpleNamespace(quality_root=str(tmp / "q"), out=str(out), arms=["GOOD", "BAD"], prompts=2))
                plan = json.loads((out / "plan.json").read_text())
                self.assertEqual(plan["prompts"], ["0", "1", "2"])
                qt.pair(types.SimpleNamespace(out=str(out), model="m", device="cpu", shard_index=0, shard_count=1))
                qt.report(types.SimpleNamespace(out=str(out), old_cache=[str(old)]))
            rows = list(csv.DictReader(open(out / "tri_by_pair.csv")))
            by = {(r["kind"], r["arm"], r["prompt_id"]): r for r in rows}
            self.assertEqual(by[("identity", "FULL", "0")]["out_sharp"], "same")
            self.assertAlmostEqual(float(by[("identity", "FULL", "0")]["bias_sharp"]), 0.04, places=4)
            self.assertAlmostEqual(float(by[("identity", "FULL", "0")]["net"]), 0.0, places=4)
            self.assertEqual(by[("seed", "FULL_SEED", "0")]["out_temporal"], "same")
            self.assertEqual(by[("arm", "BAD", "1")]["out_acceptable"], "y")   # FULL better
            self.assertAlmostEqual(float(by[("arm", "BAD", "1")]["net"]), -0.7, places=4)  # A bias cancels
            self.assertEqual(float(by[("arm", "BAD", "1")]["old_quality_kept"]), 0.0)
            self.assertEqual(float(by[("arm", "BAD", "1")]["old_not_worse"]), 0.0)
            self.assertEqual(by[("arm", "GOOD", "2")]["out_sharp"], "same")


if __name__ == "__main__":
    unittest.main()
