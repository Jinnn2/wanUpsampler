import collections
import csv
import io
import json
from contextlib import redirect_stdout
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
            "video_sha256": f"v{value}", "stem": f"{case}_{prompt}_{seed}"}


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
                qt.plan(types.SimpleNamespace(quality_root=str(tmp / "q"), out=str(out), arms=["GOOD", "BAD"], prompts=2, seeds=["42", "100"]))
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

    def test_seed_triangle(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            videos = []
            for p in range(4):  # seed 2024 always best, then 42, then 100 (a seed main effect)
                base = 30 + 50 * p
                videos += [video(p, 42, "FULL_NATIVE_50", "high", base), video(p, 100, "FULL_NATIVE_50", "high", base - 10),
                           video(p, 2024, "FULL_NATIVE_50", "high", base + 10)]
            (tmp / "q").mkdir()
            (tmp / "q" / "plan.json").write_text(json.dumps({"videos": videos}))
            out = tmp / "tri"
            buf = io.StringIO()
            with mock.patch.object(qt, "sq", fake_sq), mock.patch.object(qt, "tri_judge", lambda *a: FakeJudge()), \
                    redirect_stdout(buf):
                qt.plan(types.SimpleNamespace(quality_root=str(tmp / "q"), out=str(out), arms=[], prompts=4,
                                              seeds=["42", "100", "2024"]))
                qt.pair(types.SimpleNamespace(out=str(out), model="m", device="cpu", shard_index=0, shard_count=1))
                qt.report(types.SimpleNamespace(out=str(out), old_cache=[]))
            arms = sorted({r["arm"] for r in json.loads((out / "plan.json").read_text())["pairs"]})
            self.assertEqual(arms, ["FULL", "FULL_S2024_vs_S100", "FULL_S2024_vs_S42", "FULL_SEED"])
            text = buf.getvalue()
            self.assertIn("s2024 +0.700", text)
            self.assertIn("s100 -0.700", text)
            self.assertIn("cycles 0.00", text)

    def test_plan_all_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            videos = []
            for p in range(2):
                for s, d in ((42, 0), (100, 1), (2024, 2)):
                    base = 30 + 50 * p + 10 * d
                    videos += [video(p, s, "FULL_NATIVE_50", "high", base), video(p, s, "BAD", "high", base - 8)]
            # the same BAD clip listed under a second source: judged once, exported twice
            twin = dict(videos[1], source="s2")
            videos += [twin, dict(videos[0], source="s2")]
            (tmp / "q").mkdir()
            (tmp / "q" / "plan.json").write_text(json.dumps({"videos": videos}))
            out = tmp / "all"
            buf = io.StringIO()
            with mock.patch.object(qt, "sq", fake_sq), mock.patch.object(qt, "tri_judge", lambda *a: FakeJudge()),                     redirect_stdout(buf):
                qt.plan_all(types.SimpleNamespace(quality_root=str(tmp / "q"), out=str(out), arms=[]))
                plan = json.loads((out / "plan.json").read_text())
                self.assertEqual(collections.Counter(r["kind"] for r in plan["pairs"]), {"arm": 6, "seed": 6})
                qt.pair(types.SimpleNamespace(out=str(out), model="m", device="cpu", shard_index=0, shard_count=1))
                qt.export(types.SimpleNamespace(out=str(out)))
            rows = list(csv.DictReader(open(out / "quality_tri_by_video.csv")))
            self.assertEqual(len(rows), 7)
            self.assertEqual({r["source"] for r in rows if r["stem"] == "BAD_0_42"}, {"s", "s2"})
            self.assertAlmostEqual(float(rows[0]["quality_tri"]), 0.3, places=4)
            self.assertIn("s2024 +0.700", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
