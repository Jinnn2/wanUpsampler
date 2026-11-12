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
from UNIV_adaptor.scripts.data import quality_null as qn  # noqa: E402

ASPECTS = ("sharp", "artifact", "aesthetic", "temporal", "acceptable")


def fake_sq(rule):
    """A stand-in for score_quality: score_pair(judge, X, Y) answers 'X >= Y' from rule(X_name, Y_name)."""
    m = types.SimpleNamespace(PROTOCOL="quality_v1", ASPECTS=ASPECTS)
    m.Judge = lambda *a, **k: object()
    m.verified = lambda v: (v["video_path"], 16.0)
    m.decode = lambda path: (np.load(path + ".npy"), 16.0)

    def score_pair(judge, x, y):
        p = rule(Path(x[0]).stem, Path(y[0]).stem)
        return {a: {"p_full_first": p, "p_accel_first": p} for a in ASPECTS}
    m.score_pair = score_pair
    return m


def video(tmp, name, prompt, seed, case, motion, source="src"):
    path = tmp / f"{name}.mp4"
    path.write_text(name)
    np.save(str(path) + ".npy", np.full((4, 2, 2, 3), hash(name) % 200, dtype=np.uint8))
    return {"source": source, "group_id": f"main_p{prompt:03d}_s{seed}", "prompt_id": prompt, "base_seed": seed,
            "case_id": case, "motion_level": motion, "detail_level": "low", "family_id": "f", "video_path": str(path),
            "video_sha256": name}


class NullTest(unittest.TestCase):
    def test_outcomes_from_both_directions(self):
        def rule(x, y):
            if x == y:
                return 0.9                       # identical: yes
            if x.startswith("F") and y.startswith("F"):
                return 0.3                       # two FULL seeds: the judge says neither is as good
            if x.startswith("S"):                # SVG2 >= FULL: no on high motion, yes on low
                return 0.2 if x.endswith("h") else 0.8
            return 0.8                           # FULL >= SVG2: yes
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            videos = []
            for p, m in ((0, "high"), (1, "low")):
                tag = m[0]
                videos += [video(tmp, f"F42_{p}", p, 42, "FULL_NATIVE_50", m), video(tmp, f"F100_{p}", p, 100, "FULL_NATIVE_50", m),
                           video(tmp, f"S42_{p}{tag}", p, 42, "SVG2", m)]
            qroot = tmp / "q"
            qroot.mkdir()
            (qroot / "plan.json").write_text(json.dumps({"videos": videos}))
            out = tmp / "null"
            with mock.patch.object(qn, "sq", lambda: fake_sq(rule)):
                qn.plan(types.SimpleNamespace(quality_root=str(qroot), out=str(out), arms=["SVG2"], identity=1))
                plan = json.loads((out / "plan.json").read_text())
                self.assertEqual(sorted(r["kind"] for r in plan["pairs"]).count("seed"), 4)
                qn.pair(types.SimpleNamespace(out=str(out), model="m", device="cpu", shard_index=0, shard_count=1))
                qn.psnr(types.SimpleNamespace(out=str(out), shard_index=0, shard_count=1))
                qn.report(types.SimpleNamespace(out=str(out)))
            rows = list(csv.DictReader(open(out / "null_by_pair.csv")))
            by = {(r["kind"], r["arm"], r["motion_level"]): r for r in rows}
            self.assertEqual(by[("identity", "FULL", "high")]["fwd_sharp"], "1")
            self.assertEqual(by[("seed", "FULL_SEED", "high")]["out_sharp"], "undecided")
            self.assertEqual(by[("arm", "SVG2", "high")]["out_sharp"], "worse")
            self.assertEqual(by[("arm", "SVG2", "low")]["out_sharp"], "equal")
            self.assertEqual(float(by[("arm", "SVG2", "high")]["not_worse"]), 0.0)
            self.assertEqual(float(by[("seed", "FULL_SEED", "low")]["quality_kept"]), 0.0)
            self.assertEqual(float(by[("seed", "FULL_SEED", "low")]["not_worse"]), 1.0)
            self.assertEqual(float(by[("identity", "FULL", "high")]["psnr"]), 99.0)


if __name__ == "__main__":
    unittest.main()
