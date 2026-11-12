import collections
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from UNIV_adaptor.data_protocol import sha256_file  # noqa: E402
from UNIV_adaptor.scripts.data import annotation_tri as tri  # noqa: E402

TRI_CSV = REPO / "results/attention/quality_tri_seeds/tri_by_pair.csv"
COMMITTED = REPO / "prompts/annotation_tri_v1/pairs.json"


class TriAnnotationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            tri.select(Namespace(tri_csv=str(TRI_CSV), out=str(self.tmp / "sel")))
        self.pairs_path = self.tmp / "sel" / "pairs.json"
        self.value = json.loads(self.pairs_path.read_text(encoding="utf-8"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_select_is_fixed_and_follows_the_design(self):
        committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
        self.assertEqual(committed["pairs"], self.value["pairs"])   # the committed list is what select draws
        self.assertEqual(committed["tasks"], self.value["tasks"])
        pairs = self.value["pairs"]
        self.assertEqual(collections.Counter(q["set"] for q in pairs), collections.Counter(tri.SET_SIZE))
        self.assertFalse(any(64 <= q["prompt_id"] <= 79 for q in pairs))
        s100 = [q for q in pairs if q["set"] == "seed100"]
        self.assertTrue(all(100 in (q["x_seed"], q["y_seed"]) and q["motion_level"] == "high" for q in s100))
        self.assertEqual(len({q["prompt_id"] for q in s100}), 12)
        for p in tri.NAMED:
            q = next(q for q in s100 if q["prompt_id"] == p)
            self.assertEqual((q["x_seed"], q["y_seed"]), (100, 42))
        null = {q["prompt_id"] for q in pairs if q["set"] == "null"}
        self.assertEqual(len(null | {q["prompt_id"] for q in s100}), 16)   # every high-motion pilot prompt once
        self.assertFalse(null & {q["prompt_id"] for q in s100})
        self.assertTrue(all(q["motion_level"] == "high" for q in pairs if q["set"] in ("svg2", "null")))
        tasks = self.value["tasks"]
        self.assertEqual(len(tasks), 46)
        x_left = {q["pair"]: q["x_left"] for q in pairs}
        for t in tasks:
            self.assertEqual(t["x_left"], x_left[t["pair"]] != t["repeat"])   # repeats swap sides
        for who in ("A", "B"):
            at = collections.defaultdict(list)
            for i, t in enumerate(tri.order(self.value, who)):
                at[t["pair"]].append(i)
            self.assertTrue(all(len(v) == 1 or abs(v[0] - v[1]) > tri.MIN_REPEAT_GAP for v in at.values()))
        self.assertNotEqual(tri.order(self.value, "A"), tri.order(self.value, "B"))

    def fake_plan(self):
        rows = []
        for q in self.value["pairs"]:
            def clip(seed, case):
                path = self.tmp / "clips" / f"p{q['prompt_id']}_s{seed}_{case}.mp4"
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(f"{q['prompt_id']}-{seed}-{case}".encode())
                return {"prompt_id": q["prompt_id"], "video_path": str(path), "video_sha256": sha256_file(path)}
            rows.append({"kind": q["kind"], "arm": q["arm"], "x": clip(q["x_seed"], q["x_case"]),
                         "y": clip(q["y_seed"], q["y_case"])})
        plan = self.tmp / "plan.json"
        plan.write_text(json.dumps({"pairs": rows}), encoding="utf-8")
        return plan

    def test_kit_hides_method_and_seed(self):
        plan = self.fake_plan()
        with mock.patch.object(tri, "encode", lambda src, dst, ffmpeg: shutil.copy(src, dst)), \
                contextlib.redirect_stdout(io.StringIO()):
            tri.kit(Namespace(pairs=str(self.pairs_path), plan=str(plan), out=str(self.tmp / "kits"), annotator="A",
                              ffmpeg="ffmpeg", jobs=2))
        kdir = self.tmp / "kits" / "kit_tri_A"
        text = (kdir / "tasks_tri.js").read_text(encoding="utf-8")
        for word in ("SVG2", "RCM1", "SCALING10", "LR_BICUBIC", "FULL", "seed", "x_left"):
            self.assertNotIn(word, text)
        payload = json.loads(text[len("window.ANNOTATION = "):].rstrip().rstrip(";"))
        self.assertEqual(payload["kind"], "tri")
        self.assertEqual(payload["sample_sha256"], sha256_file(self.pairs_path))
        self.assertEqual(len(payload["tasks"]), 46)
        self.assertEqual(len(list((kdir / "videos").iterdir())), 92)
        tasks = {t["tid"]: t for t in self.value["tasks"]}
        pairs = {q["pair"]: q for q in self.value["pairs"]}
        for t in payload["tasks"][:10]:   # the left file is the x clip exactly when x_left
            task = tasks[t["gid"]]
            q = pairs[task["pair"]]
            left = (kdir / "videos" / f"{t['gid']}_L.mp4").read_bytes().decode()
            x = f"{q['prompt_id']}-{q['x_seed']}-{q['x_case']}"
            self.assertEqual(left == x, task["x_left"])
        self.assertIn('src="tasks_tri.js"', (kdir / "annotate_tri.html").read_text(encoding="utf-8"))
        self.assertTrue((kdir / "标注说明.txt").exists())

    def labels(self, who, flip_bad=False):
        """Labels that copy the judge's answer on every question (both presentations)."""
        pairs = {q["pair"]: q for q in self.value["pairs"]}
        answers = {}
        for t in self.value["tasks"]:
            q = pairs[t["pair"]]
            ans = {}
            for qid in tri.QIDS:
                out = q["judge"][qid]["out"]
                if flip_bad and q["set"] == "bad":
                    out = "x"
                side = "same" if out == "same" else ("left" if (out == "x") == t["x_left"] else "right")
                ans[qid] = {"better": side}
            answers[t["tid"]] = ans
        path = self.tmp / f"labels_tri-{who}.json"
        path.write_text(json.dumps({"annotator": who, "kind": "tri", "sample_sha256": sha256_file(self.pairs_path),
                                    "answers": answers}), encoding="utf-8")
        return str(path)

    def test_analyze_matches_labels_that_copy_the_judge(self):
        labels = [self.labels("A"), self.labels("B")]
        with contextlib.redirect_stdout(io.StringIO()):
            tri.analyze(Namespace(pairs=str(self.pairs_path), labels=labels, out=str(self.tmp / "res")))
        report = (self.tmp / "res" / "human_tri_analysis.md").read_text(encoding="utf-8")
        self.assertIn("| A | 6 | 0 | 0 | yes |", report)
        for qid in tri.QIDS:
            self.assertIn(f"| A | {qid} | 1.00 | 6 |", report)   # self-consistency
            self.assertRegex(report, rf"\| consensus \| {qid} \| hard \| 28 \| 1.00 \| 1.00 \|")
        n = sum(q["set"] == "seed100" and q["judge"]["overall"]["out"] != "same" for q in self.value["pairs"])
        self.assertRegex(report, r"\| seed100 \| consensus \| (\d+) \| \d+ \| (\d+) \| [0-9.]+ \| \1 / \d+ / \2 \|")
        self.assertGreater(n, 0)
        rows = (self.tmp / "res" / "human_tri_by_pair.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 1 + 2 * 46 * len(tri.QIDS))

    def test_annotator_who_fails_the_bad_set_is_excluded(self):
        labels = [self.labels("A"), self.labels("B", flip_bad=True)]
        with contextlib.redirect_stdout(io.StringIO()):
            tri.analyze(Namespace(pairs=str(self.pairs_path), labels=labels, out=str(self.tmp / "res")))
        report = (self.tmp / "res" / "human_tri_analysis.md").read_text(encoding="utf-8")
        self.assertIn("| B | 0 | 0 | 6 | NO (excluded below) |", report)
        self.assertNotIn("| consensus |", report)
        self.assertNotIn("| B | overall | hard |", report)

    def test_foreign_labels_are_refused(self):
        path = Path(self.labels("A"))
        data = json.loads(path.read_text(encoding="utf-8"))
        data["sample_sha256"] = "0" * 64
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            tri.analyze(Namespace(pairs=str(self.pairs_path), labels=[str(path)], out=str(self.tmp / "res")))

    def test_sign_test(self):
        self.assertAlmostEqual(tri.sign_test(10, 12), 79 / 4096)
        self.assertAlmostEqual(tri.sign_test(7, 8), 9 / 256)
        self.assertAlmostEqual(tri.sign_test(0, 5), 1.0)


if __name__ == "__main__":
    unittest.main()
