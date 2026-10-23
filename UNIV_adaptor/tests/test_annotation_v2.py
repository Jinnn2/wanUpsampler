import contextlib
import csv
import io
import json
import random
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
from UNIV_adaptor.scripts.data import annotation_v2 as an  # noqa: E402
from UNIV_adaptor.scripts.data import requirements_v2 as rv2  # noqa: E402

PROMPTS = [REPO / "prompts/univ_controlled_factor_v1.jsonl", REPO / "prompts/univ_controlled_factor_dev2.jsonl"]
FROZEN = REPO / "prompts/univ_requirements_v2_frozen.jsonl"


def fake_dynamic(root, clip):
    """A dynamic-layer root over every frozen prompt, one seed, FULL plus all 15 arms, random judge results."""
    rng = random.Random(0)
    prompts = rv2.load_prompts(PROMPTS)
    checklist = {r["prompt_id"]: r for r in rv2.load_jsonl(FROZEN)}
    videos, rows = [], []
    for pid, rec in sorted(checklist.items()):
        p = prompts[pid]
        for case in (an.FULL, *an.FAMILY):
            v = {"source": "src", "stem": f"p{pid}_{case}", "group_id": f"p{pid}_s42", "prompt_id": pid,
                 "split": p["split"], "motion_level": p["motion_level"], "detail_level": p["detail_level"],
                 "base_seed": 42, "case_id": case, "video_path": str(clip), "video_sha256": sha256_file(clip)}
            videos.append(v)
            for q in rec["questions"]:
                vlm = rv2.JUDGE[q["type"]].startswith("vlm") and case != an.FULL
                rows.append({"source": "src", "stem": v["stem"], "case_id": case, "qid": q["id"], "type": q["type"],
                             "value": round(rng.random() * 2, 4),
                             "pass_raw": int(rng.random() < 0.8), "kept": rng.choice(["", "0", "1"]),
                             "p_full_first": round(rng.random(), 3) if vlm else "",
                             "p_accel_first": round(rng.random(), 3) if vlm else ""})
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({"checklist": str(FROZEN), "checklist_sha256": sha256_file(FROZEN),
                                                "videos": videos}), encoding="utf-8")
    with (root / "dynamic_by_question.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {(r["stem"], r["qid"]): r for r in rows}


class AnnotationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.clip = self.tmp / "clip.mp4"
        self.clip.write_bytes(b"not really a video")
        self.judged = fake_dynamic(self.tmp / "dyn", self.clip)
        self.out = self.tmp / "ann"
        with contextlib.redirect_stdout(io.StringIO()):
            an.sample(Namespace(dyn_root=str(self.tmp / "dyn"), prompts=[str(p) for p in PROMPTS], out=str(self.out)))
        self.value = json.loads((self.out / "sample.json").read_text(encoding="utf-8"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_sample_is_stratified_blind_and_fixed(self):
        groups = self.value["groups"]
        self.assertEqual(len(groups), 5 * 4 * an.PER_CELL)
        cells = {(g["family"], g["cell"]) for g in groups}
        self.assertEqual(len(cells), 20)
        prompts = rv2.load_prompts(PROMPTS)
        self.assertFalse([g for g in groups if prompts[g["prompt_id"]]["split"] == "test"])
        self.assertTrue(all(1 <= len(g["questions"]) <= an.QUESTIONS_PER_GROUP for g in groups))
        a, b = set(self.value["assignment"]["A"]), set(self.value["assignment"]["B"])
        self.assertEqual(a & b, set(self.value["assignment"]["overlap"]))
        self.assertEqual(a | b, {g["gid"] for g in groups})
        self.assertTrue(40 < sum(g["accel_left"] for g in groups) < 160)
        # Same inputs, same sample; an existing sample is never overwritten.
        again = self.tmp / "ann2"
        with contextlib.redirect_stdout(io.StringIO()):
            an.sample(Namespace(dyn_root=str(self.tmp / "dyn"), prompts=[str(p) for p in PROMPTS], out=str(again)))
        self.assertEqual((again / "sample.json").read_text(encoding="utf-8"),
                         (self.out / "sample.json").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            an.sample(Namespace(dyn_root=str(self.tmp / "dyn"), prompts=[str(p) for p in PROMPTS], out=str(self.out)))

    def test_kit_hides_the_method(self):
        with mock.patch.object(an, "encode", lambda src, dst, ffmpeg: shutil.copy(src, dst)), \
                contextlib.redirect_stdout(io.StringIO()):
            an.kit(Namespace(out=str(self.out), annotator="A", ffmpeg="ffmpeg", jobs=4))
        kit = self.out / "kit_A"
        tasks = (kit / "tasks.js").read_text(encoding="utf-8")
        self.assertEqual(len(list((kit / "videos").glob("*.mp4"))), 2 * len(self.value["assignment"]["A"]))
        for word in list(an.FAMILY) + [an.FULL, "accel_left", "single_pair_disagree", "video_sha256"]:
            self.assertNotIn(word, tasks)
        self.assertTrue((kit / "annotate.html").exists())

    def test_analyze_matches_labels_that_copy_the_judge(self):
        sample_sha = sha256_file(self.out / "sample.json")
        by_gid = {g["gid"]: g for g in self.value["groups"]}
        labels = []
        for who in ("A", "B"):
            answers = {}
            for gid in self.value["assignment"][who]:
                g = by_gid[gid]
                acc, ful = ("left", "right") if g["accel_left"] else ("right", "left")
                answers[gid] = {}
                for q in g["questions"]:
                    ra = self.judged[(g["accel"]["stem"], q["id"])]
                    rf = self.judged[(g["full"]["stem"], q["id"])]
                    answers[gid][q["id"]] = {acc: "yes" if ra["pass_raw"] == 1 else "no",
                                             ful: "yes" if rf["pass_raw"] == 1 else "no", "better": "same"}
            path = self.tmp / f"labels_{who}.json"
            path.write_text(json.dumps({"annotator": who, "sample_sha256": sample_sha, "answers": answers}))
            labels.append(str(path))
        with contextlib.redirect_stdout(io.StringIO()):
            an.analyze(Namespace(out=str(self.out), labels=labels))
        report = (self.out / "analysis.md").read_text(encoding="utf-8")
        single = report.split("## Single-video judges")[1].split("By type")[0]
        self.assertIn("| all | ", single)
        self.assertIn("| 1.00 |", [l for l in single.splitlines() if l.startswith("| all")][0])
        kappas = report.split("## Annotator agreement")[1].split("##")[0]
        self.assertNotIn("| 0.", kappas.replace("| 0 |", ""))  # identical annotators: kappa 1.00 or undefined

    def test_quality_kit_reuses_the_m3_videos(self):
        with mock.patch.object(an, "encode", lambda src, dst, ffmpeg: shutil.copy(src, dst)),                 contextlib.redirect_stdout(io.StringIO()):
            an.kit(Namespace(out=str(self.out), annotator="A", ffmpeg="ffmpeg", jobs=4))
            an.quality_kit(Namespace(out=str(self.out), annotator="A", groups=an.QUALITY_GROUPS, kit_dir=None,
                                     zh="", allow_missing=False))
        kit = self.out / "kit_A"
        text = (kit / "tasks_quality.js").read_text(encoding="utf-8")
        payload = json.loads(text[len("window.ANNOTATION = "):].rstrip().rstrip(";"))
        self.assertEqual(payload["kind"], "quality")
        gids = [t["gid"] for t in payload["tasks"]]
        self.assertEqual(len(gids), an.QUALITY_GROUPS)
        self.assertTrue(set(self.value["assignment"]["overlap"]) <= set(gids) <= set(self.value["assignment"]["A"]))
        self.assertTrue(all(len(t["questions"]) == 5 for t in payload["tasks"]))
        for word in list(an.FAMILY) + [an.FULL, "accel_left", "video_sha256"]:
            self.assertNotIn(word, text)
        html = (kit / "annotate_quality.html").read_text(encoding="utf-8")
        self.assertIn('src="tasks_quality.js"', html)
        self.assertNotIn('src="tasks.js"', html)
        with self.assertRaises(ValueError):  # no videos for B yet
            an.quality_kit(Namespace(out=str(self.out), annotator="B", groups=10, kit_dir=None, zh="", allow_missing=False))

    def test_quality_analyze_matches_labels_that_copy_the_judge(self):
        from UNIV_adaptor.scripts.data.score_quality import ASPECTS

        rng = random.Random(1)
        rows = {}
        for g in self.value["groups"]:
            for side in ("accel", "full"):
                r = {"source": g["source"], "stem": g[side]["stem"]}
                for a in ASPECTS:
                    r[f"p_{a}"], r[f"full_p_{a}"], r[f"pair_{a}"] = rng.random(), rng.random(), rng.random()
                    r[f"kept_{a}"] = int(r[f"pair_{a}"] >= 0.5)
                r["d_sharpness"] = rng.random()
                rows[(r["source"], r["stem"])] = r
        csv_path = self.tmp / "quality_by_video.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(next(iter(rows.values()))))
            writer.writeheader()
            writer.writerows(rows.values())
        by_gid = {g["gid"]: g for g in self.value["groups"]}
        labels = []
        for who in ("A", "B"):
            answers = {}
            for gid in self.value["assignment"][who][:60]:
                g = by_gid[gid]
                acc, ful = ("left", "right") if g["accel_left"] else ("right", "left")
                ra, rf = rows[(g["source"], g["accel"]["stem"])], rows[(g["source"], g["full"]["stem"])]
                answers[gid] = {a: {acc: "yes" if ra[f"p_{a}"] >= 0.5 else "no", ful: "yes" if rf[f"p_{a}"] >= 0.5 else "no",
                                    "better": acc if ra[f"pair_{a}"] >= 0.5 else ful} for a in ASPECTS}
            path = self.tmp / f"labels_quality_{who}.json"
            path.write_text(json.dumps({"annotator": who, "kind": "quality",
                                        "sample_sha256": sha256_file(self.out / "sample.json"), "answers": answers}))
            labels.append(str(path))
        with contextlib.redirect_stdout(io.StringIO()):
            an.quality_analyze(Namespace(out=str(self.out), quality_csv=str(csv_path), labels=labels))
        report = (self.out / "quality_analysis.md").read_text(encoding="utf-8")
        for a in ASPECTS:
            self.assertRegex(report, rf"\| pairwise VLM {a} \| \d+ \| [0-9.]+ \| 1.00 \|")
            self.assertRegex(report, rf"\| {a} \| \d+ \| [0-9.]+ \| 1.00 \|")
        with self.assertRaises(ValueError):  # M3 labels are refused
            json_path = Path(labels[0])
            data = json.loads(json_path.read_text())
            data.pop("kind")
            json_path.write_text(json.dumps(data))
            an.quality_analyze(Namespace(out=str(self.out), quality_csv=str(csv_path), labels=labels))

    def test_calibrate(self):
        auc, cut, acc = an.calibrate([(0.1, True), (0.2, True), (0.9, False)], lower_is_yes=True)
        self.assertEqual((auc, cut, acc), (1.0, 0.2, 1.0))
        auc, cut, acc = an.calibrate([(0.3, False), (0.7, True), (0.8, True)])
        self.assertEqual((auc, cut, acc), (1.0, 0.7, 1.0))

    def test_kappa(self):
        self.assertAlmostEqual(an.kappa([("yes", "yes"), ("no", "no")])[0], 1.0)
        self.assertAlmostEqual(an.kappa([("yes", "no"), ("no", "yes")])[0], -1.0)


if __name__ == "__main__":
    unittest.main()
