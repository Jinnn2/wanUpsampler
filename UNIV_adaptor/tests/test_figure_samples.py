import contextlib
import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from UNIV_adaptor.data_protocol import sha256_file  # noqa: E402
from UNIV_adaptor.scripts.data import figure_samples as fs  # noqa: E402


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def video(path, shift):
    """Small clip with a bright bar moving `shift` pixels per frame."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 16, (104, 60))
    for t in range(9):
        frame = np.zeros((60, 104, 3), np.uint8)
        x = (10 + shift * t) % 90
        frame[25:35, x:x + 8] = 255
        writer.write(frame)
    writer.release()


class FigureSamplesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def build(self):
        """Two published sources sharing FULL clips, 4 cells x 3 seeds, arms with known scores."""
        t = self.tmp
        videos, fixed, dyn, qual, vb = [], [], [], [], {"plat": [], "two": []}
        arms = {"plat": ("STEP20",), "two": ("F41_DUP", "F41_RIFE", "RCM4")}
        clip = t / "v.mp4"
        video(clip, 4)
        sha = sha256_file(clip)
        prompts = []
        for pid, cell in enumerate(fs.CELLS):
            prompts.append({"prompt_id": pid, "prompt": f"prompt {cell}"})
            for seed in (1, 2, 3):
                gid = f"main_p{pid:03d}_s{seed}"
                for src in ("plat", "two"):
                    for case in (fs.FULL, *arms[src]):
                        stem = f"{src}_{gid}_{case}"
                        videos.append({"source": src, "stem": stem, "group_id": gid, "prompt_id": pid, "family_id": "f",
                                       "split": "dev", "motion_level": "high" if cell[0] == "h" else "low",
                                       "detail_level": "high" if cell[1] == "h" else "low", "base_seed": seed,
                                       "case_id": case, "video_path": str(clip), "video_sha256": sha})
                        # DUP: VBench equal to RIFE but judder; RCM4 on low motion: flow x3 at seed 1..3 severity rising
                        lr = np.log(3.0) * seed if case == "RCM4" and cell[0] == "l" else 0.0
                        fixed.append({"source": src, "stem": stem, "case_id": case, "logratio_flow_mag": lr,
                                      "flow_mag": 1.0, "d_flow_alternation": seed if case == "F41_DUP" else 0.0})
                        ret = {"STEP20": 0.9 if cell[1] == "h" else 0.6, "F41_RIFE": 0.8, "RCM4": 0.5 + 0.1 * seed,
                               "F41_DUP": 0.7}.get(case, "")
                        dyn.append({"source": src, "stem": stem, "retention": ret})
                        qual.append({"source": src, "stem": stem, "quality_kept": "" if case == fs.FULL else 0.8,
                                     "full_acceptable": 1, "judder": int(case == "F41_DUP")})
                        vb[src].append({"observation_id": stem, "vbench5": 0.80})
        sources = {s: {"kind": "published", "root": str(t / s)} for s in vb}
        for s, rows in vb.items():
            write(t / s / "metrics" / "quality_by_video.csv", rows)
        (t / "fixed").mkdir()
        (t / "fixed" / "plan.json").write_text(json.dumps({"sources": sources, "videos": videos}), encoding="utf-8")
        write(t / "fixed" / "fixed_by_video.csv", fixed)
        write(t / "dyn.csv", dyn)
        write(t / "q.csv", qual)
        (t / "p.jsonl").write_text("\n".join(json.dumps(p) for p in prompts), encoding="utf-8")

    def test_select_and_render(self):
        self.build()
        t = self.tmp
        out = t / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            fs.main(["select", "--fixed-root", str(t / "fixed"), "--dynamic", str(t / "dyn.csv"), "--quality", str(t / "q.csv"),
                     "--prompts", str(t / "p.jsonl"), "--arms-b", "STEP20", "F41_RIFE", "--out", str(out)])
        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        panels = {p["id"]: p for p in m["panels"]}
        # judder: 4 cells x 3 seeds all failing and blind vs RIFE; median of 12 sorted by severity is a seed-2 clip
        self.assertEqual(panels["A2_judder"]["failing_and_vbench_blind"], 12)
        self.assertEqual([r["label"] for r in panels["A2_judder"]["rows"]], ["FULL", "F41_DUP", "F41_RIFE"])
        self.assertEqual(panels["A2_judder"]["seed"], 2)
        # static motion: 2 low-motion cells x 3 seeds; flow ratio 3^seed, median of 6 -> seed 2
        self.assertEqual(panels["A3_static_motion"]["failing"], 6)
        self.assertEqual(panels["A3_static_motion"]["seed"], 2)
        self.assertNotIn("A1_freeze", panels)  # no DVG arm in this plan
        self.assertNotIn("A4_semantic_loss", panels)  # RCM4 retention 0.6-0.8, never <= 0.5
        self.assertEqual(sorted(p for p in panels if p.startswith("B_")), ["B_hh", "B_hl", "B_lh", "B_ll"])
        for arm, value in {"STEP20": 0.8, "F41_RIFE": 0.8}.items():  # min(retention 0.9 / 0.8, quality 0.8)
            self.assertAlmostEqual(panels["B_hh"]["cell_mean"][arm], value)
        self.assertTrue(any("A1_freeze" in n for n in m["notes"]))
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not on PATH")
        with contextlib.redirect_stdout(io.StringIO()):
            fs.main(["render", "--out", str(out), "--width", "208", "--frames", "4", "--panels", "A2_judder", "B_hh"])
        files = {p.name for p in (out / "render" / "A2_judder").iterdir()}
        self.assertTrue({"0_FULL_strip.png", "1_F41_DUP_xt.png", "2_F41_RIFE_strip.png", "compare.mp4"} <= files)
        import cv2

        strip = cv2.imread(str(out / "render" / "A2_judder" / "0_FULL_strip.png"))
        xt = cv2.imread(str(out / "render" / "A2_judder" / "0_FULL_xt.png"))
        self.assertEqual(strip.shape[:2], (120, 208 * 4))
        self.assertEqual(xt.shape[:2], (9, 208))
        self.assertTrue((out / "render" / "index.html").exists())

    def test_pins_dvg(self):
        from UNIV_adaptor.scripts.data import dvg_wan21_study as dvg

        root = self.tmp / "dvg"
        clip = self.tmp / "v.mp4"
        video(clip, 2)
        jobs = []
        for i, prompt in ((3, "a bench"), (7, "Downtown"), (9, "a long prompt about many things")):
            for tag, subset in (("repro", "repro"), ("reproext", "repro_ext")):
                for arm in dvg.REPRO_ARMS:
                    stem = f"{tag}_v{i:04d}_s42__{arm}"
                    jobs.append({"stem": stem, "set": subset, "prompt_id": i, "case_id": arm,
                                 "prompt": prompt if subset == "repro" else prompt + ", extended"})
                    receipt = {"stem": stem, "video_path": str(clip), "video_sha256": sha256_file(clip)}
                    (root / "receipts").mkdir(parents=True, exist_ok=True)
                    (root / "receipts" / f"{stem}.json").write_text(json.dumps(receipt), encoding="utf-8")
        (root / "plan.json").write_text(json.dumps({"schema": dvg.PLAN_SCHEMA, "jobs": jobs}), encoding="utf-8")
        out = self.tmp / "pins.json"
        with contextlib.redirect_stdout(io.StringIO()):
            fs.main(["pins-dvg", "--dvg-study", str(root), "--prompts", "a bench", "downtown", "--out", str(out)])
        pins = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual([p["id"] for p in pins], ["A5_collapse_v0003", "A5_collapse_v0007"])
        self.assertEqual([r["label"] for r in pins[0]["rows"]], ["FULL", "DVG_D050", "DVG_D050_ext"])
        with self.assertRaises(ValueError):
            fs.main(["pins-dvg", "--dvg-study", str(root), "--prompts", "shower", "--out", str(out)])


if __name__ == "__main__":
    unittest.main()
