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
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from UNIV_adaptor.data_protocol import sha256_file  # noqa: E402
from UNIV_adaptor.scripts.data import synthetic_degradation as sd  # noqa: E402

CELLS = [(m, d) for m in ("high", "low") for d in ("high", "low")]


class DegradeTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.frames = rng.integers(0, 256, (12, 32, 48, 3), dtype=np.uint8)

    def test_shapes_and_identity(self):
        rng = np.random.default_rng(1)
        for _, kind, _, param in sd.variants():
            out = sd.degrade(self.frames, kind, param, rng)
            self.assertEqual((out.shape, out.dtype), (self.frames.shape, np.uint8), kind)
        self.assertIs(sd.degrade(self.frames, "identity", None, rng), self.frames)

    def test_severity_is_monotone(self):
        # Each degradation moves the clip further from the original as its level rises.
        for kind, params in sd.DEGRADATIONS.items():
            err = [np.abs(sd.degrade(self.frames, kind, p, np.random.default_rng(2)).astype(float) - self.frames).mean()
                   for p in params]
            self.assertTrue(0 < err[0] < err[1] < err[2], f"{kind}: {err}")

    def test_hold_repeats_frames(self):
        out = sd.degrade(self.frames, "hold", 3, None)
        self.assertTrue((out[0] == out[2]).all() and (out[3] == self.frames[3]).all())

    def test_spearman(self):
        self.assertAlmostEqual(sd.spearman([0, 1, 2, 3], [4, 3, 2, 1]), -1.0)
        self.assertAlmostEqual(sd.spearman([0, 0, 1, 1], [1, 1, 0, 0]), -1.0)
        self.assertTrue(np.isnan(sd.spearman([0, 1, 2], [1, 1, 1])))


class PipelineTest(unittest.TestCase):
    """plan -> make (encoder mocked) -> finalize -> report, on a fake fixed plan."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        fixed = self.tmp / "fixed_src"
        fixed.mkdir()
        videos, pid = [], 0
        for m, d in CELLS:
            for _ in range(sd.PER_CELL + 2):
                for seed in (42, 100):
                    clip = self.tmp / f"p{pid}_s{seed}.mp4"
                    clip.write_bytes(f"{pid}-{seed}".encode())
                    for case in (sd.FULL, "LR_BICUBIC"):
                        videos.append({"source": "src", "stem": f"p{pid}_s{seed}_{case}", "group_id": f"p{pid}_s{seed}",
                                       "prompt_id": pid, "family_id": pid // 4, "split": "train", "motion_level": m,
                                       "detail_level": d, "base_seed": seed, "case_id": case, "video_path": str(clip),
                                       "video_sha256": sha256_file(clip)})
                pid += 1
        videos.append(dict(videos[0], split="test", prompt_id=70, stem="p70"))
        (fixed / "plan.json").write_text(json.dumps({"videos": videos}), encoding="utf-8")
        self.out = self.tmp / "deg"
        with contextlib.redirect_stdout(io.StringIO()):
            sd.plan(Namespace(fixed_root=str(fixed), out=str(self.out)))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_pipeline(self):
        value = sd.load_plan(self.out)
        clips = value["clips"]
        self.assertEqual(len(clips), 4 * sd.PER_CELL)
        self.assertTrue(all(c["base_seed"] == 42 and c["split"] != "test" for c in clips))
        self.assertEqual(len({c["prompt_id"] for c in clips}), len(clips))

        frames = np.zeros((4, 8, 8, 3), np.uint8)
        with mock.patch.object(sd, "read_video", return_value=(frames, 16.0)), \
                mock.patch.object(sd, "encode", side_effect=lambda f, fps, dst, ff: Path(dst).write_bytes(f.tobytes()[:64] + dst.name.encode())), \
                contextlib.redirect_stdout(io.StringIO()):
            for i in range(3):
                sd.make(Namespace(out=str(self.out), ffmpeg="ffmpeg", shard_index=i, shard_count=3))
            sd.finalize(Namespace(out=str(self.out)))
        fixed = json.loads((self.out / "fixed" / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(len(fixed["videos"]), len(clips) * (len(sd.variants()) + 1))
        self.assertEqual({v["source"] for v in fixed["videos"]}, {sd.SOURCE})
        self.assertEqual(sum(v["case_id"] == sd.FULL for v in fixed["videos"]), len(clips))

        # Fake layer outputs: quality_kept falls with blur level and ignores noise; VBench sees nothing.
        (self.out / "quality").mkdir()
        with (self.out / "quality" / "quality_by_video.csv").open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=["stem", "quality_kept"])
            w.writeheader()
            for v in fixed["videos"]:
                if v["case_id"] == sd.FULL:
                    continue
                level = int(v["case_id"].rsplit("_", 1)[1])
                w.writerow({"stem": v["stem"], "quality_kept": 1 - 0.3 * level if "BLUR" in v["case_id"] else 1.0})
        with (self.out / "vbench_by_video.csv").open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=["stem", "vbench5"])
            w.writeheader()
            w.writerows({"stem": v["stem"], "vbench5": 0.8} for v in fixed["videos"])
        with contextlib.redirect_stdout(io.StringIO()):
            sd.report(Namespace(out=str(self.out)))
        text = (self.out / "degradation_report.md").read_text(encoding="utf-8")
        row = next(line for line in text.splitlines() if line.startswith("| quality_kept | -1.00"))
        self.assertIn("| -1.00 |", row)  # blur column
        detect = [line for line in text.splitlines() if line.startswith("| quality_kept | 1.00")]
        self.assertEqual(len(detect), 1)  # level 3 blur below identity for every clip
        rows, _ = sd.collect(self.out)
        self.assertEqual(rows[sd.stem(clips[0], "DEG_BLUR_2")]["vbench5"], 0.0)  # delta vs FULL

    def test_changed_degradations_refuse_the_plan(self):
        plan = json.loads((self.out / "plan.json").read_text(encoding="utf-8"))
        plan["degradations"]["blur"] = [1, 2]
        (self.out / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaises(ValueError):
            sd.load_plan(self.out)


if __name__ == "__main__":
    unittest.main()
