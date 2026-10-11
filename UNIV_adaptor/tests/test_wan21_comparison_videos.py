"""Presentation/safety tests, plus optional real-FFmpeg 81-frame integration."""
import copy
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from UNIV_adaptor.scripts.data import make_wan21_comparison_videos as montage

diag = montage.diag


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "scores"
        self.out = self.root / "comparisons"
        self.clips, self.scores = [], {}
        for pid in range(12):
            for seed in (42, 3407):
                gid = f"p{pid:02d}_s{seed}"
                for i, arm in enumerate(montage.METHODS):
                    clip_id = f"pilot_{gid}_{arm}"
                    video = self.root / "originals" / (clip_id + ".mp4")
                    video.parent.mkdir(exist_ok=True)
                    video.write_bytes(clip_id.encode())
                    self.clips.append({"clip_id": clip_id, "group_id": gid, "method": arm,
                                       "prompt_id": f"p{pid:02d}", "seed": seed,
                                       "prompt": "A runner's blue tracksuit: crisp details, motion without cuts.",
                                       "frames": 81, "width": 832, "height": 480, "fps": 16,
                                       "video": diag.bound_file(video)})
                    self.scores[clip_id] = {d: .95 - i * .005 for d in montage.scoring.DIMENSIONS}
        self.score_plan = {"plan_sha256": "fixture-scoring-plan", "clips": self.clips,
                           "source_root": str(self.root / "native"), "study_root": str(self.root / "originals")}
        self.saved = {"scores_sha256": "fixture-score-seal", "scores": self.scores}
        diag.write_new(self.source / "plan.json", self.score_plan)
        diag.write_new(self.source / "scores.json", self.saved)
        executable = self.root / "ffmpeg.exe"
        executable.write_bytes(b"mock-executable")
        self.args = SimpleNamespace(score_root=self.source, out=self.out, panel_width=832, crf=12,
                                    group_ids=None, font=None, ffmpeg=str(executable), ffprobe=None, jobs=2, threads=2)

    def freeze(self, real_ffmpeg=None):
        try:
            montage.find_font()
        except FileNotFoundError:
            self.skipTest("TrueType test font unavailable")
        with patch.object(montage.scoring, "load", return_value=self.score_plan), patch.object(montage.scoring, "load_scores", return_value=self.saved):
            if real_ffmpeg:
                self.args.ffmpeg = real_ffmpeg
                montage.plan(self.args)
            else:
                with patch.object(montage.shutil, "which", return_value=self.args.ffmpeg), patch.object(montage.subprocess, "check_output", return_value="ffmpeg fixture\n"):
                    montage.plan(self.args)
        return montage.load(self.args)

    def test_24_groups_fixed_order_and_no_mean_of_seven(self):
        groups = montage.group_clips(self.clips, self.scores)
        self.assertEqual(len(groups), 24)
        self.assertTrue(all(tuple(r["method"] for r in g["clips"]) == montage.METHODS for g in groups))
        values = {d: .8 for d in montage.scoring.DIMENSIONS}
        values["dynamic_degree"], values["overall_consistency"] = 0, -.1
        self.assertAlmostEqual(montage.raw_quality5(values), .8)

    def test_duplicate_mixed_seed_and_prefix_rejected(self):
        with self.assertRaises(ValueError):
            montage.group_clips(self.clips + [self.clips[0]], self.scores)
        for key, value in (("seed", 99), ("prompt", "different prompt"), ("frames", 33)):
            bad = copy.deepcopy(self.clips)
            bad[1][key] = value
            with self.assertRaises(ValueError):
                montage.group_clips(bad, self.scores)

    def test_native_layout_no_crop_resize_overlay_on_content(self):
        p = self.freeze()
        self.assertEqual((p["layout"]["width"], p["layout"]["height"]), (4160, 968))
        command = montage.render_command(p, p["groups"][0], self.out / "header.png", self.out / "video.mp4", 2)
        filters = command[command.index("-filter_complex") + 1]
        self.assertNotIn("scale=", filters)
        self.assertNotIn("crop", filters)
        self.assertNotIn("fps=", filters)
        self.assertIn("hstack=inputs=5", filters)
        self.assertEqual(command[command.index("-frames:v") + 1], "81")
        self.assertEqual(command[command.index("-crf") + 1], "12")

    def test_smaller_variant_is_explicit_and_even(self):
        for width in (416, 640, 832):
            spec = montage.layout(width)
            self.assertTrue(all(spec[k] % 2 == 0 for k in ("width", "height", "video_height", "video_y")))
        for width in (300, 833, 641):
            with self.assertRaises(ValueError):
                montage.layout(width)

    def test_header_video_slot_is_fully_transparent(self):
        p = self.freeze()
        target = self.out / "headers/test.png"
        montage.make_header(p, p["groups"][0], target)
        with Image.open(target) as image:
            self.assertEqual(image.size, (4160, 968))
            alpha = image.getchannel("A")
            spec = p["layout"]
            crop = alpha.crop((0, spec["video_y"], spec["width"], spec["video_y"] + spec["video_height"]))
            self.assertEqual(crop.getextrema(), (0, 0))
            self.assertEqual(alpha.getpixel((0, 0)), 255)
        with self.assertRaises(FileExistsError):
            montage.make_header(p, p["groups"][0], target)

    def test_source_bytes_preserved_plan_idempotent(self):
        before = {str(p): diag.file_hash(p) for p in self.root.rglob("*.mp4")}
        original = self.freeze()
        montage.plan(self.args)
        self.assertEqual(original, montage.load(self.args))
        self.assertEqual(before, {str(p): diag.file_hash(p) for p in self.root.rglob("*.mp4")})

    def test_code_and_score_changes_rejected(self):
        self.freeze()
        with patch.object(montage, "implementation", return_value={"changed": True}):
            with self.assertRaises(ValueError):
                montage.load(self.args)
        (self.source / "scores.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            montage.load(self.args)

    def test_orphan_output_never_overwritten(self):
        p = self.freeze()
        group = p["groups"][0]
        target = self.out / "headers" / (group["group_id"] + ".png")
        target.parent.mkdir(parents=True)
        target.write_bytes(b"orphan")
        with self.assertRaisesRegex(FileExistsError, "Unreceipted"):
            montage.render_one(self.args, p, group)
        self.assertEqual(target.read_bytes(), b"orphan")

    def test_subset_and_changed_layout_require_separate_plan(self):
        self.args.group_ids = ["p02_s3407"]
        p = self.freeze()
        self.assertEqual(len(p["groups"]), 1)
        self.args.panel_width = 640
        with self.assertRaisesRegex(ValueError, "different settings"):
            montage.plan(self.args)

    def test_native_review_export_index_resumable_without_video_sources_in_package(self):
        self.args.group_ids = ["p00_s42"]
        p = self.freeze()
        group = p["groups"][0]
        video = self.out / "videos/p00_s42.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"fixture-derived-video")
        header = self.out / "headers/p00_s42.png"
        montage.make_header(p, group, header)
        diag.write_new(self.out / "receipts/p00_s42.json", {"plan_sha256": p["plan_sha256"], "group_id": "p00_s42",
                       "clip_ids": [r["clip_id"] for r in group["clips"]], "video": diag.bound_file(video),
                       "header": diag.bound_file(header), "geometry": {"width": 4160, "height": 968, "frames": 81, "fps": 16}})
        with patch.object(montage.subprocess, "run") as run:
            montage.render(self.args)
            run.assert_not_called()
        montage.export(self.args)
        montage.export(self.args)
        archive = self.out.parent / (self.out.name + "_review.tgz")
        with tarfile.open(archive) as bundle:
            names = bundle.getnames()
        self.assertEqual(len([n for n in names if n.endswith(".mp4")]), 1)
        self.assertTrue(any(n.endswith("index.html") for n in names))
        page = (self.out / "index.html").read_text(encoding="utf-8")
        self.assertIn("classList.toggle('native')", page)
        self.assertNotIn("http://", page)

    def test_real_ffmpeg_full81_native_pixels_and_frame_order(self):
        executable = os.environ.get("WAN_COMPARISON_TEST_FFMPEG") or shutil.which("ffmpeg")
        if not executable:
            self.skipTest("Set WAN_COMPARISON_TEST_FFMPEG to enable real FFmpeg integration")
        try:
            import imageio_ffmpeg
        except ImportError:
            self.skipTest("imageio_ffmpeg required for probe/decode integration without ffprobe")
        self.args.group_ids = ["p00_s42"]
        colors = ["red", "green", "blue", "yellow", "magenta"]
        for i, clip in enumerate(self.clips[:5]):
            video = Path(clip["video"]["path"])
            # A last-frame-only marker catches truncation/duplication of frame 80.
            subprocess.run([executable, "-v", "error", "-y", "-f", "lavfi", "-i",
                            f"color=c={colors[i]}:size=832x480:rate=16,drawbox=x=20:y=20:w=100:h=100:color=white:t=fill:enable='eq(n,80)'",
                            "-frames:v", "81", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(video)], check=True)
            clip["video"] = diag.bound_file(video)
        p = self.freeze(real_ffmpeg=executable)
        group = p["groups"][0]
        montage.render_one(self.args, p, group)
        receipt = montage.validate_receipt(self.args, p, group)
        self.assertEqual(receipt["geometry"]["frames"], 81)
        reader = imageio_ffmpeg.read_frames(receipt["video"]["path"], pix_fmt="rgb24")
        metadata = next(reader)
        first, rest = next(reader), list(reader)
        self.assertEqual(len(rest), 80)
        width = metadata["size"][0]
        y = p["layout"]["video_y"] + 240
        samples = []
        for i in range(5):
            offset = (y * width + i * 832 + 416) * 3
            samples.append(tuple(first[offset:offset + 3]))
        self.assertGreater(samples[0][0], 220)
        self.assertGreater(samples[2][2], 220)
        self.assertTrue(samples[4][0] > 220 and samples[4][2] > 220)
        for i in range(5):
            offset = ((p["layout"]["video_y"] + 70) * width + i * 832 + 70) * 3
            self.assertTrue(all(c > 220 for c in rest[-1][offset:offset + 3]))
            self.assertFalse(all(c > 220 for c in first[offset:offset + 3]))
        montage.index(self.args)


if __name__ == "__main__":
    unittest.main()
