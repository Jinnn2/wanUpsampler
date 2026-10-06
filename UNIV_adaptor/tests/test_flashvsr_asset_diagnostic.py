"""CPU protocol/safety tests. These do not establish CUDA/VSR correctness."""
import argparse
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from UNIV_adaptor.scripts.data import flashvsr_asset_diagnostic as diag


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "study"
        self.source.mkdir()
        self.cfg = {"prompt_ids": ["p02"], "seeds": [42], "spatial_arms": ["S_B025"], "frames": 33}
        full_job = {"id": "pilot_p02_s42_FULL50", "prompt": "A red ball.", "seed": 42,
                    "arm": {"id": "FULL50"}}
        spatial_job = {"id": "pilot_p02_s42_S_B025", "prompt": "A red ball.", "seed": 42,
                       "arm": {"id": "S_B025", "endpoint": {"transition": "rgb_sr_vae", "frames": 81,
                                                                "width": 416, "height": 240}}}
        self.frozen = {"study_schema": "published_wan21_study_plan_v2", "jobs": [full_job, spatial_job],
                       "config": {"sampling": {"size": "832*480", "frame_num": 81}}}
        self.frozen["plan_sha256"] = diag.digest(self.frozen)
        diag.write_new(self.source / "plan.json", self.frozen)
        video, endpoint = self.source / "full.mp4", self.source / "clean.pt"
        video.write_bytes(b"test-fixture-not-an-actual-video")
        endpoint.write_bytes(b"test-fixture-not-an-actual-tensor")
        self.full = {"job": full_job, "plan_sha256": self.frozen["plan_sha256"],
                     "sampling_identity": {"fps": 16}, "video_path": str(video), "video_sha256": diag.file_hash(video),
                     "runtime": {"pipeline_seconds": 100}}
        self.spatial = {"job": spatial_job, "plan_sha256": self.frozen["plan_sha256"],
                        "endpoint": {"main_steps": 50, "main_terminal_sigma": 0,
                                     "main_latent_shape": [16, 21, 30, 52],
                                     "main_clean": {"shape": [16, 21, 30, 52], "sha256": "tensor-content"},
                                     "artifacts": [{"kind": "main_clean_sigma_zero", "path": str(endpoint),
                                                    "sha256": diag.file_hash(endpoint)}],
                                     "timing": {"main_seconds": 25}}}
        self.write_receipts()

    def write_receipts(self):
        for row in (self.full, self.spatial):
            # Fixture updates use a fresh temporary test file, not production outputs.
            target = self.source / "records" / (row["job"]["id"] + ".json")
            target.parent.mkdir(exist_ok=True)
            target.write_text(__import__("json").dumps(row), encoding="utf-8")

    def test_freeze_reuses_only_full_and_saved_unrefined_spatial_endpoint(self):
        original = {p: diag.file_hash(p) for p in self.source.rglob("*") if p.is_file()}
        _, pairs = diag.freeze_pairs(self.source, self.cfg)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["lr_width"], 416)
        self.assertEqual(pairs[0]["frames"], 33)
        self.assertEqual(pairs[0]["source_frames"], 81)
        self.assertEqual(len(diag.jobs({"pairs": pairs})), 2)
        self.assertEqual(original, {p: diag.file_hash(p) for p in self.source.rglob("*") if p.is_file()})

    def test_output_cannot_overlap_source_in_either_direction(self):
        for out in (self.source, self.source / "new", self.root):
            with self.assertRaises(ValueError):
                diag.outside(out, self.source)
        diag.outside(self.root / "diagnosis", self.source)

    def test_output_cannot_hide_overlap_behind_parent_segments(self):
        with self.assertRaises(ValueError):
            diag.outside(self.root / "other" / ".." / "study" / "new", self.source)

    def test_changed_source_mp4_rejected(self):
        Path(self.full["video_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            diag.freeze_pairs(self.source, self.cfg)

    def test_changed_endpoint_rejected(self):
        Path(self.spatial["endpoint"]["artifacts"][0]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            diag.freeze_pairs(self.source, self.cfg)

    def test_wrong_plan_receipt_rejected(self):
        self.full["plan_sha256"] = "other-plan"
        self.write_receipts()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            diag.freeze_pairs(self.source, self.cfg)

    def test_wrong_endpoint_geometry_rejected(self):
        self.spatial["endpoint"]["main_latent_shape"] = [16, 5, 60, 104]
        self.write_receipts()
        with self.assertRaisesRegex(ValueError, "shape"):
            diag.freeze_pairs(self.source, self.cfg)

    def test_temporal_arm_and_duplicate_selections_rejected(self):
        for key, value in (("spatial_arms", ["T_B025"]), ("prompt_ids", ["p02", "p02"]),
                           ("seeds", [42, 42]), ("spatial_arms", [])):
            cfg = copy.deepcopy(self.cfg)
            cfg[key] = value
            with self.assertRaises(ValueError):
                diag.freeze_pairs(self.source, cfg)

    def test_source_prefix_short_or_long_rejected(self):
        for frames in (0, 17, 82):
            with self.assertRaises(ValueError):
                diag.freeze_pairs(self.source, dict(self.cfg, frames=frames))

    def test_padding_preserves_all_content_and_tail_frames(self):
        for width, height in ((208, 120), (416, 240), (592, 336)):
            for frames in (25, 33, 81):
                with self.subTest(width=width, height=height, frames=frames):
                    spec = diag.padding_spec(width, height, frames)
                    self.assertGreaterEqual(spec["expected_raw_output_frames"], frames)
                    self.assertEqual(spec["flash_frames"] % 8, 1)
                    self.assertEqual(spec["flash_width"] % 128, 0)
                    self.assertEqual(spec["flash_height"] % 128, 0)
                    self.assertEqual(spec["content_width"], width*4)
                    self.assertEqual(spec["content_height"], height*4)
                    self.assertGreaterEqual(spec["flash_width"], spec["content_width"])
                    self.assertGreaterEqual(spec["flash_height"], spec["content_height"])
        self.assertEqual(diag.padding_spec(208, 120, 81)["flash_frames"], 89)
        self.assertEqual(diag.padding_spec(416, 240, 33)["flash_frames"], 41)

    def test_not_a_silent_x2_recipe(self):
        with self.assertRaises(ValueError):
            diag.padding_spec(416, 240, 33, scale=2)

    def test_immutable_outputs(self):
        path = self.root / "receipt.json"
        diag.write_new(path, {"value": 1})
        diag.write_new(path, {"value": 1})
        with self.assertRaises(ValueError):
            diag.write_new(path, {"value": 2})
        self.assertEqual(diag.read(path), {"value": 1})

    def test_reconstruction_exact_match_has_no_nonjson_infinity(self):
        frames = np.zeros((3, 4, 4, 3), dtype=np.uint8)
        result = diag.reconstruction(frames, frames)
        self.assertIsNone(result["psnr_db"])
        self.assertTrue(result["psnr_exact_match"])
        self.assertEqual(result["mae_0_1"], 0)
        diag.digest(result)

    def test_reconstruction_and_signal_stats_are_not_a_total_quality_score(self):
        black = np.zeros((3, 4, 4, 3), dtype=np.uint8)
        white = np.full_like(black, 255)
        result = diag.reconstruction(black, white)
        self.assertEqual(result["psnr_db"], 0)
        self.assertEqual(result["mae_0_1"], 1)
        self.assertIn("not native-LR", result["scope"])
        stats = diag.signal_stats(black)
        self.assertEqual(stats["mean_temporal_pixel_change"], 0)
        self.assertIn("NOT quality", stats["caution"])

    def test_reconstruction_rejects_geometry_mismatch(self):
        with self.assertRaises(ValueError):
            diag.reconstruction(np.zeros((3, 4, 4, 3)), np.zeros((2, 4, 4, 3)))

    def test_resize_preserves_dtype_frames_and_constant_content(self):
        frames = np.full((3, 4, 4, 3), 120, dtype=np.uint8)
        resized = diag.resize_frames(frames, (8, 12), Image.Resampling.BICUBIC)
        self.assertEqual(resized.shape, (3, 12, 8, 3))
        self.assertEqual(resized.dtype, np.uint8)
        self.assertTrue(np.all(resized == 120))

    def test_prepared_frame_content_hash_gate(self):
        path = self.root / "frames.npz"
        frames = np.zeros((3, 4, 4, 3), dtype=np.uint8)
        np.savez_compressed(path, frames=frames)
        identity = {"npz": diag.bound_file(path), "shape": list(frames.shape)}
        self.assertTrue(np.array_equal(diag.load_frames(identity), frames))
        identity["shape"][0] = 7
        with self.assertRaises(ValueError):
            diag.load_frames(identity)

    def test_partial_sr_outputs_blocked_before_loading_models(self):
        pair = {"id": "p02_s42_S_B025"}
        folder = self.root / "media" / pair["id"]
        folder.mkdir(parents=True)
        (folder / "HR_DOWN4_FLASH.mp4").write_bytes(b"partial")
        args = argparse.Namespace(out=self.root, ngpus=1, gpu=0)
        # Production imports torch lazily, but this safety gate is CPU-only.
        with patch.dict("sys.modules", {"torch": SimpleNamespace()}), \
                patch.object(diag, "load_plan", return_value={"pairs": [pair]}), \
                patch.object(diag, "init_flash") as initialize:
            with self.assertRaises(FileExistsError):
                diag.worker(args)
            initialize.assert_not_called()

    def test_report_export_and_filled_review_are_preserved(self):
        _, pairs = diag.freeze_pairs(self.source, self.cfg)
        pair = pairs[0]
        pair.update(width=8, height=8, lr_width=4, lr_height=4)
        out = self.root / "diagnosis"
        p = {"schema": "flashvsr_asset_diagnostic_plan_v1", "source_root": str(self.source),
             "pairs": [pair], "implementation": diag.implementation(), "config": {"color_fix": True}}
        p["plan_sha256"] = diag.digest(p)
        diag.write_new(out / "plan.json", p)
        assets = {}
        for name in ("FULL", "HR_DOWN4", "NATIVE_LR", "HR_DOWN4_BICUBIC", "NATIVE_LR_BICUBIC", "HR_DOWN4_FLASH", "NATIVE_LR_FLASH"):
            size = 2 if name == "HR_DOWN4" else 4 if name == "NATIVE_LR" else 8
            frames = np.full((33, size, size, 3), 120, dtype=np.uint8)
            folder = out / "media" / pair["id"]
            folder.mkdir(parents=True, exist_ok=True)
            npz, video = folder / (name + ".npz"), folder / (name + ".mp4")
            np.savez_compressed(npz, frames=frames)
            video.write_bytes(b"fixture-not-video")
            assets[name] = {"npz": diag.bound_file(npz), "video": diag.bound_file(video), "shape": list(frames.shape), "fps": 16}
        prepared = {"plan_sha256": p["plan_sha256"], "assets": assets, "native_decode_full_clip_seconds": .2}
        diag.write_new(out / "prepared" / (pair["id"] + ".json"), prepared)
        for kind in ("HR_DOWN4", "NATIVE_LR"):
            wh = (2, 2) if kind == "HR_DOWN4" else (4, 4)
            row = {"plan_sha256": p["plan_sha256"], "pair_id": pair["id"], "kind": kind,
                   "input_npz_sha256": assets[kind]["npz"]["sha256"], "asset": assets[kind + "_FLASH"],
                   "sr_preprocess_model_postprocess_seconds": .2, "excluded_warmup_seconds": .3,
                   "timing": {"sr_model_seconds": .1, "actual_sparse_attention_calls": 30,
                              "successful_color_correction_calls": 1, "padding": diag.padding_spec(*wh, 33)}}
            diag.write_new(out / "results" / f"{pair['id']}_{kind}.json", row)
        args = argparse.Namespace(out=out)
        diag.report(args)
        result = diag.read(out / "report.json")
        self.assertIsNone(result["rows"][0]["native_lr_vs_full_pixel_quality"])
        self.assertIsNone(result["rows"][0]["end_to_end_speedup"])
        self.assertTrue(result["rows"][0]["hr_down4_flash_reconstruction"]["psnr_exact_match"])
        review = out / "manual_review.csv"
        review.write_text("my-filled-review", encoding="utf-8")
        diag.report(args)
        self.assertEqual(review.read_text(encoding="utf-8"), "my-filled-review")
        diag.export(args)
        import tarfile
        with tarfile.open(out.parent / (out.name + "_analysis.tgz")) as bundle:
            names = bundle.getnames()
            self.assertTrue(any(n.endswith("review.html") for n in names))
            self.assertTrue(any(n.endswith("report.json") for n in names))
            self.assertTrue(any(n.endswith(".mp4") for n in names))
            self.assertFalse(any(n.endswith(".npz") for n in names))
        with self.assertRaises(FileExistsError):
            diag.export(args)

    def test_receipt_rejects_unexecuted_sparse_or_failed_color_path(self):
        p = {"plan_sha256": "frozen", "config": {"color_fix": True}}
        pair = {"id": "pair", "frames": 33, "width": 832, "height": 480, "fps": 16}
        prepared = {"assets": {"HR_DOWN4": {"npz": {"sha256": "input"}}}}
        row = {"plan_sha256": "frozen", "pair_id": "pair", "kind": "HR_DOWN4", "input_npz_sha256": "input",
               "asset": {"shape": [33, 480, 832, 3], "fps": 16},
               "timing": {"actual_sparse_attention_calls": 30, "successful_color_correction_calls": 1,
                          "sr_model_seconds": 1, "padding": diag.padding_spec(208, 120, 33)}}
        diag.validate_result(row, p, pair, "HR_DOWN4", prepared)
        for key in ("actual_sparse_attention_calls", "successful_color_correction_calls"):
            wrong = copy.deepcopy(row)
            wrong["timing"][key] = 0
            with self.assertRaises(ValueError):
                diag.validate_result(wrong, p, pair, "HR_DOWN4", prepared)
        wrong = copy.deepcopy(row)
        wrong["asset"]["shape"][0] = 29
        with self.assertRaises(ValueError):
            diag.validate_result(wrong, p, pair, "HR_DOWN4", prepared)


class CheckoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.relative = "examples/WanVSR/prompt_tensor/posi_prompt.pth"
        self.file = self.root / self.relative
        self.file.parent.mkdir(parents=True)
        self.file.write_bytes(b"fixture-author-context-raw-bytes")
        self.commit = "f" * 40
        self.blob = "a" * 40
        self.context = {"relative_path": self.relative, "git_blob": self.blob,
                        "bytes": self.file.stat().st_size, "sha256": diag.file_hash(self.file)}
        self.changes = ""
        self.index = f"100644 {self.blob} 0\t{self.relative}\0"
        self.current = self.commit
        self.top = str(self.root)

        def fake_git(command, **kwargs):
            operation = command[3:]
            if operation == ["rev-parse", "--show-toplevel"]:
                return self.top + "\n"
            if operation == ["rev-parse", "HEAD"]:
                return self.current + "\n"
            if operation == ["rev-parse", "HEAD:" + self.relative]:
                return self.blob + "\n"
            if operation[:1] == ["ls-files"]:
                return self.index
            if operation[:1] == ["status"]:
                return self.changes
            if operation[:1] == ["check-attr"]:
                return self.relative + ": filter: lfs\n"
            raise AssertionError(f"Unexpected git command: {command}")

        for mocked in (patch.object(diag, "FLASH_AUTHOR_COMMIT", self.commit),
                       patch.object(diag, "FLASH_CONTEXT", self.context),
                       patch.object(diag.subprocess, "check_output", side_effect=fake_git)):
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_clean_context_is_still_raw_hash_bound(self):
        result = diag.checkout(self.root, self.commit)
        self.assertEqual(result["author_prompt_tensor"]["sha256"], self.context["sha256"])
        self.assertEqual(result["verified_git_metadata_anomalies"], [])

    def test_only_byte_identical_unstaged_context_anomaly_is_accepted(self):
        self.changes = " M " + self.relative + "\0"
        before = self.file.read_bytes()
        result = diag.checkout(self.root, self.commit)
        anomalies = result["verified_git_metadata_anomalies"]
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["status"], " M")
        self.assertEqual(anomalies[0]["raw_sha256"], self.context["sha256"])
        self.assertIn("filter: lfs", anomalies[0]["git_attributes"])
        self.assertEqual(self.file.read_bytes(), before)

    def test_actual_context_modification_is_rejected_even_when_status_clean(self):
        self.file.write_bytes(b"modified")
        for status in ("", " M " + self.relative + "\0"):
            self.changes = status
            with self.assertRaisesRegex(ValueError, "RAW bytes differ"):
                diag.checkout(self.root, self.commit)

    def test_staged_context_blob_change_is_rejected(self):
        self.index = f"100644 {'b'*40} 0\t{self.relative}\0"
        self.changes = "M  " + self.relative + "\0"
        with self.assertRaisesRegex(ValueError, "HEAD/index"):
            diag.checkout(self.root, self.commit)

    def test_source_edit_and_untracked_code_cannot_hide_behind_context_anomaly(self):
        for additional in (" M diffsynth/pipelines/flashvsr_tiny.py\0", "?? custom.py\0"):
            self.changes = " M " + self.relative + "\0" + additional
            with self.assertRaisesRegex(ValueError, "dirty"):
                diag.checkout(self.root, self.commit)

    def test_staged_status_and_rename_never_allowed(self):
        for status in ("M  ", "MM ", "?? "):
            self.changes = status + self.relative + "\0"
            with self.assertRaisesRegex(ValueError, "dirty"):
                diag.checkout(self.root, self.commit)
        self.changes = "R  new.pth\0" + self.relative + "\0"
        with self.assertRaisesRegex(ValueError, "dirty"):
            diag.checkout(self.root, self.commit)

    def test_wrong_commit_or_parent_repository_rejected(self):
        self.current = "b"*40
        with self.assertRaisesRegex(ValueError, "commit mismatch"):
            diag.checkout(self.root, self.commit)
        self.current = self.commit
        self.top = str(self.root.parent)
        with self.assertRaisesRegex(ValueError, "independent source"):
            diag.checkout(self.root, self.commit)


if __name__ == "__main__":
    unittest.main()
