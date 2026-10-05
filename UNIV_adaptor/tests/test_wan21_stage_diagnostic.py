"""CPU safety/metadata tests, not a VAE/CUDA numerical validation."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from UNIV_adaptor.scripts.data import diagnose_wan21_endpoint_stages as probe


class StageDiagnosticTests(unittest.TestCase):
    def test_png_indices_include_non_anchor_frames_without_out_of_bounds(self):
        self.assertEqual(probe.png_indices(17), [0, 4, 8, 16])
        self.assertEqual(probe.png_indices(81), [0, 4, 8, 16, 24, 40, 56, 80])
        self.assertEqual(probe.png_indices(1), [0])

    def test_job_paths_rejected_before_loading_any_assets(self):
        for job in ("../pilot_p00", "records/pilot_p00", "cal_mug_s12345"):
            with self.subTest(job=job), patch.object(probe.study, "load_plan") as load:
                with self.assertRaises(ValueError):
                    probe.load_inputs(Path("missing"), job)
                load.assert_not_called()

    def test_record_must_belong_to_frozen_plan(self):
        job = {"id": "pilot_p00_s3407_T_B025"}
        row = {"job": job, "plan_sha256": "different"}
        with patch.object(probe.study, "load_plan", return_value={"jobs": [job], "plan_sha256": "frozen"}), \
                patch.object(probe.study.base, "read", return_value=row):
            with self.assertRaisesRegex(ValueError, "frozen plan"):
                probe.load_inputs(Path("missing"), job["id"])

    def test_changed_original_video_rejected(self):
        job = {"id": "pilot_p00_s3407_T_B025", "arm": {"endpoint": {}}}
        job["arm"]["endpoint"] = {"transition": "dvg_latent_anchor"}
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "original.mp4"
            video.write_bytes(b"fixture-not-an-actual-video")
            row = {"job": job, "plan_sha256": "frozen", "video_path": str(video), "video_sha256": "wrong"}
            with patch.object(probe.study, "load_plan", return_value={"jobs": [job], "plan_sha256": "frozen"}), \
                    patch.object(probe.study.base, "read", return_value=row), \
                    patch.object(probe.study, "validate_endpoint_receipt"):
                with self.assertRaisesRegex(ValueError, "changed"):
                    probe.load_inputs(Path(directory), job["id"])

    def test_existing_output_rejected_before_importing_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "already_exists"
            out.mkdir()
            row = {"job": {}, "endpoint": {"hr": None}, "video_path": "final.mp4",
                   "video_sha256": "hash", "environment": {}}
            args = argparse.Namespace(study_root=root / "study", job="pilot_p00_s3407_T_B025",
                                      out=out, inspect_only=False)
            with patch.object(probe, "load_inputs", return_value=({"plan_sha256": "frozen"}, row)), \
                    patch.object(probe, "file_hash", return_value="hash"):
                with self.assertRaises(FileExistsError):
                    probe.run(args)
            self.assertEqual(list(out.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
