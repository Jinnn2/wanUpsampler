"""CPU-only full-duration protocol, provenance and four-GPU orchestration tests."""
import copy
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from UNIV_adaptor.scripts.data import complete_flashvsr_wan21 as complete

diag = complete.diag


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.out = self.root / "study", self.root / "complete"
        self.cfg = diag.read(complete.CONFIG)
        self.args = SimpleNamespace(out=self.out, study_root=self.source, config=complete.CONFIG,
                                    flash_root=self.root / "flash", kernel_root=self.root / "kernel",
                                    weights=self.root / "weights", gpu_ids="4,5,6,7", slot=0,
                                    native_python="native-python", sr_python="sr-python")
        self.make_source()

    def file(self, path, data=b"fixture-not-real-media"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def make_source(self):
        jobs, rows = [], []
        for pid in self.cfg["prompt_ids"]:
            for seed in self.cfg["seeds"]:
                for arm in (*complete.BASE_ARMS, "S_B025"):
                    key = f"pilot_{pid}_s{seed}_{arm}"
                    job = {"id": key, "seed": seed, "prompt": f"Prompt {pid}", "arm": {"id": arm}}
                    jobs.append(job)
                    video = self.file(self.source / "videos" / (key + ".mp4"), key.encode())
                    row = {"job": job, "sampling_identity": {"fps": 16}, "video_path": str(video),
                           "video_sha256": diag.file_hash(video), "runtime": {"pipeline_seconds": 100}}
                    if arm == "S_B025":
                        job["arm"]["endpoint"] = {"transition": "rgb_sr_vae", "frames": 81, "width": 416, "height": 240}
                        endpoint = self.file(self.source / "endpoints" / (key + ".pt"))
                        row["endpoint"] = {"main_steps": 50, "main_terminal_sigma": 0,
                                           "main_latent_shape": [16, 21, 30, 52],
                                           "main_clean": {"shape": [16, 21, 30, 52], "sha256": "clean-content"},
                                           "artifacts": [{"kind": "main_clean_sigma_zero", "path": str(endpoint),
                                                          "sha256": diag.file_hash(endpoint)}],
                                           "timing": {"main_seconds": 25}}
                    rows.append(row)
        self.file(self.root / "models/Wan2.1_VAE.pth")
        self.file(self.root / "UNIV_adaptor/external/wan21/wan/modules/vae.py", b"# fixture")
        frozen = {"study_schema": "published_wan21_study_plan_v2", "jobs": jobs,
                  "model_root": str(self.root / "models"),
                  "config": {"sampling": {"size": "832*480", "frame_num": 81}}}
        frozen["plan_sha256"] = diag.digest(frozen)
        diag.write_new(self.source / "plan.json", frozen)
        for row in rows:
            row["plan_sha256"] = frozen["plan_sha256"]
            diag.write_new(self.source / "records" / (row["job"]["id"] + ".json"), row)
        files = []
        for name in self.cfg["weights_files"]:
            path = self.file(self.args.weights / name)
            entry = diag.bound_file(path)
            entry["mtime_ns"] = path.stat().st_mtime_ns
            files.append(entry)
        diag.write_new(self.args.weights / "diagnostic_weight_binding.json",
                       {"repository": self.cfg["weights_repository"], "revision": "pinned-revision", "files": files})

    def freeze(self):
        with patch.object(complete, "ROOT", self.root), patch.object(diag, "checkout", side_effect=lambda p, c: {"path": str(p.resolve()), "commit": c}):
            complete.plan(self.args)
        return complete.load_plan(self.out)

    def asset(self, stem, shape):
        folder = self.out / "media" / "p00_s42_S_B025"
        video = self.file(folder / (stem + ".mp4"))
        npz = self.file(folder / (stem + ".npz"))
        return {"video": diag.bound_file(video), "npz": diag.bound_file(npz), "shape": shape, "fps": 16}

    def preparation(self, p, pair):
        return {"plan_sha256": p["plan_sha256"], "pair_id": pair["id"],
                "source_endpoint_sha256": pair["endpoint"]["sha256"],
                "assets": {"NATIVE_LR": self.asset("NATIVE_LR", [81, 240, 416, 3])},
                "native_decode_full_clip_seconds": 1}

    def test_plan_24_native_only_full81_preserves_originals(self):
        before = {str(p): diag.file_hash(p) for p in self.source.rglob("*") if p.is_file()}
        p = self.freeze()
        self.assertEqual(len(p["pairs"]), 24)
        self.assertTrue(all(x["frames"] == x["source_frames"] == 81 for x in p["pairs"]))
        self.assertTrue(all(set(x["comparison"]) == set(complete.BASE_ARMS) for x in p["pairs"]))
        self.assertEqual(before, {str(p): diag.file_hash(p) for p in self.source.rglob("*") if p.is_file()})

    def test_plan_resume(self):
        original = self.freeze()
        complete.plan(self.args)
        self.assertEqual(original, complete.load_plan(self.out))

    def test_resume_wrong_root_rejected(self):
        self.freeze()
        self.args.study_root = self.root / "other-source"
        with self.assertRaisesRegex(ValueError, "different inputs"):
            complete.plan(self.args)

    def test_code_changes_rejected(self):
        self.freeze()
        with patch.object(complete, "implementation", return_value={"changed": True}):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                complete.load_plan(self.out)

    def test_prefix_and_parameter_changes_rejected(self):
        for key, value in (("frames", 33), ("spatial_arms", ["T_B025"]), ("color_fix", False), ("sparse_ratio", 1)):
            with self.assertRaises(ValueError):
                complete.validate_config(dict(self.cfg, **{key: value}))

    def test_gpu_selection_exact_four_and_unique(self):
        self.assertEqual(complete.gpu_ids("4,5,6,7"), ["4", "5", "6", "7"])
        for value in ("0", "0,1", "0,1,2,3,4", "0,0,1,2", "0,00,1,2", "0,1,2,-1", "0,1,2,x"):
            with self.assertRaises(ValueError):
                complete.gpu_ids(value)

    def test_four_gpu_shards_cover_24_once(self):
        p = self.freeze()
        shards = [complete.pending_pairs(self.out, p, "prepare", slot) for slot in range(4)]
        self.assertEqual([len(x) for x in shards], [6, 6, 6, 6])
        self.assertEqual(len({x["id"] for s in shards for x in s}), 24)

    def test_worker_gpu_mapping_and_separate_interpreters(self):
        cmd, env = complete.worker_command(self.args, "prepare", 2, "6")
        self.assertEqual(cmd[0], "native-python")
        self.assertIn("prepare-worker", cmd)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "6")
        self.assertEqual(cmd[-1], "2")
        cmd, env = complete.worker_command(self.args, "sr", 2, "6")
        self.assertEqual(cmd[0], "sr-python")
        self.assertIn("sr-worker", cmd)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "6")

    def test_padding_keeps_full_tail_and_canvas(self):
        spec = diag.padding_spec(416, 240, 81)
        self.assertEqual(spec["flash_frames"], 89)
        self.assertEqual(spec["expected_raw_output_frames"], 85)
        self.assertEqual((spec["content_width"], spec["content_height"]), (1664, 960))
        self.assertEqual(spec["tail_pad_frames"], 8)

    def test_prepared_prefix_endpoint_mismatch_rejected(self):
        p = self.freeze()
        pair = p["pairs"][0]
        row = self.preparation(p, pair)
        complete.validate_prepared(row, p, pair)
        for key in ("shape", "endpoint"):
            bad = copy.deepcopy(row)
            if key == "shape":
                bad["assets"]["NATIVE_LR"]["shape"][0] = 33
            else:
                bad["source_endpoint_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "geometry mismatch"):
                complete.validate_prepared(bad, p, pair)

    def test_orphan_artifacts_not_overwritten(self):
        p = self.freeze()
        self.file(self.out / "media" / p["pairs"][0]["id"] / "NATIVE_LR.npz")
        with self.assertRaisesRegex(FileExistsError, "Unreceipted"):
            complete.pending_pairs(self.out, p, "prepare", 0)

    def test_verified_prepared_skipped(self):
        p = self.freeze()
        pair = p["pairs"][0]
        diag.write_new(complete.prepared_path(self.out, pair), self.preparation(p, pair))
        self.assertEqual(len(complete.pending_pairs(self.out, p, "prepare", 0)), 5)

    def test_prepared_media_modified_rejected(self):
        p = self.freeze()
        pair = p["pairs"][0]
        row = self.preparation(p, pair)
        self.file(Path(row["assets"]["NATIVE_LR"]["video"]["path"]), b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            complete.validate_prepared(row, p, pair)

    def test_changed_comparison_video_rejected_during_plan(self):
        self.file(self.source / "videos/pilot_p00_s42_TEA008.mp4", b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.freeze()

    def test_finalization_requires_all24(self):
        self.freeze()
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            complete.finalize(self.args)
        self.assertFalse((self.out / "dataset_manifest.json").exists())

    def test_full_sr_result_has_no_prefix_or_dense_fallback(self):
        p = self.freeze()
        pair = p["pairs"][0]
        prepared = self.preparation(p, pair)
        diag.write_new(complete.prepared_path(self.out, pair), prepared)
        row = {"plan_sha256": p["plan_sha256"], "pair_id": pair["id"], "kind": "NATIVE_LR",
               "asset": self.asset("NATIVE_LR_FLASH", [81, 480, 832, 3]),
               "input_npz_sha256": prepared["assets"]["NATIVE_LR"]["npz"]["sha256"],
               "timing": {"padding": diag.padding_spec(416, 240, 81), "sr_model_seconds": 1,
                          "actual_sparse_attention_calls": 10, "successful_color_correction_calls": 1}}
        diag.write_new(complete.result_path(self.out, pair), row)
        self.assertEqual(complete.completed_result(self.out, p, pair), row)
        for change in ("prefix", "dense"):
            bad = copy.deepcopy(row)
            if change == "prefix":
                bad["asset"]["shape"][0] = 33
            else:
                bad["timing"]["actual_sparse_attention_calls"] = 0
            with self.assertRaises(ValueError):
                diag.validate_result(bad, p, pair, "NATIVE_LR", prepared)

    def make_results(self):
        p = self.freeze()
        for pair in p["pairs"]:
            folder = self.out / "media" / pair["id"]
            prep_asset = {"video": diag.bound_file(self.file(folder / "NATIVE_LR.mp4")),
                          "npz": diag.bound_file(self.file(folder / "NATIVE_LR.npz")),
                          "shape": [81, 240, 416, 3], "fps": 16}
            prepared = {"plan_sha256": p["plan_sha256"], "pair_id": pair["id"],
                        "source_endpoint_sha256": pair["endpoint"]["sha256"],
                        "assets": {"NATIVE_LR": prep_asset}, "native_decode_full_clip_seconds": 1}
            diag.write_new(complete.prepared_path(self.out, pair), prepared)
            asset = {"video": diag.bound_file(self.file(folder / "NATIVE_LR_FLASH.mp4")),
                     "npz": diag.bound_file(self.file(folder / "NATIVE_LR_FLASH.npz")),
                     "shape": [81, 480, 832, 3], "fps": 16}
            result = {"plan_sha256": p["plan_sha256"], "pair_id": pair["id"], "kind": "NATIVE_LR",
                      "asset": asset, "input_npz_sha256": prep_asset["npz"]["sha256"],
                      "timing": {"padding": diag.padding_spec(416, 240, 81), "sr_model_seconds": 1,
                                 "actual_sparse_attention_calls": 10, "successful_color_correction_calls": 1},
                      "sr_preprocess_model_postprocess_seconds": 2}
            diag.write_new(complete.result_path(self.out, pair), result)
        return p

    def test_finalize_120_matched_videos_no_fake_speedup(self):
        self.make_results()
        complete.finalize(self.args)
        manifest = diag.read(self.out / "dataset_manifest.json")
        self.assertEqual(manifest["matched_comparison_videos"], 120)
        self.assertEqual(len(manifest["videos"]), 24)
        self.assertEqual(manifest["matched_groups"], 24)
        self.assertTrue(all(x["end_to_end_speedup"] is None for x in manifest["videos"]))
        complete.finalize(self.args)

    def test_all_complete_no_gpu_relaunch(self):
        self.make_results()
        with patch.object(complete.subprocess, "Popen") as popen:
            complete.launch(self.args, "prepare")
            complete.launch(self.args, "run")
            popen.assert_not_called()

    def test_export_only24_final_videos_no_lossless_arrays_and_resume_verified(self):
        self.make_results()
        complete.export(self.args)
        archive = self.out.parent / (self.out.name + "_videos.tgz")
        with tarfile.open(archive) as bundle:
            names = bundle.getnames()
        self.assertEqual(len([name for name in names if name.endswith(".mp4")]), 24)
        self.assertFalse(any(name.endswith((".npz", ".pt")) for name in names))
        self.assertTrue(all("NATIVE_LR_FLASH" in name for name in names if name.endswith(".mp4")))
        original = diag.file_hash(archive)
        complete.export(self.args)
        self.assertEqual(original, diag.file_hash(archive))
        archive.write_bytes(b"corrupt archive")
        with self.assertRaisesRegex(ValueError, "changed"):
            complete.export(self.args)

    def test_partial_export_rejected_not_silently_reused(self):
        self.make_results()
        self.file(self.out.parent / (self.out.name + "_videos.tgz"), b"partial")
        with self.assertRaisesRegex(FileExistsError, "incomplete export"):
            complete.export(self.args)


if __name__ == "__main__":
    unittest.main()
