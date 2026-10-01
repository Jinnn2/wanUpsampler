"""CPU protocol tests. Mock workers/scoring do not validate CUDA kernels."""
import argparse
import copy
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from UNIV_adaptor.scripts.data import published_wan21_pilot as pilot
from UNIV_adaptor.scripts.data import published_wan21_worker as worker


class PilotTests(unittest.TestCase):
    def setUp(self):
        quiet = patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.cfg = pilot.read(pilot.DEFAULT_CONFIG)
        self.plan = pilot.build_plan(self.cfg, "/models/Wan2.1-T2V-1.3B", 8, {}, {}, [])
        pilot.write(self.out / "plan.json", self.plan)
        self.args = SimpleNamespace(out=self.out, allow_partial=False, ngpus=8,
            vbench_root=self.out / "vbench", vbench_python="python", expected_vbench_commit="abc")

    def receipt(self, job, delta=0.0, noise="same"):
        video = self.out / "test_videos" / (job["id"] + ".mp4")
        video.parent.mkdir(exist_ok=True)
        video.write_bytes(job["id"].encode())
        sample = self.out / "samples" / (job["id"] + ".npz")
        sample.parent.mkdir(exist_ok=True)
        np.savez_compressed(sample, frames=np.zeros((3, 2, 2, 2), dtype=np.float32) + delta)
        row = {"plan_sha256": self.plan["plan_sha256"], "job": job,
            "video_path": str(video), "video_sha256": pilot.file_hash(video),
            "sample_path": str(sample), "sample_sha256": pilot.file_hash(sample),
            "noise": {"sha256": noise, "shape": [16, 21, 60, 104]},
            "sampling_identity": {"negative_prompt": "same", "dtype": "same"},
            "runtime": {"pipeline_seconds": 10.0}, "environment": {"gpu_name": "A800", "wan_module": job["arm"]["source"]}}
        pilot.write(self.out / "records" / (job["id"] + ".json"), row)
        return row

    def calibration_records(self, bad_arm=None, bad_noise=False):
        for job in self.plan["jobs"]:
            if job["phase"] == "calibration":
                self.receipt(job, delta=0.1 if job["arm"]["id"] == bad_arm else 0,
                             noise="different" if bad_noise and job["arm"]["id"] == "SCALING10" else "same")

    def test_counts_balance_and_deterministic_plan(self):
        self.assertEqual(len(self.plan["jobs"]), 136)
        self.assertEqual(sum(j["phase"] == "pilot" for j in self.plan["jobs"]), 120)
        self.assertEqual(sum(j["phase"] == "calibration" for j in self.plan["jobs"]), 16)
        counts = [sum(j["phase"] == "pilot" and j["gpu"] == g for j in self.plan["jobs"]) for g in range(8)]
        self.assertEqual(counts, [15] * 8)
        self.assertEqual(self.plan, pilot.build_plan(self.cfg, "/models/Wan2.1-T2V-1.3B", 8, {}, {}, []))

    def test_config_rejects_overlap_and_common_overrides(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["calibration"]["prompts"][0]["prompt"] = cfg["prompts"][0]["prompt"]
        with self.assertRaises(ValueError):
            pilot.validate_config(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["arms"][2]["flags"] += ["--base_seed", "2"]
        with self.assertRaises(ValueError):
            pilot.validate_config(cfg)

    def test_frozen_file_cannot_change(self):
        pilot.immutable(self.out / "frozen.json", {"a": 1})
        pilot.immutable(self.out / "frozen.json", {"a": 1})
        with self.assertRaises(ValueError):
            pilot.immutable(self.out / "frozen.json", {"a": 2})

    def test_plan_hash_tamper(self):
        plan = copy.deepcopy(self.plan)
        plan["jobs"][0]["seed"] += 1
        pilot.write(self.out / "plan.json", plan)
        with self.assertRaises(ValueError):
            pilot.load_plan(self.out, verify_implementation=False)

    def test_foreign_output_root_rejected_even_if_pilot_plan_exists(self):
        pilot.write(self.out / "sparse_dataset_manifest.json", {"schema": "old_phase3"})
        with self.assertRaisesRegex(ValueError, "another experiment"):
            pilot.validate_output_root(self.out)
        with self.assertRaises(ValueError):
            pilot.load_plan(self.out, verify_implementation=False)

    def test_log_excerpt_has_actual_exception(self):
        path = self.out / "worker.log"
        path.write_text("noise\n" * 80 + "ModuleNotFoundError: No module named 'gilbert'\n", encoding="utf-8")
        excerpt = pilot.log_excerpt(path, lines=5)
        self.assertEqual(len(excerpt.splitlines()), 5)
        self.assertIn("No module named 'gilbert'", excerpt)

    def test_old_frozen_plan_rejected_after_worker_update(self):
        with patch.object(pilot, "file_hash", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "implementation changed"):
                pilot.load_plan(self.out)

    def test_receipt_video_tamper_and_resumption(self):
        job = self.plan["jobs"][0]
        self.assertFalse(pilot.receipt_valid(self.out, self.plan, job))
        row = self.receipt(job)
        self.assertTrue(pilot.receipt_valid(self.out, self.plan, job))
        Path(row["video_path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            pilot.receipt_valid(self.out, self.plan, job)

    def test_sample_tamper(self):
        job = self.plan["jobs"][-1]
        row = self.receipt(job)
        Path(row["sample_path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            pilot.receipt_valid(self.out, self.plan, job)

    def test_calibration_incomplete_and_failed_and_passed(self):
        with patch.object(pilot, "load_plan", return_value=self.plan):
            with self.assertRaises(RuntimeError):
                pilot.audit(self.args)
            self.calibration_records(bad_arm="JENGA_OFF")
            with self.assertRaises(RuntimeError):
                pilot.audit(self.args)
            self.assertFalse(pilot.read(self.out / "calibration_audit.json")["passed"])
            self.calibration_records()
            pilot.audit(self.args)
            self.assertTrue(pilot.read(self.out / "calibration_audit.json")["passed"])

    def test_calibration_active_noise_mismatch_rejected(self):
        self.calibration_records(bad_noise=True)
        with patch.object(pilot, "load_plan", return_value=self.plan):
            with self.assertRaises(RuntimeError):
                pilot.audit(self.args)

    def test_complete_group_only_and_partial_flag(self):
        for job in self.plan["jobs"][:5]:
            self.receipt(job)
        self.receipt(self.plan["jobs"][5])
        with patch.object(pilot, "load_plan", return_value=self.plan), patch.object(pilot, "audit"):
            with self.assertRaises(RuntimeError):
                pilot.finalize(self.args)
            self.args.allow_partial = True
            self.args.allow_in_place_partial = True  # test-only fixture
            pilot.finalize(self.args)
        saved = pilot.read(self.out / "dataset_manifest.json")
        self.assertTrue(saved["partial_exploratory"])
        self.assertEqual(len(saved["records"]), 5)
        self.assertEqual(len(saved["complete_groups"]), 1)

    def test_finalize_seed_noise_pair_mismatch(self):
        for job in self.plan["jobs"][:5]:
            self.receipt(job, noise=job["arm"]["id"])
        self.args.allow_partial = True
        with patch.object(pilot, "load_plan", return_value=self.plan), patch.object(pilot, "audit"):
            with self.assertRaises(ValueError):
                pilot.finalize(self.args)

    def test_worker_argv_preserves_published_presets(self):
        job = next(j for j in self.plan["jobs"] if j["arm"]["id"] == "JENGA_BASE")
        argv = worker.argv_for(self.plan, job, self.out / "video.mp4")
        self.assertIn("--use_ret_steps", argv)
        self.assertIn("--sa_drop_rates", argv)
        self.assertNotIn("--enable_turbo", argv)
        self.assertEqual(argv[argv.index("--sample_steps") + 1], "50")
        self.assertEqual(argv[argv.index("--sample_guide_scale") + 1], "6.0")
        self.assertEqual(argv[argv.index("--offload_model") + 1], "false")
        tea = next(j for j in self.plan["jobs"] if j["arm"]["id"] == "TEA008")
        self.assertNotIn("--use_ret_steps", worker.argv_for(self.plan, tea, self.out / "tea.mp4"))

    def test_score_csv_and_report_binding(self):
        records = [self.receipt(j) for j in self.plan["jobs"][:5]]
        dataset = {"records": records, "partial_exploratory": True}
        rows = []
        for r in records:
            j = r["job"]
            rows.append({"observation_id": j["id"], "group_id": j["group_id"], "prompt_key": j["prompt_id"],
                "prompt": j["prompt"], "action_id": j["arm"]["id"], "seed": j["seed"],
                "video_sha256": r["video_sha256"], "pipeline_seconds": 10, "origin": "vbench", "motion": "low", "detail": "low",
                **{d: .9 for d in ["vbench5"] + self.cfg["evaluation"]["quality_dimensions"] + self.cfg["evaluation"]["diagnostic_dimensions"]}})
        pilot.csv_write(self.out / "metrics/quality_by_video.csv", rows)
        pilot.write(self.out / "metrics/score_provenance.json", {"plan_sha256": self.plan["plan_sha256"],
            "quality_csv_sha256": pilot.file_hash(self.out / "metrics/quality_by_video.csv")})
        with patch.object(pilot, "finalized", return_value=(self.plan, dataset)):
            pilot.report(self.args)
        report = pilot.read(self.out / "metrics/report.json")
        self.assertEqual(len(report["summary"]), 4)
        self.assertTrue(all(r["mean_delta_vbench5"] == 0 for r in report["summary"]))
        rows[0]["seed"] += 1
        pilot.csv_write(self.out / "metrics/quality_by_video.csv", rows)
        with self.assertRaises(ValueError):
            pilot.finalized_score_rows(self.args, self.plan, dataset)

    def test_mock_strict_score_absolute_maps_and_repeatability(self):
        from changing_resolution_uni.scripts.data import batch_vbench_score_dataset as scorer
        records = [self.receipt(j) for j in self.plan["jobs"][:5]]
        dataset = {"records": records, "partial_exploratory": True}
        dimensions = self.cfg["evaluation"]["quality_dimensions"] + self.cfg["evaluation"]["diagnostic_dimensions"]
        def fake_case(vbench_root, python, directory, prompt_map, out, dims, quality, diagnostic, ngpus, force, identity):
            request, mapping = scorer.build_case_request(video_dir=directory, prompt_map=prompt_map,
                dimensions=dims, quality_dimensions=quality, diagnostic_dimensions=diagnostic,
                python_bin=python, vbench_identity=identity)
            self.assertEqual(dims, dimensions)
            self.assertEqual(len(mapping), 1)
            self.assertTrue(all(Path(p).is_absolute() for p in pilot.read(prompt_map)))
            return scorer.CaseScoreBundle(scores={stem:{d:.9 for d in dims} for stem in mapping},
                                         provenance={"request_sha256": request["request_sha256"]})
        with patch.object(pilot, "finalized", return_value=(self.plan, dataset)), \
                patch.object(scorer, "inspect_vbench_checkout", return_value={"commit": "locked"}), \
                patch.object(scorer, "warmup_vbench_cache"), patch.object(scorer, "score_case_directory", side_effect=fake_case) as calls:
            pilot.score(self.args)
            pilot.score(self.args)
            self.assertEqual(calls.call_count, 10)
        saved = pilot.read(self.out / "metrics/score_provenance.json")
        self.assertEqual(saved["quality_csv_sha256"], pilot.file_hash(self.out / "metrics/quality_by_video.csv"))
        self.assertIn("NOT", saved["aggregate"])

    def test_partial_snapshot_keeps_live_output_unfrozen(self):
        for job in self.plan["jobs"][:5]:
            self.receipt(job)
        self.calibration_records()
        pilot.write(self.out / "calibration_audit.json", {"passed": True})
        snapshot_temp = tempfile.TemporaryDirectory()
        self.addCleanup(snapshot_temp.cleanup)
        snapshot = Path(snapshot_temp.name)
        self.args.allow_partial, self.args.snapshot_out = True, snapshot
        original = self.args.out
        with patch.object(pilot, "load_plan", return_value=self.plan), patch.object(pilot, "audit"):
            pilot.finalize(self.args)
        self.assertEqual(self.args.out, snapshot)
        self.assertTrue((snapshot / "dataset_manifest.json").exists())
        self.assertFalse((original / "dataset_manifest.json").exists())
        self.assertEqual(len(list((snapshot / "records").glob("*.json"))), 21)

    def test_blind_pair_coverage_and_repeats_are_score_independent(self):
        from UNIV_adaptor.scripts.data import acceleration_blind_audit as human
        source = []
        for job in self.plan["jobs"]:
            if job["phase"] == "pilot":
                source.append({"source": "published_wan21", "id": job["id"], "group": job["group_id"],
                    "action": job["arm"]["id"], "prompt": job["prompt"], "prompt_key": pilot.digest(job["prompt"]),
                    "seed": str(job["seed"]), "sha256": pilot.digest(job["id"]), "cell": "cell", "scores": {"vbench5": 0}})
        dataset = {"partial_exploratory": False, "complete_groups": [str(i) for i in range(24)]}
        with patch.object(pilot, "finalized", return_value=(self.plan, dataset)), patch.object(pilot, "finalized_score_rows"), \
                patch.object(human, "load_source", return_value=(source, {})), patch.object(human, "package"):
            pilot.blind(self.args)
        saved = pilot.read(self.out / "blind/private/plan.json")
        self.assertEqual(sum(p["kind"] == "real" for p in saved["pairs"]), 96)
        self.assertEqual(sum(p["kind"] == "reliability_repeat" for p in saved["pairs"]), 6)
        self.assertEqual(len({p["id"] for p in saved["pairs"]}), 102)
        pair_keys = [(p["a"]["id"], p["b"]["id"]) for p in saved["pairs"] if p["kind"] == "real"]
        for row in source:
            row["scores"] = {"vbench5": 999}
        again = human.make_plan(saved["config"], {"published_wan21": source})
        self.assertEqual(pair_keys, [(p["a"]["id"], p["b"]["id"]) for p in again])

    def test_public_export_contains_no_labels_scores_or_private_files(self):
        from UNIV_adaptor.scripts.data import acceleration_blind_audit as human
        import tarfile
        study = self.out / "blind"
        media = study / "media"
        media.mkdir(parents=True)
        clip = "a" * 24
        (media / (clip + ".mp4")).write_bytes(b"fake video")
        body = {"config": {}, "pairs": []}
        plan = body | {"plan_sha256": human.digest(body)}
        package_body = {"plan_sha256": plan["plan_sha256"], "implementation_sha256": "private",
            "pairs": [{"id": "b" * 20, "a": clip, "b": clip, "prompt": "public"}],
            "clips": {clip: {"sha256": pilot.file_hash(media / (clip + ".mp4")), "transform": {"method": "private"}}}}
        package = package_body | {"package_sha256": human.digest(package_body)}
        pilot.write(study / "private/plan.json", plan)
        pilot.write(study / "private/package.json", package)
        pilot.export(self.args)
        public = pilot.read(study / "public_study.json")
        self.assertEqual(public["clips"][clip].keys(), {"sha256"})
        with tarfile.open(self.out / "exports/blind_media_000.tgz") as archive:
            self.assertFalse(any("private" in n for n in archive.getnames()))
            self.assertIn("study/acceleration_blind_audit.py", archive.getnames())

    def test_human_method_breakdown_unblinds_repeats_and_keeps_ties(self):
        from UNIV_adaptor.scripts.data import acceleration_blind_audit as human
        pair = {"id": "real", "kind": "real", "cell": "low/high", "source": "wan",
                "a": {"id": "full", "prompt": "test prompt", "seed": "42"},
                "b": {"id": "tea", "action": "TEA008"}}
        repeat = pair | {"id": "repeat", "kind": "reliability_repeat"}
        plan = {"plan_sha256": "frozen", "config": {"minimum_raters": 3}, "pairs": [pair, repeat]}
        package = {"package_sha256": "package", "pairs": [
            {"id": key, "prompt": "test prompt", "a": "clip_a", "b": "clip_b"} for key in ("real", "repeat")]}
        study = self.out / "blind"
        for participant in ("rater01", "rater02", "rater03"):
            session = human.session(plan, package, participant)
            answers = {r["id"]:{"votes": {d:("B" if r["swap"] else "A") for d in human.DIMENSIONS}} for r in session}
            pilot.write(study / "private/ratings" / (participant + ".json"), {"participant": participant, "answers": answers})
        human_rows = [{"pair": "real", "kind": "real", "source": "wan", "cluster": "cluster", "dimension": d,
                       "raters": 3, "consensus": "tie" if d == "prompt" else "A"} for d in human.DIMENSIONS]
        pilot.csv_write(study / "analysis/human_consensus.csv", human_rows)
        pilot.csv_write(study / "analysis/metric_pairs.csv", [{"pair": "real", "scope": "presented", "dimension": "overall", "metric": "vbench5", "correct": 0, "miss": 1, "reversed": 0}])
        with patch.object(human, "report"), patch.object(human, "load_plan", return_value=plan), patch.object(human, "load_package", return_value=package):
            pilot.blind_report(self.args)
        result = pilot.read(study / "analysis/method_content_report.json")
        self.assertEqual(len(result["human_preference_by_method_cell"]), 4)
        prompt = next(r for r in result["human_preference_by_method_cell"] if r["dimension"] == "prompt")
        self.assertEqual(prompt["ties"], 1)
        self.assertEqual(prompt["reference_wins"], 0)
        self.assertTrue(all(r["within_rater_agreement"] == 1 for r in result["repeat_reliability"]))
        self.assertEqual(result["presented_metric_agreement_by_method_cell"][0]["metric_tie_rate"], 1)


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)
        self.ndim = self.array.ndim
    def __getitem__(self, index):
        return FakeTensor(self.array[index])
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.array


class WorkerTests(unittest.TestCase):
    setUp = PilotTests.setUp

    def test_scaling_import_guards_reuse_real_classes_and_forbid_parallel(self):
        t2v, i2v = ModuleType("stub_t2v"), ModuleType("stub_i2v")
        t2v.CustomWanT2V, i2v.CustomWanI2V = object(), object()
        with patch.dict("sys.modules", {"scaling_cache.adapter.wan.text2video": t2v,
                                       "scaling_cache.adapter.wan.image2video": i2v}):
            worker.install_scaling_import_guards(self.out)
            import sys
            facade = sys.modules["scaling_cache.adapter.wan"]
            self.assertIs(facade.CustomWanT2V, t2v.CustomWanT2V)
            self.assertIs(facade.CustomWanI2V, i2v.CustomWanI2V)
            with self.assertRaises(RuntimeError):
                sys.modules["xfuser.core.distributed"].get_sequence_parallel_world_size()
    def test_mock_worker_reuses_model_warmup_and_resume(self):
        # A minimal torch facade exercises orchestration, not numerical inference.
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "A800",
            synchronize=lambda: None, reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 100)
        fake_torch.version = SimpleNamespace(cuda="12.4")
        fake_torch.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=True))
        fake_torch.manual_seed = lambda _: None
        fake_torch.randn = lambda *dims, **kw: FakeTensor(np.ones(dims, dtype=np.float32))
        module = SimpleNamespace()
        stats = {"constructors": 0, "generate_calls": 0, "saves": 0}
        def constructor():
            stats["constructors"] += 1
            return SimpleNamespace(config=SimpleNamespace(sample_fps=16, t5_dtype="bf16"),
                                   param_dtype="bf16", sample_neg_prompt="same")
        module.wan = SimpleNamespace(WanT2V=constructor)
        def parse_args():
            parser = argparse.ArgumentParser()
            parser.add_argument("--save_file")
            result, _ = parser.parse_known_args()
            result.ulysses_size = result.ring_size = 1
            result.dit_fsdp = result.t5_fsdp = False
            return result
        module._parse_args = parse_args
        def save_video(**kw):
            stats["saves"] += 1
            Path(kw["save_file"]).write_bytes(b"mock video")
        module.cache_video = save_video
        def generate(options):
            stats["generate_calls"] += 1
            module.wan.WanT2V()
            fake_torch.randn(16, 2, 2, 2, generator=object())
            module.cache_video(tensor=FakeTensor(np.zeros((1, 3, 81, 8, 8))), save_file=options.save_file)
        module.generate = generate
        args = SimpleNamespace(out=self.out, gpu=0, arm="FULL50", calibration=False, probe=False)
        initial_randn = fake_torch.randn
        with patch.object(worker, "load_plan", return_value=self.plan), patch.object(worker, "check_sources"), \
                patch.object(worker, "load_entrypoint", return_value=module), patch.object(worker, "versions", return_value={}), \
                patch.object(worker, "configure_attention_backend", return_value=(None, {"dense_backend": "flash_attention_2", "flash_attn_interface_module": "fake-interface"})), \
                patch.object(worker.importlib.util, "find_spec", return_value=object()), patch.dict("sys.modules", {"torch": fake_torch, "wan": SimpleNamespace(__file__="mock/wan/__init__.py")}), \
                patch("UNIV_adaptor.scripts.data.acceleration_blind_audit.probe", return_value={"width": 832, "height": 480, "duration": 81 / 16}):
            worker.run(args)
            self.assertEqual(stats["constructors"], 1)
            self.assertEqual(stats["generate_calls"], 4)  # 3 assigned FULL jobs + excluded warmup
            self.assertEqual(stats["saves"], 3)
            records = list((self.out / "records").glob("*.json"))
            self.assertEqual(len(records), 3)
            for path in records:
                receipt = pilot.read(path)
                self.assertTrue(receipt["runtime"]["warmed"])
                self.assertGreater(receipt["runtime"]["pipeline_seconds"], 0)
            worker.run(args)
            self.assertEqual(stats["generate_calls"], 4)  # completed jobs are skipped
        fake_torch.randn = initial_randn


if __name__ == "__main__":
    unittest.main()
