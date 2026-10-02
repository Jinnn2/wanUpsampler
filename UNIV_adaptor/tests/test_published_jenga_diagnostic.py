"""CPU-only protocol and intervention tests; not GPU performance validation."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from UNIV_adaptor.scripts.data import published_wan21_pilot as pilot
from UNIV_adaptor.scripts.data import published_wan21_jenga_diagnostic as diagnostic
from UNIV_adaptor.scripts.data import published_wan21_jenga_diagnostic_worker as diag_worker
from UNIV_adaptor.tests import test_published_wan21_pilot as fixtures


class DiagnosticPlanTests(unittest.TestCase):
    setUp = fixtures.PilotTests.setUp
    receipt = fixtures.PilotTests.receipt
    calibration_records = fixtures.PilotTests.calibration_records

    def freeze(self):
        self.calibration_records(bad_arm="JENGA_OFF")
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.source = self.out
        self.diag_out = Path(directory.name)
        self.diag_args = SimpleNamespace(source=self.source, out=self.diag_out, ngpus=8, python="python")
        with patch.object(pilot, "check_sources", return_value={}), patch.object(pilot, "weight_inventory", return_value=[]):
            return diagnostic.freeze(self.diag_args)

    def diagnostic_receipt(self, plan, job, delta=0, hits=0):
        video = self.diag_out / (job["id"] + ".mp4")
        video.write_bytes(job["id"].encode())
        sample = self.diag_out / (job["id"] + ".npz")
        np.savez_compressed(sample, frames=np.zeros((3, 2, 2, 2), dtype=np.float32) + delta)
        row = copy.deepcopy(pilot.read(self.source / "records" / (f"calibration_{job['group_id']}_FULL50.json")))
        n = 2 * job["arm"]["steps"]
        row.update(plan_sha256=plan["plan_sha256"], job=job, video_path=str(video), video_sha256=pilot.file_hash(video),
                   sample_path=str(sample), sample_sha256=pilot.file_hash(sample))
        row["jenga_diagnostics"] = {"settings": job["diagnostic"], "counters": {
            "forward_calls": n, "hard_off_calls": n if job["diagnostic"]["cache"] == "hard_off" else 0,
            "cache_enabled_calls": n if job["diagnostic"]["cache"] == "upstream_zero" else 0,
            "cache_hits": hits, "identity_order_calls": n if job["diagnostic"]["order"] == "identity" else 0,
            "dense_attention_calls": (n - hits) * 60, "sparse_attention_calls": 0}}
        pilot.write(self.diag_out / "records" / (job["id"] + ".json"), row)
        return row

    def test_six_runs_three_persistent_workers_source_read_only(self):
        before = pilot.file_hash(self.out / "plan.json")
        plan = self.freeze()
        self.assertEqual(len(plan["jobs"]), 6)
        self.assertEqual({j["gpu"] for j in plan["jobs"]}, {0, 1, 2})
        self.assertEqual(len(plan["source_records"]), 16)
        self.assertEqual(pilot.file_hash(self.source / "plan.json"), before)
        self.assertEqual(diagnostic.load_plan(self.diag_out), plan)
        self.assertFalse((self.source / "diagnostic_report.json").exists())
        self.assertTrue(all(j["arm"]["flags"] == self.cfg["disabled_arms"][-1]["flags"] for j in plan["jobs"]))

    def test_does_not_accept_old_worker_hash_bypass(self):
        self.calibration_records()
        old = pilot.read(self.out / "plan.json")
        old["implementation"]["published_wan21_worker.py"] = "unknown-old-worker"
        old["plan_sha256"] = pilot.digest({k:v for k,v in old.items() if k != "plan_sha256"})
        pilot.write(self.out / "plan.json", old)
        with self.assertRaisesRegex(ValueError, "implementation changed"):
            diagnostic.source_rows(self.out)

    def test_reject_nested_output(self):
        for target in (self.out, self.out / "nested", self.out.parent):
            with self.assertRaises(ValueError):
                diagnostic.separate_roots(self.out, target)

    def test_source_sample_and_receipt_tamper_rejected(self):
        plan = self.freeze()
        ref = plan["source_records"][0]
        row = pilot.read(ref["path"])
        Path(row["sample_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "video/sample changed"):
            diagnostic.load_plan(self.diag_out)

    def test_diagnostic_hash_tamper_and_implementation_change(self):
        plan = self.freeze()
        with patch.object(diagnostic, "implementation", return_value={}):
            with self.assertRaisesRegex(ValueError, "implementation changed"):
                diagnostic.load_plan(self.diag_out)
        plan["jobs"][0]["diagnostic"]["cache"] = "hard_off"
        pilot.write(self.diag_out / "plan.json", plan)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            diagnostic.load_plan(self.diag_out)

    def test_report_separates_causes_without_releasing_original_pilot(self):
        plan = self.freeze()
        before = pilot.file_hash(self.source / "plan.json")
        for job in plan["jobs"]:
            self.diagnostic_receipt(plan, job, delta=0 if job["diagnostic"]["order"] == "identity" else .1)
        diagnostic.report(self.diag_args)
        report = pilot.read(self.diag_out / "diagnostic_report.json")
        self.assertFalse(report["pilot_released"])
        self.assertEqual(len(report["checks"]), 6)
        for finding in report["group_findings"]:
            self.assertTrue(finding["legacy_zero_repeat_vs_original_compatible"])
            self.assertEqual(finding["legacy_zero_cache_hits"], 0)
            self.assertFalse(finding["hard_off_vs_full_compatible"])
            self.assertTrue(finding["identity_off_vs_full_compatible"])
        self.assertEqual(pilot.file_hash(self.source / "plan.json"), before)
        self.assertEqual(report["thresholds"]["sample_max_tolerance"], .02)

    def test_report_rejects_missing_runs_and_failed_intervention(self):
        plan = self.freeze()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            diagnostic.report(self.diag_args)
        for job in plan["jobs"]:
            self.diagnostic_receipt(plan, job, hits=1 if job["diagnostic"]["cache"] == "hard_off" else 0)
        with self.assertRaisesRegex(ValueError, "count validation failed"):
            diagnostic.report(self.diag_args)

    def test_environment_noise_nonfinite_rejected(self):
        plan = self.freeze()
        job = plan["jobs"][0]
        row = self.diagnostic_receipt(plan, job)
        ref = pilot.read(self.source / "records" / f"calibration_{job['group_id']}_FULL50.json")
        self.assertTrue(diagnostic.compare(row, ref, self.cfg["calibration"])["compatible"])
        row["environment"]["torch"] = "different"
        self.assertFalse(diagnostic.compare(row, ref, self.cfg["calibration"])["compatible"])
        np.savez_compressed(row["sample_path"], frames=np.full((3, 2, 2, 2), np.nan))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            diagnostic.compare(row, ref, self.cfg["calibration"])

    def test_worker_attaches_only_diagnostic_receipts(self):
        plan = self.freeze()
        args = SimpleNamespace(out=self.diag_out, gpu=1, arm="JENGA_HARD_OFF", probe=False)
        module = object()
        counters = {"forward_calls": 100, "cache_hits": 0}
        def fake_run(options):
            self.assertTrue(options.calibration)
            self.assertEqual(diag_worker.worker.load_plan(options.out), plan)
            self.assertIs(diag_worker.worker.load_entrypoint("jenga"), module)
            diag_worker.worker.immutable(self.diag_out / "receipt-test.json", {"original_field": "preserved"})
        with patch.object(diag_worker.worker, "load_plan"), \
                patch.object(diag_worker.worker, "load_entrypoint", return_value=module), \
                patch.object(diag_worker.worker, "immutable", side_effect=pilot.immutable), \
                patch.object(diag_worker.worker, "run", side_effect=fake_run), \
                patch.object(diag_worker, "install_diagnostic_hooks", return_value=counters):
            diag_worker.run(args)
        receipt = pilot.read(self.diag_out / "receipt-test.json")
        self.assertEqual(receipt["original_field"], "preserved")
        self.assertEqual(receipt["jenga_diagnostics"]["counters"], counters)
        self.assertEqual(receipt["jenga_diagnostics"]["settings"], {"cache": "hard_off", "order": "gilbert"})

    def test_launch_three_persistent_workers_and_resume_skips_completed(self):
        plan = self.freeze()
        with patch.object(pilot, "check_sources", return_value={}), \
                patch.object(pilot, "weight_inventory", return_value=[]), \
                patch.object(diagnostic.subprocess, "Popen", side_effect=lambda *a, **k: SimpleNamespace(poll=lambda: 0, returncode=0)) as start, \
                patch.object(diagnostic.time, "sleep"):
            diagnostic.launch(self.diag_args)
            self.assertEqual(start.call_count, 3)
            visible = [call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for call in start.call_args_list]
            self.assertEqual(visible, ["0", "1", "2"])
            for job in plan["jobs"]:
                self.diagnostic_receipt(plan, job)
            diagnostic.launch(self.diag_args)
            self.assertEqual(start.call_count, 3)  # no new generation for valid receipts

    def test_launch_failure_stops_only_its_started_process(self):
        self.freeze()
        process = Mock()
        with patch.object(pilot, "check_sources", return_value={}), \
                patch.object(pilot, "weight_inventory", return_value=[]), \
                patch.object(diagnostic.subprocess, "Popen", side_effect=[process, OSError("failed to start")]):
            with self.assertRaisesRegex(OSError, "failed to start"):
                diagnostic.launch(self.diag_args)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=15)


class FakeOrder:
    dtype = "int64"
    device = "cpu"
    def __init__(self, values): self.values = list(values)
    def numel(self): return len(self.values)


class HookTests(unittest.TestCase):
    def hooks(self, cache, order):
        model = SimpleNamespace(enable_teacache=True, use_cache=False, hilbert_order=FakeOrder([2,0,1]),
                                linear_to_hilbert=FakeOrder([1,2,0]))
        original_order, original_inverse = model.hilbert_order, model.linear_to_hilbert
        state = {"results": [], "permutations": []}
        attention = SimpleNamespace(flash_attention=lambda: "real-dense", block_sparse_attention=lambda: "real-sparse")
        def forward(model):
            state["permutations"].append(model.hilbert_order.values[:])
            model.use_cache = model.enable_teacache
            if model.use_cache:
                return "cached"
            return attention.flash_attention()
        module = SimpleNamespace(teacache_forward=forward)
        def generate(options):
            # Mimics upstream assigning True on every generation; the control
            # must override at the forward boundary, not before generate().
            model.enable_teacache = True
            for _ in range(4):
                state["results"].append(module.teacache_forward(model))
        module.generate = generate
        facade = SimpleNamespace(arange=lambda n, **kw: FakeOrder(range(n)))
        with patch.dict("sys.modules", {"wan.modules.model_mul": attention}):
            counters = diag_worker.install_diagnostic_hooks(module, {"cache": cache, "order": order}, facade)
            module.generate(None)
        self.assertIs(model.hilbert_order, original_order)
        self.assertIs(model.linear_to_hilbert, original_inverse)
        self.assertTrue(model.enable_teacache)
        return module, state, counters, attention

    def test_legacy_counts_without_disabling_cache_or_reordering(self):
        module, state, c, _ = self.hooks("upstream_zero", "gilbert")
        self.assertEqual(c["cache_hits"], 4)
        self.assertEqual(c["cache_enabled_calls"], 4)
        self.assertEqual(c["hard_off_calls"], 0)
        self.assertEqual(state["permutations"], [[2,0,1]] * 4)
        module.generate(None)
        self.assertEqual(c["forward_calls"], 4)  # counters reset per video, including after warmup

    def test_hard_off_survives_upstream_reenabling_and_preserves_kernel(self):
        _, state, c, _ = self.hooks("hard_off", "gilbert")
        self.assertEqual(c["cache_hits"], 0)
        self.assertEqual(c["hard_off_calls"], 4)
        self.assertEqual(c["dense_attention_calls"], 4)
        self.assertEqual(c["sparse_attention_calls"], 0)
        self.assertEqual(state["results"], ["real-dense"] * 4)
        self.assertEqual(state["permutations"], [[2,0,1]] * 4)

    def test_identity_changes_only_diagnostic_permutation(self):
        _, state, c, _ = self.hooks("hard_off", "identity")
        self.assertEqual(c["identity_order_calls"], 4)
        self.assertEqual(state["permutations"], [[0,1,2]] * 4)
        self.assertEqual(c["cache_hits"], 0)

    def test_rejects_unplanned_settings(self):
        with self.assertRaises(ValueError):
            diag_worker.install_diagnostic_hooks(None, {"cache": "on", "order": "identity"}, object())


if __name__ == "__main__":
    unittest.main()
