"""CPU protocol/asset-reuse tests. Mocked sampling is NOT CUDA validation."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from UNIV_adaptor.scripts.data import published_wan21_pilot as base
from UNIV_adaptor.scripts.data import published_wan21_study as study
from UNIV_adaptor.scripts.data import published_wan21_study_worker as study_worker
from UNIV_adaptor.scripts.data import published_wan21_worker as old_worker
from UNIV_adaptor.tests import test_published_wan21_pilot as fixtures
from UNIV_adaptor.wan21_endpoint_runtime import (
    geometry, latent_shape, fresh_hr_scheduler, coarse_noise, NativeCodec, asdict_without_tensor,
    EndpointRuntime,
)
from UNIV_adaptor.transition import dvg_rounded_anchors


class StudyTests(unittest.TestCase):
    setUp = fixtures.PilotTests.setUp
    receipt = fixtures.PilotTests.receipt
    calibration_records = fixtures.PilotTests.calibration_records

    def freeze(self):
        self.old_root, self.old_plan = self.out, self.plan
        self.args.model_root = Path(self.old_plan["model_root"]).resolve()
        self.old_plan["model_root"] = str(self.args.model_root)
        self.old_plan["plan_sha256"] = base.digest({k:v for k,v in self.old_plan.items() if k != "plan_sha256"})
        base.write(self.old_root / "plan.json", self.old_plan)
        self.calibration_records(bad_arm="JENGA_OFF")
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        self.out = Path(other.name)
        self.args.out = self.out
        self.args.config = study.CONFIG
        self.args.sr_checkpoint = self.out / "RealESRGAN_x2plus.pth"
        self.args.sr_checkpoint.write_bytes(b"test-only-SR-weights")
        self.args.reuse_calibration_root = self.old_root
        with patch.object(base, "check_sources", return_value={}), patch.object(base, "weight_inventory", return_value=[]):
            study.plan(self.args)
        self.plan = base.read(self.out / "plan.json")
        return self.plan

    def endpoint_receipt(self, job, delta=.25):
        row = self.receipt(job, delta=0 if job["arm"]["id"].endswith("_OFF") else delta)
        case = job["arm"].get("endpoint")
        if case:
            info = geometry(case, (832, 480, 81))
            artifact = self.out / (job["id"] + ".pt")
            artifact.write_bytes(b"fake-clean-latent-not-pickle")
            hr = {"noise": {"shape": [16, 21, 60, 104], "sha256": "repair-shared"},
                  "sigmas": [.2, .15, .1, .05, 0], "timesteps": [200, 150, 100, 50],
                  "noise_seed": job["seed"] + 1000000007, "fresh_solver_history": True,
                  "formula": "(1-sigma)*clean_hr + sigma*noise", "shift": 1} if case["refine"] else None
            row["endpoint"] = {"schema": "native_wan21_endpoint_receipt_v1", **info,
                "main_steps": 50, "main_terminal_sigma": 0, "hr": hr,
                "main_clean": {"shape": info["main_latent_shape"], "sha256": "main-shared"},
                "restored_clean": {"shape": info["target_latent_shape"], "sha256": "restored"},
                "executed_main_noise": {"shape": info["main_latent_shape"], "sha256": "small"},
                "model_forward_counts": {"main": 100, "hr": 8 if case["refine"] else 0},
                "transition": {"baseline": case["transition"]},
                "noise_alignment": {"policy": "nested_iid_full_field_anchor_subsample_v1",
                    "anchors": {str(a):list(dvg_rounded_anchors(info["main_latent_shape"][a], info["target_latent_shape"][a])) for a in (1, 2, 3)}},
                "artifacts": [{"kind": "main_clean_sigma_zero", "path": str(artifact), "sha256": base.file_hash(artifact)}]}
        base.write(self.out / "records" / (job["id"] + ".json"), row)
        return row

    def load_patch(self):
        return patch.object(study, "load_plan", return_value=self.plan)

    def test_counts_and_old_pipeline_untouched(self):
        old_impl = self.plan["implementation"]
        self.freeze()
        self.assertEqual(sum(j["phase"] == "pilot" for j in self.plan["jobs"]), 240)
        self.assertEqual(sum(j["phase"] == "calibration" for j in self.plan["jobs"]), 26)
        self.assertEqual(len(self.plan["config"]["arms"]), 10)
        self.assertNotIn("jenga", [a["source"] for a in self.plan["config"]["arms"]])
        self.assertEqual(old_impl, {p.name:base.file_hash(p) for p in (Path(base.__file__), study.LEGACY_WORKER)})
        self.assertEqual(base.read(self.old_root / "plan.json"), self.old_plan)
        self.assertTrue(all(j["arm"]["steps"] == 50 for j in self.plan["jobs"] if j["arm"].get("endpoint")))

    def test_geometry_budgets_and_frame_legality(self):
        cfg = study.resolve_config(study.CONFIG, "dummy")
        densities = {a["id"]:geometry(a["endpoint"], (832, 480, 81))["realized_main_token_density"] for a in cfg["arms"] if a.get("endpoint")}
        self.assertAlmostEqual(densities["S_B050"], 21*37/(30*52))
        self.assertEqual(densities["S_B025"], .25)
        self.assertAlmostEqual(densities["T_B050"], 11/21)
        self.assertAlmostEqual(densities["T_B025"], 5/21)
        self.assertEqual(latent_shape(832, 480, 81), (16, 21, 60, 104))
        for geometry_args in [(592, 335, 81), (832, 480, 40), (832, 480, 0)]:
            with self.assertRaises(ValueError):
                latent_shape(*geometry_args)

    def test_overlay_rejects_wrong_sigma_and_bicubic_and_mixed_axes(self):
        cfg = study.resolve_config(study.CONFIG, "dummy")
        for field, value in [("sigma", .3), ("steps", 5)]:
            bad = copy.deepcopy(cfg)
            bad["endpoint_protocol"]["hr"][field] = value
            with self.assertRaises(ValueError):
                study.validate_endpoint_config(bad)
        bad = copy.deepcopy(cfg)
        bad["endpoint_protocol"]["sr"]["wan_rgb_sr_backend"] = "bicubic"
        with self.assertRaises(ValueError):
            study.validate_endpoint_config(bad)
        bad = copy.deepcopy(cfg)
        next(a for a in bad["arms"] if a["id"] == "S_B050")["endpoint"]["frames"] = 41
        with self.assertRaises(ValueError):
            study.validate_endpoint_config(bad)

    def test_old_output_rejected(self):
        with self.assertRaisesRegex(ValueError, "older published pilot"):
            study.validate_output_root(self.out)

    def test_hash_tamper_and_implementation_binding(self):
        self.freeze()
        with patch.object(base, "check_sources", return_value={}):
            self.assertEqual(study.load_plan(self.out), self.plan)
            with patch.object(study, "implementation", return_value={}):
                with self.assertRaisesRegex(ValueError, "implementation"):
                    study.load_plan(self.out)
            self.args.sr_checkpoint.write_bytes(b"different-SR")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                study.load_plan(self.out)
        bad = copy.deepcopy(self.plan)
        bad["jobs"][0]["seed"] += 1
        base.write(self.out / "plan.json", bad)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            study.load_plan(self.out, verify_implementation=False)

    def test_calibration_reuses_12_videos_despite_old_jenga_failure(self):
        self.freeze()
        old_bytes = (self.old_root / "plan.json").read_bytes()
        with self.load_patch():
            study.reuse_calibration(self.args)
            study.reuse_calibration(self.args)  # idempotent
        rows = list((self.out / "records").glob("*.json"))
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(base.read(r)["reused_from"]["scope"].startswith("calibration only") for r in rows))
        self.assertEqual((self.old_root / "plan.json").read_bytes(), old_bytes)
        self.assertFalse((self.out / "calibration_audit.json").exists())
        job = next(j for j in self.plan["jobs"] if j["id"] == base.read(rows[0])["job"]["id"])
        self.assertTrue(study.receipt_valid(self.out, self.plan, job))
        source = Path(base.read(rows[0])["reused_from"]["receipt_path"])
        source.write_text("{}")
        with self.assertRaisesRegex(ValueError, "provenance"):
            study.receipt_valid(self.out, self.plan, job)

    def test_incompatible_calibration_weights_rejected(self):
        self.freeze()
        self.plan["weight_inventory"] = [{"path": "wrong"}]
        with self.load_patch(), self.assertRaisesRegex(ValueError, "identity differs"):
            study.reuse_calibration(self.args)

    def test_endpoint_protocol_and_artifact_tamper(self):
        self.freeze()
        job = next(j for j in self.plan["jobs"] if j["arm"]["id"] == "S_B025")
        row = self.endpoint_receipt(job)
        study.validate_endpoint_receipt(row)
        for field, value in [("main_steps", 49), ("main_terminal_sigma", .02)]:
            bad = copy.deepcopy(row)
            bad["endpoint"][field] = value
            with self.assertRaises(ValueError):
                study.validate_endpoint_receipt(bad)
        for field, value in [("shift", 8), ("fresh_solver_history", False), ("timesteps", [200, 150, 100, 49])]:
            bad = copy.deepcopy(row)
            bad["endpoint"]["hr"][field] = value
            with self.assertRaises(ValueError):
                study.validate_endpoint_receipt(bad)
        bad = copy.deepcopy(row)
        bad["endpoint"]["noise_alignment"]["anchors"]["2"][0] = 1
        with self.assertRaises(ValueError):
            study.validate_endpoint_receipt(bad)
        Path(row["endpoint"]["artifacts"][0]["path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "endpoint changed"):
            study.validate_endpoint_receipt(row)

    def fill_calibration(self):
        with self.load_patch():
            study.reuse_calibration(self.args)
        for job in self.plan["jobs"]:
            if job["phase"] == "calibration" and job["arm"].get("endpoint"):
                self.endpoint_receipt(job)

    def test_audit_accepts_different_st_quality_but_requires_adapter_equality(self):
        self.freeze()
        self.fill_calibration()
        with self.load_patch(), patch.object(base, "receipt_valid", study.receipt_valid):
            study.audit(self.args)
        self.assertTrue(base.read(self.out / "calibration_audit.json")["passed"])
        job = next(j for j in self.plan["jobs"] if j["phase"] == "calibration" and j["arm"]["id"] == "NATIVE_ADAPTER_OFF")
        row = self.endpoint_receipt(job)
        np.savez_compressed(row["sample_path"], frames=np.ones((3,2,2,2), dtype=np.float32))
        row["sample_sha256"] = base.file_hash(row["sample_path"])
        base.write(self.out / "records" / (job["id"] + ".json"), row)
        with self.load_patch(), patch.object(base, "receipt_valid", study.receipt_valid):
            with self.assertRaisesRegex(RuntimeError, "Calibration failed"):
                study.audit(self.args)

    def test_audit_shared_hr_noise_and_full_endpoint_checks(self):
        self.freeze()
        self.fill_calibration()
        job = next(j for j in self.plan["jobs"] if j["phase"] == "calibration" and j["arm"]["id"] == "FULL50_HR4")
        path = self.out / "records" / (job["id"] + ".json")
        row = base.read(path)
        row["endpoint"]["main_clean"]["sha256"] = "different"
        row["endpoint"]["hr"]["noise"]["sha256"] = "different"
        base.write(path, row)
        with self.load_patch(), patch.object(base, "receipt_valid", study.receipt_valid):
            with self.assertRaises(RuntimeError):
                study.audit(self.args)
        failed = next(c for c in base.read(self.out / "calibration_audit.json")["checks"] if c["arm"] == "FULL50_HR4")
        self.assertFalse(failed["full_main_clean_matches_adapter"])

    def test_formal_finalize_rejects_unmatched_hr_noise(self):
        self.freeze()
        for job in self.plan["jobs"][:10]:
            self.endpoint_receipt(job)
        job = self.plan["jobs"][5]
        path = self.out / "records" / (job["id"] + ".json")
        row = base.read(path)
        row["endpoint"]["hr"]["noise"]["sha256"] = "different"
        base.write(path, row)
        with self.load_patch(), patch.object(base, "receipt_valid", study.receipt_valid):
            with self.assertRaisesRegex(ValueError, "HR repair noise"):
                study.finalize(self.args)

    def test_blind_keeps_all_216_pairs_and_control_roles_without_score_selection(self):
        from UNIV_adaptor.scripts.data import acceleration_blind_audit as human
        self.freeze()
        source = [{"source":"published_wan21","id":j["id"],"group":j["group_id"],
            "action":j["arm"]["id"],"prompt":j["prompt"],"prompt_key":base.digest(j["prompt"]),
            "seed":str(j["seed"]),"sha256":base.digest(j["id"]),"cell":"cell","scores":{"vbench5":0}}
            for j in self.plan["jobs"] if j["phase"] == "pilot"]
        dataset = {"partial_exploratory":False,"complete_groups":[str(i) for i in range(24)]}
        with patch.object(base,"finalized",return_value=(self.plan,dataset)), patch.object(base,"finalized_score_rows"), \
                patch.object(human,"load_source",return_value=(source,{})), patch.object(human,"package"):
            study.blind(self.args)
        saved = base.read(self.out / "blind/private/plan.json")
        primary = [p for p in saved["pairs"] if p["kind"] == "real"]
        self.assertEqual(len(primary),216)
        self.assertEqual(sum(p["kind"] == "reliability_repeat" for p in saved["pairs"]),6)
        self.assertEqual(sum(p["method_role"] == "refinement_control" for p in primary),24)
        self.assertEqual(sum(p["method_role"] == "vae_roundtrip_control" for p in primary),24)
        self.assertEqual(sum(p["method_role"] == "published_pipeline" for p in primary),48)
        for row in source:
            row["scores"] = {"vbench5":999}
        again = human.make_plan(saved["config"],{"published_wan21":source})
        self.assertEqual([(p["a"]["id"],p["b"]["id"]) for p in primary],[(p["a"]["id"],p["b"]["id"]) for p in again])

    def test_new_hooks_finalize_and_score_ten_arms_end_to_end_with_fake_metrics(self):
        from changing_resolution_uni.scripts.data import batch_vbench_score_dataset as scorer
        self.freeze()
        self.fill_calibration()
        for job in self.plan["jobs"][:10]:
            self.endpoint_receipt(job)
        self.args.allow_partial, self.args.allow_in_place_partial = True, True
        names = ("WORKER","load_plan","plan","validate_output_root","receipt_valid","audit","blind","finalize","report")
        originals = {name:getattr(base,name) for name in names}

        def fake_case(vbench_root, python, directory, prompt_map, out, dims, quality, diagnostic, ngpus, force, identity):
            mapping = base.read(prompt_map)
            self.assertEqual(len(mapping),1)
            self.assertTrue(all(Path(p).is_absolute() for p in mapping))
            return scorer.CaseScoreBundle(scores={Path(p).stem:{d:.9 for d in dims} for p in mapping}, provenance={"fake":True})

        with patch.multiple(base,**originals), self.load_patch(), \
                patch.object(scorer,"inspect_vbench_checkout",return_value={"commit":"locked"}), \
                patch.object(scorer,"warmup_vbench_cache"), patch.object(scorer,"score_case_directory",side_effect=fake_case) as calls:
            study.activate()
            base.finalize(self.args)
            base.score(self.args)
            self.assertEqual(calls.call_count,10)
        dataset = base.read(self.out / "dataset_manifest.json")
        self.assertEqual(len(dataset["records"]),10)
        result = base.read(self.out / "metrics/report.json")
        self.assertEqual(len(result["summary"]),9)
        self.assertTrue(all("method_role" in r for r in result["summary"]))
        self.assertIn("custom_not_official_dvg",result["method_scope"])


class NumpyTensor(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def cpu(self):
        return self

    def to(self, *args, **kwargs):
        return NumpyTensor(self.astype(kwargs["dtype"])) if "dtype" in kwargs else self

    @property
    def device(self):
        return "cpu"

    def index_select(self, axis, indices):
        return NumpyTensor(np.take(self, indices, axis=axis))

    def contiguous(self):
        return self

    def detach(self):
        return self

    def float(self):
        return NumpyTensor(self.astype(np.float32))

    def numpy(self):
        return np.asarray(self)

    def unsqueeze(self, axis):
        return NumpyTensor(np.expand_dims(self, axis))


class WorkerAttachmentTests(unittest.TestCase):
    def test_only_native_endpoint_is_adapted_and_instrumentation_is_subtracted(self):
        arm = {"id":"S_B025","source":"wan21","endpoint":{"transition":"rgb_sr_vae"}}
        frozen = {"config":{"arms":[arm],"disabled_arms":[],"endpoint_protocol":{"sr":{}},"warmup":True}}
        original_generate = Mock()
        pipeline = SimpleNamespace(generate=original_generate)
        factory = Mock(return_value=pipeline)
        module = SimpleNamespace(wan=SimpleNamespace(WanT2V=factory))
        controller = SimpleNamespace(generate=Mock(),record={"schema":"test"},instrument_seconds=7.0)
        saved = []
        args = SimpleNamespace(out=Path("/new-study"),gpu=0,arm="S_B025",calibration=True,probe=False)

        def fake_run(options):
            entry = old_worker.load_entrypoint("wan21")
            created = entry.wan.WanT2V("config",checkpoint="native-weights")
            self.assertIs(created.generate,controller.generate)
            row = {"runtime":{"pipeline_seconds":100.0,"instrumentation_seconds":2.0,"scope":"main"}}
            old_worker.immutable(args.out / "records/r.json",row)

        originals = {name:getattr(old_worker,name) for name in ("load_plan","receipt_valid","load_entrypoint","immutable","run")}
        with patch.multiple(old_worker,**originals), patch.object(study,"load_plan",return_value=frozen), \
                patch.object(old_worker,"load_entrypoint",return_value=module), \
                patch.object(old_worker,"immutable",side_effect=lambda path,row:saved.append(row)), \
                patch.object(old_worker,"run",side_effect=fake_run), \
                patch.object(study_worker,"EndpointRuntime",return_value=controller) as runtime, \
                patch.object(study,"validate_endpoint_receipt") as validate:
            study_worker.run(args)
            self.assertEqual(runtime.call_count,1)
            self.assertEqual(validate.call_count,1)
        self.assertEqual(saved[0]["runtime"]["pipeline_seconds"],93)
        self.assertEqual(saved[0]["runtime"]["instrumentation_seconds"],9)
        self.assertEqual(saved[0]["endpoint"],controller.record)
        factory.assert_called_once_with("config",checkpoint="native-weights")

    def test_published_cache_constructor_and_forward_are_not_wrapped(self):
        arm = {"id":"TEA008","source":"teacache"}
        frozen = {"config":{"arms":[arm],"disabled_arms":[],"endpoint_protocol":{},"warmup":True}}
        factory = Mock()
        module = SimpleNamespace(wan=SimpleNamespace(WanT2V=factory))
        args = SimpleNamespace(out=Path("/new-study"),gpu=0,arm="TEA008",calibration=True,probe=False)
        def fake_run(options):
            entry = old_worker.load_entrypoint("teacache")
            self.assertIs(entry.wan.WanT2V,factory)
        originals = {name:getattr(old_worker,name) for name in ("load_plan","receipt_valid","load_entrypoint","immutable","run")}
        with patch.multiple(old_worker,**originals), patch.object(study,"load_plan",return_value=frozen), \
                patch.object(old_worker,"load_entrypoint",return_value=module), \
                patch.object(old_worker,"run",side_effect=fake_run), patch.object(study_worker,"EndpointRuntime") as runtime:
            study_worker.run(args)
            runtime.assert_not_called()


class NativeSchedulerContractTests(unittest.TestCase):
    def test_actual_pinned_set_timesteps_with_numpy_tensor_facade(self):
        """Execute the pinned setter AST, not a hand-retyped sigma formula.

        This checks protocol/API/timestep rounding. It is NOT a Torch solver
        numerical or GPU-kernel test.
        """
        source = study.ROOT / "UNIV_adaptor/external/wan21/wan/utils/fm_solvers_unipc.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlowUniPCMultistepScheduler")
        method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "set_timesteps"))
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        facade = SimpleNamespace(from_numpy=NumpyTensor, int64=np.int64)
        namespace = {"np":np, "torch":facade}
        module = ast.Module(body=[method], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)

        class Scheduler:
            set_timesteps = namespace["set_timesteps"]

            def __init__(self, num_train_timesteps, shift, use_dynamic_shifting):
                self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps,
                    shift=shift, use_dynamic_shifting=use_dynamic_shifting,
                    final_sigmas_type="zero", solver_order=2)
                self.solver_p = None
                self.sigma_max, self.sigma_min = .999, 0

            @property
            def step_index(self):
                return self._step_index

        hr = fresh_hr_scheduler(Scheduler, device="cpu", num_train_timesteps=1000, sigma=.2, steps=4)
        np.testing.assert_allclose(hr.sigmas, [.2,.15,.1,.05,0], rtol=0, atol=1e-7)
        self.assertEqual(hr.timesteps.tolist(), [200,150,100,50])
        self.assertEqual(hr.model_outputs, [None,None])
        self.assertEqual(hr.lower_order_nums, 0)
        self.assertIsNone(hr.last_sample)
        main = Scheduler(1000,1,False)
        main.set_timesteps(50, device="cpu", shift=8)
        self.assertEqual(len(main.timesteps), 50)
        self.assertEqual(float(main.sigmas[-1]), 0)
        self.assertIsNot(hr, main)

    def test_noise_subsample_preserves_entries_and_full_control(self):
        fake_torch = ModuleType("torch")
        fake_torch.tensor = lambda value, **kw:np.asarray(value)
        full = NumpyTensor(np.arange(1*5*6*8).reshape(1,5,6,8))
        with patch.dict(sys.modules, {"torch":fake_torch}):
            small, anchors = coarse_noise(full, (1,3,3,4))
            same, _ = coarse_noise(full, full.shape)
        expected = np.asarray(full)[np.ix_([0], anchors["1"], anchors["2"], anchors["3"])]
        np.testing.assert_array_equal(small, expected)
        np.testing.assert_array_equal(same, full)
        self.assertEqual(len(np.unique(small)), small.size)

    def test_native_codec_uses_deterministic_vae_lists(self):
        calls = []
        vae = SimpleNamespace(decode=lambda zs:calls.append(("decode",zs)) or ["rgb"],
                              encode=lambda xs:calls.append(("encode",xs)) or ["latent"])
        codec = NativeCodec(vae)
        self.assertEqual(codec.decode("clean"), "rgb")
        self.assertEqual(codec.encode(["video"]), ["latent"])
        self.assertEqual(calls, [("decode",["clean"]), ("encode",["video"])])

    def test_transition_metadata_does_not_deepcopy_cuda_tensor(self):
        class NeverCopy:
            def __deepcopy__(self, memo):
                raise AssertionError("Do not clone large tensors for a JSON receipt")
        from UNIV_adaptor.transition import TransitionResult
        result = TransitionResult("identity", NeverCopy(), (1,1,1,1), (1,1,1,1), None,None,None,None,None,None,False,False)
        self.assertNotIn("clean_hr", asdict_without_tensor(result))

    def test_mock_complete_endpoint_runtime_warmup_then_hr4(self):
        """Run the entire adapter against fake CPU tensors/model/solver.

        This catches orchestration mistakes but makes NO numerical quality or
        CUDA claim. The real pinned setter has a separate AST conformance test.
        """
        from contextlib import nullcontext
        import hashlib

        class Generator:
            def __init__(self, **kw):
                pass

            def manual_seed(self, seed):
                self.rng = np.random.default_rng(seed)
                return self

        class Scheduler:
            def __init__(self, **kw):
                self.step_index = None
                self.lower_order_nums = 0
                self.model_outputs = [None,None]

            def set_timesteps(self, steps=None, *, sigmas=None, **kw):
                grid = np.linspace(.999,.01,steps) if sigmas is None else sigmas
                self.sigmas = NumpyTensor([*grid,0]).astype(np.float32)
                self.timesteps = NumpyTensor(grid * 1000).astype(np.int64)

            def step(self, prediction, t, sample, **kw):
                self.step_index = (self.step_index or 0) + 1
                return [sample]  # deliberately fake zero-velocity trajectory

        class Model:
            def __init__(self):
                self.seen = []

            def to(self, device):
                return self

            def __call__(self, inputs, **kw):
                self.seen.append((int(kw["t"][0]), kw["seq_len"], tuple(inputs[0].shape)))
                return [NumpyTensor(np.zeros_like(inputs[0]))]

        encoder_calls = []

        class Encoder:
            model = SimpleNamespace(to=lambda device:None)

            def __call__(self, prompts, device):
                encoder_calls.append(prompts)
                return ["fake-context"]

        cuda, amp = ModuleType("torch.cuda"), ModuleType("torch.cuda.amp")
        cuda.synchronize = lambda:None
        amp.autocast = lambda **kw:nullcontext()
        cuda.amp = amp
        torch = ModuleType("torch")
        torch.cuda, torch.Generator, torch.float32 = cuda, Generator, np.float32
        torch.no_grad = nullcontext
        torch.tensor = lambda value, **kw:NumpyTensor(value)
        torch.stack = lambda items:NumpyTensor(np.stack(items))
        torch.randn = lambda *shape, generator, **kw:NumpyTensor(generator.rng.standard_normal(shape).astype(np.float32))
        saved = []
        def save(payload, path):
            saved.append(payload)
            Path(path).write_bytes(b"mock-clean-artifact")
        torch.save = save
        module = ModuleType("wan.text2video")
        module.FlowUniPCMultistepScheduler = Scheduler
        tqdm = ModuleType("tqdm")
        tqdm.tqdm = lambda iterable, **kw:iterable
        pipeline = SimpleNamespace(device="cpu", t5_cpu=False, sp_size=1, rank=0,
            vae_stride=(4,8,8), patch_size=(1,2,2), num_train_timesteps=1000,
            param_dtype="bf16", sample_neg_prompt="negative", model=Model(), text_encoder=Encoder(),
            vae=SimpleNamespace(model=SimpleNamespace(z_dim=16), decode=lambda zs:[zs[0]]))
        case = {"width":32,"height":16,"frames":9,"transition":"identity","refine":True}
        protocol = {"hr":{"sigma":.2,"steps":4,"noise_seed_offset":1000000007},
                    "calibration_seed":12345,"calibration_prompts":["cal"]}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
                "torch":torch,"torch.cuda":cuda,"torch.cuda.amp":amp,"wan.text2video":module,"tqdm":tqdm}):
            runtime = EndpointRuntime(pipeline,case,protocol,directory,"FULL50_HR4",True)
            runtime.generate("cal",size=(32,16),frame_num=9,seed=12345)
            self.assertEqual(len(saved),0)  # no warmup artifact leak
            self.assertEqual(runtime.record["artifacts"],[])
            output = runtime.generate("study",size=(32,16),frame_num=9,seed=42)
            self.assertEqual(len(saved),1)
            record = runtime.record
            self.assertEqual(record["model_forward_counts"],{"main":100,"hr":8})
            self.assertEqual(record["hr"]["timesteps"],[200,150,100,50])
            self.assertEqual(tuple(output.shape),(16,3,2,4))
            self.assertEqual(record["main_clean"],record["restored_clean"])
            self.assertEqual(record["main_clean"],record["executed_main_noise"])
            expected = np.random.default_rng(42).standard_normal((16,3,2,4)).astype(np.float32)
            self.assertEqual(record["main_clean"]["sha256"],hashlib.sha256(expected.tobytes()).hexdigest())
            self.assertEqual(encoder_calls,[["cal"],["negative"],["study"],["negative"]])
            self.assertEqual([v[0] for v in pipeline.model.seen[-8:]],[200,200,150,150,100,100,50,50])
            self.assertTrue(all(np.isfinite(v) and v>=0 for v in record["timing"].values()))


if __name__ == "__main__":
    unittest.main()
