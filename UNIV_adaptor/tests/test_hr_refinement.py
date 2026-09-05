from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from UNIV_adaptor.hr_refinement import install_hr_grid, resample_hr_sigmas
from UNIV_adaptor.scripts.validation.run_hr_refinement_ablation import (
    DEFAULT_PROMPT, REPO_ROOT, build_plan, prepare, configure_fixed_total_case,
)

try:
    import numpy as np
    import torch
except ImportError:
    torch = None

LIGHTX2V = Path(os.environ.get("LIGHTX2V_REPO", REPO_ROOT.parent / "LightX2V"))


def reference_sigmas():
    raw = [0.999 * (1 - i / 50) for i in range(50)]
    return [8 * s / (1 + 7 * s) for s in raw] + [0.0]


def load_source_class(path, name, namespace):
    """Execute real scheduler methods on CPU without loading inference backends."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def cpu_scheduler():
    namespace = {
        "torch": torch, "np": np, "AI_DEVICE": "cpu",
        "GET_DTYPE": lambda: torch.float32,
        "GET_SENSITIVE_DTYPE": lambda: torch.float32,
        "logger": SimpleNamespace(info=lambda *a, **k: None),
    }
    load_source_class(LIGHTX2V / "lightx2v/models/schedulers/scheduler.py", "BaseScheduler", namespace)
    cls = load_source_class(LIGHTX2V / "lightx2v/models/schedulers/wan/scheduler.py", "WanScheduler", namespace)
    scheduler = cls({
        "infer_steps": 50, "target_video_length": 81, "sample_shift": 8,
        "sample_guide_scale": 6, "seq_parallel": False, "dim": 12, "num_heads": 3,
        "task": "t2v", "model_cls": "wan2.1",
    })
    # Use the actual UNIV reset implementation, not a copied reset in the test.
    namespace["WanScheduler"] = cls
    univ_cls = load_source_class(REPO_ROOT / "UNIV_adaptor/wan_runner.py", "WanUniversalScheduler", namespace)
    scheduler.reset_solver_history = lambda: univ_cls.reset_solver_history(scheduler)
    scheduler.prepare(42, (1, 2, 2, 2))
    return scheduler


class HRGridTest(unittest.TestCase):
    def test_fixed_total_plan_keeps_reference_suffix_and_moves_boundary(self):
        args = SimpleNamespace(
            template_config=str(REPO_ROOT / "UNIV_adaptor/configs/univ_hr_refinement_ablation.json"),
            out_dir="/fixed-total", model_root="/model", prompt=DEFAULT_PROMPT,
            negative_prompt="", seed=42, comparison="fixed-total",
        )
        plan = build_plan(args)
        base = plan["cases"][0]["planned_sigmas"]
        for case in plan["cases"]:
            self.assertEqual(case["lr_steps"] + case["hr_steps"], 50)
            self.assertEqual(case["planned_sigmas"], base[-(case["hr_steps"] + 1):])
            self.assertEqual(case["reference_schedule"]["lr_compute_steps"], list(range(case["lr_steps"])))
            self.assertEqual(case["reference_schedule"]["hr_compute_steps"], list(range(case["lr_steps"], 50)))
        self.assertEqual(len({c["boundary_path"] for c in plan["cases"]}), 4)

    def test_reference_preserved_and_reduced_grids_cover_same_interval(self):
        reference = reference_sigmas()
        self.assertEqual(resample_hr_sigmas(reference, boundary_step=40, hr_steps=10), tuple(reference[40:]))
        for steps in (6, 4, 2):
            values = resample_hr_sigmas(reference, boundary_step=40, hr_steps=steps)
            self.assertEqual(len(values), steps + 1)
            self.assertEqual(values[0], reference[40])
            self.assertEqual(values[-1], 0)
            self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        self.assertEqual(resample_hr_sigmas(reference, boundary_step=40, hr_steps=2),
                         (reference[40], reference[45], 0))

    def test_invalid_grids_are_rejected(self):
        for reference, boundary, steps in (([.5, .5, 0], 0, 2), ([.5, .1], 0, 2),
                                           ([.5, 0], 1, 2), ([.5, 0], 0, 0)):
            with self.assertRaises(ValueError):
                resample_hr_sigmas(reference, boundary_step=boundary, hr_steps=steps)

    def test_plan_is_paired_and_refuses_different_prompt_in_same_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                template_config=str(REPO_ROOT / "UNIV_adaptor/configs/univ_hr_refinement_ablation.json"),
                out_dir=directory, model_root="/model", prompt=DEFAULT_PROMPT,
                negative_prompt="", seed=42,
            )
            plan, _ = prepare(args)
            self.assertEqual([c["hr_steps"] for c in plan["cases"]], [10, 6, 4, 2])
            self.assertEqual(len({c["planned_sigmas"][0] for c in plan["cases"]}), 1)
            self.assertEqual(len(plan["reference_schedule"]["lr_compute_steps"]), 40)
            self.assertEqual(build_plan(args), plan)
            args.prompt = "changed"
            with self.assertRaisesRegex(RuntimeError, "different comparison"):
                prepare(args)


@unittest.skipUnless(torch is not None and LIGHTX2V.is_dir(), "requires Torch and local LightX2V scheduler source")
class HRActualWanSolverTest(unittest.TestCase):
    def integrate(self, scheduler, indices, *, constant_clean=None):
        outputs = []
        for index in indices:
            scheduler.step_pre(index)
            sigma = scheduler.sigmas[index]
            if constant_clean is None:
                scheduler.noise_pred = .15 * scheduler.latents + .05 * torch.sin(3 * sigma)
            else:
                scheduler.noise_pred = (scheduler.latents - constant_clean) / sigma
            scheduler.step_post()
            outputs.append(scheduler.latents.clone())
        return outputs

    def test_hr10_matches_original_solver_suffix_exactly(self):
        original = cpu_scheduler()
        changed = cpu_scheduler()
        reference = original.sigmas.tolist()
        # Both start from identical arbitrary HR states and empty solver history.
        original.reset_solver_history()
        grid = install_hr_grid(changed, reference_sigmas=reference, boundary_step=40, hr_steps=10)
        self.assertTrue(torch.equal(original.timesteps, changed.timesteps))
        first = self.integrate(original, range(40, 50))
        second = self.integrate(changed, grid["compute_indices"])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))

    def test_all_grids_reach_analytic_clean_endpoint_and_clear_lr_history(self):
        for steps in (10, 6, 4, 2):
            scheduler = cpu_scheduler()
            reference = scheduler.sigmas.tolist()
            clean = torch.full_like(scheduler.latents, .25)
            noise = scheduler.latents.clone()
            scheduler.latents = (1 - reference[40]) * clean + reference[40] * noise
            scheduler.model_outputs = [torch.zeros(9), torch.zeros(9)]
            scheduler.last_sample = torch.zeros(9)
            scheduler.lower_order_nums = 2
            grid = install_hr_grid(scheduler, reference_sigmas=reference, boundary_step=40, hr_steps=steps)
            self.assertIsNone(scheduler.last_sample)
            self.assertEqual(scheduler.model_outputs, [None, None])
            result = self.integrate(scheduler, grid["compute_indices"], constant_clean=clean)
            self.assertEqual(len(result), steps)
            self.assertEqual(len(scheduler.timesteps), 40 + steps)
            self.assertTrue(torch.isfinite(result[-1]).all())
            torch.testing.assert_close(result[-1], clean, rtol=1e-5, atol=1e-6)

    def test_real_runner_branches_reuse_one_transition_and_only_run_requested_hr_nfe(self):
        self.check_runner_branches(fixed_total=False)

    def test_fixed_total_runner_executes_all_prefixes_and_original_suffixes(self):
        self.check_runner_branches(fixed_total=True)

    def check_runner_branches(self, *, fixed_total):
        from UNIV_adaptor import UniversalAction, resolve_schedule
        from UNIV_adaptor.flow import wan_clean_from_velocity, wan_renoise
        from UNIV_adaptor.transition import WanDVGAnchorTransition

        def tensor_digest(value):
            return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

        namespace = {
            "torch": torch, "copy": copy, "hashlib": hashlib, "time": time,
            "Path": Path, "json": json, "AI_DEVICE": "cpu",
            "GET_DTYPE": lambda: torch.float32,
            "logger": SimpleNamespace(info=lambda *a, **k: None),
            "RUNNER_REGISTER": lambda name: lambda cls: cls,
            "WanRunner": object,
            "wan_clean_from_velocity": wan_clean_from_velocity, "wan_renoise": wan_renoise,
            "install_hr_grid": install_hr_grid, "tensor_sha256": tensor_digest,
            "synchronize": lambda tensor: None,
        }
        base = load_source_class(REPO_ROOT / "UNIV_adaptor/wan_runner.py", "WanUniversalRGBPipelineRunner", namespace)
        namespace["WanUniversalRGBPipelineRunner"] = base
        cls = load_source_class(REPO_ROOT / "UNIV_adaptor/hr_ablation_runner.py", "WanHRRefinementAblationRunner", namespace)
        runner = object.__new__(cls)
        runner.shared_boundary = runner.shared_identity = runner.shared_record = None
        runner.progress_callback = None
        runner.check_stop = lambda: None
        runner.video_segment_num = 1
        runner._univ_transition = WanDVGAnchorTransition()
        calls = []
        action = UniversalAction(.5, .5, 1.0, .8)
        with tempfile.TemporaryDirectory() as directory:
            runner.config = {
                "univ_action": dict(spatial_ratio=.5, temporal_ratio=.5, lr_nfe_ratio=1.0, switch_ratio=.8),
                "univ_hr_boundary_path": str(Path(directory) / "boundary.pt"),
                "univ_enable_transition_diagnostics": False,
            }
            for count in (10, 6, 4, 2):
                scheduler = cpu_scheduler()
                if fixed_total:
                    action = UniversalAction(.5, .5, 1.0, (50 - count) / 50)
                scheduler.univ_schedule = resolve_schedule(action, reference_nfe=50, target_latent_shape=(1, 3, 4, 4))
                scheduler.univ_seed = 42
                scheduler.univ_hr_noise = torch.zeros((1, 3, 4, 4))
                runner.model = SimpleNamespace(scheduler=scheduler, config={})
                runner.scheduler = scheduler
                reference = scheduler.sigmas.clone()
                if fixed_total:
                    configure_fixed_total_case(runner, {
                        "config": {"univ_action": dict(spatial_ratio=.5, temporal_ratio=.5,
                                                        lr_nfe_ratio=1., switch_ratio=(50-count)/50)},
                        "boundary_path": str(Path(directory) / f"boundary{count}.pt"),
                    })
                def infer(inputs):
                    calls.append(scheduler.step_index)
                    scheduler.noise_pred = .15 * scheduler.latents + .05 * torch.sin(3 * scheduler.sigmas[scheduler.step_index])
                runner.model.infer = infer
                runner.inputs = {}
                runner.input_info = SimpleNamespace(prompt="fox", negative_prompt="", seed=42,
                                                   save_result_path=str(Path(directory) / f"HR{count}.mp4"))
                runner.hr_steps = count
                before = len(calls)
                result = runner.run_segment(None)
                self.assertEqual(len(calls) - before, 50 if fixed_total or count == 10 else count)
                self.assertEqual(scheduler.infer_steps, 50)
                self.assertEqual(tuple(result.shape), (1, 3, 4, 4))
                self.assertEqual(runner.univ_runtime_record["hr_schedule"]["hr_steps"], count)
                self.assertEqual(runner.univ_runtime_record["shared_boundary"]["reused"], not fixed_total and count != 10)
                if fixed_total:
                    self.assertEqual(calls[before:], list(range(50)))
                    self.assertTrue(torch.equal(scheduler.sigmas, reference))
                    self.assertEqual(runner.hr_grid["boundary_step"], 50 - count)
                self.assertEqual(tensor_digest(runner.shared_boundary), runner.boundary_sha256)
            saved = torch.load(Path(runner.config["univ_hr_boundary_path"]), weights_only=True)
            self.assertEqual(tensor_digest(saved["state"]), runner.boundary_sha256)
            runner.input_info.prompt = "different prompt"
            with self.assertRaisesRegex(ValueError, "different request"):
                runner.run_segment(None)


if __name__ == "__main__":
    unittest.main()
