from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from UNIV_adaptor.hr_refinement import direct_hr_sigmas, install_direct_hr_grid
from UNIV_adaptor.scripts.validation.run_mrflow_refinement_ablation import (
    CONTROL_ID,
    DEFAULT_HR_STEPS,
    DEFAULT_PROMPT,
    DEFAULT_SIGMAS,
    build_plan,
    case_id,
    parse_hr_steps,
    parse_sigmas,
    prepare,
)

try:
    import torch
except ImportError:
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIGHTX2V = REPO_ROOT.parent / "LightX2V" / "LightX2V"
LIGHTX2V = Path(os.environ.get("LIGHTX2V_REPO", DEFAULT_LIGHTX2V))


def args_for(out_dir):
    return SimpleNamespace(
        template_config=str(
            REPO_ROOT / "UNIV_adaptor/configs/univ_mrflow_refinement_ablation.json"
        ),
        out_dir=str(out_dir),
        model_root="/model",
        prompt=DEFAULT_PROMPT,
        negative_prompt="",
        seed=42,
        sigmas=DEFAULT_SIGMAS,
        hr_steps=DEFAULT_HR_STEPS,
    )


def load_source_class(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class DirectSigmaPlanTest(unittest.TestCase):
    def test_default_plan_has_control_and_full_factorial(self):
        plan = build_plan(args_for("/mrflow"))
        self.assertEqual(plan["reference_schedule"]["switch_step"], 50)
        self.assertEqual(plan["reference_schedule"]["lr_compute_steps"], list(range(50)))
        self.assertEqual(plan["reference_schedule"]["hr_compute_steps"], [])
        self.assertEqual(len(plan["cases"]), 10)
        self.assertEqual(plan["cases"][0]["id"], CONTROL_ID)
        self.assertEqual(plan["cases"][0]["planned_sigmas"], [0.0])
        self.assertEqual(
            {(row["refine_sigma"], row["hr_steps"]) for row in plan["cases"][1:]},
            {(sigma, steps) for sigma in DEFAULT_SIGMAS for steps in DEFAULT_HR_STEPS},
        )

    def test_direct_grids_and_case_ids_are_stable(self):
        self.assertEqual(case_id(.12, 4), "S0120_HR04")
        self.assertEqual(direct_hr_sigmas(start_sigma=.12, hr_steps=4),
                         (.12, .09, .06, .03, 0.0))
        self.assertEqual(parse_sigmas(".12, .2,.3"), DEFAULT_SIGMAS)
        self.assertEqual(parse_hr_steps("1,2,4"), DEFAULT_HR_STEPS)
        for sigma, steps in ((0, 1), (1, 1), (.1, 0)):
            with self.assertRaises(ValueError):
                direct_hr_sigmas(start_sigma=sigma, hr_steps=steps)
        with self.assertRaises(ValueError):
            case_id(.1234, 1)

    def test_plan_is_content_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(directory)
            plan, _ = prepare(args)
            self.assertEqual(build_plan(args), plan)
            args.sigmas = (.12, .25)
            with self.assertRaisesRegex(RuntimeError, "different comparison"):
                prepare(args)


@unittest.skipUnless(torch is not None and LIGHTX2V.is_dir(), "requires Torch and LightX2V")
class DirectSigmaWanSolverTest(unittest.TestCase):
    @staticmethod
    def scheduler():
        import numpy as np

        namespace = {
            "torch": torch,
            "np": np,
            "AI_DEVICE": "cpu",
            "GET_DTYPE": lambda: torch.float32,
            "GET_SENSITIVE_DTYPE": lambda: torch.float32,
            "logger": SimpleNamespace(info=lambda *args, **kwargs: None),
        }
        load_source_class(
            LIGHTX2V / "lightx2v/models/schedulers/scheduler.py", "BaseScheduler", namespace
        )
        cls = load_source_class(
            LIGHTX2V / "lightx2v/models/schedulers/wan/scheduler.py", "WanScheduler", namespace
        )
        scheduler = cls({
            "infer_steps": 50,
            "target_video_length": 81,
            "sample_shift": 8,
            "sample_guide_scale": 6,
            "seq_parallel": False,
            "dim": 12,
            "num_heads": 3,
            "task": "t2v",
            "model_cls": "wan2.1",
        })
        scheduler.prepare(42, (1, 2, 2, 2))
        scheduler.reset_solver_history = lambda: (
            setattr(scheduler, "model_outputs", [None] * scheduler.solver_order),
            setattr(scheduler, "timestep_list", [None] * scheduler.solver_order),
            setattr(scheduler, "last_sample", None),
            setattr(scheduler, "noise_pred", None),
            setattr(scheduler, "this_order", None),
            setattr(scheduler, "lower_order_nums", 0),
        )
        return scheduler

    def test_direct_grid_uses_explicit_sigmas_and_reaches_clean(self):
        for sigma, steps in ((.12, 1), (.2, 2), (.3, 4)):
            scheduler = self.scheduler()
            clean = torch.full_like(scheduler.latents, .25)
            noise = scheduler.latents.clone()
            scheduler.latents = (1 - sigma) * clean + sigma * noise
            grid = install_direct_hr_grid(
                scheduler, start_sigma=sigma, hr_steps=steps
            )
            self.assertEqual(grid["compute_indices"], list(range(steps)))
            self.assertEqual(grid["model_timesteps"][0], int(sigma * 1000))
            for index in grid["compute_indices"]:
                scheduler.step_pre(index)
                scheduler.noise_pred = (scheduler.latents - clean) / scheduler.sigmas[index]
                scheduler.step_post()
            torch.testing.assert_close(scheduler.latents, clean, rtol=1e-5, atol=1e-6)

    def test_runner_completes_lr50_once_then_reuses_clean_transition(self):
        from UNIV_adaptor import UniversalAction, resolve_schedule
        from UNIV_adaptor.flow import wan_renoise
        from UNIV_adaptor.transition import WanDVGAnchorTransition

        def tensor_sha256(value):
            raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            return hashlib.sha256(raw).hexdigest()

        class FakeBase:
            def _build_transition(self, *, spatial_needed):
                return self._univ_transition

            @staticmethod
            def _renoise(clean_hr, noise, sigma):
                return wan_renoise(clean_hr.float(), noise.float(), sigma).to(clean_hr.dtype)

            def _write_runtime_record(self):
                pass

        namespace = {
            "torch": torch,
            "copy": __import__("copy"),
            "time": __import__("time"),
            "Path": Path,
            "logger": SimpleNamespace(info=lambda *args, **kwargs: None),
            "GET_DTYPE": lambda: torch.float32,
            "RUNNER_REGISTER": lambda name: lambda cls: cls,
            "WanUniversalRGBPipelineRunner": FakeBase,
            "install_direct_hr_grid": install_direct_hr_grid,
            "synchronize": lambda tensor: None,
            "tensor_sha256": tensor_sha256,
        }
        cls = load_source_class(
            REPO_ROOT / "UNIV_adaptor/mrflow_ablation_runner.py",
            "WanMrFlowRefinementAblationRunner",
            namespace,
        )
        runner = object.__new__(cls)
        runner.shared_clean_lr = runner.shared_clean_hr = runner.shared_hr_noise = None
        runner.shared_identity = runner.shared_record = None
        runner.progress_callback = None
        runner.check_stop = lambda: None
        runner.video_segment_num = 1
        runner._univ_transition = WanDVGAnchorTransition()
        runner.input_info = SimpleNamespace(prompt="fox", negative_prompt="", seed=42)
        runner.inputs = {}
        calls = []
        action = UniversalAction(.5, .5, 1.0, 1.0)

        with tempfile.TemporaryDirectory() as directory:
            runner.config = {
                "univ_action": dict(spatial_ratio=.5, temporal_ratio=.5,
                                    lr_nfe_ratio=1.0, switch_ratio=1.0),
                "univ_mrflow_boundary_path": str(Path(directory) / "shared.pt"),
            }
            for branch, (sigma, steps) in enumerate(((0.0, 0), (.12, 2), (.12, 4))):
                scheduler = self.scheduler()
                scheduler.univ_schedule = resolve_schedule(
                    action, reference_nfe=50, target_latent_shape=(1, 3, 4, 4)
                )
                scheduler.univ_hr_noise = torch.zeros((1, 3, 4, 4))
                post_calls = []
                original_post = scheduler.step_post

                def step_post():
                    post_calls.append(scheduler.step_index)
                    original_post()

                scheduler.step_post = step_post

                def infer(inputs):
                    calls.append(scheduler.step_index)
                    scheduler.noise_pred = .1 * scheduler.latents

                runner.model = SimpleNamespace(scheduler=scheduler, infer=infer)
                runner.refine_sigma = sigma
                runner.hr_steps = steps
                before = len(calls)
                runner.run_segment(None)
                expected = 50 if branch == 0 else steps
                self.assertEqual(len(calls) - before, expected)
                self.assertEqual(len(post_calls), expected)
                self.assertEqual(runner.univ_runtime_record["shared_clean_hr"]["reused"], branch > 0)
                self.assertEqual(runner.univ_runtime_record["hr_schedule"]["hr_steps"], steps)
                self.assertEqual(scheduler.infer_steps, 50)
            self.assertEqual(post_calls, [0, 1, 2, 3])
            self.assertTrue(Path(runner.config["univ_mrflow_boundary_path"]).is_file())
            try:
                saved = torch.load(
                    runner.config["univ_mrflow_boundary_path"], weights_only=True
                )
            except TypeError:
                saved = torch.load(runner.config["univ_mrflow_boundary_path"])
            self.assertEqual(saved["schema"], "univ_mrflow_clean_transition_v1")
            self.assertEqual(tensor_sha256(saved["clean_hr"]), runner.clean_hr_sha256)
            starts = [
                runner._renoise(runner.shared_clean_hr, runner.shared_hr_noise, .12)
                for _ in range(2)
            ]
            self.assertEqual(tensor_sha256(starts[0]), tensor_sha256(starts[1]))


if __name__ == "__main__":
    unittest.main()
