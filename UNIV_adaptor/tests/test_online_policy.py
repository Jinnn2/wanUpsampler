from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from UNIV_adaptor.data_protocol import write_json_atomic
from UNIV_adaptor.online_policy import (
    bootstrap_oracle_gain_ci,
    full_compute_steps,
    suffix_compute_steps,
    validate_spec,
)
from UNIV_adaptor.scripts.validation.run_online_policy_existence import (
    analyze_payload,
    expected_video,
    prepare,
    write_analysis_reports,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "UNIV_adaptor/configs/univ_online_policy_existence.json"
TEMPLATE_PATH = REPO_ROOT / "UNIV_adaptor/configs/univ_mrflow_refinement_ablation.json"


class OnlinePolicyProtocolTest(unittest.TestCase):
    def test_oracle_bootstrap_reselects_fixed_action(self):
        rows = [
            {"a": 1.0, "b": 0.0},
            {"a": 0.0, "b": 1.0},
        ]
        lo, hi = bootstrap_oracle_gain_ci(rows, ["a", "b"], repetitions=500)
        self.assertEqual(lo, 0.0)
        self.assertEqual(hi, 0.5)

    def test_exact_suffix_budget_preserves_boundaries(self):
        self.assertEqual(
            suffix_compute_steps(
                decision_step=12, reference_nfe=50, remaining_full_compute=1
            ),
            (49,),
        )
        sparse = full_compute_steps(
            decision_step=12, reference_nfe=50, remaining_full_compute=8
        )
        self.assertEqual(len(sparse), 20)
        self.assertEqual(sparse[:12], tuple(range(12)))
        self.assertEqual(sparse[12], 12)
        self.assertEqual(sparse[-1], 49)
        dense = full_compute_steps(
            decision_step=12, reference_nfe=50, remaining_full_compute=38
        )
        self.assertEqual(dense, tuple(range(50)))

    def test_spec_is_complete_three_by_two_matrix_plus_two_restarts(self):
        spec = validate_spec(json.loads(SPEC_PATH.read_text(encoding="utf-8")))
        continues = [case for case in spec["cases"] if case["role"] == "continue"]
        restarts = [case for case in spec["cases"] if case["role"] == "restart"]
        self.assertEqual(len(continues), 6)
        self.assertEqual(len(restarts), 2)
        self.assertEqual(
            {
                (case["remaining_lr_full_compute"], case["hr_steps"])
                for case in continues
            },
            {(lr, hr) for lr in (8, 20, 38) for hr in (0, 4)},
        )

    def test_spec_rejects_broken_counterfactual_layout(self):
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        mismatched_prefix = copy.deepcopy(spec)
        mismatched_prefix["cases"][0]["initial_action"]["spatial_ratio"] = 0.6
        with self.assertRaisesRegex(ValueError, "share one initial action"):
            validate_spec(mismatched_prefix)

        mismatched_suffix = copy.deepcopy(spec)
        mismatched_suffix["cases"][-1]["hr_refine_sigma"] = 0.2
        with self.assertRaisesRegex(ValueError, "matched LR/HR suffix"):
            validate_spec(mismatched_suffix)

    def test_prepare_materializes_eight_immutable_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.txt"
            prompts.write_text(
                "\n".join(f"prompt {i}" for i in range(5)), encoding="utf-8"
            )
            args = SimpleNamespace(
                spec=str(SPEC_PATH),
                template_config=str(TEMPLATE_PATH),
                prompts=str(prompts),
                out_root=str(root / "out"),
                model_root=str(root / "model"),
                lightx2v_repo=str(root / "lightx2v"),
                prompt_offset=1,
                limit=3,
                timing_warmup=1,
                seed=9700,
            )
            manifest = prepare(args)
            self.assertEqual(len(manifest["cases"]), 8)
            self.assertEqual(
                manifest["selected_prompts"], ["prompt 1", "prompt 2", "prompt 3"]
            )
            for case in manifest["cases"]:
                config = json.loads(
                    Path(case["config_path"]).read_text(encoding="utf-8")
                )
                self.assertEqual(config["univ_online_case_id"], case["name"])
                self.assertEqual(config["univ_online_decision_step"], 12)
            ratios_by_group = {
                case["initial_group"]: case["low_latent_token_ratio"]
                for case in manifest["cases"]
            }
            self.assertEqual(len(ratios_by_group), 3)
            balanced_ratio = ratios_by_group["balanced"]
            for ratio in ratios_by_group.values():
                self.assertLessEqual(abs(ratio / balanced_ratio - 1.0), 0.05)
            self.assertEqual(
                prepare(args)["manifest_sha256"], manifest["manifest_sha256"]
            )


class OnlinePolicySyntheticAnalysisTest(unittest.TestCase):
    def test_oracle_analysis_detects_lr_hr_and_restart_headroom(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = validate_spec(json.loads(SPEC_PATH.read_text(encoding="utf-8")))
            manifest = {
                "schema": "univ_online_policy_generation_manifest_v1",
                "manifest_sha256": "manifest-test",
                "spec": spec,
                "out_root": str(root),
                "prompt_offset": 0,
                "prompt_count": 16,
                "selected_prompts": [f"prompt {index}" for index in range(16)],
                "timing_warmup": 0,
                "seed_base": 100,
                "cases": [
                    {
                        **case,
                        "low_latent_shape": [16, 14, 56, 96],
                        "low_latent_token_ratio": (
                            case["initial_action"]["spatial_ratio"] ** 2
                            * case["initial_action"]["temporal_ratio"]
                        ),
                    }
                    for case in spec["cases"]
                ],
            }
            scores = {
                "schema": "univ_online_policy_vbench_v1",
                "manifest_sha256": "manifest-test",
                "cases": {},
            }
            winners = [
                "bal_lr20_hr04",
                "bal_lr20_hr04",
                "bal_lr20_hr04",
                "bal_lr20_hr04",
                "restart_spatial_lr20_hr04",
                "restart_spatial_lr20_hr04",
                "restart_spatial_lr20_hr04",
                "restart_spatial_lr20_hr04",
                "restart_temporal_lr20_hr04",
                "restart_temporal_lr20_hr04",
                "restart_temporal_lr20_hr04",
                "restart_temporal_lr20_hr04",
                "bal_lr08_hr00",
                "bal_lr08_hr00",
                "bal_lr38_hr00",
                "bal_lr38_hr00",
            ]
            for case in manifest["cases"]:
                name = case["name"]
                timing_path = root / "timings" / f"{name}.jsonl"
                timing_path.parent.mkdir(parents=True, exist_ok=True)
                timing = [{"kind": "initialization", "case": name}]
                per_video = {}
                for index in range(16):
                    video = expected_video(manifest, case, index)
                    video.parent.mkdir(parents=True, exist_ok=True)
                    video.write_bytes(b"v" * 2048)
                    endpoint = video.with_suffix(video.suffix + ".endpoint.pt")
                    endpoint.write_bytes(b"e" * 2048)
                    quality = 0.9 if winners[index] == name else 0.5
                    dimensions = {
                        key: quality
                        for key in (
                            "subject_consistency",
                            "background_consistency",
                            "motion_smoothness",
                            "aesthetic_quality",
                            "imaging_quality",
                        )
                    }
                    dimensions["dynamic_degree"] = 0.5
                    dimensions["overall_consistency"] = 0.5
                    per_video[video.stem] = dimensions
                    lr = case["remaining_lr_full_compute"]
                    runtime = {
                        "schema": "wan_univ_online_policy_existence_v1",
                        "artifact_id": name,
                        "seed": 100 + index,
                        "online_decision": {
                            "schema": "univ_online_decision_observation_v1",
                            "case_id": name,
                            "case_role": case["role"],
                            "initial_group": case["initial_group"],
                            "decision_step": 12,
                            "remaining_lr_full_compute": lr,
                            "remaining_lr_compute_steps": [
                                step
                                for step in full_compute_steps(
                                    decision_step=12,
                                    reference_nfe=50,
                                    remaining_full_compute=lr,
                                )
                                if step >= 12
                            ],
                            "state_sha256": "a" * 64
                            if case["role"] == "continue"
                            else "c" * 64,
                            "predicted_clean_sha256": "b" * 64
                            if case["role"] == "continue"
                            else "d" * 64,
                        },
                        "endpoint_state": {
                            "schema": "univ_mrflow_clean_transition_v1",
                            "path": str(endpoint),
                        },
                        "lr_endpoint": {
                            "clean_lr_sha256": (
                                f"{lr:064x}" if case["role"] == "continue" else "f" * 64
                            )
                        },
                        "timing_seconds": {
                            "decision_prefix_lr": 1.0,
                            "online_observation": 0.1,
                            "candidate_denoise": 4.0,
                            "remaining_lr": 2.0,
                            "transition": 0.5,
                            "hr_full_compute": 0.5,
                        },
                    }
                    write_json_atomic(
                        video.with_suffix(video.suffix + ".univ.json"), runtime
                    )
                    timing.append(
                        {
                            "kind": "video",
                            "case": name,
                            "prompt_index": index,
                            "segment_elapsed_s": 5.0,
                            "output": str(video),
                        }
                    )
                timing_path.write_text(
                    "\n".join(json.dumps(row) for row in timing) + "\n",
                    encoding="utf-8",
                )
                scores["cases"][name] = {"per_video": per_video}
            payload = analyze_payload(manifest, scores, bootstrap_repetitions=200)
            result = next(
                row for row in payload["lambda_results"] if row["lambda"] == 0.0
            )
            self.assertTrue(result["initial_strategy_exists_in_sampled_space"])
            self.assertTrue(result["online_strategy_exists_in_sampled_space"])
            self.assertTrue(result["restart_strategy_exists_in_sampled_space"])
            self.assertGreater(result["single_attempt_gain_vs_fixed_global"], 0)
            self.assertGreater(result["restart_incremental_gain_vs_online_oracle"], 0)
            self.assertGreater(result["restart_incremental_gain_ci95"][0], 0)
            write_analysis_reports(manifest, payload)
            report = (root / "reports/POLICY_EXISTENCE.md").read_text(encoding="utf-8")
            self.assertIn("initial-strategy gain", report)
            self.assertIn("online LR+HR gain", report)
            self.assertIn("restart-only increment", report)

            for case in manifest["cases"]:
                if case["role"] != "restart":
                    continue
                for position, dimensions in enumerate(
                    scores["cases"][case["name"]]["per_video"].values()
                ):
                    quality = (
                        0.9
                        if case["name"] == "restart_spatial_lr20_hr04" and position == 4
                        else 0.1
                    )
                    for key in (
                        "subject_consistency",
                        "background_consistency",
                        "motion_smoothness",
                        "aesthetic_quality",
                        "imaging_quality",
                    ):
                        dimensions[key] = quality
            sparse_restart = analyze_payload(
                manifest, scores, bootstrap_repetitions=200
            )
            sparse_result = next(
                row for row in sparse_restart["lambda_results"] if row["lambda"] == 0.0
            )
            self.assertTrue(sparse_result["online_strategy_exists_in_sampled_space"])
            self.assertEqual(sparse_result["restart_incremental_gain_ci95"][0], 0.0)
            self.assertFalse(sparse_result["restart_strategy_exists_in_sampled_space"])


if __name__ == "__main__":
    unittest.main()
