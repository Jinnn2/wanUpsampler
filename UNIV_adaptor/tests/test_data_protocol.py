from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from UNIV_adaptor.data_protocol import (
    PLAN_SCHEMA,
    PROTOCOL_SCHEMA,
    RECORD_SCHEMA,
    action_key,
    build_collection_plan,
    enumerate_action_pool,
    proxy_compute_density,
    validate_collection_plan,
    validate_protocol,
    validate_trajectory_record,
)


def protocol() -> dict:
    presets = [
        ("B30", 0.3, 0.50, 0.50, 0.40, 1.00),
        ("B40", 0.4, 0.625, 0.67, 0.55, 0.95),
        ("B50", 0.5, 0.75, 0.67, 0.70, 0.90),
        ("B60", 0.6, 0.875, 0.80, 0.85, 0.85),
        ("B70", 0.7, 1.00, 1.00, 1.00, 0.80),
    ]
    return {
        "schema": PROTOCOL_SCHEMA,
        "controller_factorization": "prompt_x_budget_quality_curve",
        "observation_mode": "prompt_only",
        "trajectory_origin": "independent_step0",
        "preset_status": "frozen_after_measured_cost",
        "reference_nfe": 50,
        "target_latent_shape": [16, 21, 90, 156],
        "transition": "dvg_latent_anchor",
        "calibration_action_space": {
            "spatial_ratios": [0.5, 0.75, 1.0],
            "temporal_ratios": [0.5, 1.0],
            "lr_nfe_ratios": [0.5, 1.0],
            "switch_ratios": [0.8, 0.9, 1.0],
        },
        "budget_presets": [
            {
                "id": budget_id,
                "target_cost_ratio": target,
                "allocation_source": "test_dvg_proxy",
                "action": {
                    "spatial_ratio": spatial,
                    "temporal_ratio": temporal,
                    "lr_nfe_ratio": nfe,
                    "switch_ratio": switch,
                },
            }
            for budget_id, target, spatial, temporal, nfe, switch in presets
        ],
        "splits": [
            {
                "name": "train",
                "prompt_count": 2,
                "base_seeds": [42],
                "collection_mode": "full_budget_curve",
            },
            {
                "name": "validation",
                "prompt_count": 1,
                "base_seeds": [42, 100],
                "collection_mode": "full_budget_curve",
            },
            {
                "name": "test",
                "prompt_count": 1,
                "base_seeds": [42, 100],
                "collection_mode": "full_budget_curve",
            },
        ],
        "quality_dimensions": [
            "subject_consistency",
            "background_consistency",
            "motion_smoothness",
            "aesthetic_quality",
            "imaging_quality",
            "native_fidelity",
        ],
    }


class DataProtocolTest(unittest.TestCase):
    def test_checked_in_pilot_resolves_to_5400_videos(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "configs/univ_prompt_budget_pilot.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        plan = build_collection_plan(
            value,
            [f"unique pilot prompt {index}" for index in range(500)],
        )
        self.assertEqual(len(plan["assignments"]), 900)
        self.assertEqual(
            sum(len(row["budget_candidates"]) for row in plan["assignments"]),
            4500,
        )
        switches = {
            candidate["requested_action"]["switch_ratio"]
            for candidate in plan["assignments"][0]["budget_candidates"]
        }
        self.assertTrue(switches <= {0.8, 0.9, 1.0})

    def test_action_pool_and_proxy_density(self):
        value = protocol()
        pool = enumerate_action_pool(value)
        self.assertEqual(len(pool), 3 * 2 * 2 * 3)
        self.assertEqual(len({candidate.key for candidate in pool}), len(pool))
        full = next(
            candidate
            for candidate in pool
            if candidate.action.spatial_ratio == 1.0
            and candidate.action.temporal_ratio == 1.0
            and candidate.action.lr_nfe_ratio == 1.0
            and candidate.action.switch_ratio == 1.0
        )
        self.assertAlmostEqual(proxy_compute_density(full.action), 1.0)
        self.assertEqual(full.key, action_key(full.action, full.transition))

    def test_plan_contains_all_five_budgets_for_every_prompt_seed(self):
        prompts = ["first", "second", "third", "fourth"]
        first = build_collection_plan(protocol(), prompts)
        second = build_collection_plan(protocol(), prompts)
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], PLAN_SCHEMA)
        self.assertEqual(len(first["assignments"]), 6)
        for row in first["assignments"]:
            self.assertTrue(row["native_teacher_required"])
            self.assertEqual(
                [item["budget_id"] for item in row["budget_candidates"]],
                ["B30", "B40", "B50", "B60", "B70"],
            )
        validate_collection_plan(first)

    def test_plan_hash_detects_mutation(self):
        plan = build_collection_plan(protocol(), ["a", "b", "c", "d"])
        changed = copy.deepcopy(plan)
        changed["assignments"][0]["prompt"] = "changed"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_collection_plan(changed)

    def test_protocol_rejects_switch_below_point_eight(self):
        value = protocol()
        value["calibration_action_space"]["switch_ratios"][0] = 0.6
        with self.assertRaisesRegex(ValueError, r"\[0.8, 1.0\]"):
            validate_protocol(value)
        value = protocol()
        value["budget_presets"][0]["action"]["switch_ratio"] = 0.7
        with self.assertRaisesRegex(ValueError, r"\[0.8, 1.0\]"):
            validate_protocol(value)

    def test_duplicate_prompts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            build_collection_plan(protocol(), ["a", "b", "c", "a"])

    def test_generated_record_can_be_validated_before_scoring(self):
        plan = build_collection_plan(protocol(), ["a", "b", "c", "d"])
        assignment = plan["assignments"][0]
        artifact = {
            "video_path": "/data/video.mp4",
            "video_sha256": "a" * 64,
            "cost": {"pipeline_seconds": 10.0},
        }
        candidates = [
            {**candidate, **artifact}
            for candidate in assignment["budget_candidates"]
        ]
        record = {
            "schema": RECORD_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "trajectory_key": assignment["trajectory_key"],
            "split": assignment["split"],
            "prompt_id": assignment["prompt_id"],
            "prompt": assignment["prompt"],
            "seed": assignment["seed"],
            "native_teacher": artifact,
            "budget_candidates": candidates,
            "provenance": {"manifest_sha256": "b" * 64},
        }
        validate_trajectory_record(
            record,
            expected_plan_sha256=plan["plan_sha256"],
            require_scores=False,
        )
        with self.assertRaisesRegex(ValueError, "quality"):
            validate_trajectory_record(record, require_scores=True)


if __name__ == "__main__":
    unittest.main()
