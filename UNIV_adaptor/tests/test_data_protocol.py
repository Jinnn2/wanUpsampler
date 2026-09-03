from __future__ import annotations

import copy
import unittest

from UNIV_adaptor.data_protocol import (
    PLAN_SCHEMA,
    PROTOCOL_SCHEMA,
    action_key,
    build_collection_plan,
    build_probe_selection_plan,
    diverse_candidates,
    enumerate_action_pool,
    proxy_compute_density,
    validate_collection_plan,
    validate_protocol,
    validate_trajectory_record,
)


def protocol() -> dict:
    return {
        "schema": PROTOCOL_SCHEMA,
        "reference_nfe": 50,
        "target_latent_shape": [16, 21, 90, 156],
        "split_seed": 42,
        "transitions": ["dvg_latent_anchor"],
        "action_space": {
            "spatial_ratios": [0.5, 0.75, 1.0],
            "temporal_ratios": [0.5, 1.0],
            "lr_nfe_ratios": [0.5, 1.0],
            "switch_ratios": [0.6, 0.8, 1.0],
        },
        "budgets": {
            "target_densities": [0.3, 0.5, 0.7],
            "proxy_tolerance": 0.05,
        },
        "common_probe": {
            "selection_status": "selected",
            "selected": {
                "id": "probe",
                "spatial_ratio": 0.5,
                "temporal_ratio": 0.5,
                "stop_step": 10,
                "full_compute_steps": 4,
            },
            "candidates": [
                {
                    "id": "probe",
                    "spatial_ratio": 0.5,
                    "temporal_ratio": 0.5,
                    "stop_step": 10,
                    "full_compute_steps": 4,
                }
            ],
        },
        "probe_selection": {
            "prompt_offset": 0,
            "prompt_count": 2,
            "base_seeds": [42, 100],
            "downstream_candidate_count": 4,
        },
        "splits": [
            {
                "name": "train",
                "prompt_count": 3,
                "base_seeds": [42],
                "collection_mode": "sparse_train",
                "candidate_count": 2,
            },
            {
                "name": "validation",
                "prompt_count": 1,
                "base_seeds": [42, 100],
                "collection_mode": "dense_oracle",
                "candidate_count": 3,
            },
        ],
    }


class DataProtocolTest(unittest.TestCase):
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
        )
        self.assertAlmostEqual(proxy_compute_density(full.action), 1.0)
        self.assertEqual(full.key, action_key(full.action, full.transition))

    def test_pending_probe_blocks_controller_plan(self):
        value = protocol()
        value["common_probe"]["selected"] = None
        validate_protocol(value, require_selected_probe=False)
        with self.assertRaisesRegex(ValueError, "common_probe.selected"):
            build_collection_plan(value, ["a", "b", "c", "d"])

    def test_plan_is_deterministic_and_has_deferred_dvg_slot(self):
        value = protocol()
        prompts = ["first", "second", "third", "fourth"]
        first = build_collection_plan(value, prompts)
        second = build_collection_plan(value, prompts)
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], PLAN_SCHEMA)
        self.assertEqual(len(first["assignments"]), 5)
        train = [row for row in first["assignments"] if row["split"] == "train"]
        self.assertEqual(len(train), 3)
        for row in train:
            self.assertEqual(len(row["candidate_slots"]), 2)
            self.assertEqual(row["candidate_slots"][0]["selector"], "dvg_runtime")
            self.assertEqual(
                row["candidate_slots"][0]["selection_status"],
                "deferred_until_common_probe",
            )
        validate_collection_plan(first)

    def test_plan_hash_detects_mutation(self):
        plan = build_collection_plan(protocol(), ["a", "b", "c", "d"])
        changed = copy.deepcopy(plan)
        changed["assignments"][0]["prompt"] = "changed"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_collection_plan(changed)

    def test_probe_plan_uses_same_downstream_actions_for_each_probe(self):
        value = protocol()
        value["common_probe"]["selected"] = None
        value["common_probe"]["candidates"].append(
            {
                "id": "probe_late",
                "spatial_ratio": 0.5,
                "temporal_ratio": 0.5,
                "stop_step": 15,
                "full_compute_steps": 4,
            }
        )
        plan = build_probe_selection_plan(value, ["a", "b", "c", "d"])
        self.assertEqual(plan["schema"], "univ_probe_selection_plan_v1")
        self.assertEqual(len(plan["assignments"]), 4)
        for row in plan["assignments"]:
            branches = row["probe_branches"]
            self.assertEqual(len(branches), 2)
            first = [item["action_key"] for item in branches[0]["downstream_candidates"]]
            second = [item["action_key"] for item in branches[1]["downstream_candidates"]]
            self.assertEqual(first, second)
        validate_collection_plan(plan)

    def test_diverse_selection_is_repeatable_and_unique(self):
        pool = enumerate_action_pool(protocol())
        first = diverse_candidates(pool, count=8, seed=7)
        second = diverse_candidates(pool, count=8, seed=7)
        self.assertEqual([item.key for item in first], [item.key for item in second])
        self.assertEqual(len({item.key for item in first}), 8)

    def test_completed_record_requires_quality_cost_and_propensity(self):
        plan = build_collection_plan(protocol(), ["a", "b", "c", "d"])
        quality = {
            "subject_consistency": 0.8,
            "background_consistency": 0.8,
            "motion_smoothness": 0.8,
            "aesthetic_quality": 0.7,
            "imaging_quality": 0.7,
            "native_fidelity": 0.9,
        }
        artifact = {
            "video_path": "/data/video.mp4",
            "video_sha256": "a" * 64,
            "quality": quality,
            "cost": {"warm_pipeline_seconds": 10.0},
        }
        candidates = []
        for candidate in diverse_candidates(enumerate_action_pool(protocol()), count=2, seed=9):
            payload = candidate.as_dict(
                reference_nfe=50,
                target_latent_shape=(16, 21, 90, 156),
            )
            payload.update(artifact)
            payload["selection"] = {"source": "test", "propensity": 0.5}
            candidates.append(payload)
        record = {
            "schema": "univ_sparse_trajectory_record_v1",
            "plan_sha256": plan["plan_sha256"],
            "trajectory_key": "train_p000000_s42_b0",
            "split": "train",
            "prompt_id": 0,
            "prompt": "a",
            "seed": 42,
            "common_probe": {
                "id": "probe",
                "boundary_step": 10,
                "boundary_sigma": 0.8,
                "feature_path": "/data/probe.npz",
            },
            "native_teacher": artifact,
            "candidates": candidates,
            "provenance": {
                "code_commit": "abc",
                "model_sha256": "b" * 64,
                "config_sha256": "c" * 64,
            },
        }
        validate_trajectory_record(
            record,
            expected_plan_sha256=plan["plan_sha256"],
        )
        changed = copy.deepcopy(record)
        changed["candidates"][0]["selection"]["propensity"] = 0.0
        with self.assertRaisesRegex(ValueError, "propensity"):
            validate_trajectory_record(changed)


if __name__ == "__main__":
    unittest.main()
