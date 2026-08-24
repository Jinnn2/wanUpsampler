from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from changing_resolution_uni.scripts.data.oracle_record_schema import (
    QUALITY5_DIMENSIONS,
    OracleRecordError,
    aggregate_prompt_records,
    validate_scored_record,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from changing_resolution_uni.scripts.router.dataset_router import RouterDataset


STEPS = [40, 45, 50]


def make_record(
    prompt_id: int,
    seed: int,
    qualities: list[float],
    latencies: list[float],
) -> dict:
    cases = {
        case: {
            "request_sha256": "a" * 64,
            "result_sha256": "b" * 64,
            "full_info_sha256": "c" * 64,
            "run_manifest_path": "/strict/run/score_run_manifest.json",
        }
        for case in ["native_hr", *(f"step{step}" for step in STEPS)]
    }
    return {
        "prompt_id": prompt_id,
        "seed": seed,
        "prompt_text": f"prompt {prompt_id}",
        "native_vbench5": 0.95,
        "native_latency_seconds": 200.0,
        "native_dimensions": {name: 0.95 for name in QUALITY5_DIMENSIONS},
        "native_latency_source": "warm_pipeline_seconds",
        "candidates": [
            {
                "step": step,
                "vbench5": quality,
                "latency_seconds": latency,
                "latency_source": "estimated_warm_pipeline_seconds",
                "dimensions": {name: quality for name in QUALITY5_DIMENSIONS},
            }
            for step, quality, latency in zip(STEPS, qualities, latencies)
        ],
        "scoring_provenance": {
            "schema": "strict_vbench5_record_provenance_v1",
            "quality_dimensions": QUALITY5_DIMENSIONS,
            "diagnostic_dimensions": [],
            "quality_aggregation": "arithmetic_mean_raw_vbench5_float64",
            "vbench": {
                "git_commit": "1" * 40,
                "tracked_dirty": False,
                "evaluate_py_sha256": "2" * 64,
            },
            "cases": cases,
        },
    }


class RouterDatasetTest(unittest.TestCase):
    def test_prompt_label_averages_utility_across_seeds(self) -> None:
        records = {
            7: [
                make_record(7, 42, [0.90, 0.91, 0.88], [100.0, 120.0, 80.0]),
                make_record(7, 100, [0.94, 0.88, 0.89], [100.0, 120.0, 80.0]),
            ]
        }
        samples, seeds = aggregate_prompt_records(
            records,
            candidate_steps=STEPS,
            primary_lambda=0.01,
            expected_seeds=[42, 100],
        )

        sample = samples[7]
        self.assertEqual(seeds, [42, 100])
        self.assertEqual(sample["seed_count"], 2)
        self.assertEqual(STEPS[int(np.argmax(sample["utilities"]))], 40)
        self.assertGreater(
            sample["seed_oracle_utility"], float(np.max(sample["utilities"]))
        )

    def test_scored_record_requires_every_quality_dimension(self) -> None:
        record = make_record(1, 42, [0.90, 0.91, 0.92], [100.0, 90.0, 80.0])
        del record["candidates"][0]["dimensions"][QUALITY5_DIMENSIONS[-1]]
        with self.assertRaisesRegex(OracleRecordError, "imaging_quality"):
            validate_scored_record(record, candidate_steps=STEPS)

    def test_formal_record_rejects_unknown_latency_provenance(self) -> None:
        record = make_record(1, 42, [0.90, 0.91, 0.92], [100.0, 90.0, 80.0])
        record["candidates"][0]["latency_source"] = "unknown"
        with self.assertRaisesRegex(OracleRecordError, "not traceable"):
            validate_scored_record(
                record,
                candidate_steps=STEPS,
                require_dimensions=True,
                require_provenance=True,
            )

    def test_formal_record_recomputes_quality_mean(self) -> None:
        record = make_record(1, 42, [0.90, 0.91, 0.92], [100.0, 90.0, 80.0])
        record["candidates"][0]["vbench5"] = 0.95
        with self.assertRaisesRegex(OracleRecordError, "float64 mean"):
            validate_scored_record(
                record,
                candidate_steps=STEPS,
                require_dimensions=True,
                require_provenance=True,
            )

    def test_prompt_offset_seed_policy(self) -> None:
        records = {
            7: [
                make_record(7, 49, [0.90, 0.91, 0.92], [100.0, 90.0, 80.0]),
                make_record(7, 107, [0.91, 0.92, 0.93], [100.0, 90.0, 80.0]),
            ]
        }
        samples, seeds = aggregate_prompt_records(
            records,
            candidate_steps=STEPS,
            primary_lambda=0.01,
            expected_seeds=[42, 100],
            seed_policy="prompt_offset",
        )
        self.assertEqual(seeds, [42, 100])
        self.assertEqual(samples[7]["seeds"], [49, 107])

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_router_dataset_rejects_missing_t5_embedding(self) -> None:
        records = {
            3: [
                make_record(3, 42, [0.90, 0.91, 0.92], [100.0, 90.0, 80.0]),
                make_record(3, 100, [0.91, 0.92, 0.93], [100.0, 90.0, 80.0]),
            ]
        }
        samples, _ = aggregate_prompt_records(
            records,
            candidate_steps=STEPS,
            primary_lambda=0.01,
            expected_seeds=[42, 100],
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError, "T5 embedding coverage check failed"
            ):
                RouterDataset(
                    [samples[3]],
                    Path(directory),
                    STEPS,
                    primary_lambda=0.01,
                )

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_router_dataset_loads_valid_t5_embedding(self) -> None:
        records = {
            3: [
                make_record(3, 42, [0.90, 0.91, 0.92], [100.0, 90.0, 80.0]),
                make_record(3, 100, [0.91, 0.92, 0.93], [100.0, 90.0, 80.0]),
            ]
        }
        samples, _ = aggregate_prompt_records(
            records,
            candidate_steps=STEPS,
            primary_lambda=0.01,
            expected_seeds=[42, 100],
        )
        with tempfile.TemporaryDirectory() as directory:
            t5_dir = Path(directory)
            np.savez_compressed(
                t5_dir / "prompt_000003.npz",
                pooled_embedding=np.ones(4096, dtype=np.float16),
            )
            dataset = RouterDataset(
                [samples[3]],
                t5_dir,
                STEPS,
                primary_lambda=0.01,
            )
            item = dataset[0]
            self.assertEqual(tuple(item["pooled_t5"].shape), (4096,))
            self.assertEqual(item["seed_count"], 2)
            expected_quality = item["vbench5"] - item["vbench5"][-1]
            np.testing.assert_allclose(
                item["relative_quality_target"].numpy(),
                expected_quality.numpy(),
            )
            self.assertEqual(float(item["relative_quality_target"][-1]), 0.0)


if __name__ == "__main__":
    unittest.main()
