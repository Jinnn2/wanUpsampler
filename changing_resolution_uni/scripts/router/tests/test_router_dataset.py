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
            with self.assertRaisesRegex(ValueError, "T5 embedding coverage check failed"):
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


if __name__ == "__main__":
    unittest.main()
