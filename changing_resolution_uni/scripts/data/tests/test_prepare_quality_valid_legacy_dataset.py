from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from changing_resolution_uni.scripts.data.prepare_quality_valid_legacy_dataset import (
    FORMAL_STEPS,
    QUALITY5_DIMENSIONS,
    canonicalize_record,
)


def make_record(*, fallback: bool = False, bad_latency: bool = False) -> dict:
    native_dimensions = {name: 0.9 for name in QUALITY5_DIMENSIONS}
    candidates = []
    for index, step in enumerate(FORMAL_STEPS):
        value = 0.9 if fallback else 0.88 + index * 0.001
        latency = 100.0 - index
        if bad_latency and step == 50:
            latency = 200.0
        candidates.append(
            {
                "step": step,
                "vbench5": value,
                "dimensions": {name: value for name in QUALITY5_DIMENSIONS},
                "latency_seconds": latency,
            }
        )
    return {
        "prompt_id": 3,
        "seed": 45,
        "prompt_text": "a test prompt",
        "native_vbench5": 0.9,
        "native_dimensions": native_dimensions,
        "native_latency_seconds": 189.0,
        "candidates": candidates,
    }


class PrepareQualityValidLegacyDatasetTest(unittest.TestCase):
    def canonicalize(self, record: dict) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "record.json"
            source.write_text("{}", encoding="utf-8")
            return canonicalize_record(
                record,
                source_path=source,
                primary_lambda=0.01,
                quality_mean_tolerance=1e-5,
                equality_tolerance=1e-12,
                latency_monotonic_tolerance=1e-9,
            )

    def test_valid_record_is_recomputed_from_dimensions(self) -> None:
        canonical = self.canonicalize(make_record())
        self.assertEqual(len(canonical["candidates"]), len(FORMAL_STEPS))
        self.assertEqual(
            canonical["candidates"][0]["vbench5"],
            sum(canonical["candidates"][0]["dimensions"].values())
            / len(QUALITY5_DIMENSIONS),
        )
        self.assertEqual(
            canonical["candidates"][0]["latency_source"],
            "legacy_branch_estimate",
        )

    def test_native_fallback_signature_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "native_hr_quality_fallback_signature"):
            self.canonicalize(make_record(fallback=True))

    def test_missing_legacy_native_dimensions_remains_explicit(self) -> None:
        record = make_record()
        del record["native_dimensions"]
        canonical = self.canonicalize(record)
        self.assertEqual(canonical["native_dimensions"], {})
        self.assertEqual(
            canonical["native_quality_source"],
            "legacy_scalar_without_native_dimensions",
        )

    def test_nonmonotonic_latency_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_latency_not_monotonic"):
            self.canonicalize(make_record(bad_latency=True))


if __name__ == "__main__":
    unittest.main()
