from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from paper.aaai27.experiments.refresh_distill4_p0_results import (
    P0_CASE,
    merge_vbench,
    merge_warm,
)


QUALITY5 = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
)


class RefreshDistill4P0ResultsTest(unittest.TestCase):
    def test_merges_only_p0_vbench_case_and_updates_spec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics").mkdir(parents=True)
            canonical = self._vbench(("native_hr4", P0_CASE), 0.5)
            partial = self._vbench((P0_CASE,), 0.9)
            (root / "metrics/vbench_v1_custom.json").write_text(
                json.dumps(canonical), encoding="utf-8"
            )
            partial_path = root / "metrics/partial.json"
            partial_path.write_text(json.dumps(partial), encoding="utf-8")
            (root / "benchmark_spec.json").write_text(
                json.dumps(
                    {
                        "cases": [
                            {"name": "native_hr4"},
                            {"name": P0_CASE},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            merge_vbench(root, partial_path)
            merged = json.loads(
                (root / "metrics/vbench_v1_custom.json").read_text(
                    encoding="utf-8"
                )
            )
            spec = json.loads(
                (root / "benchmark_spec.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                merged["cases"]["native_hr4"]["numeric_metrics"][
                    "result.subject_consistency.0"
                ],
                0.5,
            )
            self.assertEqual(
                merged["cases"][P0_CASE]["numeric_metrics"][
                    "result.subject_consistency.0"
                ],
                0.9,
            )
            self.assertAlmostEqual(spec["cases"][1]["quality_value"], 0.9)
            self.assertTrue((root / "metrics/history").is_dir())

    def test_merges_p0_warm_rows_and_recomputes_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "warm_quality_efficiency"
            partial = root / "partial"
            for base in (canonical, partial):
                for subdir in ("raw", "resources", "configs"):
                    (base / subdir).mkdir(parents=True, exist_ok=True)
            (root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "cases": [{"name": "native_hr4"}, {"name": P0_CASE}],
                        "analysis_pairs": [
                            {
                                "comparison": "native_vs_rgb",
                                "left_case": "native_hr4",
                                "right_case": P0_CASE,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary_fields = [
                "case",
                "display_name",
                "pipeline_mean_s",
                "denoise_mean_s",
                "speedup_vs_native",
                "latency_reduction_vs_native_pct",
            ]
            self._write_csv(
                canonical / "quality_efficiency_warm.csv",
                summary_fields,
                [
                    self._summary("native_hr4", 10.0, 8.0),
                    self._summary(P0_CASE, 12.0, 9.0),
                ],
            )
            raw_fields = [
                "case",
                "phase",
                "repeat",
                "prompt_index",
                "seed",
                "pipeline_elapsed_s",
                "denoise_elapsed_s",
            ]
            self._write_csv(
                canonical / "quality_efficiency_warm_raw.csv",
                raw_fields,
                [
                    self._raw("native_hr4", 10.0, 8.0),
                    self._raw(P0_CASE, 12.0, 9.0),
                ],
            )
            self._write_csv(
                canonical / "quality_efficiency_warm_pairs.csv",
                ["comparison"],
                [{"comparison": "old"}],
            )
            self._write_csv(
                partial / "quality_efficiency_warm.csv",
                summary_fields,
                [self._summary(P0_CASE, 8.0, 6.0)],
            )
            self._write_csv(
                partial / "quality_efficiency_warm_raw.csv",
                raw_fields,
                [self._raw(P0_CASE, 8.0, 6.0)],
            )
            for base in (canonical, partial):
                (base / "raw" / f"{P0_CASE}.jsonl").write_text(
                    "{}\n", encoding="utf-8"
                )
                (base / "resources" / f"{P0_CASE}.json").write_text(
                    "{}", encoding="utf-8"
                )
                (base / "configs" / f"{P0_CASE}.json").write_text(
                    "{}", encoding="utf-8"
                )
            canonical_protocol = {
                "cases": ["native_hr4", P0_CASE],
                "config_sha256": {P0_CASE: "old"},
                "implementation_sha256": {"code": "old"},
                "source_manifest_sha256": "old",
                "source_spec_sha256": "old",
                "run_signature": "old",
            }
            partial_protocol = {
                **canonical_protocol,
                "cases": [P0_CASE],
                "config_sha256": {P0_CASE: "new"},
                "implementation_sha256": {"code": "new"},
                "source_manifest_sha256": "new",
                "source_spec_sha256": "new",
                "run_signature": "partial",
            }
            (canonical / "protocol.json").write_text(
                json.dumps(canonical_protocol), encoding="utf-8"
            )
            (partial / "protocol.json").write_text(
                json.dumps(partial_protocol), encoding="utf-8"
            )
            (canonical / "warm_timing_manifest.json").write_text(
                json.dumps({"settings": canonical_protocol}), encoding="utf-8"
            )
            (partial / "warm_timing_manifest.json").write_text(
                json.dumps({"settings": partial_protocol}), encoding="utf-8"
            )
            merge_warm(root, partial)
            rows = self._read_csv(canonical / "quality_efficiency_warm.csv")
            by_case = {row["case"]: row for row in rows}
            pairs = self._read_csv(
                canonical / "quality_efficiency_warm_pairs.csv"
            )
            self.assertEqual(float(by_case[P0_CASE]["pipeline_mean_s"]), 8.0)
            self.assertEqual(float(by_case[P0_CASE]["speedup_vs_native"]), 1.25)
            self.assertEqual(float(pairs[0]["pipeline_delta_mean_s"]), -2.0)
            self.assertTrue((canonical / "history").is_dir())

    @staticmethod
    def _vbench(cases: tuple[str, ...], score: float) -> dict:
        return {
            "dimensions": list(QUALITY5),
            "cases": {
                case: {
                    "numeric_metrics": {
                        f"result.{dimension}.0": score
                        for dimension in QUALITY5
                    }
                }
                for case in cases
            },
        }

    @staticmethod
    def _summary(case: str, pipeline: float, denoise: float) -> dict:
        return {
            "case": case,
            "display_name": case,
            "pipeline_mean_s": pipeline,
            "denoise_mean_s": denoise,
            "speedup_vs_native": "",
            "latency_reduction_vs_native_pct": "",
        }

    @staticmethod
    def _raw(case: str, pipeline: float, denoise: float) -> dict:
        return {
            "case": case,
            "phase": "measured",
            "repeat": 0,
            "prompt_index": 0,
            "seed": 15000,
            "pipeline_elapsed_s": pipeline,
            "denoise_elapsed_s": denoise,
        }

    @staticmethod
    def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
