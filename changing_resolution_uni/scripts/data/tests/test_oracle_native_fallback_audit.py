from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from changing_resolution_uni.scripts.data import audit_oracle_native_fallback as audit


class OracleNativeFallbackAuditTest(unittest.TestCase):
    def test_detects_exact_native_reuse_and_preserves_step_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            records_dir = root / "records"
            out_dir = Path(directory) / "report"
            records_dir.mkdir(parents=True)

            for prompt_id in (0, 1):
                for seed in (42 + prompt_id, 100 + prompt_id):
                    native_quality = 0.9 + prompt_id * 0.01
                    candidates = []
                    for index, step in enumerate(audit.FORMAL_STEPS):
                        quality = (
                            native_quality
                            if prompt_id == 0
                            else native_quality - 0.02 + index * 0.001
                        )
                        candidates.append(
                            {
                                "step": step,
                                "vbench5": quality,
                                "latency_seconds": 100.0 - index,
                            }
                        )
                    record = {
                        "prompt_id": prompt_id,
                        "seed": seed,
                        "prompt_text": f"prompt {prompt_id}",
                        "native_vbench5": native_quality,
                        "native_latency_seconds": 189.0 if prompt_id == 0 else 200.0,
                        "candidates": candidates,
                    }
                    (records_dir / f"p{prompt_id:06d}_s{seed}.json").write_text(
                        json.dumps(record), encoding="utf-8"
                    )

            argv = [
                "audit_oracle_native_fallback.py",
                "--dataset-dir",
                str(root),
                "--out-dir",
                str(out_dir),
                "--sample-count",
                "2",
            ]
            with mock.patch.object(sys, "argv", argv):
                audit.main()

            report = json.loads(
                (out_dir / "oracle_native_fallback_report.json").read_text(
                    encoding="utf-8"
                )
            )
            summary = report["summary"]
            self.assertEqual(summary["record_count"], 4)
            self.assertEqual(summary["candidate_row_count"], 52)
            self.assertEqual(summary["all_candidates_equal_native_record_count"], 2)
            self.assertEqual(summary["candidate_exact_native_reuse_count"], 26)
            self.assertEqual(summary["native_latency_exact_189_count"], 2)
            self.assertEqual(len(report["per_step"]), len(audit.FORMAL_STEPS))
            self.assertTrue((out_dir / "sampled_prompt_step_table.csv").is_file())


if __name__ == "__main__":
    unittest.main()
