from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from changing_resolution_uni.scripts.router import audit_token_attribution_inputs


class AuditTokenAttributionInputsTest(unittest.TestCase):
    def test_reports_pooled_only_and_complete_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            t5_dir = root / "t5_embeddings"
            out_dir = Path(directory) / "report"
            t5_dir.mkdir(parents=True)
            (root / "dataset_manifest.json").write_text(
                json.dumps(
                    {
                        "record_files": [
                            "p000000_s42.json",
                            "p000001_s43.json",
                        ]
                    }
                ),
                encoding="utf-8",
            )
            np.savez_compressed(
                t5_dir / "prompt_000000.npz",
                pooled_embedding=np.zeros(4096),
                seq_embedding=np.zeros((3, 4096)),
            )
            (t5_dir / "prompt_000000.json").write_text(
                json.dumps({"tokens": ["▁A", "▁scene", "</s>"]}),
                encoding="utf-8",
            )
            np.savez_compressed(
                t5_dir / "prompt_000001.npz",
                pooled_embedding=np.zeros(4096),
            )
            (t5_dir / "prompt_000001.json").write_text(
                json.dumps({"tokens": ["▁Another", "▁scene", "</s>"]}),
                encoding="utf-8",
            )

            argv = [
                "audit",
                "--dataset-dir",
                str(root),
                "--out-dir",
                str(out_dir),
            ]
            with mock.patch.object(sys, "argv", argv):
                audit_token_attribution_inputs.main()
            report = json.loads(
                (out_dir / "t5_attribution_input_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["selected_prompt_count"], 2)
            self.assertEqual(report["complete_prompt_count"], 1)
            self.assertEqual(report["issue_counts"]["missing_seq_embedding"], 1)


if __name__ == "__main__":
    unittest.main()
