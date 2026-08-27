from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from changing_resolution_uni.scripts.data.build_oracle_trajectory_dataset import (
    parse_sample_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
VERIFY_SCRIPT = (
    REPO_ROOT
    / "changing_resolution_uni"
    / "scripts"
    / "data"
    / "verify_oracle_resume_task.py"
)


class OracleResumeTaskTest(unittest.TestCase):
    def test_verifier_uses_derived_sample_seed_and_writes_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "part_00"
            prompt_id = 5
            base_seed = 42
            sample_id = f"{prompt_id:04d}_seed{base_seed + prompt_id}"
            seed_root = part / "raw_samples" / f"seed_{base_seed}"
            paths = [
                seed_root / "videos" / "step30" / f"{sample_id}_step30.mp4",
                seed_root / "latents" / "step30" / f"{sample_id}_step30.pt",
                seed_root / "videos" / "native_hr" / f"{sample_id}_native_hr.mp4",
            ]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"artifact")
            manifest = seed_root / "manifests" / f"{sample_id}.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "branches": [{"candidate_step": 30}],
                        "native_hr": {"output": str(paths[2])},
                    }
                ),
                encoding="utf-8",
            )
            record = part / "records" / f"p{prompt_id:06d}_s{base_seed}.json"
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(
                json.dumps(
                    {
                        "prompt_id": prompt_id,
                        "seed": base_seed,
                        "manifest": {"branches": [{"candidate_step": 30}]},
                    }
                ),
                encoding="utf-8",
            )
            marker = Path(temporary) / "task.done.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(VERIFY_SCRIPT),
                    "--part-root",
                    str(part),
                    "--prompt-offset",
                    str(prompt_id),
                    "--limit",
                    "1",
                    "--seeds",
                    str(base_seed),
                    "--candidate-steps",
                    "30",
                    "--include-native-hr",
                    "1",
                    "--require-latents",
                    "1",
                    "--marker",
                    str(marker),
                    "--quiet",
                ],
                check=False,
            )
            self.assertEqual(result.returncode, 0)
            self.assertTrue(json.loads(marker.read_text(encoding="utf-8"))["complete"])

    def test_parse_manifest_uses_base_seed_plus_prompt_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt_id = 1003
            base_seed = 100
            manifest = (
                root
                / "raw_samples"
                / f"seed_{base_seed}"
                / "manifests"
                / f"{prompt_id:04d}_seed{base_seed + prompt_id}.json"
            )
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps({"schema": "wan_taa_free_oracle_v1", "branches": []}),
                encoding="utf-8",
            )
            args = SimpleNamespace(candidate_steps=[30, 50], primary_lambda=0.01)
            record = parse_sample_manifest(
                args=args,
                out_root=root,
                prompt_id=prompt_id,
                prompt_text="prompt",
                seed=base_seed,
                batch_metrics={},
            )
            self.assertIn("manifest", record)
            self.assertEqual(record["seed"], base_seed)


if __name__ == "__main__":
    unittest.main()
