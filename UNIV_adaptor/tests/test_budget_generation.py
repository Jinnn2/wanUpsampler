from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from UNIV_adaptor.scripts.data.run_prompt_budget_generation import (
    generate_job,
    prepare,
    selected_jobs,
    validate_manifest,
)
from UNIV_adaptor.tests.test_data_protocol import protocol


class PromptBudgetGenerationTest(unittest.TestCase):
    def test_prepare_builds_six_cases_and_balanced_eight_worker_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            prompts_path = root / "prompts.txt"
            template_path = root / "template.json"
            protocol_payload = protocol()
            protocol_payload["preset_status"] = "frozen_for_pilot_cost_calibration"
            protocol_path.write_text(json.dumps(protocol_payload), encoding="utf-8")
            prompts_path.write_text("a\nb\nc\nd\n", encoding="utf-8")
            template_path.write_text(
                json.dumps(
                    {
                        "infer_steps": 50,
                        "target_video_length": 81,
                        "target_height": 720,
                        "target_width": 1248,
                        "feature_caching": "NoCaching",
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                protocol=str(protocol_path),
                prompts=str(prompts_path),
                template_config=str(template_path),
                model_root=str(root / "model"),
                out_root=str(root / "output"),
                job_chunk_size=1,
                worker_count=8,
            )
            manifest = prepare(args)
            validate_manifest(manifest)
            self.assertEqual(len(manifest["cases"]), 6)
            self.assertEqual(len(manifest["jobs"]), 36)
            self.assertEqual(len(selected_jobs(manifest, ["train"])), 12)
            self.assertEqual(
                {job["worker_slot"] for job in manifest["jobs"]},
                set(range(8)),
            )
            with self.assertRaisesRegex(RuntimeError, "allow-pilot-presets"):
                generate_job(
                    SimpleNamespace(
                        manifest=str(root / "output/generation_manifest.json"),
                        job_id=manifest["jobs"][0]["job_id"],
                        allow_pilot_presets=False,
                    )
                )
            previous = prepare(args)
            self.assertEqual(previous["manifest_sha256"], manifest["manifest_sha256"])

            native_config = Path(manifest["cases"][0]["config_path"])
            original = native_config.read_bytes()
            template = json.loads(template_path.read_text(encoding="utf-8"))
            template["sample_shift"] = 9
            template_path.write_text(json.dumps(template), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "new OUT_ROOT"):
                prepare(args)
            self.assertEqual(native_config.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
