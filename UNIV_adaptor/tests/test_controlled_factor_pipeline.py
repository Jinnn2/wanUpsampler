from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import canonical_sha256  # noqa: E402
from UNIV_adaptor.scripts.data.run_controlled_factor_generation import (  # noqa: E402
    build_plan,
    load_prompt_rows,
    validate_prompts,
    validate_protocol,
)
from UNIV_adaptor.scripts.data.run_prompt_budget_generation import load_json  # noqa: E402
from UNIV_adaptor.scripts.data.score_controlled_factor_dataset import (  # noqa: E402
    ACTIONS,
    ANALYSIS_SCHEMA,
)


class ControlledFactorProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol_path = ROOT / "UNIV_adaptor/configs/univ_controlled_factor_v1.json"
        self.prompts_path = ROOT / "prompts/univ_controlled_factor_v1.jsonl"

    def test_locked_factorial_and_plan(self) -> None:
        protocol = validate_protocol(load_json(self.protocol_path))
        prompts = validate_prompts(load_prompt_rows(self.prompts_path), protocol)
        plan = build_plan(protocol, prompts)
        self.assertEqual(plan["counts"]["prompts"], 80)
        self.assertEqual(plan["counts"]["families"], 20)
        self.assertEqual(plan["counts"]["prompt_seed_groups"], 240)
        self.assertEqual(plan["counts"]["videos"], 960)
        self.assertEqual([row["prompt_id"] for row in prompts[:8]], list(range(8)))
        for family in {row["family_id"] for row in prompts}:
            cells = {
                (row["motion_level"], row["detail_level"])
                for row in prompts
                if row["family_id"] == family
            }
            self.assertEqual(
                cells,
                {("low", "low"), ("low", "high"), ("high", "low"), ("high", "high")},
            )

    def test_tfidf_train_and_one_time_confirmation(self) -> None:
        protocol = validate_protocol(load_json(self.protocol_path))
        prompts = validate_prompts(load_prompt_rows(self.prompts_path), protocol)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scored = root / "scored"
            trained = root / "trained"
            scored.mkdir()
            rows = []
            for prompt in prompts:
                motion_high = prompt["motion_level"] == "high"
                detail_high = prompt["detail_level"] == "high"
                values = {
                    ACTIONS[0]: -0.020 if detail_high else -0.004,
                    ACTIONS[1]: -0.025 if motion_high else -0.003,
                    ACTIONS[2]: -0.012 if motion_high and detail_high else -0.005,
                }
                row = {
                    key: prompt[key]
                    for key in (
                        "prompt_id",
                        "family_id",
                        "split",
                        "motion_level",
                        "detail_level",
                        "factor_cell",
                        "prompt",
                        "prompt_sha256",
                    )
                }
                for action, value in values.items():
                    tag = action.lower()
                    row[f"mean_delta_vbench5__{tag}"] = value
                    row[f"mean_time_ratio__{tag}"] = 0.5
                rows.append(row)
            identity_keys = (
                "prompt_id",
                "family_id",
                "split",
                "motion_level",
                "detail_level",
                "factor_cell",
                "prompt",
                "prompt_sha256",
            )
            inputs_path = scored / "prompt_inputs.csv"
            with inputs_path.open("w", encoding="utf-8", newline="") as handle:
                input_rows = [{key: row[key] for key in identity_keys} for row in rows]
                writer = csv.DictWriter(handle, fieldnames=list(input_rows[0]))
                writer.writeheader()
                writer.writerows(input_rows)
            development_targets = scored / "prompt_targets_train_validation.csv"
            with development_targets.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(row for row in rows if row["split"] != "test")
            test_targets = scored / "prompt_targets_test.csv"
            with test_targets.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(row for row in rows if row["split"] == "test")
            body = {
                "input_sha256": "input",
                "score_payload_sha256": "scores",
                "quality_definition": "synthetic_test",
                "target_definition": "synthetic_test",
                "prompt_count": 80,
                "prompt_seed_groups": 240,
                "video_count": 960,
                "action_timing": {},
                "test_labels_scored_but_not_authorized_for_model_selection": True,
                "analysis_source_sha256": "synthetic",
            }
            (scored / "analysis.json").write_text(
                json.dumps(
                    {
                        "schema": ANALYSIS_SCHEMA,
                        "analysis_sha256": canonical_sha256(body),
                        **body,
                    }
                ),
                encoding="utf-8",
            )
            trainer = ROOT / "UNIV_adaptor/scripts/router/train_controlled_factor_prompt_prior.py"
            base = [
                sys.executable,
                str(trainer),
                "--scored-dir",
                str(scored),
                "--out-dir",
                str(trained),
                "--features",
                "tfidf",
                "--bootstrap-repetitions",
                "100",
            ]
            subprocess.run([base[0], base[1], "train", *base[2:]], cwd=ROOT, check=True)
            with (trained / "predictions.csv").open(encoding="utf-8", newline="") as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual(len(predictions), 64)
            self.assertNotIn("locked_test", {row["split"] for row in predictions})
            subprocess.run(
                [base[0], base[1], "confirm", *base[2:], "--confirm-test-access"],
                cwd=ROOT,
                check=True,
            )
            self.assertTrue((trained / "test_access_guard.json").is_file())
            self.assertTrue((trained / "test_confirmation.json").is_file())
            repeated = subprocess.run(
                [base[0], base[1], "confirm", *base[2:], "--confirm-test-access"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(repeated.returncode, 0)


if __name__ == "__main__":
    unittest.main()
