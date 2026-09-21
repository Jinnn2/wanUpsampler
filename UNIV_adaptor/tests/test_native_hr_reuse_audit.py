from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from UNIV_adaptor.scripts.data.audit_native_hr_reuse import audit


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def add_native_record(
    root: Path,
    *,
    name: str,
    prompt_sha256: str,
    seed: int,
    video_bytes: bytes,
) -> None:
    video = root / "videos" / f"{name}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(video_bytes)
    digest = hashlib.sha256(video_bytes).hexdigest()
    write_json(
        root / "records" / "train" / f"{name}.json",
        {
            "prompt_sha256": prompt_sha256,
            "seed": seed,
            "native_teacher": {
                "video_path": str(video.resolve()),
                "video_sha256": digest,
                "video_bytes": len(video_bytes),
                "cost": {"pipeline_seconds": 10.0},
            },
        },
    )


class NativeHrReuseAuditTest(unittest.TestCase):
    def test_prefers_bound_phase2_and_finds_other_compatible_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            phase4 = outputs / "univ_matched_star_phase4_v1"
            phase3 = outputs / "univ_sparse_action_phase3_v1"
            phase2 = outputs / "univ_prompt_budget_phase2_source"
            other = outputs / "univ_prompt_budget_other"
            config = {"infer_steps": 50, "target_height": 720, "target_width": 1248}
            write_json(phase2 / "configs" / "native_hr50.json", config)
            write_json(other / "configs" / "native_hr50.json", config)
            write_json(
                phase3 / "sparse_action_plan.json",
                {"plan_sha256": "phase3", "source_phase2": {"root": str(phase2)}},
            )
            groups = [
                {
                    "group_id": f"g{index}",
                    "prompt_key": f"p{index}",
                    "prompt_sha256": f"hash{index}",
                    "base_seed": index,
                    "seed": 100 + index,
                }
                for index in range(3)
            ]
            write_json(
                phase4 / "sparse_action_plan.json",
                {
                    "plan_sha256": "phase4",
                    "source_phase3": {"root": str(phase3)},
                    "groups": groups,
                },
            )
            add_native_record(
                phase2,
                name="bound",
                prompt_sha256="hash0",
                seed=100,
                video_bytes=b"bound-video",
            )
            add_native_record(
                other,
                name="other",
                prompt_sha256="hash1",
                seed=101,
                video_bytes=b"other-video",
            )

            report = audit(
                phase4_root=phase4,
                search_roots=[outputs],
                verify_video_sha=True,
            )

            self.assertEqual(report["counts"]["expected_groups"], 3)
            self.assertEqual(report["counts"]["reused_groups"], 2)
            self.assertEqual(report["counts"]["bound_phase2_reused_groups"], 1)
            self.assertEqual(report["counts"]["other_reused_groups"], 1)
            self.assertEqual(report["counts"]["missing_groups"], 1)
            self.assertEqual(report["missing"][0]["group_id"], "g2")
            self.assertTrue(
                all(row["artifact"]["video_sha256_verified"] for row in report["selected"])
            )

    def test_rejects_mismatched_native_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase4 = root / "phase4"
            phase3 = root / "phase3"
            phase2 = root / "univ_prompt_budget_phase2"
            other = root / "univ_prompt_budget_bad"
            write_json(phase2 / "configs" / "native_hr50.json", {"infer_steps": 50})
            write_json(other / "configs" / "native_hr50.json", {"infer_steps": 40})
            write_json(
                phase3 / "sparse_action_plan.json",
                {"source_phase2": {"root": str(phase2)}},
            )
            write_json(
                phase4 / "sparse_action_plan.json",
                {
                    "source_phase3": {"root": str(phase3)},
                    "groups": [
                        {
                            "group_id": "g0",
                            "prompt_key": "p0",
                            "prompt_sha256": "hash0",
                            "base_seed": 42,
                            "seed": 42,
                        }
                    ],
                },
            )
            add_native_record(
                other,
                name="bad",
                prompt_sha256="hash0",
                seed=42,
                video_bytes=b"bad-config-video",
            )

            report = audit(
                phase4_root=phase4,
                search_roots=[root],
                verify_video_sha=True,
            )

            self.assertEqual(report["counts"]["reused_groups"], 0)
            self.assertEqual(report["counts"]["missing_groups"], 1)
            self.assertEqual(report["counts"]["invalid_candidates"], 1)
            self.assertEqual(
                report["invalid_candidates"][0]["reason"],
                "native_config_mismatch",
            )


if __name__ == "__main__":
    unittest.main()
