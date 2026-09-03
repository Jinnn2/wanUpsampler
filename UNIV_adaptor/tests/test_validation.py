from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from UNIV_adaptor.validation import (
    REPO_ROOT,
    comparison_groups,
    prepare_suite,
    resolve_cases,
    selected_manifest_cases,
    summarize_suite,
)


class ValidationSuiteTest(unittest.TestCase):
    def test_case_filter_preserves_requested_order_and_rejects_unknown(self):
        manifest = {
            "cases": [
                {"name": "native"},
                {"name": "dvg"},
                {"name": "rgb"},
            ]
        }
        self.assertEqual(
            [row["name"] for row in selected_manifest_cases(manifest, ["rgb", "native"])],
            ["rgb", "native"],
        )
        with self.assertRaisesRegex(ValueError, "absent"):
            selected_manifest_cases(manifest, ["missing"])
        with self.assertRaisesRegex(ValueError, "unique"):
            selected_manifest_cases(manifest, ["dvg", "dvg"])

    def test_profiles_have_one_native_and_expected_baselines(self):
        spec = json.loads(
            (REPO_ROOT / "UNIV_adaptor/configs/univ_validation_cases.json").read_text(
                encoding="utf-8"
            )
        )
        smoke = resolve_cases(spec, "smoke")
        core = resolve_cases(spec, "core")
        full = resolve_cases(spec, "full")
        self.assertEqual(len(smoke), 3)
        self.assertEqual(len(core), 8)
        self.assertEqual(len(full), 14)
        for cases in (smoke, core, full):
            self.assertEqual(sum(case["kind"] == "native" for case in cases), 1)
        self.assertEqual(
            {case.get("transition") for case in smoke if case["kind"] == "univ"},
            {"dvg_latent_anchor", "rgb_sr_vae"},
        )
        groups = comparison_groups({"cases": smoke})
        self.assertEqual(
            groups["joint_sw060"],
            ["native_hr50", "dvg_joint_sw060", "rgb_joint_sw060"],
        )

    def test_changed_protocol_refuses_to_overwrite_existing_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.txt"
            prompts.write_text("first prompt\nsecond prompt\nthird prompt\n", encoding="utf-8")
            args = self._prepare_args(root, prompts)
            manifest = prepare_suite(args)
            native = next(case for case in manifest["cases"] if case["kind"] == "native")
            self.assertEqual(native["model_cls"], "wan2.1_univ_native")
            config_path = Path(manifest["cases"][0]["config_path"])
            original_config = config_path.read_bytes()

            changed = self._prepare_args(root, prompts)
            changed.seed += 1
            with self.assertRaisesRegex(RuntimeError, "protocol changed"):
                prepare_suite(changed)
            self.assertEqual(config_path.read_bytes(), original_config)

    def test_summary_uses_warm_pipeline_ratio_and_quality_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "timings").mkdir()
            (root / "configs").mkdir()
            native_config = root / "configs/native.json"
            dvg_config = root / "configs/dvg.json"
            native_config.write_text(json.dumps({"enable_cfg": True}), encoding="utf-8")
            dvg_config.write_text(json.dumps({"enable_cfg": True}), encoding="utf-8")
            manifest = {
                "prompt_count": 2,
                "prompt_offset": 0,
                "seed_base": 9700,
                "cases": [
                    {
                        "name": "native",
                        "kind": "native",
                        "config_path": str(native_config),
                    },
                    {
                        "name": "dvg",
                        "kind": "univ",
                        "transition": "dvg_latent_anchor",
                        "config_path": str(dvg_config),
                        "resolved_schedule": {"total_full_dit_evaluations": 35},
                    },
                ]
            }
            self._write_timing(root, "native", [10.0, 12.0])
            self._write_timing(root, "dvg", [5.0, 6.0], univ=True)
            self._write_vbench(root)

            rows = summarize_suite(SimpleNamespace(out_root=str(root)), manifest)
            by_name = {row["case"]: row for row in rows}
            self.assertAlmostEqual(by_name["dvg"]["speedup_vs_native"], 2.0)
            self.assertAlmostEqual(by_name["dvg"]["quality5_delta_vs_native"], -0.1)
            self.assertEqual(by_name["dvg"]["full_dit_evaluations"], 35)
            self.assertTrue((root / "reports/summary.csv").is_file())
            self.assertTrue((root / "reports/per_video.csv").is_file())

    @staticmethod
    def _prepare_args(root: Path, prompts: Path) -> SimpleNamespace:
        return SimpleNamespace(
            case_spec=str(REPO_ROOT / "UNIV_adaptor/configs/univ_validation_cases.json"),
            template_config=str(
                REPO_ROOT / "UNIV_adaptor/configs/wan21_t2v_univ_rgb_720p.example.json"
            ),
            prompts=str(prompts),
            out_root=str(root / "outputs"),
            profile="smoke",
            prompt_offset=0,
            limit=2,
            timing_warmup=0,
            transition_diagnostics=False,
            seed=9700,
            model_root=str(root / "model"),
            realesrgan_checkpoint=str(root / "realesrgan.pth"),
        )

    @staticmethod
    def _write_timing(root: Path, case: str, durations: list[float], univ: bool = False) -> None:
        rows = [
            {
                "kind": "initialization",
                "case": case,
                "elapsed_s": 20.0,
            }
        ]
        for index, duration in enumerate(durations):
            row = {
                "kind": "video",
                "case": case,
                "phase": "measured",
                "repeat": index,
                "prompt_index": index,
                "seed": 9700 + index,
                "pipeline_elapsed_s": duration,
                "segment_elapsed_s": duration - 1.0,
                "peak_allocated_gib": 8.0,
                "output": str(root / f"{case}_{index:02d}_seed{9700 + index}.mp4"),
            }
            if univ:
                row["univ_stage_timing_s"] = {"lr_full_compute": 2.0, "transition": 1.0}
            rows.append(row)
        path = root / "timings" / f"{case}.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _write_vbench(root: Path) -> None:
        dimensions = {
            "subject_consistency": 0.8,
            "background_consistency": 0.8,
            "motion_smoothness": 0.8,
            "aesthetic_quality": 0.8,
            "imaging_quality": 0.8,
            "dynamic_degree": 0.7,
        }
        payload = {"cases": {}}
        for case, offset in (("native", 0.0), ("dvg", -0.1)):
            aggregate = {key: value + offset for key, value in dimensions.items()}
            payload["cases"][case] = {
                "aggregate": aggregate,
                "quality5_mean": 0.8 + offset,
                "per_video": {
                    f"{case}_{index:02d}_seed{9700 + index}": aggregate
                    for index in range(2)
                },
            }
        path = root / "metrics" / "vbench_scores.json"
        path.parent.mkdir()
        path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
