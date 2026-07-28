from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper.aaai27.experiments.export_distill4_p0_p1_p3 import (
    MAIN_CASES,
    QUALITY5,
    REPO_ROOT,
    export_bundle,
)


class Distill4P0P1P3ExportTest(unittest.TestCase):
    def test_exports_strict_metadata_bundle_without_videos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main = root / "main"
            p3 = root / "p3"
            p3_cases = tuple(
                f"talh3_s{strength}_{mode}"
                for strength in ("0p25", "0p5", "0p75", "1")
                for mode in ("random", "resize_flow")
            )
            self._write_suite(
                main,
                family="distill4_quality_efficiency",
                cases=MAIN_CASES,
                prompts=10,
                seed=9800,
            )
            self._write_suite(
                p3,
                family="distill4_talh_validation_sweep",
                cases=p3_cases,
                prompts=8,
                seed=16000,
            )
            (main / "configs/endpoint_rgb_1hr.json").write_text(
                json.dumps(
                    {
                        "wan_final_refine_steps": 1,
                        "wan_final_refine_sigma": 0.12,
                        "wan_rgb_sr_backend": "realesrgan",
                    }
                ),
                encoding="utf-8",
            )
            self._write_metrics(
                main / "metrics/vbench_v1_custom.json",
                MAIN_CASES,
                QUALITY5,
            )
            self._write_metrics(
                main / "metrics/vbench_temporal_flickering.json",
                MAIN_CASES,
                ("temporal_flickering",),
            )
            self._write_metrics(
                p3 / "metrics/vbench_v1_custom.json",
                p3_cases,
                (*QUALITY5, "temporal_flickering"),
            )
            for path in (
                main / "metrics/vbench_paired_statistics.csv",
                main / "metrics/vbench_temporal_flickering_paired_statistics.csv",
            ):
                path.write_text("comparison,metric\n", encoding="utf-8")
            (main / "quality_efficiency_warm.csv").write_text(
                "case,seconds\n", encoding="utf-8"
            )
            ranking = [
                {
                    "case": case,
                    "selected": index == 0,
                    "quality5_mean": 0.9 - index / 100,
                }
                for index, case in enumerate(p3_cases)
            ]
            (p3 / "metrics/talh_validation_selection.json").write_text(
                json.dumps(
                    {
                        "selected": ranking[0],
                        "ranking": ranking,
                    }
                ),
                encoding="utf-8",
            )
            (p3 / "metrics/talh_validation_selection.csv").write_text(
                "case,selected\n", encoding="utf-8"
            )

            output = export_bundle(
                main_root=main,
                validation_root=p3,
                output_root=root / "export",
                project_root=REPO_ROOT,
                include_videos=False,
            )
            manifest = json.loads(
                (output / "export_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["validated_video_counts"],
                {"main_suite": 180, "p3_validation": 64},
            )
            self.assertFalse((output / "main_suite/videos").exists())
            self.assertTrue(output.with_suffix(".tar.gz").is_file())
            self.assertTrue((output / "SHA256SUMS").is_file())

    @staticmethod
    def _write_suite(
        root: Path,
        *,
        family: str,
        cases: tuple[str, ...],
        prompts: int,
        seed: int,
    ) -> None:
        (root / "configs").mkdir(parents=True)
        (root / "metrics").mkdir()
        (root / "run_manifest.json").write_text(
            json.dumps(
                {
                    "family": family,
                    "seed_base": seed,
                    "prompt_offset": 0,
                    "prompts": [f"prompt {index}" for index in range(prompts)],
                    "cases": [{"name": case} for case in cases],
                }
            ),
            encoding="utf-8",
        )
        (root / "generation_schedule.json").write_text("{}", encoding="utf-8")
        for case in cases:
            (root / "configs" / f"{case}.json").write_text(
                "{}", encoding="utf-8"
            )
            video_root = root / "videos" / case
            video_root.mkdir(parents=True)
            for index in range(prompts):
                (
                    video_root / f"{case}_{index:02d}_seed{seed + index}.mp4"
                ).write_bytes(b"x" * 1025)

    @staticmethod
    def _write_metrics(
        path: Path, cases: tuple[str, ...], dimensions: tuple[str, ...]
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "dimensions": list(dimensions),
                    "cases": {
                        case: {
                            "numeric_metrics": {
                                f"result.{dimension}.0": 0.9
                                for dimension in dimensions
                            }
                        }
                        for case in cases
                    },
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
