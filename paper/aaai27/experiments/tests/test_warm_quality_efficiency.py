from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper.aaai27.experiments.benchmark_warm_quality_efficiency import (
    display_name,
    protocol_signature,
    select_cases,
    summarize_all,
    validate_raw_rows,
)


def video_row(
    *,
    model_cls: str,
    phase: str,
    repeat: int,
    prompt_index: int,
    seed: int,
    pipeline: float,
    denoise: float,
) -> dict[str, object]:
    return {
        "kind": "video",
        "phase": phase,
        "repeat": repeat,
        "prompt_index": prompt_index,
        "seed": seed,
        "model_cls": model_cls,
        "pipeline_elapsed_s": pipeline,
        "denoise_elapsed_s": denoise,
        "segment_count": 1,
        "output": f"video_{prompt_index}.mp4",
    }


class WarmQualityEfficiencyTest(unittest.TestCase):
    def test_default_order_keeps_direct_competitors_adjacent(self) -> None:
        manifest = {
            "cases": [
                {"name": "talh45"},
                {"name": "ralu_quality"},
                {"name": "full_hr50"},
                {"name": "lightx2v_cr45"},
            ]
        }
        selected = select_cases(manifest, None)
        self.assertEqual(
            [case["name"] for case in selected],
            ["full_hr50", "lightx2v_cr45", "ralu_quality", "talh45"],
        )

    def test_paper_display_names_hide_internal_talh_ids(self) -> None:
        self.assertEqual(display_name({"name": "talh40"}), "TrajScale-40")
        self.assertEqual(display_name({"name": "talh45"}), "TrajScale-45")
        self.assertEqual(display_name({"name": "lightx2v_cr45"}), "LightX2V-45")
        self.assertEqual(display_name({"name": "ralu_quality"}), "RALU-Quality")
        self.assertEqual(
            display_name({"name": "full_lr50_stage2_5hr"}), "Endpoint-5HR"
        )

    def test_protocol_signature_ignores_embedded_signature(self) -> None:
        protocol = {"gpu": 0, "cases": ["full_hr50"]}
        signature = protocol_signature(protocol)
        protocol["run_signature"] = "stale"
        self.assertEqual(protocol_signature(protocol), signature)

    def test_validates_one_persistent_process_timing_layout(self) -> None:
        case = {"name": "talh45", "model_cls": "trajscale"}
        rows = [{"kind": "initialization", "model_cls": "trajscale", "elapsed_s": 10.0}]
        rows.append(
            video_row(
                model_cls="trajscale",
                phase="warmup",
                repeat=0,
                prompt_index=0,
                seed=100,
                pipeline=4.0,
                denoise=3.0,
            )
        )
        for repeat in range(2):
            rows.append(
                video_row(
                    model_cls="trajscale",
                    phase="measured",
                    repeat=repeat,
                    prompt_index=repeat + 1,
                    seed=101 + repeat,
                    pipeline=4.0,
                    denoise=3.0,
                )
            )
        validate_raw_rows(rows, case, 1, 2, 100, 0)
        rows[-1]["seed"] = 999
        with self.assertRaises(RuntimeError):
            validate_raw_rows(rows, case, 1, 2, 100, 0)

    def test_summarizes_warm_latency_and_native_speedup(self) -> None:
        cases = [
            {
                "name": "full_hr50",
                "method": "native",
                "model_cls": "native_cls",
                "lr_evaluations": 0,
                "hr_evaluations": 50,
                "total_evaluations": 50,
                "handoff_step": None,
                "refinement_steps": None,
                "reschedule_mode": "canonical",
            },
            {
                "name": "talh45",
                "method": "talh",
                "model_cls": "trajscale_cls",
                "lr_evaluations": 45,
                "hr_evaluations": 5,
                "total_evaluations": 50,
                "handoff_step": 45,
                "refinement_steps": None,
                "reschedule_mode": "canonical",
            },
        ]
        spec = {
            case["name"]: {
                "family": "wan50",
                "quality_metric": "vbench",
                "quality_value": 0.8,
                "quality_components": {"quality": 0.8},
                "vbench_source": "scores.json",
            }
            for case in cases
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            (root / "resources").mkdir()
            for case, times in zip(cases, ((10.0, 8.0), (5.0, 4.0))):
                rows = [
                    {
                        "kind": "initialization",
                        "model_cls": case["model_cls"],
                        "elapsed_s": 20.0,
                    },
                    video_row(
                        model_cls=case["model_cls"],
                        phase="warmup",
                        repeat=0,
                        prompt_index=0,
                        seed=100,
                        pipeline=times[0],
                        denoise=times[1],
                    ),
                ]
                for repeat in range(2):
                    rows.append(
                        video_row(
                            model_cls=case["model_cls"],
                            phase="measured",
                            repeat=repeat,
                            prompt_index=repeat + 1,
                            seed=101 + repeat,
                            pipeline=times[0],
                            denoise=times[1],
                        )
                    )
                (root / "raw" / f"{case['name']}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                (root / "resources" / f"{case['name']}.json").write_text(
                    json.dumps({"physical_gpu": 0, "peak_memory_delta_gib": 26.0}),
                    encoding="utf-8",
                )

            summary, raw = summarize_all(cases, spec, root, warmup=1, repeats=2)
            by_case = {row["case"]: row for row in summary}
            self.assertEqual(len(raw), 6)
            self.assertEqual(by_case["talh45"]["display_name"], "TrajScale-45")
            self.assertEqual(by_case["talh45"]["pipeline_mean_s"], 5.0)
            self.assertEqual(by_case["talh45"]["speedup_vs_native"], 2.0)
            self.assertEqual(
                by_case["talh45"]["latency_reduction_vs_native_pct"], 50.0
            )


if __name__ == "__main__":
    unittest.main()
