from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from changing_resolution_uni.scripts.data import batch_vbench_score_dataset as scorer
from changing_resolution_uni.scripts.data.oracle_record_schema import (
    validate_scored_record,
)


class BatchVBenchBackfillTest(unittest.TestCase):
    VBENCH_IDENTITY = {
        "git_commit": "1" * 40,
        "tracked_dirty": False,
        "tracked_dirty_paths": [],
        "evaluate_py_sha256": "2" * 64,
    }

    def _fixture(self, root: Path) -> Path:
        seed_dir = root / "seed_42"
        manifest_dir = seed_dir / "manifests"
        videos_dir = seed_dir / "videos"
        manifest_dir.mkdir(parents=True)
        branches = []
        for index, step in enumerate(scorer.FORMAL_STEPS):
            case_dir = videos_dir / f"step{step}"
            case_dir.mkdir(parents=True)
            (case_dir / f"0000_seed42_step{step}.mp4").write_bytes(b"video")
            branches.append(
                {
                    "candidate_step": step,
                    "estimated_warm_pipeline_seconds": 100.0 - index,
                }
            )
        native_dir = videos_dir / "native_hr"
        native_dir.mkdir(parents=True)
        (native_dir / "0000_seed42_native_hr.mp4").write_bytes(b"video")
        (manifest_dir / "0000_seed42.json").write_text(
            json.dumps(
                {
                    "prompt_index": 0,
                    "prompt": "a test prompt",
                    "seed": 42,
                    "branches": branches,
                    "native_hr": {"warm_pipeline_seconds": 200.0},
                }
            ),
            encoding="utf-8",
        )
        return seed_dir

    @staticmethod
    def _score_case(
        video_dir: Path, dimensions: list[str], **_: object
    ) -> scorer.CaseScoreBundle:
        case = video_dir.name
        stem = f"0000_seed42_{case}"
        return scorer.CaseScoreBundle(
            scores={stem: {dimension: 0.9 for dimension in dimensions}},
            provenance={
                "request_sha256": "a" * 64,
                "result_sha256": "b" * 64,
                "full_info_sha256": "c" * 64,
                "run_manifest_path": "/strict/run/score_run_manifest.json",
            },
        )

    def test_complete_scores_produce_strict_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seed_dir = self._fixture(Path(directory))
            with mock.patch.object(
                scorer, "score_case_directory", side_effect=self._score_case
            ):
                result = scorer.backfill_seed_records(
                    seed_dir=seed_dir,
                    vbench_root=Path(directory),
                    python_bin="python",
                    quality_dimensions=scorer.QUALITY5_DIMENSIONS,
                    diagnostic_dimensions=[],
                    ngpus=1,
                    primary_lambda=0.01,
                    force_rescore=False,
                    vbench_identity=self.VBENCH_IDENTITY,
                )
            record = result["records"][0]
            self.assertEqual(len(record["candidates"]), len(scorer.FORMAL_STEPS))
            self.assertEqual(
                set(record["native_dimensions"]), set(scorer.QUALITY5_DIMENSIONS)
            )
            self.assertIn("optimal_step_lambda_001", record)
            self.assertEqual(
                record["candidates"][0]["latency_source"],
                "estimated_warm_pipeline_seconds",
            )
            validate_scored_record(
                record,
                require_dimensions=True,
                require_provenance=True,
            )

    def test_missing_dimension_fails_closed(self) -> None:
        def incomplete(
            video_dir: Path, dimensions: list[str], **kwargs: object
        ) -> scorer.CaseScoreBundle:
            bundle = self._score_case(video_dir, dimensions, **kwargs)
            result = bundle.scores
            if video_dir.name == "step40":
                del result["0000_seed42_step40"][dimensions[-1]]
            return scorer.CaseScoreBundle(
                scores=result, provenance=bundle.provenance
            )

        with tempfile.TemporaryDirectory() as directory:
            seed_dir = self._fixture(Path(directory))
            with mock.patch.object(
                scorer, "score_case_directory", side_effect=incomplete
            ):
                with self.assertRaisesRegex(RuntimeError, "step 40 missing"):
                    scorer.backfill_seed_records(
                        seed_dir=seed_dir,
                        vbench_root=Path(directory),
                        python_bin="python",
                        quality_dimensions=scorer.QUALITY5_DIMENSIONS,
                        diagnostic_dimensions=[],
                        ngpus=1,
                        primary_lambda=0.01,
                        force_rescore=False,
                        vbench_identity=self.VBENCH_IDENTITY,
                    )

    def test_case_request_changes_when_video_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_dir = root / "step40"
            video_dir.mkdir()
            video = video_dir / "0000_seed42_step40.mp4"
            video.write_bytes(b"first-video")
            prompt_map = root / "prompt_map.json"
            prompt_map.write_text(
                json.dumps({str(video): "a test prompt"}), encoding="utf-8"
            )
            first, _ = scorer.build_case_request(
                video_dir=video_dir,
                prompt_map=prompt_map,
                dimensions=scorer.QUALITY5_DIMENSIONS,
                quality_dimensions=scorer.QUALITY5_DIMENSIONS,
                diagnostic_dimensions=[],
                python_bin="python",
                vbench_identity=self.VBENCH_IDENTITY,
            )
            video.write_bytes(b"second-video")
            second, _ = scorer.build_case_request(
                video_dir=video_dir,
                prompt_map=prompt_map,
                dimensions=scorer.QUALITY5_DIMENSIONS,
                quality_dimensions=scorer.QUALITY5_DIMENSIONS,
                diagnostic_dimensions=[],
                python_bin="python",
                vbench_identity=self.VBENCH_IDENTITY,
            )
            self.assertNotEqual(first["request_sha256"], second["request_sha256"])

    def test_parser_rejects_incomplete_official_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "results_eval_results.json"
            result_path.write_text(
                json.dumps(
                    {
                        dimension: [
                            0.9,
                            [
                                {
                                    "video_path": "/videos/sample.mp4",
                                    "video_results": 0.9,
                                }
                            ],
                        ]
                        for dimension in scorer.QUALITY5_DIMENSIONS[:-1]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "dimension mismatch"):
                scorer.parse_vbench_eval_result(
                    result_path,
                    dimensions=scorer.QUALITY5_DIMENSIONS,
                    expected_stems={"sample"},
                )

    def test_only_exact_content_bound_run_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vbench_root = root / "VBench"
            vbench_root.mkdir()
            (vbench_root / "evaluate.py").write_text("# fixture", encoding="utf-8")
            video_dir = root / "videos" / "step40"
            video_dir.mkdir(parents=True)
            video = video_dir / "0000_seed42_step40.mp4"
            video.write_bytes(b"video")
            prompt_map = root / "prompt_map.json"
            prompt_map.write_text(
                json.dumps({str(video): "a test prompt"}), encoding="utf-8"
            )
            out_dir = root / "metrics" / "step40"

            def fake_vbench(cmd: list[str], **_: object) -> None:
                output_path = Path(cmd[cmd.index("--output_path") + 1])
                payload = {
                    dimension: [
                        0.9,
                        [
                            {
                                "video_path": str(video),
                                "video_results": 90.0
                                if dimension == "imaging_quality"
                                else 0.9,
                            }
                        ],
                    ]
                    for dimension in scorer.QUALITY5_DIMENSIONS
                }
                (output_path / "fixture_eval_results.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                (output_path / "fixture_full_info.json").write_text(
                    "[]", encoding="utf-8"
                )

            with mock.patch.object(
                scorer.subprocess, "run", side_effect=fake_vbench
            ) as run_mock:
                first = scorer.score_case_directory(
                    vbench_root=vbench_root,
                    python_bin="python",
                    video_dir=video_dir,
                    prompt_map=prompt_map,
                    out_dir=out_dir,
                    dimensions=scorer.QUALITY5_DIMENSIONS,
                    quality_dimensions=scorer.QUALITY5_DIMENSIONS,
                    diagnostic_dimensions=[],
                    ngpus=1,
                    force_rescore=False,
                    vbench_identity=self.VBENCH_IDENTITY,
                )
                second = scorer.score_case_directory(
                    vbench_root=vbench_root,
                    python_bin="python",
                    video_dir=video_dir,
                    prompt_map=prompt_map,
                    out_dir=out_dir,
                    dimensions=scorer.QUALITY5_DIMENSIONS,
                    quality_dimensions=scorer.QUALITY5_DIMENSIONS,
                    diagnostic_dimensions=[],
                    ngpus=1,
                    force_rescore=False,
                    vbench_identity=self.VBENCH_IDENTITY,
                )
            self.assertEqual(run_mock.call_count, 1)
            self.assertEqual(first.scores, second.scores)
            self.assertEqual(
                first.scores["0000_seed42_step40"]["imaging_quality"], 0.9
            )

    def test_equivalent_cross_second_full_info_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vbench_root = root / "VBench"
            vbench_root.mkdir()
            (vbench_root / "evaluate.py").write_text("# fixture", encoding="utf-8")
            video_dir = root / "videos" / "step42"
            video_dir.mkdir(parents=True)
            video = video_dir / "0000_seed42_step42.mp4"
            video.write_bytes(b"video")
            prompt_map = root / "prompt_map.json"
            prompt_map.write_text(
                json.dumps({str(video): "a test prompt"}), encoding="utf-8"
            )

            def fake_vbench(cmd: list[str], **_: object) -> None:
                output_path = Path(cmd[cmd.index("--output_path") + 1])
                result_payload = {
                    dimension: [
                        0.9,
                        [
                            {
                                "video_path": str(video),
                                "video_results": 90.0
                                if dimension == "imaging_quality"
                                else 0.9,
                            }
                        ],
                    ]
                    for dimension in scorer.QUALITY5_DIMENSIONS
                }
                (output_path / "results_second_eval_results.json").write_text(
                    json.dumps(result_payload), encoding="utf-8"
                )
                full_info = [
                    {
                        "prompt_en": "a test prompt",
                        "dimension": scorer.QUALITY5_DIMENSIONS,
                        "video_list": [str(video)],
                    }
                ]
                (output_path / "results_second_full_info.json").write_text(
                    json.dumps(full_info), encoding="utf-8"
                )
                (output_path / "results_first_full_info.json").write_text(
                    json.dumps(list(reversed(full_info))), encoding="utf-8"
                )

            with mock.patch.object(scorer.subprocess, "run", side_effect=fake_vbench):
                result = scorer.score_case_directory(
                    vbench_root=vbench_root,
                    python_bin="python",
                    video_dir=video_dir,
                    prompt_map=prompt_map,
                    out_dir=root / "metrics" / "step42",
                    dimensions=scorer.QUALITY5_DIMENSIONS,
                    quality_dimensions=scorer.QUALITY5_DIMENSIONS,
                    diagnostic_dimensions=[],
                    ngpus=8,
                    force_rescore=False,
                    vbench_identity=self.VBENCH_IDENTITY,
                )
            extras = result.provenance["equivalent_extra_full_info"]
            self.assertEqual([row["file"] for row in extras], ["results_first_full_info.json"])

    def test_conflicting_cross_second_full_info_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vbench_root = root / "VBench"
            vbench_root.mkdir()
            (vbench_root / "evaluate.py").write_text("# fixture", encoding="utf-8")
            video_dir = root / "videos" / "step42"
            video_dir.mkdir(parents=True)
            video = video_dir / "0000_seed42_step42.mp4"
            video.write_bytes(b"video")
            prompt_map = root / "prompt_map.json"
            prompt_map.write_text(json.dumps({str(video): "prompt"}), encoding="utf-8")

            def fake_vbench(cmd: list[str], **_: object) -> None:
                output_path = Path(cmd[cmd.index("--output_path") + 1])
                payload = {
                    dimension: [
                        0.9,
                        [{"video_path": str(video), "video_results": 90.0 if dimension == "imaging_quality" else 0.9}],
                    ]
                    for dimension in scorer.QUALITY5_DIMENSIONS
                }
                (output_path / "results_second_eval_results.json").write_text(json.dumps(payload), encoding="utf-8")
                (output_path / "results_second_full_info.json").write_text("[]", encoding="utf-8")
                (output_path / "results_first_full_info.json").write_text("[{}]", encoding="utf-8")

            with mock.patch.object(scorer.subprocess, "run", side_effect=fake_vbench):
                with self.assertRaisesRegex(RuntimeError, "not equivalent"):
                    scorer.score_case_directory(
                        vbench_root=vbench_root,
                        python_bin="python",
                        video_dir=video_dir,
                        prompt_map=prompt_map,
                        out_dir=root / "metrics" / "step42",
                        dimensions=scorer.QUALITY5_DIMENSIONS,
                        quality_dimensions=scorer.QUALITY5_DIMENSIONS,
                        diagnostic_dimensions=[],
                        ngpus=8,
                        force_rescore=False,
                        vbench_identity=self.VBENCH_IDENTITY,
                    )


if __name__ == "__main__":
    unittest.main()
