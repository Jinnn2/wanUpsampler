from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from changing_resolution_uni.scripts.data import batch_vbench_score_dataset as scorer


class BatchVBenchBackfillTest(unittest.TestCase):
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
    def _score_case(video_dir: Path, dimensions: list[str], **_: object) -> dict:
        case = video_dir.name
        stem = f"0000_seed42_{case}"
        return {stem: {dimension: 0.9 for dimension in dimensions}}

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
                    dimensions=scorer.QUALITY5_DIMENSIONS,
                    ngpus=1,
                    primary_lambda=0.01,
                    skip_existing=True,
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

    def test_missing_dimension_fails_closed(self) -> None:
        def incomplete(video_dir: Path, dimensions: list[str], **kwargs: object) -> dict:
            result = self._score_case(video_dir, dimensions, **kwargs)
            if video_dir.name == "step40":
                del result["0000_seed42_step40"][dimensions[-1]]
            return result

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
                        dimensions=scorer.QUALITY5_DIMENSIONS,
                        ngpus=1,
                        primary_lambda=0.01,
                        skip_existing=True,
                    )


if __name__ == "__main__":
    unittest.main()
