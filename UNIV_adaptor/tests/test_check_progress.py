import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import io

from UNIV_adaptor.scripts.data.check_prompt_budget_progress import inspect_progress
from UNIV_adaptor.scripts.data.check_prompt_budget_progress import (
    inspect_progress,
    print_multi_machine_summary,
)


class TestCheckPromptBudgetProgress(unittest.TestCase):
    def test_inspect_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_root = Path(tmpdir)
            timings_dir = out_root / "timings"
            timings_dir.mkdir(parents=True)
            logs_dir = out_root / "logs" / "8gpu_data"
            logs_dir.mkdir(parents=True)

            job1_timing = timings_dir / "job1.jsonl"
            job1_timing.write_text(
                json.dumps({"kind": "initialization"}) + "\n" +
                json.dumps({"kind": "video", "prompt_index": 0, "pipeline_elapsed_s": 5.2}) + "\n" +
                json.dumps({"kind": "video", "prompt_index": 1, "pipeline_elapsed_s": 4.8}) + "\n"
            )

            (logs_dir / "gpu_0.log").write_text("Generating step 20/50\nDone frame 81\n")

            manifest = {
                "jobs": [
                    {
                        "job_id": "job1",
                        "split": "train",
                        "case_id": "native_hr50",
                        "prompt_count": 2,
                        "worker_slot": 0,
                        "timing_path": str(job1_timing),
                    },
                    {
                        "job_id": "job2",
                        "split": "train",
                        "case_id": "B30",
                        "prompt_count": 5,
                        "worker_slot": 1,
                        "timing_path": str(timings_dir / "job2.jsonl"),
                    },
                ]
            }
            (out_root / "generation_manifest.json").write_text(json.dumps(manifest))

            buf = io.StringIO()
            with patch("sys.stdout", buf):
                inspect_progress(out_root, detail=True)
                summary = inspect_progress(out_root, detail=True)

            output = buf.getvalue()
            self.assertIn("UNIV 8-GPU Prompt Budget Generation Status", output)
            self.assertIn("Shard:", output)
            self.assertIn("1/2 (50.0%)", output)
            self.assertIn("GPU 0", output)
            self.assertIn("GPU 1", output)
            self.assertIsNotNone(summary)
            self.assertEqual(summary["completed_jobs"], 1)
            self.assertEqual(summary["total_jobs"], 2)

    def test_multi_machine_summary(self):
        s1 = {
            "is_running": True,
            "total_jobs": 10,
            "completed_jobs": 5,
            "total_videos": 250,
            "generated_videos": 125,
            "total_video_latency": 1250.0,
            "finalized_count": 0,
        }
        s2 = {
            "is_running": True,
            "total_jobs": 10,
            "completed_jobs": 3,
            "total_videos": 250,
            "generated_videos": 75,
            "total_video_latency": 750.0,
            "finalized_count": 0,
        }
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_multi_machine_summary([s1, s2])
        out = buf.getvalue()
        self.assertIn("MULTI-MACHINE / MULTI-SHARD AGGREGATE SUMMARY", out)
        self.assertIn("2/2 running (16 GPUs total)", out)
        self.assertIn("8/20 (40.0%)", out)
        self.assertIn("200/500 (40.0%)", out)


if __name__ == "__main__":
    unittest.main()

