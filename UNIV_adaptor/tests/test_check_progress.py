import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import io

from UNIV_adaptor.scripts.data.check_prompt_budget_progress import inspect_progress


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

            output = buf.getvalue()
            self.assertIn("UNIV 8-GPU Prompt Budget Generation Status", output)
            self.assertIn("1/2 (50.0%)", output)
            self.assertIn("GPU 0", output)
            self.assertIn("GPU 1", output)


if __name__ == "__main__":
    unittest.main()

