from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper.aaai27.experiments.export_results import export_results


ALLOWED_MISSING = {
    "sources.lora_architecture_loss",
    "sources.stage2_architecture_loss",
    "sources.generalization",
}


class ResultExportTest(unittest.TestCase):
    def test_exports_distributed_evidence_with_exact_missing_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            results = project / "outputs/aaai27_experiments"
            collection = results
            legacy = project / "outputs/legacy_endpoint"
            factorial = results / "factorial_wan50"
            (collection / "compiled_tables").mkdir(parents=True)
            legacy.mkdir(parents=True)
            (factorial / "videos/case").mkdir(parents=True)
            (factorial / "_private").mkdir()
            (results / "_state").mkdir()

            summary = legacy / "summary.csv"
            summary.write_text("metric,value\nl1,0.1\n", encoding="utf-8")
            (legacy / "raw.jsonl").write_text('{"l1": 0.1}\n', encoding="utf-8")
            (factorial / "run_manifest.json").write_text('{"family": "wan50"}\n', encoding="utf-8")
            (factorial / "videos/case/example.mp4").write_bytes(b"video")
            (factorial / "_private/key.csv").write_text("key\nsecret\n", encoding="utf-8")
            (results / "_state/audit.json").write_text("{}\n", encoding="utf-8")
            (results / "_state/task.log").write_text("private path\n", encoding="utf-8")
            (collection / "paper_tables.md").write_text("# tables\n", encoding="utf-8")
            (collection / "compiled_tables/endpoint.csv").write_text("metric,value\nl1,0.1\n", encoding="utf-8")

            inventory = {
                "schema_version": 2,
                "generated_at_utc": "2027-01-01T00:00:00+00:00",
                "canonical_results_root": str(results),
                "sources": {
                    "endpoint": {"status": "complete", "path": str(summary), "rows": []},
                    "endpoint_paired_statistics": {
                        "status": "complete",
                        "path": f"derived from {summary}",
                        "rows": [{"metric": "l1"}],
                    },
                    "lora_architecture_loss": {"status": "missing", "path": "missing.csv", "rows": []},
                    "stage2_architecture_loss": {"status": "missing", "path": "missing.csv", "rows": []},
                    "generalization": {"status": "missing", "path": "missing.csv", "rows": []},
                },
                "factorials": {"wan50": {"status": "complete", "root": str(factorial)}},
                "ablations": {},
                "final_configuration": {},
                "issues": [{"item": item, "status": "missing", "detail": "file not found"} for item in sorted(ALLOWED_MISSING)],
            }
            inventory_path = collection / "result_inventory.json"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

            output = export_results(
                project_root=project,
                inventory_path=inventory_path,
                output_root=root / "export",
                allowed_missing=ALLOWED_MISSING,
                include_code=False,
            )

            self.assertTrue((output / "core/result_inventory.json").is_file())
            self.assertTrue((output / "evidence/legacy/legacy_endpoint/raw.jsonl").is_file())
            self.assertTrue((output / "evidence/factorials/wan50/run_manifest.json").is_file())
            self.assertFalse((output / "evidence/factorials/wan50/videos/case/example.mp4").exists())
            self.assertFalse((output / "evidence/factorials/wan50/_private/key.csv").exists())
            self.assertFalse((output / "provenance/task_state/task.log").exists())
            self.assertTrue((output / "provenance/task_state/audit.json").is_file())
            self.assertTrue((output / "SHA256SUMS").is_file())
            exclusions = json.loads((output / "core/declared_exclusions.json").read_text(encoding="utf-8"))
            self.assertEqual(set(exclusions["allowed_missing"]), ALLOWED_MISSING)

            full_output = export_results(
                project_root=project,
                inventory_path=inventory_path,
                output_root=root / "export_full",
                allowed_missing=ALLOWED_MISSING,
                include_videos=True,
                include_private=True,
                include_logs=True,
                include_code=False,
            )
            self.assertTrue((full_output / "evidence/factorials/wan50/videos/case/example.mp4").is_file())
            self.assertTrue((full_output / "evidence/factorials/wan50/_private/key.csv").is_file())
            self.assertTrue((full_output / "provenance/task_state/task.log").is_file())

    def test_rejects_any_issue_outside_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            results = project / "outputs/aaai27_experiments"
            (results / "compiled_tables").mkdir(parents=True)
            inventory = {
                "schema_version": 2,
                "canonical_results_root": str(results),
                "issues": [
                    *[{"item": item} for item in sorted(ALLOWED_MISSING)],
                    {"item": "external.vbench.wan50"},
                ],
            }
            inventory_path = results / "result_inventory.json"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "unexpected issues: external.vbench.wan50"):
                export_results(
                    project_root=project,
                    inventory_path=inventory_path,
                    output_root=root / "export",
                    allowed_missing=ALLOWED_MISSING,
                    include_code=False,
                )


if __name__ == "__main__":
    unittest.main()
