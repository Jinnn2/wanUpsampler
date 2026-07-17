from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from paper.aaai27.experiments.export_results import export_results
from paper.aaai27.experiments.restore_results import build_restore_plan, restore_results, verify_checksums


class ResultRestoreTest(unittest.TestCase):
    def test_verifies_and_hardlink_restores_deleted_canonical_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            results = project / "outputs/aaai27_experiments"
            (results / "compiled_tables").mkdir(parents=True)
            (results / "factorial/videos/case").mkdir(parents=True)
            (results / "paper_tables.md").write_text("# tables\n", encoding="utf-8")
            (results / "compiled_tables/table.csv").write_text("value\n1\n", encoding="utf-8")
            (results / "factorial/run_manifest.json").write_text("{}\n", encoding="utf-8")
            (results / "factorial/videos/case/video.mp4").write_bytes(b"video")
            inventory = {
                "schema_version": 2,
                "canonical_results_root": str(results),
                "sources": {},
                "factorials": {"wan50": {"status": "complete", "root": str(results / "factorial")}},
                "ablations": {},
                "final_configuration": {},
                "issues": [],
            }
            (results / "result_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
            export = export_results(
                project_root=project,
                inventory_path=results / "result_inventory.json",
                output_root=root / "export",
                allowed_missing=set(),
                include_videos=True,
                include_code=False,
            )
            verify_checksums(export)
            plan = build_restore_plan(export, project)
            shutil.rmtree(results)

            restore_results(plan, results, hardlink=True)

            restored_video = results / "factorial/videos/case/video.mp4"
            exported_video = export / "evidence/factorials/wan50/videos/case/video.mp4"
            self.assertEqual(restored_video.read_bytes(), b"video")
            self.assertEqual(restored_video.stat().st_ino, exported_video.stat().st_ino)
            self.assertTrue((results / "compiled_tables/table.csv").is_file())
            self.assertNotEqual(
                (results / "compiled_tables/table.csv").stat().st_ino,
                (export / "core/compiled_tables/table.csv").stat().st_ino,
            )

    def test_checksum_verification_rejects_changed_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "file.txt").write_text("changed", encoding="utf-8")
            (root / "SHA256SUMS").write_text(f"{'0' * 64}  file.txt\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "checksum mismatch"):
                verify_checksums(root)


if __name__ == "__main__":
    unittest.main()
