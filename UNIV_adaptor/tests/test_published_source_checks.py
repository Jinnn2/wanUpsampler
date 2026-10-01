"""Source-lock regression tests: runtime caches are not source mutations."""
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from UNIV_adaptor.scripts.data import fetch_published_accelerators as fetcher


class SourceCheckTests(unittest.TestCase):
    def changes(self, output):
        with patch.object(fetcher.subprocess, "run", return_value=SimpleNamespace(stdout=output)):
            return fetcher.source_changes(Path("snapshot"))

    def test_untracked_bytecode_does_not_dirty_source(self):
        self.assertEqual(self.changes("?? scaling_cache/__pycache__/__init__.cpython-311.pyc\0?? wan/__pycache__/cache.pyo\0"), [])

    def test_tracked_bytecode_and_real_untracked_modules_are_rejected(self):
        self.assertEqual(self.changes(" M wan/__pycache__/model.pyc\0?? wan/model_override.py\0?? __pycache__/override.py\0"),
            [" M wan/__pycache__/model.pyc", "?? wan/model_override.py", "?? __pycache__/override.py"])

    def test_tracked_edit_and_rename_keep_status_and_paths(self):
        self.assertEqual(self.changes(" M wan/model.py\0R  new name.py\0old name.py\0"),
            [" M wan/model.py", "R  new name.py (from old name.py)"])

    def test_failed_status_is_not_considered_clean(self):
        with patch.object(fetcher.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "git")):
            with self.assertRaises(subprocess.CalledProcessError):
                fetcher.source_changes(Path("snapshot"))

    def test_gilbert_is_required_and_restored_from_pinned_commit(self):
        manifest, _ = fetcher.load_manifest()
        record = next(r for r in manifest["repositories"] if r["name"] == "jenga")
        self.assertIn("gilbert.py", record["entrypoints"])
        record = {**record, "entrypoints": ["gilbert.py"]}
        target = Path("missing_snapshot")
        with patch.object(fetcher, "check_checkout", side_effect=["missing entrypoints: ['gilbert.py']", "ok"]), \
                patch.object(fetcher, "git", return_value="blob") as git:
            fetcher.restore_missing_entrypoints(record, target)
        git.assert_any_call("restore", "--ignore-skip-worktree-bits", "--source=HEAD", "--worktree", "--", "gilbert.py", cwd=target)


if __name__ == "__main__":
    unittest.main()
