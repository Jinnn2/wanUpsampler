from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from UNIV_adaptor.model_contract import (
    WAN21_T2V_REQUIRED_FILES,
    validate_wan21_t2v_model_root,
)


class WanModelContractTest(unittest.TestCase):
    def _complete_root(self, root: Path) -> None:
        config = {
            "dim": 1536,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "in_dim": 16,
            "num_heads": 12,
            "num_layers": 30,
            "out_dim": 16,
        }
        for filename in WAN21_T2V_REQUIRED_FILES:
            path = root / filename
            if filename == "config.json":
                path.write_text(json.dumps(config), encoding="utf-8")
            else:
                path.touch()

    def test_complete_official_layout_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_root(root)
            config = validate_wan21_t2v_model_root(root)
            self.assertEqual(config["dim"], 1536)
            self.assertEqual(config["num_heads"], 12)

    def test_missing_config_is_reported_before_scheduler_init(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_root(root)
            (root / "config.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                validate_wan21_t2v_model_root(root)

    def test_missing_architecture_key_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_root(root)
            config_path = root / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            del config["dim"]
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dim"):
                validate_wan21_t2v_model_root(root)


if __name__ == "__main__":
    unittest.main()
