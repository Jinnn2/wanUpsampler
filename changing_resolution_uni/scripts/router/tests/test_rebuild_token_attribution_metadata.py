from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from changing_resolution_uni.scripts.router import rebuild_token_attribution_metadata


class FakeTokenizer:
    def convert_ids_to_tokens(self, values: list[int]) -> list[str]:
        vocabulary = {1: "▁A", 2: "▁scene", 3: "</s>"}
        return [vocabulary[value] for value in values]


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(_: str) -> FakeTokenizer:
        return FakeTokenizer()


class RebuildTokenAttributionMetadataTest(unittest.TestCase):
    def test_reconstructs_metadata_without_reencoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            records_dir = root / "records"
            t5_dir = root / "t5_embeddings"
            out_dir = root / "token_attribution_embeddings"
            records_dir.mkdir(parents=True)
            t5_dir.mkdir()
            record_name = "p000000_s42.json"
            (records_dir / record_name).write_text(
                json.dumps(
                    {
                        "prompt_id": 0,
                        "seed": 42,
                        "prompt_text": "A scene",
                    }
                ),
                encoding="utf-8",
            )
            (root / "dataset_manifest.json").write_text(
                json.dumps({"record_files": [record_name]}), encoding="utf-8"
            )
            np.savez_compressed(
                t5_dir / "prompt_000000.npz",
                pooled_embedding=np.zeros(4096),
                seq_embedding=np.zeros((3, 4096)),
                input_ids=np.asarray([1, 2, 3]),
            )
            fake_transformers = types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
            argv = [
                "rebuild",
                "--dataset-dir",
                str(root),
                "--out-dir",
                str(out_dir),
                "--tokenizer-path",
                "fixture-tokenizer",
            ]
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                with mock.patch.object(sys, "argv", argv):
                    rebuild_token_attribution_metadata.main()

            metadata = json.loads(
                (out_dir / "prompt_000000.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["tokens"], ["▁A", "▁scene", "</s>"])
            self.assertEqual(metadata["natural_word_count"], 1)
            self.assertTrue((out_dir / "prompt_000000.npz").is_file())


if __name__ == "__main__":
    unittest.main()
