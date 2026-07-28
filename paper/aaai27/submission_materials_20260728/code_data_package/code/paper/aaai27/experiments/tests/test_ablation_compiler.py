from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class AblationCompilerTest(unittest.TestCase):
    def test_compiles_real_checkpoints_and_metrics_with_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variants = []
            axes = ["target_modules", "target_modules", "rank", "rank", "loss", "loss"]
            for index in range(6):
                checkpoint = root / f"variant{index}.safetensors"
                checkpoint.write_bytes(f"checkpoint-{index}".encode())
                metrics = root / f"variant{index}.csv"
                with metrics.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["metric", "samples", "better", "lora_mean"])
                    writer.writeheader()
                    writer.writerow({"metric": "l1", "samples": 10, "better": "lower", "lora_mean": 0.1 + index})
                variants.append(
                    {
                        "axis": axes[index],
                        "variant": f"v{index}",
                        "target_modules": "qkvo+ffn",
                        "rank": 8 * (index + 1),
                        "loss": "main",
                        "train_steps": 10000,
                        "train_seed": 1,
                        "lora_strength": 0.75,
                        "checkpoint": str(checkpoint),
                        "metrics_csv": str(metrics),
                        "columns": {"value": "lora_mean"},
                    }
                )
            registry = root / "registry.json"
            registry.write_text(json.dumps({"variants": variants}), encoding="utf-8")
            output = root / "result.csv"
            script = Path(__file__).parents[1] / "compile_ablation_results.py"
            subprocess.run(
                [sys.executable, str(script), "--kind", "lora", "--registry", str(registry), "--output", str(output)],
                check=True,
                capture_output=True,
            )
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(len(row["checkpoint_sha256"]) == 64 for row in rows))
            self.assertTrue(output.with_suffix(".provenance.json").is_file())


if __name__ == "__main__":
    unittest.main()
