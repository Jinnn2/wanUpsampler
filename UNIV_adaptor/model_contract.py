from __future__ import annotations

import json
from pathlib import Path


WAN21_T2V_REQUIRED_FILES = (
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
)

WAN21_T2V_REQUIRED_CONFIG_KEYS = (
    "dim",
    "ffn_dim",
    "freq_dim",
    "in_dim",
    "num_heads",
    "num_layers",
    "out_dim",
)


def validate_wan21_t2v_model_root(model_root: str | Path) -> dict[str, object]:
    """Validate the official Wan2.1 T2V 1.3B directory contract."""

    root = Path(model_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Wan model root is not a directory: {root}")

    missing_files = [name for name in WAN21_T2V_REQUIRED_FILES if not (root / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Wan model root {root} is incomplete; missing files: {missing_files}"
        )

    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Wan config is not valid JSON: {root / 'config.json'}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Wan config must contain a JSON object: {root / 'config.json'}")

    missing_keys = [key for key in WAN21_T2V_REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise ValueError(
            f"Wan config {root / 'config.json'} is missing architecture keys: {missing_keys}"
        )
    if int(config["dim"]) <= 0 or int(config["num_heads"]) <= 0:
        raise ValueError("Wan dim and num_heads must be positive")
    if int(config["dim"]) % int(config["num_heads"]) != 0:
        raise ValueError(
            "Wan dim must be divisible by num_heads: "
            f"dim={config['dim']}, num_heads={config['num_heads']}"
        )
    return config
