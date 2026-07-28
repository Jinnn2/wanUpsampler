from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    ema: object | None = None,
    step: int = 0,
    config: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "config": config or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    ema: object | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    payload = torch.load(path, map_location=map_location)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if ema is not None and "ema" in payload:
        ema.load_state_dict(payload["ema"])
    return int(payload.get("step", 0))
