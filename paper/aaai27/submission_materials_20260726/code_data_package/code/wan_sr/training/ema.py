from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {"decay": self.decay, "shadow": copy.deepcopy(self.shadow)}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.shadow = {name: tensor.clone() for name, tensor in state["shadow"].items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        params = dict(model.named_parameters())
        for name, tensor in self.shadow.items():
            if name in params:
                params[name].copy_(tensor)
