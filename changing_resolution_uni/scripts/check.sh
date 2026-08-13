#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

python - <<'PY'
import torch
from changing_resolution_uni.model import UniversalCleanLatentUpsampler
from changing_resolution_uni.losses import UniversalCleanUpsampleLoss

model = UniversalCleanLatentUpsampler(hidden_channels=32, cond_dim=64, pre_blocks=1, post_blocks=1)
x = torch.randn(2, 16, 5, 4, 6)
for target in ((6, 9), (8, 12), (12, 18)):
    y, aux = model(x, output_size=target, return_aux=True)
    assert tuple(y.shape) == (2, 16, 5, *target), y.shape
    assert torch.isfinite(y).all()
    assert aux["subpixel_weights"] is not None
    weights = aux["subpixel_weights"]
    assert weights.shape == (2, 27, 5 * target[0] * target[1]), weights.shape
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]), atol=1e-5)
    loss, _ = UniversalCleanUpsampleLoss()(y, torch.randn_like(y), x)
    assert torch.isfinite(loss)
    loss.backward()
    model.zero_grad(set_to_none=True)
print("U-ITU model smoke test passed")
print("params", sum(p.numel() for p in model.parameters()))
PY
