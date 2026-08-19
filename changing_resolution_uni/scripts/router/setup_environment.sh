#!/usr/bin/env bash
set -euo pipefail

echo "================================================================================"
echo " [1/3] Installing All Required Dependencies..."
echo "================================================================================"

# 1. Uninstall any fake/dummy 'clip' package
pip uninstall -y clip 2>/dev/null || true

# 2. Install official OpenAI CLIP from git
echo "Installing OpenAI CLIP..."
pip install git+https://github.com/openai/CLIP.git --no-cache-dir

# 3. Install PyIQA (for MUSIQ imaging quality) and other dependencies
echo "Installing PyIQA and supporting packages..."
pip install pyiqa huggingface_hub lmdb imageio ftfy regex tqdm

echo ""
echo "================================================================================"
echo " [2/3] Cleaning Broken Hub Caches & Pre-warming All Models in Single Process..."
echo "================================================================================"

# Clean potential broken zip/dino directories from previous race conditions
rm -rf "${HOME}/.cache/torch/hub/main.zip"* \
       "${HOME}/.cache/torch/hub/facebookresearch-dino"* \
       "${HOME}/.cache/torch/hub/facebookresearch_dino"* 2>/dev/null || true

python - <<'PY'
import sys
import time
import torch

print("-> Pre-downloading and validating DINO (facebookresearch/dino:main)...")
for attempt in range(5):
    try:
        torch.hub.load("facebookresearch/dino:main", "dino_vits16")
        print("   [OK] DINO pre-warmed.")
        break
    except Exception as e:
        print(f"   [Retry {attempt+1}] DINO failed: {e}. Retrying in 3s...")
        time.sleep(3)

import clip
for m in ["ViT-B/32", "ViT-L/14"]:
    print(f"-> Pre-downloading CLIP {m}...")
    for attempt in range(5):
        try:
            clip.load(m, device="cpu")
            print(f"   [OK] CLIP {m} ready.")
            break
        except Exception as e:
            print(f"   [Retry {attempt+1}] CLIP {m} failed: {e}. Retrying in 3s...")
            time.sleep(3)

print("-> Pre-warming PyIQA MUSIQ model...")
try:
    from pyiqa.archs.musiq_arch import MUSIQ
    print("   [OK] PyIQA MUSIQ imported.")
except Exception as e:
    print(f"   [Warning] PyIQA import: {e}")
PY

echo ""
echo "================================================================================"
echo " [3/3] Environment Verification..."
echo "================================================================================"
python - <<'PY'
import torch
import clip
import pyiqa

assert hasattr(clip, "load"), "CLIP load function missing!"
print(" Environment Setup & Pre-warming 100% Successful!")
PY
