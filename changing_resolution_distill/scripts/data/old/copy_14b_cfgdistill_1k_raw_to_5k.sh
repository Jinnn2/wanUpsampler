#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

SRC_RAW="${SRC_RAW:-${PROJECT_ROOT}/data/changing_resolution_distill/raw_wan21_14b_cfgdistill_720p_1k}"
DST_RAW="${DST_RAW:-${PROJECT_ROOT}/data/changing_resolution_distill/raw_wan21_14b_cfgdistill_720p_5k}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-5000}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
START_SEED="${START_SEED:-620000}"

if [[ ! -d "${SRC_RAW}" ]]; then
  echo "Source raw video dir not found: ${SRC_RAW}" >&2
  exit 1
fi

mkdir -p "${DST_RAW}"

echo "Copy 14B CfgDistill 1k raw videos"
echo "  src: ${SRC_RAW}"
echo "  dst: ${DST_RAW}"
echo "  total_samples: ${TOTAL_SAMPLES}"
echo "  gpu_ids: ${GPU_IDS}"

python - "${SRC_RAW}" "${DST_RAW}" "${TOTAL_SAMPLES}" "${GPU_IDS}" "${START_SEED}" <<'PY'
import re
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
total = int(sys.argv[3])
gpus = [item.strip() for item in sys.argv[4].split(",") if item.strip()]
start_seed = int(sys.argv[5])
num_parts = len(gpus)
if num_parts < 1:
    raise SystemExit("GPU_IDS is empty")

base = total // num_parts
remainder = total % num_parts
ranges = []
offset = 0
for rank in range(num_parts):
    count = base + (1 if rank < remainder else 0)
    ranges.append((rank, offset, offset + count))
    offset += count

def rank_for_index(index: int) -> int:
    for rank, start, end in ranges:
        if start <= index < end:
            return rank
    raise ValueError(f"sample index {index} is outside 0..{total - 1}")

copied = 0
skipped = 0
for path in sorted(src.rglob("*.mp4")):
    match = re.search(r"_(\d{3,8})_seed\d+", path.stem)
    if match is None:
        skipped += 1
        print(f"[skip] cannot infer sample index: {path}", file=sys.stderr)
        continue
    index = int(match.group(1))
    if index >= total:
        skipped += 1
        print(f"[skip] index outside target range: {path}", file=sys.stderr)
        continue
    rank = rank_for_index(index)
    part_dir = dst / f"part_{rank:02d}"
    part_dir.mkdir(parents=True, exist_ok=True)
    target = part_dir / f"wan21_14b_cfgdistill_720p_{index:06d}_seed{start_seed + index}.mp4"
    if target.exists():
        continue
    shutil.copy2(path, target)
    copied += 1

print(f"copied={copied} skipped={skipped}")
PY

count="$(find "${DST_RAW}" -type f -name '*.mp4' | wc -l)"
echo "Raw video copy ready: ${DST_RAW} (${count} mp4 files)"
