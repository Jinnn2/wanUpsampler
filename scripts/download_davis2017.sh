#!/usr/bin/env bash
set -euo pipefail

# Download DAVIS 2017 TrainVal 480p and convert image sequences to mp4 files
# consumable by scripts/build_latent_pairs.py.
#
# Usage:
#   bash scripts/download_davis2017.sh
#
# Override paths:
#   DATASET_ROOT=/data/yongyang/Jin/datasets RAW_VIDEO_DIR=/data/yongyang/Jin/wanUpsampler/data/raw_videos \
#   bash scripts/download_davis2017.sh

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/datasets}"
RAW_VIDEO_DIR="${RAW_VIDEO_DIR:-${PROJECT_ROOT}/data/raw_videos}"
DAVIS_URL="${DAVIS_URL:-https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip}"
DAVIS_ZIP="${DAVIS_ZIP:-${DATASET_ROOT}/DAVIS-2017-trainval-480p.zip}"
DAVIS_EXTRACT_DIR="${DAVIS_EXTRACT_DIR:-${DATASET_ROOT}/DAVIS}"
DAVIS_FPS="${DAVIS_FPS:-24}"
VIDEO_CRF="${VIDEO_CRF:-12}"

mkdir -p "${DATASET_ROOT}" "${RAW_VIDEO_DIR}"

download_file() {
  if [[ -f "${DAVIS_ZIP}" ]]; then
    echo "Found existing archive: ${DAVIS_ZIP}"
    return
  fi

  echo "Downloading DAVIS 2017 TrainVal 480p..."
  if command -v wget >/dev/null 2>&1; then
    wget -c "${DAVIS_URL}" -O "${DAVIS_ZIP}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --continue-at - "${DAVIS_URL}" -o "${DAVIS_ZIP}"
  else
    echo "Neither wget nor curl is available." >&2
    exit 1
  fi
}

extract_archive() {
  if [[ -d "${DAVIS_EXTRACT_DIR}/JPEGImages/480p" ]]; then
    echo "Found extracted DAVIS frames: ${DAVIS_EXTRACT_DIR}/JPEGImages/480p"
    return
  fi

  echo "Extracting ${DAVIS_ZIP}..."
  if command -v unzip >/dev/null 2>&1; then
    unzip -q "${DAVIS_ZIP}" -d "${DATASET_ROOT}"
  else
    python - "${DAVIS_ZIP}" "${DATASET_ROOT}" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(out_dir)
PY
  fi
}

convert_sequences_with_ffmpeg() {
  local frame_root="${DAVIS_EXTRACT_DIR}/JPEGImages/480p"
  local output_root="${RAW_VIDEO_DIR}/davis2017_480p"
  mkdir -p "${output_root}"

  shopt -s nullglob
  local sequence_dir
  local count=0
  for sequence_dir in "${frame_root}"/*; do
    [[ -d "${sequence_dir}" ]] || continue
    local name
    name="$(basename "${sequence_dir}")"
    local output="${output_root}/${name}.mp4"
    if [[ -f "${output}" ]]; then
      echo "Skip existing video: ${output}"
      continue
    fi
    echo "Converting ${name} -> ${output}"
    ffmpeg -hide_banner -loglevel error -y \
      -framerate "${DAVIS_FPS}" \
      -pattern_type glob \
      -i "${sequence_dir}/*.jpg" \
      -vf "format=yuv420p" \
      -c:v libx264 \
      -crf "${VIDEO_CRF}" \
      "${output}"
    count=$((count + 1))
  done
  echo "Converted ${count} DAVIS sequences into ${output_root}"
}

convert_sequences_with_python() {
  python - "${DAVIS_EXTRACT_DIR}/JPEGImages/480p" "${RAW_VIDEO_DIR}/davis2017_480p" "${DAVIS_FPS}" <<'PY'
import sys
from pathlib import Path

import imageio.v3 as iio

frame_root = Path(sys.argv[1])
output_root = Path(sys.argv[2])
fps = int(sys.argv[3])
output_root.mkdir(parents=True, exist_ok=True)

converted = 0
for seq_dir in sorted(p for p in frame_root.iterdir() if p.is_dir()):
    out_path = output_root / f"{seq_dir.name}.mp4"
    if out_path.exists():
        print(f"Skip existing video: {out_path}")
        continue
    frames = [iio.imread(path) for path in sorted(seq_dir.glob("*.jpg"))]
    if not frames:
        continue
    print(f"Converting {seq_dir.name} -> {out_path}")
    iio.imwrite(out_path, frames, fps=fps, codec="libx264", macro_block_size=1)
    converted += 1

print(f"Converted {converted} DAVIS sequences into {output_root}")
PY
}

download_file
extract_archive

if command -v ffmpeg >/dev/null 2>&1; then
  convert_sequences_with_ffmpeg
else
  convert_sequences_with_python
fi

echo
echo "DAVIS raw videos are ready under:"
echo "  ${RAW_VIDEO_DIR}/davis2017_480p"
echo
echo "Next step:"
echo "  RAW_VIDEO_DIR=${RAW_VIDEO_DIR} bash scripts/run_lightx2v_training.sh build"
