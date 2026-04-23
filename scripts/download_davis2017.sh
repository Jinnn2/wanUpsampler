#!/usr/bin/env bash
set -euo pipefail

# Minimal DAVIS 2017 TrainVal 480p downloader.
# Proxy variables such as http_proxy / https_proxy are inherited from the shell.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/datasets}"
RAW_VIDEO_DIR="${RAW_VIDEO_DIR:-${PROJECT_ROOT}/data/raw_videos}"

DAVIS_URL="${DAVIS_URL:-https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip}"
DAVIS_ZIP="${DAVIS_ZIP:-${DATASET_ROOT}/DAVIS-2017-trainval-480p.zip}"
DAVIS_DIR="${DAVIS_DIR:-${DATASET_ROOT}/DAVIS}"
OUTPUT_DIR="${OUTPUT_DIR:-${RAW_VIDEO_DIR}/davis2017_480p}"
FPS="${FPS:-24}"
CRF="${CRF:-12}"

mkdir -p "${DATASET_ROOT}" "${OUTPUT_DIR}"

if [[ ! -f "${DAVIS_ZIP}" ]]; then
  wget -c "${DAVIS_URL}" -O "${DAVIS_ZIP}"
fi

if [[ ! -d "${DAVIS_DIR}/JPEGImages/480p" ]]; then
  unzip -q "${DAVIS_ZIP}" -d "${DATASET_ROOT}"
fi

if [[ ! -d "${DAVIS_DIR}/JPEGImages/480p" ]]; then
  echo "DAVIS frame directory not found: ${DAVIS_DIR}/JPEGImages/480p" >&2
  echo "Check extracted layout with: find ${DATASET_ROOT} -maxdepth 4 -type d | head -50" >&2
  exit 1
fi

for seq_dir in "${DAVIS_DIR}/JPEGImages/480p"/*; do
  [[ -d "${seq_dir}" ]] || continue
  name="$(basename "${seq_dir}")"
  out="${OUTPUT_DIR}/${name}.mp4"
  [[ -f "${out}" ]] && continue

  ffmpeg -hide_banner -loglevel error -y \
    -framerate "${FPS}" \
    -pattern_type glob \
    -i "${seq_dir}/*.jpg" \
    -vf "format=yuv420p" \
    -c:v libx264 \
    -crf "${CRF}" \
    "${out}"
done

video_count="$(find "${OUTPUT_DIR}" -type f -name '*.mp4' | wc -l)"
if [[ "${video_count}" == "0" ]]; then
  echo "No mp4 files were produced under ${OUTPUT_DIR}. Check ffmpeg and DAVIS frames." >&2
  exit 1
fi

echo "DAVIS videos are ready: ${OUTPUT_DIR} (${video_count} mp4 files)"
