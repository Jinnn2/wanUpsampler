#!/usr/bin/env bash
set -euo pipefail

# Minimal DAVIS 2017 TrainVal 480p downloader.
# Proxy variables such as http_proxy / https_proxy are inherited from the shell.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PATH_CONFIG="${SCRIPT_DIR}/../configs/local_paths.sh"
PATH_CONFIG="${PATH_CONFIG:-${DEFAULT_PATH_CONFIG}}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/datasets}"
RAW_VIDEO_DIR="${RAW_VIDEO_DIR:-${PROJECT_ROOT}/data/raw_videos}"

DAVIS_URL="${DAVIS_URL:-https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip}"
DAVIS_ZIP="${DAVIS_ZIP:-${DATASET_ROOT}/DAVIS-2017-trainval-480p.zip}"
DAVIS_DIR="${DAVIS_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${RAW_VIDEO_DIR}/davis2017_480p}"
FPS="${FPS:-24}"
CRF="${CRF:-12}"

mkdir -p "${DATASET_ROOT}" "${OUTPUT_DIR}"

if [[ ! -f "${DAVIS_ZIP}" ]]; then
  wget -c "${DAVIS_URL}" -O "${DAVIS_ZIP}"
fi

FRAME_ROOT=""
if [[ -n "${DAVIS_DIR}" && -d "${DAVIS_DIR}/JPEGImages/480p" ]]; then
  FRAME_ROOT="${DAVIS_DIR}/JPEGImages/480p"
else
  FRAME_ROOT="$(find "${DATASET_ROOT}" -path '*/JPEGImages/480p' -type d -print -quit)"
fi

if [[ -z "${FRAME_ROOT}" ]]; then
  unzip -q "${DAVIS_ZIP}" -d "${DATASET_ROOT}"
fi

if [[ -n "${DAVIS_DIR}" && -d "${DAVIS_DIR}/JPEGImages/480p" ]]; then
  FRAME_ROOT="${DAVIS_DIR}/JPEGImages/480p"
else
  FRAME_ROOT="$(find "${DATASET_ROOT}" -path '*/JPEGImages/480p' -type d -print -quit)"
fi

if [[ -z "${FRAME_ROOT}" || ! -d "${FRAME_ROOT}" ]]; then
  echo "DAVIS frame directory not found under: ${DATASET_ROOT}" >&2
  echo "Check extracted layout with: find ${DATASET_ROOT} -maxdepth 6 -type d | head -80" >&2
  exit 1
fi

for seq_dir in "${FRAME_ROOT}"/*; do
  [[ -d "${seq_dir}" ]] || continue
  name="$(basename "${seq_dir}")"
  out="${OUTPUT_DIR}/${name}.mp4"
  if [[ -f "${out}" ]]; then
    if ffprobe -v error -select_streams v:0 -show_entries stream=codec_type -of csv=p=0 "${out}" >/dev/null 2>&1; then
      continue
    fi
    echo "Removing invalid video: ${out}" >&2
    rm -f "${out}"
  fi

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
