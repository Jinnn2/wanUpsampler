#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PATH_CONFIG="${PATH_CONFIG:-${PROJECT_ROOT}/configs/local_paths.sh}"
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

DATASET_DIR="${CR_DATASET_DIR:-${PROJECT_ROOT}/datasets/open_video_720p}"
RAW_VIDEO_DIR="${CR_RAW_VIDEO_DIR:-${PROJECT_ROOT}/data/changing_resolution/raw_open_video_720p}"
SOURCES_FILE="${CR_SOURCES_FILE:-${PROJECT_ROOT}/changing_resolution/configs/open_video_720p_sources.tsv}"
MIN_VIDEOS="${MIN_VIDEOS:-2}"

mkdir -p "${DATASET_DIR}" "${RAW_VIDEO_DIR}"

download_file() {
  local url="$1"
  local dst="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 3 --connect-timeout 30 \
      -C - -o "${dst}" "${url}"
    return
  fi

  wget -c -O "${dst}" "${url}"
}

download_one() {
  local name="$1"
  local url="$2"
  local ext="${url##*.}"
  local dst="${DATASET_DIR}/${name}.${ext}"
  local out="${RAW_VIDEO_DIR}/${name}.mp4"

  if [[ ! -f "${dst}" ]]; then
    echo "Downloading ${name}"
    if ! download_file "${url}" "${dst}"; then
      echo "[warn] download failed, skip: ${name}" >&2
      rm -f "${dst}"
      return 0
    fi
  fi

  if ! ffprobe -v error "${dst}" >/dev/null 2>&1; then
    echo "[warn] invalid downloaded video, skip: ${dst}" >&2
    rm -f "${dst}"
    return 0
  fi

  if [[ ! -f "${out}" ]] || ! ffprobe -v error "${out}" >/dev/null 2>&1; then
    ffmpeg -hide_banner -loglevel error -y -i "${dst}" \
      -map 0:v:0 -an -vf "fps=16" -c:v libx264 -pix_fmt yuv420p -crf 18 "${out}"
  fi
}

if [[ ! -f "${SOURCES_FILE}" ]]; then
  echo "Sources file not found: ${SOURCES_FILE}" >&2
  exit 1
fi

while IFS=$'\t' read -r name url; do
  [[ -z "${name}" || "${name}" == \#* ]] && continue
  download_one "${name}" "${url}"
done < "${SOURCES_FILE}"

count="$(find "${RAW_VIDEO_DIR}" -type f -name '*.mp4' | wc -l)"
if (( count < MIN_VIDEOS )); then
  echo "Only ${count} mp4 files are ready, expected at least ${MIN_VIDEOS}." >&2
  echo "Check proxy/network or edit: ${SOURCES_FILE}" >&2
  exit 1
fi

echo "Open video dataset ready: ${RAW_VIDEO_DIR} (${count} mp4 files)"
