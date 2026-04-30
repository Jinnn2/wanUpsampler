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

mkdir -p "${DATASET_DIR}" "${RAW_VIDEO_DIR}"

download_one() {
  local name="$1"
  local url="$2"
  local dst="${DATASET_DIR}/${name}.mp4"
  local out="${RAW_VIDEO_DIR}/${name}.mp4"

  if [[ ! -f "${dst}" ]]; then
    echo "Downloading ${name}"
    wget -c -O "${dst}" "${url}"
  fi

  if ! ffprobe -v error "${dst}" >/dev/null 2>&1; then
    echo "Invalid downloaded video: ${dst}" >&2
    exit 1
  fi

  if [[ ! -f "${out}" ]]; then
    ffmpeg -hide_banner -loglevel error -y -i "${dst}" \
      -map 0:v:0 -an -vf "fps=16" -c:v libx264 -pix_fmt yuv420p -crf 18 "${out}"
  fi
}

download_one "big_buck_bunny" "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
download_one "elephants_dream" "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4"
download_one "sintel" "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4"
download_one "tears_of_steel" "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4"
download_one "for_bigger_blazes" "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
download_one "for_bigger_escapes" "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4"
download_one "for_bigger_fun" "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"
download_one "for_bigger_joyrides" "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4"
download_one "for_bigger_meltdowns" "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4"
download_one "subaru_outback" "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackOnStreetAndDirt.mp4"
download_one "volkswagen_gti" "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/VolkswagenGTIReview.mp4"

count="$(find "${RAW_VIDEO_DIR}" -type f -name '*.mp4' | wc -l)"
echo "Open video dataset ready: ${RAW_VIDEO_DIR} (${count} mp4 files)"
