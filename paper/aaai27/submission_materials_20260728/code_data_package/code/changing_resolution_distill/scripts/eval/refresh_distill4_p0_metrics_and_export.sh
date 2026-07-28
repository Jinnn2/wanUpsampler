#!/usr/bin/env bash
set -euo pipefail

# Refresh only Endpoint-RGB-1HR after the P0 sigma=0.12 protocol change,
# merge it into the canonical 18-case metrics, then run the strict exporter.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
WAN_PYTHON="${WAN_PYTHON:-/opt/conda/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/opt/conda/envs/vbench/bin/python}"
VBENCH_ROOT="${VBENCH_ROOT:-/path/to/VBench}"
MAIN_ROOT="${DISTILL4_FINAL_QUALITY_EFFICIENCY:-${PROJECT_ROOT}/outputs/aaai27_experiments/quality_efficiency_distill4}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if (( ${#GPUS[@]} != 4 )); then
  echo "Exactly four comma-separated GPU ids are required; got GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi
WARM_GPU="${WARM_GPU:-${GPUS[0]}}"
REFRESH_VBENCH="${REFRESH_VBENCH:-1}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
RAW_SUBDIR="vbench_raw_p0_sigma012_${timestamp}"
PARTIAL_JSON_NAME="vbench_p0_rgb1_sigma012_${timestamp}.json"
PARTIAL_JSON="${MAIN_ROOT}/metrics/${PARTIAL_JSON_NAME}"
WARM_REFRESH_ROOT="${MAIN_ROOT}/warm_quality_efficiency_p0_sigma012_${timestamp}"
WARM_PROTOCOL="${MAIN_ROOT}/warm_quality_efficiency/protocol.json"

for path in \
  "${MAIN_ROOT}/run_manifest.json" \
  "${MAIN_ROOT}/benchmark_spec.json" \
  "${MAIN_ROOT}/metrics/vbench_v1_custom.json" \
  "${WARM_PROTOCOL}" \
  "${VBENCH_ROOT}/evaluate.py"; do
  [[ -f "${path}" ]] || { echo "Required file not found: ${path}" >&2; exit 1; }
done

if [[ "${REFRESH_VBENCH}" == "1" ]]; then
  echo "[1/5] VBench-5 refresh: endpoint_rgb_1hr only"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${VBENCH_PYTHON}" \
    "${PROJECT_ROOT}/paper/aaai27/experiments/run_vbench_factorials.py" run \
    --factorial-root "${MAIN_ROOT}" \
    --vbench-root "${VBENCH_ROOT}" \
    --python "${VBENCH_PYTHON}" \
    --ngpus 4 \
    --cases endpoint_rgb_1hr \
    --raw-subdir "${RAW_SUBDIR}" \
    --output-name "${PARTIAL_JSON_NAME}"

  echo "[2/5] Merge VBench case and rebuild paired statistics"
  "${WAN_PYTHON}" \
    "${PROJECT_ROOT}/paper/aaai27/experiments/refresh_distill4_p0_results.py" \
    merge-vbench \
    --suite-root "${MAIN_ROOT}" \
    --partial-json "${PARTIAL_JSON}"
  "${WAN_PYTHON}" \
    "${PROJECT_ROOT}/paper/aaai27/experiments/compile_vbench_paired_statistics.py" \
    --factorial-root "${MAIN_ROOT}"
else
  echo "[1-2/5] Reusing the already merged P0 VBench refresh"
fi

mapfile -t WARM_SETTINGS < <(
  "${WAN_PYTHON}" - "${WARM_PROTOCOL}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
settings = payload.get("settings", payload)
for key in (
    "warmup_videos",
    "measured_videos",
    "seed_base",
    "prompt_offset",
    "num_frames",
    "model_root",
    "prompts_file",
    "negative_prompt",
):
    print(settings.get(key, ""))
PY
)
if (( ${#WARM_SETTINGS[@]} != 8 )); then
  echo "Failed to read canonical warm timing protocol: ${WARM_PROTOCOL}" >&2
  exit 1
fi

echo "[3/5] Warm timing refresh: endpoint_rgb_1hr only on GPU ${WARM_GPU}"
warm_command=(
  "${WAN_PYTHON}"
  "${PROJECT_ROOT}/paper/aaai27/experiments/benchmark_warm_quality_efficiency.py"
  --suite-root "${MAIN_ROOT}"
  --output-root "${WARM_REFRESH_ROOT}"
  --python "${WAN_PYTHON}"
  --gpu "${WARM_GPU}"
  --warmup "${WARM_SETTINGS[0]}"
  --repeats "${WARM_SETTINGS[1]}"
  --seed "${WARM_SETTINGS[2]}"
  --prompt-offset "${WARM_SETTINGS[3]}"
  --num-frames "${WARM_SETTINGS[4]}"
  --model-root "${WARM_SETTINGS[5]}"
  --prompts "${WARM_SETTINGS[6]}"
  --negative-prompt "${WARM_SETTINGS[7]}"
  --cases endpoint_rgb_1hr
)
if [[ "${ALLOW_BUSY_GPU:-0}" == "1" ]]; then
  warm_command+=(--allow-busy-gpu)
fi
"${warm_command[@]}"

echo "[4/5] Merge warm timing case into canonical 18-case tables"
"${WAN_PYTHON}" \
  "${PROJECT_ROOT}/paper/aaai27/experiments/refresh_distill4_p0_results.py" \
  merge-warm \
  --suite-root "${MAIN_ROOT}" \
  --partial-root "${WARM_REFRESH_ROOT}"

echo "[5/5] Strict P0/P1/P3 export"
bash "${PROJECT_ROOT}/changing_resolution_distill/scripts/eval/export_distill4_p0_p1_p3_final.sh"
