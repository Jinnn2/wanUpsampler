#!/usr/bin/env bash
# Fixed-path exporter for the original experiment machine reached via the
# Remote Tunnel named "temp".
#
# This operational script intentionally lives outside the anonymous AAAI
# Code/Data ZIP because it contains the original machine's absolute paths.

set -Eeuo pipefail
umask 077

usage() {
  cat <<'EOF'
Usage:
  bash export_temp_fixed_paths.sh [metadata|full|all]

Modes:
  metadata  Export environment, Git state, LMDB metadata/splits, raw-video
            inventories, final-output inventories, and small training records.
  full      Export the five checksum-verified custom checkpoints, source
            snapshot, Git state, and full environment.
  all       Run both exports.

Optional environment:
  PACKAGE_ROOT  Directory containing reproduction_assets.json and tools/.
                Defaults to the fixed project-side Code/Data package.
EOF
}

MODE="${1:-metadata}"
case "${MODE}" in
  metadata|full|all) ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown mode: ${MODE}" >&2; usage >&2; exit 2 ;;
esac

# Fixed original-machine paths.
PROJECT_ROOT="/mnt/afs_2/houze/wanUpsampler"
LIGHTX2V_REPO="/mnt/afs_2/houze/LightX2V"
DIFFSYNTH_REPO="/mnt/afs_2/houze/DiffSynth-Studio"
VBENCH_REPO="/mnt/afs_2/houze/VBench"
REALESRGAN_REPO="/mnt/afs_2/houze/Real-ESRGAN"
MODEL_ROOT="/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B"
DISTILL_MODEL_ROOT="/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill"
WAN_PYTHON="/opt/conda/bin/python"
VBENCH_PYTHON="/opt/conda/envs/vbench/bin/python"

WAN50_RAW_VIDEO_DIR="${PROJECT_ROOT}/data/changing_resolution/raw_wan21_720p_1k"
DISTILL4_RAW_VIDEO_DIR="${PROJECT_ROOT}/data/changing_resolution_distill/raw_wan21_14b_cfgdistill_720p_5k"
WAN50_ITU_LMDB="${PROJECT_ROOT}/data/changing_resolution/lmdb_368x640_720x1248_1k"
WAN50_TTD40_LMDB="${PROJECT_ROOT}/data/changing_resolution/lmdb_tail_skip_lora_step40_to_step50"
WAN50_TTD45_LMDB="${PROJECT_ROOT}/data/changing_resolution/lmdb_tail_skip_lora_step45_to_step50"
DISTILL4_ITU_LMDB="${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_clean_368x640_720x1248_14b_cfgdistill_5k"
DISTILL4_TTD3_LMDB="${PROJECT_ROOT}/data/changing_resolution_distill/lmdb_last_step_skip_lora_368x640_14b_cfgdistill_5k_step3"

WAN50_ITU_OUTPUT="${PROJECT_ROOT}/outputs/changing_resolution_clean_368x640_720x1248_stage2_lmdb"
WAN50_TTD40_OUTPUT="${PROJECT_ROOT}/outputs/changing_resolution_tail_skip_lora_step40_to_step50_temporal"
WAN50_TTD45_OUTPUT="${PROJECT_ROOT}/outputs/changing_resolution_tail_skip_lora_step45_to_step50"
DISTILL4_ITU_OUTPUT="${PROJECT_ROOT}/outputs/changing_resolution_distill_clean_368x640_720x1248_stage2_14b_cfgdistill_5k_lmdb"
DISTILL4_TTD3_OUTPUT="${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_368x640_14b_cfgdistill_5k_step3"
DISTILL4_TTD3_LEGACY_OUTPUT="${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3"

PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/paper/aaai27/submission_materials_20260728/code_data_package}"
METADATA_EXPORTER="${PACKAGE_ROOT}/tools/export_missing_repro_metadata.sh"
FULL_EXPORTER="${PACKAGE_ROOT}/tools/export_full_repro_bundle.sh"
VERIFY_CHECKPOINT_HASHES="${VERIFY_CHECKPOINT_HASHES:-1}"

EXPORT_PARENT="${PROJECT_ROOT}/outputs/aaai27_repro_exports"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="intrascale_temp_${MODE}_${STAMP}"
RUN_ROOT="${EXPORT_PARENT}/${RUN_NAME}"
ARCHIVE="${EXPORT_PARENT}/${RUN_NAME}.tar.gz"

export PROJECT_ROOT LIGHTX2V_REPO DIFFSYNTH_REPO VBENCH_REPO REALESRGAN_REPO
export MODEL_ROOT DISTILL_MODEL_ROOT WAN_PYTHON VBENCH_PYTHON

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "Fixed project root is unavailable: ${PROJECT_ROOT}" >&2
  exit 1
fi
if [[ -e "${RUN_ROOT}" || -e "${ARCHIVE}" ]]; then
  echo "Refusing to overwrite an existing export: ${RUN_ROOT} or ${ARCHIVE}" >&2
  exit 1
fi
mkdir -p "${EXPORT_PARENT}" "${RUN_ROOT}/audit"

PREFLIGHT="${RUN_ROOT}/audit/preflight_paths.tsv"
printf 'kind\tlogical_name\trequired\tstatus\tpath\n' > "${PREFLIGHT}"
MISSING_REQUIRED=0

record_path() {
  local kind="$1"
  local name="$2"
  local required="$3"
  local path="$4"
  local status="missing"
  case "${kind}" in
    dir) [[ -d "${path}" ]] && status="present" ;;
    file) [[ -f "${path}" ]] && status="present" ;;
    exe) [[ -x "${path}" ]] && status="present" ;;
    *) echo "Unknown path kind: ${kind}" >&2; exit 2 ;;
  esac
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${kind}" "${name}" "${required}" "${status}" "${path}" >> "${PREFLIGHT}"
  if [[ "${required}" == "yes" && "${status}" != "present" ]]; then
    MISSING_REQUIRED=$((MISSING_REQUIRED + 1))
  fi
}

record_path dir project yes "${PROJECT_ROOT}"
record_path dir lightx2v yes "${LIGHTX2V_REPO}"
record_path dir diffsynth yes "${DIFFSYNTH_REPO}"
record_path dir vbench yes "${VBENCH_REPO}"
record_path dir realesrgan no "${REALESRGAN_REPO}"
record_path dir wan13b yes "${MODEL_ROOT}"
record_path dir distill14b yes "${DISTILL_MODEL_ROOT}"
record_path exe wan_python yes "${WAN_PYTHON}"
record_path exe vbench_python yes "${VBENCH_PYTHON}"
record_path dir wan50_raw_videos yes "${WAN50_RAW_VIDEO_DIR}"
record_path dir distill4_raw_videos yes "${DISTILL4_RAW_VIDEO_DIR}"
record_path dir wan50_itu_lmdb yes "${WAN50_ITU_LMDB}"
record_path dir wan50_ttd40_lmdb yes "${WAN50_TTD40_LMDB}"
record_path dir wan50_ttd45_lmdb yes "${WAN50_TTD45_LMDB}"
record_path dir distill4_itu_lmdb yes "${DISTILL4_ITU_LMDB}"
record_path dir distill4_ttd3_lmdb yes "${DISTILL4_TTD3_LMDB}"
record_path file metadata_exporter yes "${METADATA_EXPORTER}"
record_path file full_exporter yes "${FULL_EXPORTER}"
record_path file asset_manifest yes "${PACKAGE_ROOT}/reproduction_assets.json"

printf 'missing_required\t%s\n' "${MISSING_REQUIRED}" \
  > "${RUN_ROOT}/audit/preflight_summary.tsv"

write_video_inventory() {
  local name="$1"
  local root="$2"
  local out="${RUN_ROOT}/audit/${name}_videos.tsv"
  printf 'relative_path\tsize_bytes\tmtime_epoch\n' > "${out}"
  if [[ ! -d "${root}" ]]; then
    printf '0\n' > "${RUN_ROOT}/audit/${name}_video_count.txt"
    printf '0\n' > "${RUN_ROOT}/audit/${name}_video_bytes.txt"
    return
  fi
  find "${root}" -type f -iname '*.mp4' \
    -printf '%P\t%s\t%T@\n' | LC_ALL=C sort >> "${out}"
  awk 'NR>1 {count += 1; bytes += $2} END {
    print count + 0 > count_file
    print bytes + 0 > bytes_file
  }' \
    count_file="${RUN_ROOT}/audit/${name}_video_count.txt" \
    bytes_file="${RUN_ROOT}/audit/${name}_video_bytes.txt" \
    "${out}"
}

write_output_inventory() {
  local name="$1"
  local root="$2"
  local out="${RUN_ROOT}/audit/${name}_output_inventory.tsv"
  printf 'relative_path\tsize_bytes\tmtime_epoch\n' > "${out}"
  if [[ -d "${root}" ]]; then
    find "${root}" -maxdepth 3 -type f \
      -printf '%P\t%s\t%T@\n' | LC_ALL=C sort >> "${out}"
  fi
}

copy_small_training_records() {
  local name="$1"
  local root="$2"
  local out="${RUN_ROOT}/training_records/${name}"
  [[ -d "${root}" ]] || return
  mkdir -p "${out}"
  while IFS= read -r -d '' source; do
    local relative="${source#${root}/}"
    local target="${out}/${relative}"
    mkdir -p "$(dirname "${target}")"
    cp --preserve=timestamps "${source}" "${target}"
  done < <(
    find "${root}" -maxdepth 3 -type f \
      \( -iname '*.json' -o -iname '*.yaml' -o -iname '*.yml' \
         -o -iname '*.csv' -o -iname '*.jsonl' -o -iname '*.log' \
         -o -iname '*.txt' \) \
      -size -64M -print0
  )
}

copy_project_file() {
  local relative="$1"
  local source="${PROJECT_ROOT}/${relative}"
  local target="${RUN_ROOT}/project_records/${relative}"
  [[ -f "${source}" ]] || return
  mkdir -p "$(dirname "${target}")"
  cp --preserve=timestamps "${source}" "${target}"
}

write_model_inventory() {
  local name="$1"
  local root="$2"
  local out="${RUN_ROOT}/audit/${name}_model_inventory.tsv"
  printf 'relative_path\tsize_bytes\tmtime_epoch\n' > "${out}"
  if [[ -d "${root}" ]]; then
    find "${root}" -maxdepth 3 -type f \
      -printf '%P\t%s\t%T@\n' | LC_ALL=C sort >> "${out}"
  fi
}

verify_custom_checkpoints() {
  local manifest="${PACKAGE_ROOT}/reproduction_assets.json"
  local output="${RUN_ROOT}/audit/custom_checkpoint_verification.json"
  [[ -f "${manifest}" ]] || return
  "${WAN_PYTHON}" - \
    "${manifest}" "${PROJECT_ROOT}" "${output}" "${VERIFY_CHECKPOINT_HASHES}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, project_root, output_path, verify_hashes = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
root = Path(project_root)
rows = {}
for name, spec in manifest["custom_checkpoints"].items():
    path = root / spec["source_relative_to_project"]
    row = {
        "relative_path": spec["source_relative_to_project"],
        "exists": path.is_file(),
        "expected_size_bytes": spec["size_bytes"],
        "expected_sha256": spec["sha256"],
    }
    if path.is_file():
        row["actual_size_bytes"] = path.stat().st_size
        row["size_matches"] = row["actual_size_bytes"] == spec["size_bytes"]
        if verify_hashes == "1":
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                    digest.update(chunk)
            row["actual_sha256"] = digest.hexdigest()
            row["sha256_matches"] = row["actual_sha256"] == spec["sha256"]
    rows[name] = row
Path(output_path).write_text(
    json.dumps({"schema_version": 1, "checkpoints": rows}, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY
}

if [[ "${MODE}" == "metadata" || "${MODE}" == "all" ]]; then
  if [[ ! -f "${METADATA_EXPORTER}" ]]; then
    echo "Metadata exporter is missing: ${METADATA_EXPORTER}" >&2
    exit 1
  fi
  bash "${METADATA_EXPORTER}" \
    --project-root "${PROJECT_ROOT}" \
    --output "${RUN_ROOT}/metadata"

  write_video_inventory "wan50" "${WAN50_RAW_VIDEO_DIR}"
  write_video_inventory "distill4" "${DISTILL4_RAW_VIDEO_DIR}"
  write_model_inventory "wan13b" "${MODEL_ROOT}"
  write_model_inventory "distill14b" "${DISTILL_MODEL_ROOT}"
  write_output_inventory "wan50_itu" "${WAN50_ITU_OUTPUT}"
  write_output_inventory "wan50_ttd40" "${WAN50_TTD40_OUTPUT}"
  write_output_inventory "wan50_ttd45" "${WAN50_TTD45_OUTPUT}"
  write_output_inventory "distill4_itu" "${DISTILL4_ITU_OUTPUT}"
  write_output_inventory "distill4_ttd3" "${DISTILL4_TTD3_OUTPUT}"
  write_output_inventory "distill4_ttd3_legacy" "${DISTILL4_TTD3_LEGACY_OUTPUT}"

  copy_small_training_records "wan50_itu" "${WAN50_ITU_OUTPUT}"
  copy_small_training_records "wan50_ttd40" "${WAN50_TTD40_OUTPUT}"
  copy_small_training_records "wan50_ttd45" "${WAN50_TTD45_OUTPUT}"
  copy_small_training_records "distill4_itu" "${DISTILL4_ITU_OUTPUT}"
  copy_small_training_records "distill4_ttd3" "${DISTILL4_TTD3_OUTPUT}"
  copy_small_training_records "distill4_ttd3_legacy" "${DISTILL4_TTD3_LEGACY_OUTPUT}"
  copy_small_training_records "wan50_final_results" \
    "${PROJECT_ROOT}/outputs/aaai27_experiments/quality_efficiency_final"
  copy_small_training_records "distill4_final_results" \
    "${PROJECT_ROOT}/outputs/aaai27_experiments/quality_efficiency_distill4"

  for relative in \
    changing_resolution/configs/train_clean_368x640_to_720x1248_lmdb_stage2.yaml \
    changing_resolution/configs/train_tail_skip_lora_step40_temporal.yaml \
    changing_resolution/configs/train_tail_skip_lora_step45.yaml \
    changing_resolution/configs/wan_t2v_stage3_compare_10_prompts.txt \
    changing_resolution_distill/configs/train_clean_368x640_to_720x1248_lmdb_stage2_distill.yaml \
    changing_resolution_distill/configs/train_last_step_skip_lora_368x640_distill.yaml \
    paper/aaai27/experiments/distill4_talh_validation_prompts_8.txt \
    paper/aaai27/experiments/experiment_manifest.json; do
    copy_project_file "${relative}"
  done

  verify_custom_checkpoints
fi

if [[ "${MODE}" == "full" || "${MODE}" == "all" ]]; then
  if [[ ! -f "${FULL_EXPORTER}" ]]; then
    echo "Full exporter is missing: ${FULL_EXPORTER}" >&2
    exit 1
  fi
  bash "${FULL_EXPORTER}" \
    --project-root "${PROJECT_ROOT}" \
    --output "${RUN_ROOT}/full_repro"
fi

{
  echo "schema_version=1"
  echo "mode=${MODE}"
  echo "created_utc=${STAMP}"
  echo "hostname=$(hostname)"
  echo "project_root=${PROJECT_ROOT}"
  echo "package_root=${PACKAGE_ROOT}"
  echo "missing_required_paths=${MISSING_REQUIRED}"
  echo "checkpoint_hash_verification=${VERIFY_CHECKPOINT_HASHES}"
} > "${RUN_ROOT}/EXPORT_SUMMARY.txt"

(
  cd "${RUN_ROOT}"
  find . -type f ! -name SHA256SUMS -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

tar -C "${EXPORT_PARENT}" -czf "${ARCHIVE}" "${RUN_NAME}"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"

echo "EXPORT_DIR=${RUN_ROOT}"
echo "ARCHIVE=${ARCHIVE}"
echo "ARCHIVE_SHA256=${ARCHIVE}.sha256"
if (( MISSING_REQUIRED > 0 )); then
  echo "WARNING: ${MISSING_REQUIRED} required paths were missing; inspect ${PREFLIGHT}" >&2
fi
