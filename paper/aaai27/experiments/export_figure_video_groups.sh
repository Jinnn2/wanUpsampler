#!/usr/bin/env bash
set -euo pipefail

# Export the TALH-Q spatial-detail and TALH-E motion candidates used by the
# paper figure. The 2026-07-17 export contains the native-HR and Wan50
# factorial videos. The incremental export contains metadata only, so an
# external directory is required if the final TALH-Q s=0.75 videos were not
# preserved elsewhere.

FINAL_EXPORT="/mnt/afs_2/houze/wanUpsampler/exports/aaai27_final_20260717"
INCREMENTAL_EXPORT="/mnt/afs_2/houze/wanUpsampler/exports/aaai27_closure_20260718_incremental"
OUTPUT_ARCHIVE="${PWD}/talh_figure_video_candidates.tar.gz"
Q_FINAL_DIR=""
ALLOW_INCOMPLETE=0

usage() {
  cat <<'EOF'
Usage:
  bash export_figure_video_groups.sh [options]

Options:
  --final-export DIR       aaai27_final_20260717 export directory
  --incremental-export DIR aaai27_closure_20260718_incremental directory
  --q-final-dir DIR        Directory containing step40_lora_s0p75_stage2_*.mp4
                           or talh40_*.mp4
  --output FILE.tar.gz     Output archive path
  --allow-incomplete       Package the available Q rows if final TALH-Q is absent
  -h, --help               Show this help

Examples:
  # Strict, complete export when regenerated/preserved TALH-Q videos exist:
  bash export_figure_video_groups.sh \
    --q-final-dir /path/to/videos/step40_lora_s0p75_stage2 \
    --output /mnt/afs_2/houze/talh_figure_videos.tar.gz

  # Package everything currently retained by the two exports:
  bash export_figure_video_groups.sh --allow-incomplete
EOF
}

while (($#)); do
  case "$1" in
    --final-export)
      FINAL_EXPORT="$2"
      shift 2
      ;;
    --incremental-export)
      INCREMENTAL_EXPORT="$2"
      shift 2
      ;;
    --q-final-dir)
      Q_FINAL_DIR="$2"
      shift 2
      ;;
    --output)
      OUTPUT_ARCHIVE="$2"
      shift 2
      ;;
    --allow-incomplete)
      ALLOW_INCOMPLETE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$FINAL_EXPORT" ]]; then
  echo "Final export directory not found: $FINAL_EXPORT" >&2
  exit 1
fi

if [[ -f "$INCREMENTAL_EXPORT/export_manifest.json" ]] && \
   grep -Eq '"include_videos"[[:space:]]*:[[:space:]]*false' \
     "$INCREMENTAL_EXPORT/export_manifest.json"; then
  echo "Info: incremental export is metadata-only (include_videos=false)." >&2
fi

NATIVE_DIR="$FINAL_EXPORT/evidence/legacy/changing_resolution_tail_skip_lora_clean_pred_compare_360p_368x640/videos/ori_50"
FACTORIAL_DIR="$FINAL_EXPORT/evidence/factorials/wan50/videos"

required_dirs=(
  "$NATIVE_DIR"
  "$FACTORIAL_DIR/step40_base_interp"
  "$FACTORIAL_DIR/step40_base_stage2"
  "$FACTORIAL_DIR/step45_base_interp"
  "$FACTORIAL_DIR/step45_base_stage2"
  "$FACTORIAL_DIR/step45_lora_stage2"
)
for directory in "${required_dirs[@]}"; do
  if [[ ! -d "$directory" ]]; then
    echo "Required video directory not found: $directory" >&2
    exit 1
  fi
done

# Look for TALH-Q only under names that are known to use strength 0.75. Never
# fall back to factorial_wan50/step40_lora_stage2 because that export used 1.0.
if [[ -z "$Q_FINAL_DIR" ]]; then
  q_candidates=(
    "$FINAL_EXPORT/evidence/canonical/wan50_step40_strength/videos/step40_lora_s0p75_stage2"
    "$INCREMENTAL_EXPORT/evidence/canonical/wan50_step40_strength/videos/step40_lora_s0p75_stage2"
    "$FINAL_EXPORT/evidence/canonical/quality_efficiency_final/videos/talh40"
    "$INCREMENTAL_EXPORT/evidence/canonical/quality_efficiency_final/videos/talh40"
  )
  for directory in "${q_candidates[@]}"; do
    if [[ -d "$directory" ]]; then
      Q_FINAL_DIR="$directory"
      break
    fi
  done
fi

if [[ -n "$Q_FINAL_DIR" && ! -d "$Q_FINAL_DIR" ]]; then
  echo "TALH-Q final directory not found: $Q_FINAL_DIR" >&2
  exit 1
fi

if [[ -z "$Q_FINAL_DIR" && "$ALLOW_INCOMPLETE" -ne 1 ]]; then
  cat >&2 <<'EOF'
Final TALH-Q s=0.75 videos are absent from both exports.
Pass --q-final-dir with a preserved/regenerated step40_lora_s0p75_stage2 or
talh40 directory. Use --allow-incomplete only if a three-row Q group is useful.
EOF
  exit 1
fi

OUTPUT_ARCHIVE="$(realpath -m "$OUTPUT_ARCHIVE")"
mkdir -p "$(dirname "$OUTPUT_ARCHIVE")"
STAGING_PARENT="$(mktemp -d)"
PACKAGE_NAME="talh_figure_video_candidates"
PACKAGE_DIR="$STAGING_PARENT/$PACKAGE_NAME"
mkdir -p "$PACKAGE_DIR/TALH-Q_spatial_detail" "$PACKAGE_DIR/TALH-E_motion"
trap 'rm -rf -- "$STAGING_PARENT"' EXIT

copy_required() {
  local source="$1"
  local destination="$2"
  if [[ ! -s "$source" ]]; then
    echo "Required video missing or empty: $source" >&2
    exit 1
  fi
  cp -p -- "$source" "$destination"
}

find_q_final() {
  local index="$1"
  local seed="$2"
  local candidate
  for candidate in \
    "$Q_FINAL_DIR/step40_lora_s0p75_stage2_${index}_seed${seed}.mp4" \
    "$Q_FINAL_DIR/talh40_${index}_seed${seed}.mp4"; do
    if [[ -s "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

write_group() {
  local group="$1"
  local index="$2"
  local seed="$3"
  local prompt="$4"
  local out="$PACKAGE_DIR/$group/prompt_${index}_seed${seed}"
  mkdir -p "$out"

  copy_required "$NATIVE_DIR/ori_50_${index}_seed${seed}.mp4" \
    "$out/01_Native-HR.mp4"

  if [[ "$group" == "TALH-Q_spatial_detail" ]]; then
    copy_required "$FACTORIAL_DIR/step40_base_interp/step40_base_interp_${index}_seed${seed}.mp4" \
      "$out/02_Trilinear-at-40.mp4"
    copy_required "$FACTORIAL_DIR/step40_base_stage2/step40_base_stage2_${index}_seed${seed}.mp4" \
      "$out/03_CLL-only-at-40.mp4"
    local q_video=""
    if [[ -n "$Q_FINAL_DIR" ]]; then
      q_video="$(find_q_final "$index" "$seed" || true)"
    fi
    if [[ -n "$q_video" ]]; then
      copy_required "$q_video" "$out/04_TALH-Q-at-40.mp4"
    elif [[ "$ALLOW_INCOMPLETE" -ne 1 ]]; then
      echo "Final TALH-Q video missing for prompt $index / seed $seed" >&2
      exit 1
    else
      printf '%s\n' "MISSING: final TALH-Q s=0.75 video was not included in either export." \
        > "$out/04_TALH-Q-at-40_MISSING.txt"
    fi
  else
    copy_required "$FACTORIAL_DIR/step45_base_interp/step45_base_interp_${index}_seed${seed}.mp4" \
      "$out/02_Trilinear-at-45.mp4"
    copy_required "$FACTORIAL_DIR/step45_base_stage2/step45_base_stage2_${index}_seed${seed}.mp4" \
      "$out/03_CLL-only-at-45.mp4"
    copy_required "$FACTORIAL_DIR/step45_lora_stage2/step45_lora_stage2_${index}_seed${seed}.mp4" \
      "$out/04_TALH-E-at-45.mp4"
  fi

  cat > "$out/metadata.txt" <<EOF
prompt_index=$index
seed=$seed
prompt=$prompt
infer_steps=50
sample_shift=8
cfg=6
frames=81
resolution=720x1248
EOF
}

# TALH-Q spatial-detail candidates.
write_group "TALH-Q_spatial_detail" "00" "9700" \
  "A cinematic night market street after rain, vendors cooking under warm lanterns, reflections on wet pavement, realistic crowd motion."
write_group "TALH-Q_spatial_detail" "05" "9705" \
  "A robot arm assembling a circuit board in a clean laboratory, tiny components moving precisely, sharp industrial lighting."
write_group "TALH-Q_spatial_detail" "08" "9708" \
  "A glass greenhouse filled with tropical flowers, sunlight beams through mist, slow dolly movement, rich color detail."

# TALH-E motion candidates.
write_group "TALH-E_motion" "06" "9706" \
  "A dancer performing under neon signs in a narrow alley, fast footwork, handheld cinematic framing, colorful reflections."
write_group "TALH-E_motion" "02" "9702" \
  "A mountain biker riding along a narrow forest trail, dust and leaves flying behind the wheels, dynamic tracking shot."
write_group "TALH-E_motion" "07" "9707" \
  "A golden retriever running through shallow ocean waves at sunset, splashing water, natural playful motion."

cat > "$PACKAGE_DIR/README.txt" <<EOF
TALH paper figure video candidates

Source final export: $FINAL_EXPORT
Source incremental export: $INCREMENTAL_EXPORT
TALH-Q final source: ${Q_FINAL_DIR:-MISSING}

Within every prompt directory, prompt, seed, scheduler settings, frame count,
and output resolution are aligned. Native-HR is named ori_50 in the legacy
export. The old factorial step40_lora_stage2 result is intentionally excluded
because its LoRA strength is 1.0 rather than the final 0.75.
EOF

(
  cd "$STAGING_PARENT"
  tar -czf "$OUTPUT_ARCHIVE" "$PACKAGE_NAME"
)

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$OUTPUT_ARCHIVE" > "${OUTPUT_ARCHIVE}.sha256"
fi

echo "Created: $OUTPUT_ARCHIVE"
echo "Size: $(du -h "$OUTPUT_ARCHIVE" | awk '{print $1}')"
if [[ "$ALLOW_INCOMPLETE" -eq 1 && -z "$Q_FINAL_DIR" ]]; then
  echo "Warning: TALH-Q final row is marked missing in the archive." >&2
fi
