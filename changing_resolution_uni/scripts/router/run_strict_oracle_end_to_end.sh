#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RAW_DATASET_DIR="${RAW_DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1k}"
STRICT_DATASET_DIR="${STRICT_DATASET_DIR:-${PROJECT_ROOT}/data/changing_resolution_uni/oracle_dataset_1k_strict}"
T5_SOURCE_DIR="${T5_SOURCE_DIR:-${RAW_DATASET_DIR}/t5_embeddings}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/afs_2/houze/VBench}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/strict_oracle_e2e_1k}"

NGPUS="${NGPUS:-4}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-1000}"
EXPECTED_SEEDS="${EXPECTED_SEEDS:-42 100 2024}"
PRIMARY_LAMBDA="${PRIMARY_LAMBDA:-0.01}"
PILOT_VIDEO_COUNT="${PILOT_VIDEO_COUNT:-8}"
PILOT_CASES="${PILOT_CASES:-step49 step50}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
DIAGNOSTIC_DIMENSIONS="${DIAGNOSTIC_DIMENSIONS:-overall_consistency}"
RUN_TRAINING="${RUN_TRAINING:-0}"
FORCE_RESCORE="${FORCE_RESCORE:-0}"
FROM_STEP="${FROM_STEP:-1}"
TO_STEP="${TO_STEP:-8}"

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
SEED="${SEED:-42}"

if ! [[ "${FROM_STEP}" =~ ^[1-8]$ && "${TO_STEP}" =~ ^[1-8]$ ]]; then
  echo "FROM_STEP and TO_STEP must be integers in [1, 8]." >&2
  exit 2
fi
if (( FROM_STEP > TO_STEP )); then
  echo "FROM_STEP must not exceed TO_STEP." >&2
  exit 2
fi
if ! [[ "${PILOT_VIDEO_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PILOT_VIDEO_COUNT must be positive." >&2
  exit 2
fi

should_run() {
  local step="$1"
  (( step >= FROM_STEP && step <= TO_STEP ))
}

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${RUN_ROOT}/logs"
log_file="${RUN_ROOT}/logs/run_${timestamp}_steps${FROM_STEP}-${TO_STEP}.log"
exec > >(tee -a "${log_file}") 2>&1

export PROJECT_ROOT RAW_DATASET_DIR STRICT_DATASET_DIR T5_SOURCE_DIR VBENCH_ROOT
export NGPUS EXPECTED_PROMPTS EXPECTED_SEEDS PRIMARY_LAMBDA FORCE_RESCORE
export PYTHONPATH="${PROJECT_ROOT}:${VBENCH_ROOT}:${PYTHONPATH:-}"

echo "Strict oracle end-to-end pipeline"
echo "  project       : ${PROJECT_ROOT}"
echo "  raw dataset   : ${RAW_DATASET_DIR}"
echo "  strict dataset: ${STRICT_DATASET_DIR}"
echo "  VBench        : ${VBENCH_ROOT}"
echo "  steps         : ${FROM_STEP}..${TO_STEP}"
echo "  log           : ${log_file}"

if [[ ! -f "${VBENCH_ROOT}/evaluate.py" ]]; then
  echo "VBench is missing: ${VBENCH_ROOT}/evaluate.py" >&2
  exit 2
fi
if [[ ! -d "${RAW_DATASET_DIR}" ]]; then
  echo "Raw oracle dataset is missing: ${RAW_DATASET_DIR}" >&2
  exit 2
fi

tracked_changes="$(git -C "${VBENCH_ROOT}" status --porcelain --untracked-files=no)"
if [[ -n "${tracked_changes}" ]]; then
  echo "VBench has tracked modifications; formal scoring refuses to continue:" >&2
  echo "${tracked_changes}" >&2
  exit 2
fi
export EXPECTED_VBENCH_COMMIT
EXPECTED_VBENCH_COMMIT="$(git -C "${VBENCH_ROOT}" rev-parse HEAD)"

if should_run 1; then
  echo "[1/8] Preflight environment verification"
  python - <<'PY'
import importlib
import os

import clip
import pyiqa
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
assert hasattr(clip, "load"), "Incorrect CLIP package: clip.load is missing"
for name in [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
]:
    importlib.import_module(f"vbench.{name}")
    print(f"[OK] vbench.{name}")
print("Python:", os.sys.executable)
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
print("CLIP:", clip.__file__)
PY
  echo "WanUpsampler commit: $(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
  echo "VBench commit      : ${EXPECTED_VBENCH_COMMIT}"
fi

if should_run 2; then
  echo "[2/8] Strict step49/step50 pilot"
  export SOURCE_SEED_DIR
  SOURCE_SEED_DIR="$(
    find "${RAW_DATASET_DIR}" \
      -type d \
      -path '*/raw_samples/seed_42' \
      -print -quit
  )"
  if [[ -z "${SOURCE_SEED_DIR}" ]]; then
    echo "Could not find raw_samples/seed_42 under ${RAW_DATASET_DIR}" >&2
    exit 2
  fi
  export PILOT_ROOT="${RUN_ROOT}/pilot_${timestamp}"
  export PILOT_VIDEO_COUNT PILOT_CASES
  python - <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import (
    QUALITY5_DIMENSIONS,
    inspect_vbench_checkout,
    score_case_directory,
)

vbench_root = Path(os.environ["VBENCH_ROOT"]).resolve()
source_seed_dir = Path(os.environ["SOURCE_SEED_DIR"]).resolve()
pilot_root = Path(os.environ["PILOT_ROOT"]).resolve()
pilot_count = int(os.environ["PILOT_VIDEO_COUNT"])
case_names = os.environ["PILOT_CASES"].split()
identity = inspect_vbench_checkout(
    vbench_root,
    expected_commit=os.environ["EXPECTED_VBENCH_COMMIT"],
)

manifest_files = sorted((source_seed_dir / "manifests").glob("*.json"))[:pilot_count]
if len(manifest_files) != pilot_count:
    raise RuntimeError(
        f"Expected {pilot_count} pilot manifests, found {len(manifest_files)}"
    )

summaries = {}
for case_name in case_names:
    case_root = pilot_root / case_name
    video_dir = case_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=False)
    prompt_map = {}
    for manifest_path in manifest_files:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prompt_id = int(manifest["prompt_index"])
        seed = int(manifest["seed"])
        source_video = (
            source_seed_dir
            / "videos"
            / case_name
            / f"{prompt_id:04d}_seed{seed}_{case_name}.mp4"
        )
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        pilot_video = video_dir / source_video.name
        shutil.copy2(source_video, pilot_video)
        prompt_map[str(pilot_video.resolve())] = str(manifest["prompt"])

    prompt_map_path = case_root / "prompt_map.json"
    prompt_map_path.write_text(
        json.dumps(prompt_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    bundle = score_case_directory(
        vbench_root=vbench_root,
        python_bin=sys.executable,
        video_dir=video_dir,
        prompt_map=prompt_map_path,
        out_dir=case_root / "metrics",
        dimensions=QUALITY5_DIMENSIONS,
        quality_dimensions=QUALITY5_DIMENSIONS,
        diagnostic_dimensions=[],
        ngpus=int(os.environ["NGPUS"]),
        force_rescore=True,
        vbench_identity=identity,
    )
    matrix = np.asarray(
        [
            [bundle.scores[stem][name] for name in QUALITY5_DIMENSIONS]
            for stem in sorted(bundle.scores)
        ],
        dtype=np.float64,
    )
    summaries[case_name] = {
        "video_count": int(matrix.shape[0]),
        "vbench5": float(matrix.mean()),
        "dimensions": {
            name: float(matrix[:, index].mean())
            for index, name in enumerate(QUALITY5_DIMENSIONS)
        },
        "request_sha256": bundle.provenance["request_sha256"],
        "result_sha256": bundle.provenance["result_sha256"],
        "run_manifest_path": bundle.provenance["run_manifest_path"],
    }

if "step49" in summaries and "step50" in summaries:
    delta = {
        name: summaries["step50"]["dimensions"][name]
        - summaries["step49"]["dimensions"][name]
        for name in QUALITY5_DIMENSIONS
    }
    summaries["step50_minus_step49"] = {
        "vbench5": summaries["step50"]["vbench5"]
        - summaries["step49"]["vbench5"],
        "dimensions": delta,
    }
    if max(abs(value) for value in delta.values()) <= 1e-12:
        raise RuntimeError(
            "Pilot produced identical step49/step50 scores in every dimension"
        )

output = pilot_root / "pilot_comparison.json"
output.write_text(
    json.dumps(summaries, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(summaries, indent=2, ensure_ascii=False))
print(f"Pilot result: {output}")
PY
fi

if should_run 3; then
  echo "[3/8] Full strict VBench-5 rescore"
  DATASET_DIR="${STRICT_DATASET_DIR}" \
  SOURCE_DATASET_DIRS="${RAW_DATASET_DIR}" \
  DIAGNOSTIC_DIMENSIONS="" \
  EXPECTED_VBENCH_COMMIT="${EXPECTED_VBENCH_COMMIT}" \
  bash "${SCRIPT_DIR}/run_strict_vbench_rescore.sh"
fi

if should_run 4; then
  echo "[4/8] Quality-5 dimension analysis and lambda sweep"
  DATASET_DIR="${STRICT_DATASET_DIR}" \
  OUT_DIR="${RUN_ROOT}/analysis_quality5" \
  bash "${SCRIPT_DIR}/run_strict_oracle_analysis.sh"
fi

if should_run 5; then
  if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
    echo "[5/8] Independent diagnostic scoring: ${DIAGNOSTIC_DIMENSIONS}"
    DATASET_DIR="${STRICT_DATASET_DIR}" \
    SOURCE_DATASET_DIRS="${RAW_DATASET_DIR}" \
    DIAGNOSTIC_DIMENSIONS="${DIAGNOSTIC_DIMENSIONS}" \
    EXPECTED_VBENCH_COMMIT="${EXPECTED_VBENCH_COMMIT}" \
    bash "${SCRIPT_DIR}/run_strict_vbench_rescore.sh"
  else
    echo "[5/8] Diagnostics disabled"
  fi
fi

if should_run 6; then
  echo "[6/8] Final dimension analysis and lambda sweep"
  DATASET_DIR="${STRICT_DATASET_DIR}" \
  OUT_DIR="${RUN_ROOT}/analysis_final" \
  bash "${SCRIPT_DIR}/run_strict_oracle_analysis.sh"

  export FINAL_ANALYSIS_DIR="${RUN_ROOT}/analysis_final"
  python - <<'PY'
import csv
import os
from pathlib import Path

root = Path(os.environ["FINAL_ANALYSIS_DIR"])
print("\nPer-dimension discriminability")
with (root / "dimensions" / "dimension_discriminability.csv").open(
    encoding="utf-8"
) as handle:
    for row in csv.DictReader(handle):
        print(
            row["metric"],
            "median_range=", row["median_prompt_range"],
            "flat=", row["flat_prompt_fraction"],
            "ties=", row["tie_fraction"],
            "50-49=", row["step50_minus_step49_mean"],
            "50-30=", row["step50_minus_step30_mean"],
            "endpoints=", row["endpoint_unique_winner_fraction"],
        )

print("\nLambda nearest to 0.01")
with (root / "lambda_sweep" / "lambda_sweep_summary.csv").open(
    encoding="utf-8"
) as handle:
    rows = list(csv.DictReader(handle))
row = min(rows, key=lambda item: abs(float(item["lambda"]) - 0.01))
for key, value in row.items():
    print(f"{key}: {value}")
PY
fi

if should_run 7; then
  echo "[7/8] T5 embedding coverage"
  if [[ ! -e "${STRICT_DATASET_DIR}/t5_embeddings" ]]; then
    if [[ ! -d "${T5_SOURCE_DIR}" ]]; then
      echo "T5 source directory is missing: ${T5_SOURCE_DIR}" >&2
      exit 2
    fi
    ln -s "${T5_SOURCE_DIR}" "${STRICT_DATASET_DIR}/t5_embeddings"
  fi
  test -d "${STRICT_DATASET_DIR}/t5_embeddings"
  echo "T5 embeddings: ${STRICT_DATASET_DIR}/t5_embeddings"
fi

if should_run 8; then
  if [[ "${RUN_TRAINING}" != "1" ]]; then
    echo "[8/8] Training intentionally gated. Review ${RUN_ROOT}/analysis_final first."
    echo "To train only, run:"
    echo "FROM_STEP=8 TO_STEP=8 RUN_TRAINING=1 bash ${BASH_SOURCE[0]}"
  else
    echo "[8/8] Router training"
    DATASET_DIR="${STRICT_DATASET_DIR}" \
    PRIMARY_LAMBDA="${PRIMARY_LAMBDA}" \
    OUT_DIR="${RUN_ROOT}/router_lambda${PRIMARY_LAMBDA//./}" \
    EPOCHS="${EPOCHS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    LR="${LR}" \
    SEED="${SEED}" \
    bash "${SCRIPT_DIR}/run_train_and_benchmark.sh"
  fi
fi

echo "Pipeline invocation complete. Log: ${log_file}"
