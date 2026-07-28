#!/usr/bin/env bash
# Minimal follow-up for the fixed original experiment machine reached through
# the VS Code Remote Tunnel named "temp".
#
# It does not copy latent tensors or generated videos. It checks the two gaps
# found in the 2026-07-28 metadata export, copies the small Distill4 TTD3
# training records, and verifies the five custom-checkpoint hashes.

set -Eeuo pipefail
umask 077

PROJECT_ROOT="/mnt/afs_2/houze/wanUpsampler"
VBENCH_REPO="/mnt/afs_2/houze/VBench"
PACKAGE_ROOT="${PROJECT_ROOT}/paper/aaai27/submission_materials_20260728/code_data_package"
WAN_PYTHON="/opt/conda/bin/python"
DISTILL4_TTD3_OUTPUT="${PROJECT_ROOT}/outputs/changing_resolution_distill_last_step_skip_lora_14b_cfgdistill_5k_step3"
VERIFY_CHECKPOINT_HASHES="${VERIFY_CHECKPOINT_HASHES:-1}"

EXPORT_PARENT="${PROJECT_ROOT}/outputs/aaai27_repro_exports"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="intrascale_temp_followup_${STAMP}"
RUN_ROOT="${EXPORT_PARENT}/${RUN_NAME}"
ARCHIVE="${EXPORT_PARENT}/${RUN_NAME}.tar.gz"

[[ -d "${PROJECT_ROOT}" ]] || {
  echo "Missing fixed project root: ${PROJECT_ROOT}" >&2
  exit 1
}
[[ -x "${WAN_PYTHON}" ]] || {
  echo "Missing fixed Wan Python: ${WAN_PYTHON}" >&2
  exit 1
}
[[ ! -e "${RUN_ROOT}" && ! -e "${ARCHIVE}" ]] || {
  echo "Refusing to overwrite: ${RUN_ROOT} or ${ARCHIVE}" >&2
  exit 1
}
mkdir -p \
  "${RUN_ROOT}/audit" \
  "${RUN_ROOT}/metadata" \
  "${RUN_ROOT}/training_records/distill4_ttd3_legacy"

# Record conda environments and test every registered environment against the
# checked-out VBench revision. The fixed historical path is checked first, but
# its absence does not abort the remaining evidence export.
if command -v conda >/dev/null 2>&1; then
  conda env list --json > "${RUN_ROOT}/audit/conda_env_list.json"
else
  printf '{"envs":[]}\n' > "${RUN_ROOT}/audit/conda_env_list.json"
fi

{
  printf 'python\tstatus\tvbench_import\n'
  printf '/opt/conda/envs/vbench/bin/python\t%s\t%s\n' \
    "$([[ -x /opt/conda/envs/vbench/bin/python ]] && echo present || echo missing)" \
    "not_tested"
} > "${RUN_ROOT}/audit/vbench_python_candidates.tsv"

mapfile -t CONDA_PYTHONS < <(
  "${WAN_PYTHON}" - "${RUN_ROOT}/audit/conda_env_list.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for env in payload.get("envs", []):
    candidate = Path(env) / "bin" / "python"
    if candidate.is_file():
        print(candidate)
PY
)

VBENCH_PYTHON_RESOLVED=""
for candidate in "${CONDA_PYTHONS[@]}"; do
  status="present"
  import_status="failed"
  if PYTHONPATH="${VBENCH_REPO}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${candidate}" -c 'import vbench' >/dev/null 2>&1; then
    import_status="ok"
    if [[ -z "${VBENCH_PYTHON_RESOLVED}" ]]; then
      VBENCH_PYTHON_RESOLVED="${candidate}"
    fi
  fi
  printf '%s\t%s\t%s\n' "${candidate}" "${status}" "${import_status}" \
    >> "${RUN_ROOT}/audit/vbench_python_candidates.tsv"
done

if [[ -n "${VBENCH_PYTHON_RESOLVED}" ]]; then
  mkdir -p "${RUN_ROOT}/metadata/vbench_environment"
  "${VBENCH_PYTHON_RESOLVED}" -VV \
    > "${RUN_ROOT}/metadata/vbench_environment/python.txt" 2>&1
  "${VBENCH_PYTHON_RESOLVED}" -m pip freeze --all \
    > "${RUN_ROOT}/metadata/vbench_environment/pip_freeze.txt"
  printf '%s\n' "${VBENCH_PYTHON_RESOLVED}" \
    > "${RUN_ROOT}/metadata/vbench_environment/resolved_python.txt"
fi

# Inspect the canonical and legacy Distill4 TTD3 LMDB candidates and select the
# first one that actually contains data.mdb. Directory existence alone is not
# accepted as evidence.
"${WAN_PYTHON}" - \
  "${PROJECT_ROOT}" \
  "${RUN_ROOT}/metadata/distill4_ttd3_lmdb_manifest.json" <<'PY'
import hashlib
import json
import random
import sys
from pathlib import Path

import lmdb

project = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
candidates = [
    "data/changing_resolution_distill/lmdb_last_step_skip_lora_368x640_14b_cfgdistill_5k_step3",
    "data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3",
]

selected = None
shards = []
for relative in candidates:
    found = sorted((project / relative).rglob("data.mdb"))
    if found:
        selected = relative
        shards = [path.parent for path in found]
        break

rows = []
total = 0
prompt_hashes = set()
seed_min = None
seed_max = None
schemas = set()
for shard in shards:
    env = lmdb.open(
        str(shard), readonly=True, lock=False, readahead=False, meminit=False
    )
    with env.begin() as txn:
        metadata_raw = txn.get(b"metadata")
        metadata = json.loads(metadata_raw.decode("utf-8")) if metadata_raw else {}
        count_raw = txn.get(b"num_samples")
        count = (
            int(count_raw.decode("utf-8"))
            if count_raw
            else int(metadata.get("num_samples", 0))
        )
        schema = metadata.get("schema")
        if schema:
            schemas.add(str(schema))
        for index in range(count):
            prompt = txn.get(f"prompt_{index:08d}_data".encode())
            if prompt:
                prompt_hashes.add(hashlib.sha256(prompt).hexdigest())
            seed = txn.get(f"seed_{index:08d}_data".encode())
            if seed:
                value = int(seed.decode("utf-8"))
                seed_min = value if seed_min is None else min(seed_min, value)
                seed_max = value if seed_max is None else max(seed_max, value)
    env.close()
    total += count
    rows.append(
        {
            "name": shard.relative_to(project / selected).as_posix(),
            "samples": count,
            "data_mdb_bytes": (shard / "data.mdb").stat().st_size,
            "schema": schema,
        }
    )

validation_samples = 0
validation_indices = []
if total >= 2:
    validation_samples = min(max(1, round(total * 0.02)), 64, total - 1)
    indices = list(range(total))
    random.Random(1234).shuffle(indices)
    validation_indices = sorted(indices[:validation_samples])

report = {
    "schema_version": 1,
    "candidate_relative_paths": candidates,
    "selected_relative_path": selected,
    "exists": bool(shards),
    "shards": rows,
    "shard_count": len(rows),
    "total_samples": total,
    "unique_prompt_hashes": len(prompt_hashes),
    "seed_min": seed_min,
    "seed_max": seed_max,
    "schemas": sorted(schemas),
    "split": {
        "algorithm": "random.Random(1234).shuffle(indices)",
        "training_samples": total - validation_samples,
        "validation_samples": validation_samples,
        "validation_indices": validation_indices,
    },
}
output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
if not shards:
    print("No data.mdb found in either Distill4 TTD3 candidate", file=sys.stderr)
PY

# Copy only small, human-readable training records from the realized legacy run.
for name in metrics.jsonl train_config.yaml best_val.json; do
  source="${DISTILL4_TTD3_OUTPUT}/${name}"
  if [[ -f "${source}" ]]; then
    cp --preserve=timestamps \
      "${source}" \
      "${RUN_ROOT}/training_records/distill4_ttd3_legacy/${name}"
  fi
done

# Verify sizes and, by default, SHA-256 for the five exact custom checkpoints.
"${WAN_PYTHON}" - \
  "${PACKAGE_ROOT}/reproduction_assets.json" \
  "${PROJECT_ROOT}" \
  "${RUN_ROOT}/audit/custom_checkpoint_verification.json" \
  "${VERIFY_CHECKPOINT_HASHES}" <<'PY'
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

{
  echo "schema_version=1"
  echo "created_utc=${STAMP}"
  echo "vbench_python_resolved=$([[ -n "${VBENCH_PYTHON_RESOLVED}" ]] && echo yes || echo no)"
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
