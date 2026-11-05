#!/usr/bin/env bash
# Attention arms on the platform's official-Wan path (attention_wan.py), first check on 4 natural_ext prompts, seed 42.
#   bash UNIV_adaptor/scripts/run_attention_wan.sh            # all: gate + SpargeAttn sweep, then SVG2, report, tgz
#   bash UNIV_adaptor/scripts/run_attention_wan.sh <step>     # gate | svg2 | pack
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
TAG="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9')"
# The interpreter that generated the platform FULL50 (WAN_PYTHON of install_twostage_wan21_env.sh). The env file is
# named after the GPU model, so on a new machine type fall back to any other machine's env file on the shared afs.
ENV_FILE="outputs/twostage_env/${TAG}.env"
for f in "${ENV_FILE}" outputs/twostage_env/*.env; do
  [[ -f "$f" ]] || continue
  WAN_PYTHON=""
  # shellcheck disable=SC1090
  source "$f"
  if [[ -n "${WAN_PYTHON}" && -x "${WAN_PYTHON}" ]]; then ENV_FILE="$f"; break; fi
done
[[ -x "${WAN_PYTHON:-}" ]] || WAN_PYTHON=""
[[ -z "${WAN_PYTHON}" && -x .venv_published_wan21/bin/python ]] && WAN_PYTHON="${REPO_ROOT}/.venv_published_wan21/bin/python"
export PYTHON_WAN21="${PYTHON_WAN21:-${WAN_PYTHON:-/opt/conda/bin/python}}"
PY=/opt/conda/bin/python
A=UNIV_adaptor/scripts/data/attention_wan.py
CFG=UNIV_adaptor/configs/published_wan21_platform_natural_ext.json
OUT=outputs/attention_wan
REF=outputs/published_wan21_platform_natural_ext
COMMON=(--config "${CFG}" --out "${OUT}" --first 4 --seeds 42 --gpus "${GPUS:-0,1,2,3}")
echo "worker python: ${PYTHON_WAN21} (from ${ENV_FILE}; env files here: $(ls outputs/twostage_env/ 2>/dev/null | xargs))"

gate() {
  "${PY}" "${A}" run "${COMMON[@]}" --arms DENSE SPARGE50 SPARGE65 SPARGE80
  "${PY}" "${A}" gate --out "${OUT}" --reference "${REF}" | tee "${OUT}/gate.txt"
}
svg2() { "${PY}" "${A}" run "${COMMON[@]}" --arms SVG2; }
pack() {
  "${PY}" "${A}" report --out "${OUT}"
  echo "worker python: ${PYTHON_WAN21}" > "${OUT}/python.txt"
  tar czf outputs/attention_wan.tgz -C outputs attention_wan
  echo "SEND THIS: ${REPO_ROOT}/outputs/attention_wan.tgz"
}

case "${1:-all}" in
  all) gate; svg2 || echo "SVG2 FAILED (see ${OUT}/logs); packing the rest"; pack ;;
  gate|svg2|pack) "$1" ;;
  *) echo "unknown step $1" >&2; exit 2 ;;
esac
