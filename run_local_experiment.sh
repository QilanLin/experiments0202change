#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
PROJECT_PARENT="$(cd "${PROJECT_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENTS_RUN_FILE="${EXPERIMENTS_RUN_FILE:-${PROJECT_ROOT}/run_local_experiment.py}"

MODEL=""
for ((i = 1; i <= $#; i++)); do
  arg="${!i}"
  case "${arg}" in
    --model=*)
      MODEL="${arg#--model=}"
      ;;
    --model)
      if ((i < $#)); then
        next_index=$((i + 1))
        MODEL="${!next_index}"
      fi
      ;;
  esac
done

export HF_HOME="${HF_HOME:-${PROJECT_PARENT}/.cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/huggingface/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ "${MODEL}" == "timesfm" ]]; then
  TIMESFM_OFFICIAL_SRC="${TIMESFM_OFFICIAL_SRC:-${PROJECT_PARENT}/timesfm_official/src}"
  if [[ ! -f "${TIMESFM_OFFICIAL_SRC}/timesfm/__init__.py" ]]; then
    echo "[ERROR] timesfm official source not found: ${TIMESFM_OFFICIAL_SRC}" >&2
    echo "[ERROR] Expected the official clone at ../timesfm_official/src relative to the repo." >&2
    exit 1
  fi
  export PYTHONPATH="${TIMESFM_OFFICIAL_SRC}:${PROJECT_PARENT}${PYTHONPATH:+:${PYTHONPATH}}"
  echo "[INFO] TimesFM official source: ${TIMESFM_OFFICIAL_SRC}"
else
  export PYTHONPATH="${PROJECT_PARENT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

echo "[INFO] Repo root: ${PROJECT_ROOT}"
echo "[INFO] Run file: ${EXPERIMENTS_RUN_FILE}"
echo "[INFO] HF_HOME: ${HF_HOME}"
echo "[INFO] HUGGINGFACE_HUB_CACHE: ${HUGGINGFACE_HUB_CACHE}"
echo "[INFO] HF_HUB_OFFLINE: ${HF_HUB_OFFLINE}"
echo "[INFO] TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE}"

exec "${PYTHON_BIN}" "${EXPERIMENTS_RUN_FILE}" "$@"
