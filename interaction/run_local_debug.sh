#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/configs/minicpmo_debug_eval_only.sh"
LAUNCHER_SCRIPT="${SCRIPT_DIR}/actually-run-shell.sh"

CHECK_ONLY=0
LOCAL_EVAL_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --check-only) CHECK_ONLY=1 ;;
    --eval-only) LOCAL_EVAL_ONLY=1 ;;
    *)
      echo "Usage: $0 [--check-only] [--eval-only]" >&2
      exit 2
      ;;
  esac
done

for path in \
  /mnt/shared-storage-user/p1-shared \
  /mnt/shared-storage-user/cuiganqu \
  /mnt/shared-storage-gpfs2/gpfs2-shared-public \
  /mnt/shared-storage-gpfs2/p1-shared-2; do
  if [[ ! -d "${path}" ]]; then
    echo "Missing rjob-equivalent mount: ${path}" >&2
    exit 2
  fi
  echo "[ok] mount ${path}"
done

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing experiment config: ${CONFIG_FILE}" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
if [[ "${LOCAL_EVAL_ONLY}" == "1" ]]; then
  # Export these overrides because actually-run-shell.sh sources the config again.
  export EVAL_ONLY=1
  export NO_EVAL=0
else
  EVAL_ONLY="${EVAL_ONLY:-0}"
fi

required_files=(
  "${PROJECT_ROOT}/minicpmo_sft/train_text_policy_fsdp.py"
  "${PROJECT_ROOT}/minicpmo_sft/minicpmo_dataset.py"
  "${PROJECT_ROOT}/minicpmo_sft/sft_logging.py"
  "${TRAIN_ENV_PREFIX}/bin/python"
  "${TRAIN_ENV_PREFIX}/bin/torchrun"
  "${EVAL_DATA}"
)
if [[ "${EVAL_ONLY}" != "1" ]]; then
  required_files+=("${TRAIN_DATA}/manifest.jsonl")
fi
for path in "${required_files[@]}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Missing or empty required file: ${path}" >&2
    exit 2
  fi
  echo "[ok] file ${path}"
done

if [[ ! -d "${MODEL}" ]]; then
  echo "Missing model directory: ${MODEL}" >&2
  exit 2
fi
echo "[ok] model ${MODEL}"

if [[ ! -x "${TRAIN_ENV_PREFIX}/bin/python" || ! -x "${TRAIN_ENV_PREFIX}/bin/torchrun" ]]; then
  echo "Python environment is not executable: ${TRAIN_ENV_PREFIX}" >&2
  exit 2
fi
if [[ ! -w "${PROJECT_ROOT}/minicpmo_runs" ]]; then
  echo "Run directory is not writable: ${PROJECT_ROOT}/minicpmo_runs" >&2
  exit 2
fi
echo "[ok] writable ${PROJECT_ROOT}/minicpmo_runs"

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "Local preflight passed; no job was started (eval_only=${EVAL_ONLY})."
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; run this script inside an rlaunch GPU session." >&2
  exit 2
fi
gpu_count="$(nvidia-smi --list-gpus | wc -l)"
if (( gpu_count < LOCAL_NPROC_PER_NODE )); then
  echo "Only ${gpu_count} GPUs are visible, but LOCAL_NPROC_PER_NODE=${LOCAL_NPROC_PER_NODE}." >&2
  exit 2
fi
echo "[ok] visible_gpus=${gpu_count} local_processes=${LOCAL_NPROC_PER_NODE}"

exec "${LAUNCHER_SCRIPT}" --config "${CONFIG_FILE}" --local
