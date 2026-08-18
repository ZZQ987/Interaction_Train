#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/shared-storage-user/leihaodi/Interaction_Train"
PYTHON_BIN="/mnt/shared-storage-user/p1-shared/leihaodi/miniconda3/envs/minicpmo45_duplex_flashattn/bin/python"
EVAL_SCRIPT="${PROJECT_ROOT}/minicpmo_sft/eval_spokenwoz_duplex_tts.py"

CHECKPOINT="/mnt/shared-storage-user/leihaodi/interaction-save/ckpt-2100"
BASE_MODEL="/mnt/shared-storage-gpfs2/p1-shared-2/luoyun/construct_demo/models/openbmb/MiniCPM-o-4_5"
TEST_PARQUET="${PROJECT_ROOT}/data/data/test-00000-of-00005.parquet"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/shared-storage-user/leihaodi/interaction-save/eval-result/ckpt-2100}"
REF_AUDIO="${BASE_MODEL}/assets/system_ref_audio.wav"
PYARROW_SITE_PACKAGES="/mnt/shared-storage-user/p1-shared/leihaodi/miniconda3/envs/specbench/lib/python3.12/site-packages"
NUM_GPUS="${NUM_GPUS:-0}"
MAX_NEW_SPEAK_TOKENS="${MAX_NEW_SPEAK_TOKENS:-128}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "Missing evaluation script: ${EVAL_SCRIPT}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PROJECT_ROOT}/minicpmo_sft:${PYTHONPATH:-}"
export PATH="$(dirname "${PYTHON_BIN}"):/usr/local/nvidia/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

extra_args=()
if [[ "${RESUME:-0}" == "1" ]]; then
  extra_args+=(--resume)
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  extra_args+=(--overwrite)
fi
if [[ -n "${MAX_DIALOGS:-}" ]]; then
  extra_args+=(--max-dialogs "${MAX_DIALOGS}")
fi
if [[ "${GOLD_ONLY:-0}" == "1" ]]; then
  extra_args+=(--gold-only)
fi

exec "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
  --checkpoint "${CHECKPOINT}" \
  --base-model "${BASE_MODEL}" \
  --test-parquet "${TEST_PARQUET}" \
  --output-dir "${OUTPUT_DIR}" \
  --ref-audio "${REF_AUDIO}" \
  --pyarrow-site-packages "${PYARROW_SITE_PACKAGES}" \
  --audio-chunk-seconds 1.0 \
  --input-sample-rate 16000 \
  --attn-implementation flash_attention_2 \
  --decode-mode greedy \
  --max-new-speak-tokens-per-chunk "${MAX_NEW_SPEAK_TOKENS}" \
  --num-gpus "${NUM_GPUS}" \
  "${extra_args[@]}"
