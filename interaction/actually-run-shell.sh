#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

MODE="rjob"
if [[ "$#" -eq 3 && "$1" == "--config" && "$3" == "--local" ]]; then
  MODE="local"
elif [[ "$#" -ne 2 || "$1" != "--config" ]]; then
  echo "Usage: $0 --config CONFIG_FILE [--local]" >&2
  exit 2
fi

CONFIG_FILE="$2"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing experiment config: ${CONFIG_FILE}" >&2
  exit 2
fi
CONFIG_FILE="$(cd -- "$(dirname -- "${CONFIG_FILE}")" &>/dev/null && pwd)/$(basename -- "${CONFIG_FILE}")"

EVAL_ONLY_OVERRIDE="${EVAL_ONLY-}"
NO_EVAL_OVERRIDE="${NO_EVAL-}"
OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR-}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
if [[ -n "${EVAL_ONLY_OVERRIDE}" ]]; then
  EVAL_ONLY="${EVAL_ONLY_OVERRIDE}"
fi
if [[ -n "${NO_EVAL_OVERRIDE}" ]]; then
  NO_EVAL="${NO_EVAL_OVERRIDE}"
fi
REQUESTED_OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-${OUTPUT_DIR-}}"
MAX_CKPT_LIMIT="${MAX_CKPT_LIMIT:-3}"
NO_EVAL="${NO_EVAL:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
if [[ "${EVAL_ONLY}" == "1" && "${NO_EVAL}" == "1" ]]; then
  echo "EVAL_ONLY=1 requires NO_EVAL=0 (${CONFIG_FILE})" >&2
  exit 2
fi

CONFIG_VARS=(
  RJOB_NAMESPACE KUBEBRAIN_CLUSTER_ENTRY RJOB_TASK_TYPE RJOB_GPU RJOB_CPU RJOB_MEMORY
  RJOB_CHARGED_GROUP RJOB_IMAGE NODE_COUNT NPROC_PER_NODE
  LOCAL_NPROC_PER_NODE MASTER_PORT
  RJOB_NAME_PREFIX WANDB_MODE NCCL_DEBUG NCCL_IB_DISABLE
  TORCH_NCCL_ASYNC_ERROR_HANDLING PYTORCH_ALLOC_CONF
  PYTORCH_CUDA_ALLOC_CONF TRAIN_ENV_PREFIX HF_HOME MODEL TOKENIZER_MODEL
  PROCESSOR_MODEL SOURCE_ROOT TRAIN_DATA EVAL_DATA OUTPUT_NAME_PREFIX
  INPUT_SCHEMA MAX_LENGTH BATCH_SIZE EVAL_BATCH_SIZE GRAD_ACCUM_STEPS EPOCHS
  LEARNING_RATE WEIGHT_DECAY GRADIENT_CHECKPOINTING NO_SAVE_CHECKPOINTS
  MAX_SLICE_NUMS MAX_IMAGE_PIXELS FORCE_IMAGE_SIZE IMAGE_SCALE_RESOLUTION
  VISION_BATCH_SIZE ATTN_IMPLEMENTATION TRAJECTORY_MAX_TURNS
  TRAJECTORY_CHUNK_STRIDE EVAL_TRAJECTORY_MAX_TURNS
  EVAL_TRAJECTORY_CHUNK_STRIDE GENERATED_CHUNK_STRIDE
  GENERATED_MAX_IMAGES_PER_TURN GENERATED_IMAGE_SELECTION
  TRAIN_EXCLUDE_TASK_TYPES EVAL_EXCLUDE_TASK_TYPES
  TRAIN_SPEAK_SAMPLING_RATIO TRAIN_BALANCE_SPEAK_LISTEN_KEEP_DELEGATE
  TRAIN_LISTEN_TO_SPEAK_RATIO_KEEP_DELEGATE
  TRAIN_DROP_CHAT_ALL_SILENCE_CHUNKS
  TRAIN_REQUIRE_PRIOR_INSTRUCTION_FOR_ACTION_CHUNKS
  COLLAPSE_REPEATED_SPEAK_SEGMENTS DROP_PLACEHOLDER_SPEAK_CHUNKS
  CLEAN_EVENT_GROUNDING_TEMPLATES LISTEN_WEIGHT SPEAK_WEIGHT
  SPEAK_BOUNDARY_WEIGHT DELEGATE_WEIGHT MAX_TRAIN_BATCHES
  NO_EVAL EVAL_ONLY EVAL_BEFORE_TRAIN EVAL_EVERY_STEPS EVAL_SAVE_PREDICTIONS_LIMIT
  SPEAK_THRESHOLD_SWEEP SAVE_EVERY_STEPS LOG_EVERY MAX_EVAL_BATCHES
  KEEP_ALIVE
)
for name in "${CONFIG_VARS[@]}"; do
  if [[ ! -v "${name}" ]]; then
    echo "Missing required config value: ${name} (${CONFIG_FILE})" >&2
    exit 2
  fi
done

case "${RJOB_TASK_TYPE}" in
  normal|idle) ;;
  *)
    echo "Invalid RJOB_TASK_TYPE: ${RJOB_TASK_TYPE}; expected normal or idle (${CONFIG_FILE})" >&2
    exit 2
    ;;
esac

TS="$(date +%Y%m%d_%H%M%S)"
USER_ROOT="${PROJECT_ROOT}"
CODE_DIR="${PROJECT_ROOT}/minicpmo_sft"
RUNS_ROOT="${PROJECT_ROOT}/minicpmo_runs"
TRAIN_PYTHON_SCRIPT="${CODE_DIR}/train_text_policy_fsdp.py"
if [[ "${MODE}" == "local" ]]; then
  RJOB_NAME="${RJOB_NAME_PREFIX}-local-${TS}"
  NNODES=1
  NPROC_PER_NODE="${LOCAL_NPROC_PER_NODE}"
  NODE_RANK=0
  MASTER_ADDR="127.0.0.1"
else
  RJOB_NAME="${RJOB_NAME_PREFIX}-${TS}"
fi
OUTPUT_ROOT="${RUNS_ROOT}/outputs"
if [[ -n "${REQUESTED_OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${REQUESTED_OUTPUT_DIR}"
else
  OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_NAME_PREFIX}_${TS}"
fi
LOG_DIR="${RUNS_ROOT}/logs/${RJOB_NAME}"
WANDB_DIR="${RUNS_ROOT}/wandb/${RJOB_NAME}"

export KUBEBRAIN_CLUSTER_ENTRY
export KUBEBRAIN_NAMESPACE="${RJOB_NAMESPACE}"

RUN_CMD=$(cat <<'EOF'
set -euo pipefail

first_host() {
  local raw="${1:-}"
  raw="${raw//;/,}"
  raw="${raw// /,}"
  raw="${raw//|/,}"
  echo "${raw%%,*}"
}

if [[ -z "${NODE_RANK:-}" ]]; then
  for var in NODE_RANK RANK OMPI_COMM_WORLD_RANK PMI_RANK BRAINPP_TASK_INDEX BRAINPP_ROLE_INDEX PET_NNI_CURRENT_TASK_ROLE_CURRENT_INSTANCE_INDEX RJOB_REPLICA_INDEX JOB_COMPLETION_INDEX KUBEBRAIN_REPLICA; do
    val="${!var:-}"
    if [[ "${val}" =~ ^[0-9]+$ ]]; then
      export NODE_RANK="${val}"
      break
    fi
  done
fi

if [[ -z "${NODE_RANK:-}" && "$(hostname)" =~ -([0-9]+)$ ]]; then
  export NODE_RANK="${BASH_REMATCH[1]}"
fi

if [[ -z "${MASTER_ADDR:-}" ]]; then
  for var in MASTER_ADDR RJOB_MASTER_ADDR BRAINPP_MASTER_ADDR VC_WORKER_HOSTS WORKER_HOSTS HOSTS POD_IPS RJOB_WORKER_HOSTS; do
    val="${!var:-}"
    if [[ -n "${val}" ]]; then
      export MASTER_ADDR="$(first_host "${val}")"
      break
    fi
  done
fi

if [[ -z "${MASTER_ADDR:-}" && "${NODE_RANK:-}" == "0" ]]; then
  export MASTER_ADDR="$(hostname -i | awk '{print $1}')"
fi

if [[ -z "${NODE_RANK:-}" || -z "${MASTER_ADDR:-}" ]]; then
  echo "Cannot infer NODE_RANK or MASTER_ADDR for multi-node torchrun." >&2
  env | sort | grep -E '^(MASTER|NODE|RANK|WORLD|LOCAL|HOST|POD|RJOB|BRAIN|PET|VC_|WORKER|OMPI|PMI|KUBEBRAIN)' || true
  exit 2
fi

if [[ ! -x "${TRAIN_ENV_PREFIX}/bin/python" || ! -x "${TRAIN_ENV_PREFIX}/bin/torchrun" ]]; then
  echo "Invalid TRAIN_ENV_PREFIX: ${TRAIN_ENV_PREFIX}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_PYTHON_SCRIPT}" ]]; then
  echo "Missing training script: ${TRAIN_PYTHON_SCRIPT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${WANDB_DIR}"
cd "${CODE_DIR}"

export NNODES NPROC_PER_NODE MASTER_PORT
export PYTHONUNBUFFERED=1
export WANDB_MODE NCCL_DEBUG NCCL_IB_DISABLE TORCH_NCCL_ASYNC_ERROR_HANDLING
export PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="${TRAIN_ENV_PREFIX}/bin:/usr/local/nvidia/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

echo "[minicpmo_generated_rjob] hostname=$(hostname)"
echo "[minicpmo_generated_rjob] TRAIN_ENV_PREFIX=${TRAIN_ENV_PREFIX}"
echo "[minicpmo_generated_rjob] TRAIN_PYTHON_SCRIPT=${TRAIN_PYTHON_SCRIPT}"
echo "[minicpmo_generated_rjob] CODE_DIR=${CODE_DIR}"
echo "[minicpmo_generated_rjob] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[minicpmo_generated_rjob] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[minicpmo_generated_rjob] LOG_DIR=${LOG_DIR}"
echo "[minicpmo_generated_rjob] WANDB_DIR=${WANDB_DIR}"
echo "[minicpmo_generated_rjob] NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[minicpmo_generated_rjob] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[minicpmo_generated_rjob] MAX_LENGTH=${MAX_LENGTH} BATCH_SIZE=${BATCH_SIZE} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "[minicpmo_generated_rjob] INPUT_SCHEMA=${INPUT_SCHEMA}"
echo "[minicpmo_generated_rjob] MAX_IMAGE_PIXELS=${MAX_IMAGE_PIXELS}"
echo "[minicpmo_generated_rjob] FORCE_IMAGE_SIZE=${FORCE_IMAGE_SIZE}"
echo "[minicpmo_generated_rjob] IMAGE_SCALE_RESOLUTION=${IMAGE_SCALE_RESOLUTION}"
echo "[minicpmo_generated_rjob] VISION_BATCH_SIZE=${VISION_BATCH_SIZE}"
echo "[minicpmo_generated_rjob] ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[minicpmo_generated_rjob] TRAIN_DATA=${TRAIN_DATA} EVAL_DATA=${EVAL_DATA}"
echo "[minicpmo_generated_rjob] NO_EVAL=${NO_EVAL} EVAL_ONLY=${EVAL_ONLY} EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS} MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES}"
echo "[minicpmo_generated_rjob] SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS}"
echo "[minicpmo_generated_rjob] MAX_CKPT_LIMIT=${MAX_CKPT_LIMIT}"
echo "[minicpmo_generated_rjob] TRAJECTORY_MAX_TURNS=${TRAJECTORY_MAX_TURNS} TRAJECTORY_CHUNK_STRIDE=${TRAJECTORY_CHUNK_STRIDE} GENERATED_MAX_IMAGES_PER_TURN=${GENERATED_MAX_IMAGES_PER_TURN}"
echo "[minicpmo_generated_rjob] EVAL_TRAJECTORY_MAX_TURNS=${EVAL_TRAJECTORY_MAX_TURNS} EVAL_TRAJECTORY_CHUNK_STRIDE=${EVAL_TRAJECTORY_CHUNK_STRIDE}"
echo "[minicpmo_generated_rjob] TRAIN_EXCLUDE_TASK_TYPES=${TRAIN_EXCLUDE_TASK_TYPES}"
echo "[minicpmo_generated_rjob] EVAL_EXCLUDE_TASK_TYPES=${EVAL_EXCLUDE_TASK_TYPES}"
echo "[minicpmo_generated_rjob] TRAIN_SPEAK_SAMPLING_RATIO=${TRAIN_SPEAK_SAMPLING_RATIO}"
echo "[minicpmo_generated_rjob] TRAIN_BALANCE_SPEAK_LISTEN_KEEP_DELEGATE=${TRAIN_BALANCE_SPEAK_LISTEN_KEEP_DELEGATE}"
echo "[minicpmo_generated_rjob] TRAIN_LISTEN_TO_SPEAK_RATIO_KEEP_DELEGATE=${TRAIN_LISTEN_TO_SPEAK_RATIO_KEEP_DELEGATE}"
echo "[minicpmo_generated_rjob] TRAIN_DROP_CHAT_ALL_SILENCE_CHUNKS=${TRAIN_DROP_CHAT_ALL_SILENCE_CHUNKS}"
echo "[minicpmo_generated_rjob] TRAIN_REQUIRE_PRIOR_INSTRUCTION_FOR_ACTION_CHUNKS=${TRAIN_REQUIRE_PRIOR_INSTRUCTION_FOR_ACTION_CHUNKS}"
echo "[minicpmo_generated_rjob] COLLAPSE_REPEATED_SPEAK_SEGMENTS=${COLLAPSE_REPEATED_SPEAK_SEGMENTS}"
echo "[minicpmo_generated_rjob] DROP_PLACEHOLDER_SPEAK_CHUNKS=${DROP_PLACEHOLDER_SPEAK_CHUNKS}"
echo "[minicpmo_generated_rjob] CLEAN_EVENT_GROUNDING_TEMPLATES=${CLEAN_EVENT_GROUNDING_TEMPLATES}"
echo "[minicpmo_generated_rjob] SOURCE_ROOT=${SOURCE_ROOT}"
echo "[minicpmo_generated_rjob] OUTPUT_DIR=${OUTPUT_DIR}"

if [[ "${EVAL_ONLY}" != "1" && ! -s "${TRAIN_DATA}/manifest.jsonl" ]]; then
  echo "Missing generated trajectory train manifest: ${TRAIN_DATA}/manifest.jsonl" >&2
  exit 2
fi
if [[ -d "${EVAL_DATA}" ]]; then
  if [[ ! -s "${EVAL_DATA}/manifest.jsonl" ]] && ! find "${EVAL_DATA}" -name '*.jsonl' -not -name '*.tmp' -print -quit | grep -q .; then
    echo "Missing generated trajectory eval manifest/jsonl under: ${EVAL_DATA}" >&2
    exit 2
  fi
elif [[ ! -s "${EVAL_DATA}" ]]; then
  echo "Missing generated trajectory eval file: ${EVAL_DATA}" >&2
  exit 2
fi

extra_args=()
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  extra_args+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi
if [[ "${NO_SAVE_CHECKPOINTS}" == "1" ]]; then
  extra_args+=(--no-save-checkpoints)
fi
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  extra_args+=(--gradient-checkpointing)
fi
if [[ "${TRAIN_BALANCE_SPEAK_LISTEN_KEEP_DELEGATE}" == "1" ]]; then
  extra_args+=(--train-balance-speak-listen-keep-delegate)
fi
if [[ "${TRAIN_DROP_CHAT_ALL_SILENCE_CHUNKS}" == "1" ]]; then
  extra_args+=(--train-drop-chat-all-silence-chunks)
fi
if [[ "${TRAIN_REQUIRE_PRIOR_INSTRUCTION_FOR_ACTION_CHUNKS}" == "1" ]]; then
  extra_args+=(--train-require-prior-instruction-for-action-chunks)
fi
if [[ "${COLLAPSE_REPEATED_SPEAK_SEGMENTS}" == "1" ]]; then
  extra_args+=(--collapse-repeated-speak-segments)
fi
if [[ "${DROP_PLACEHOLDER_SPEAK_CHUNKS}" == "1" ]]; then
  extra_args+=(--drop-placeholder-speak-chunks)
fi
if [[ "${CLEAN_EVENT_GROUNDING_TEMPLATES}" == "1" ]]; then
  extra_args+=(--clean-event-grounding-templates)
fi
if [[ "${EVAL_BEFORE_TRAIN}" == "1" ]]; then
  extra_args+=(--eval-before-train)
fi
if [[ "${NO_EVAL}" == "1" ]]; then
  extra_args+=(--no-eval)
fi
if [[ "${EVAL_ONLY}" == "1" ]]; then
  extra_args+=(--eval-only)
fi

"${TRAIN_ENV_PREFIX}/bin/python" -c 'import flash_attn, torch; print(f"flash_attn={flash_attn.__version__} torch={torch.__version__}")'

"${TRAIN_ENV_PREFIX}/bin/torchrun" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${TRAIN_PYTHON_SCRIPT}" \
  --model "${MODEL}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --processor-model "${PROCESSOR_MODEL}" \
  --train-data "${TRAIN_DATA}" \
  --eval-data "${EVAL_DATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-dir "${LOG_DIR}" \
  --input-schema "${INPUT_SCHEMA}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --max-length "${MAX_LENGTH}" \
  --max-slice-nums "${MAX_SLICE_NUMS}" \
  --max-image-pixels "${MAX_IMAGE_PIXELS}" \
  --force-image-size "${FORCE_IMAGE_SIZE}" \
  --image-scale-resolution "${IMAGE_SCALE_RESOLUTION}" \
  --vision-batch-size "${VISION_BATCH_SIZE}" \
  --listen-weight "${LISTEN_WEIGHT}" \
  --speak-weight "${SPEAK_WEIGHT}" \
  --speak-boundary-weight "${SPEAK_BOUNDARY_WEIGHT}" \
  --delegate-weight "${DELEGATE_WEIGHT}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --max-eval-batches "${MAX_EVAL_BATCHES}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --eval-every-steps "${EVAL_EVERY_STEPS}" \
  --eval-save-predictions-limit "${EVAL_SAVE_PREDICTIONS_LIMIT}" \
  --speak-threshold-sweep="${SPEAK_THRESHOLD_SWEEP}" \
  --save-every-steps "${SAVE_EVERY_STEPS}" \
  --max-ckpt-limit "${MAX_CKPT_LIMIT}" \
  --log-every "${LOG_EVERY}" \
  --generated-trajectory-mode \
  --trajectory-max-turns "${TRAJECTORY_MAX_TURNS}" \
  --trajectory-chunk-stride "${TRAJECTORY_CHUNK_STRIDE}" \
  --eval-trajectory-max-turns "${EVAL_TRAJECTORY_MAX_TURNS}" \
  --eval-trajectory-chunk-stride "${EVAL_TRAJECTORY_CHUNK_STRIDE}" \
  --generated-max-images-per-turn "${GENERATED_MAX_IMAGES_PER_TURN}" \
  --generated-image-selection "${GENERATED_IMAGE_SELECTION}" \
  --train-exclude-task-types "${TRAIN_EXCLUDE_TASK_TYPES}" \
  --eval-exclude-task-types "${EVAL_EXCLUDE_TASK_TYPES}" \
  --train-speak-sampling-ratio "${TRAIN_SPEAK_SAMPLING_RATIO}" \
  --train-listen-to-speak-ratio-keep-delegate "${TRAIN_LISTEN_TO_SPEAK_RATIO_KEEP_DELEGATE}" \
  "${extra_args[@]}"

if [[ "${KEEP_ALIVE}" == "1" && "${EVAL_ONLY}" != "1" ]]; then
  sleep infinity
fi
EOF
)

# Cluster webhook only allows hostNetwork for 8-GPU tasks.
if [[ "${RJOB_GPU}" -eq 8 ]]; then
  HOST_NETWORK=true
else
  HOST_NETWORK=false
fi

if [[ "${MODE}" == "local" ]]; then
  echo "Running MiniCPM-o generated trajectory SFT locally:"
else
  echo "Submitting MiniCPM-o generated trajectory SFT rjob:"
fi
echo "  config_file=${CONFIG_FILE}"
echo "  name=${RJOB_NAME}"
if [[ "${MODE}" == "local" ]]; then
  echo "  nodes=${NNODES} local_processes=${NPROC_PER_NODE} master_addr=${MASTER_ADDR}"
else
  echo "  namespace=${RJOB_NAMESPACE} task_type=${RJOB_TASK_TYPE} charged_group=${RJOB_CHARGED_GROUP}"
  echo "  nodes=${NODE_COUNT} gpu_per_node=${RJOB_GPU} host_network=${HOST_NETWORK}"
fi
echo "  source_root=${SOURCE_ROOT}"
echo "  code_dir=${CODE_DIR}"
echo "  train_python_script=${TRAIN_PYTHON_SCRIPT}"
echo "  train_env_prefix=${TRAIN_ENV_PREFIX}"
echo "  model=${MODEL}"
echo "  tokenizer_model=${TOKENIZER_MODEL}"
echo "  processor_model=${PROCESSOR_MODEL}"
echo "  train_data=${TRAIN_DATA}"
echo "  eval_data=${EVAL_DATA}"
echo "  max_length=${MAX_LENGTH} batch_size=${BATCH_SIZE} grad_accum_steps=${GRAD_ACCUM_STEPS}"
echo "  input_schema=${INPUT_SCHEMA}"
echo "  eval_batch_size=${EVAL_BATCH_SIZE} eval_only=${EVAL_ONLY} eval_before_train=${EVAL_BEFORE_TRAIN} max_train_batches=${MAX_TRAIN_BATCHES}"
echo "  max_image_pixels=${MAX_IMAGE_PIXELS}"
echo "  force_image_size=${FORCE_IMAGE_SIZE}"
echo "  image_scale_resolution=${IMAGE_SCALE_RESOLUTION}"
echo "  vision_batch_size=${VISION_BATCH_SIZE}"
echo "  attn_implementation=${ATTN_IMPLEMENTATION}"
echo "  eval_every_steps=${EVAL_EVERY_STEPS} max_eval_batches=${MAX_EVAL_BATCHES}"
echo "  save_every_steps=${SAVE_EVERY_STEPS}"
echo "  trajectory_max_turns=${TRAJECTORY_MAX_TURNS} trajectory_chunk_stride=${TRAJECTORY_CHUNK_STRIDE} generated_max_images_per_turn=${GENERATED_MAX_IMAGES_PER_TURN}"
echo "  eval_trajectory_max_turns=${EVAL_TRAJECTORY_MAX_TURNS} eval_trajectory_chunk_stride=${EVAL_TRAJECTORY_CHUNK_STRIDE}"
echo "  train_exclude_task_types=${TRAIN_EXCLUDE_TASK_TYPES} eval_exclude_task_types=${EVAL_EXCLUDE_TASK_TYPES}"
echo "  train_speak_sampling_ratio=${TRAIN_SPEAK_SAMPLING_RATIO}"
echo "  train_balance_speak_listen_keep_delegate=${TRAIN_BALANCE_SPEAK_LISTEN_KEEP_DELEGATE}"
echo "  train_listen_to_speak_ratio_keep_delegate=${TRAIN_LISTEN_TO_SPEAK_RATIO_KEEP_DELEGATE}"
echo "  train_drop_chat_all_silence_chunks=${TRAIN_DROP_CHAT_ALL_SILENCE_CHUNKS}"
echo "  train_require_prior_instruction_for_action_chunks=${TRAIN_REQUIRE_PRIOR_INSTRUCTION_FOR_ACTION_CHUNKS}"
echo "  collapse_repeated_speak_segments=${COLLAPSE_REPEATED_SPEAK_SEGMENTS}"
echo "  drop_placeholder_speak_chunks=${DROP_PLACEHOLDER_SPEAK_CHUNKS}"
echo "  clean_event_grounding_templates=${CLEAN_EVENT_GROUNDING_TEMPLATES}"
echo "  listen_weight=${LISTEN_WEIGHT} speak_weight=${SPEAK_WEIGHT} speak_boundary_weight=${SPEAK_BOUNDARY_WEIGHT} delegate_weight=${DELEGATE_WEIGHT}"
echo "  speak_threshold_sweep=${SPEAK_THRESHOLD_SWEEP}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  log_dir=${LOG_DIR}"
echo "  wandb_dir=${WANDB_DIR}"

RJOB_ENV_VARS=(
  USER_ROOT CODE_DIR RUNS_ROOT TRAIN_ENV_PREFIX TRAIN_PYTHON_SCRIPT HF_HOME
  WANDB_MODE NCCL_DEBUG NCCL_IB_DISABLE TORCH_NCCL_ASYNC_ERROR_HANDLING
  PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
  MODEL TOKENIZER_MODEL PROCESSOR_MODEL SOURCE_ROOT TRAIN_DATA EVAL_DATA
  OUTPUT_DIR OUTPUT_ROOT LOG_DIR WANDB_DIR INPUT_SCHEMA MAX_LENGTH MAX_IMAGE_PIXELS
  FORCE_IMAGE_SIZE IMAGE_SCALE_RESOLUTION VISION_BATCH_SIZE
  ATTN_IMPLEMENTATION BATCH_SIZE GRAD_ACCUM_STEPS EPOCHS LEARNING_RATE
  WEIGHT_DECAY MAX_SLICE_NUMS TRAJECTORY_MAX_TURNS
  TRAJECTORY_CHUNK_STRIDE EVAL_TRAJECTORY_MAX_TURNS
  EVAL_TRAJECTORY_CHUNK_STRIDE GENERATED_CHUNK_STRIDE
  GENERATED_MAX_IMAGES_PER_TURN GENERATED_IMAGE_SELECTION
  TRAIN_EXCLUDE_TASK_TYPES EVAL_EXCLUDE_TASK_TYPES
  TRAIN_SPEAK_SAMPLING_RATIO TRAIN_BALANCE_SPEAK_LISTEN_KEEP_DELEGATE
  TRAIN_LISTEN_TO_SPEAK_RATIO_KEEP_DELEGATE
  TRAIN_DROP_CHAT_ALL_SILENCE_CHUNKS
  TRAIN_REQUIRE_PRIOR_INSTRUCTION_FOR_ACTION_CHUNKS
  COLLAPSE_REPEATED_SPEAK_SEGMENTS DROP_PLACEHOLDER_SPEAK_CHUNKS
  CLEAN_EVENT_GROUNDING_TEMPLATES LISTEN_WEIGHT SPEAK_WEIGHT
  SPEAK_BOUNDARY_WEIGHT DELEGATE_WEIGHT EVAL_BATCH_SIZE MAX_TRAIN_BATCHES
  NO_EVAL EVAL_ONLY EVAL_BEFORE_TRAIN EVAL_EVERY_STEPS EVAL_SAVE_PREDICTIONS_LIMIT
  SPEAK_THRESHOLD_SWEEP SAVE_EVERY_STEPS GRADIENT_CHECKPOINTING
  NO_SAVE_CHECKPOINTS LOG_EVERY MAX_EVAL_BATCHES MAX_CKPT_LIMIT KEEP_ALIVE
)
rjob_env_args=()
for name in "${RJOB_ENV_VARS[@]}"; do
  rjob_env_args+=(-e "${name}=${!name}")
done

if [[ "${MODE}" == "local" ]]; then
  for name in "${RJOB_ENV_VARS[@]}"; do
    export "${name}"
  done
  export NNODES NPROC_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT
  # Drop inherited nounset (SHELLOPTS) so /etc/profile.d scripts that
  # reference ZSH_VERSION do not abort. Skip -l: training uses explicit paths.
  exec env -u SHELLOPTS bash -c "${RUN_CMD}"
fi

rjob submit --name="${RJOB_NAME}" \
  --namespace="${RJOB_NAMESPACE}" \
  --task-type="${RJOB_TASK_TYPE}" \
  --gpu="${RJOB_GPU}" --memory="${RJOB_MEMORY}" --cpu="${RJOB_CPU}" \
  --charged-group="${RJOB_CHARGED_GROUP}" \
  --private-machine=group \
  --mount=gpfs://gpfs1/p1-shared:/mnt/shared-storage-user/p1-shared \
  --mount=gpfs://gpfs1/cuiganqu:/mnt/shared-storage-user/cuiganqu \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --mount=gpfs://gpfs2/p1-shared-2:/mnt/shared-storage-gpfs2/p1-shared-2 \
  --image="${RJOB_IMAGE}" \
  -P "${NODE_COUNT}" --host-network="${HOST_NETWORK}" \
  -e DISTRIBUTED_JOB=true \
  -e GROUP="${RJOB_NAMESPACE}" \
  -e NNODES="${NODE_COUNT}" \
  -e NPROC_PER_NODE="${NPROC_PER_NODE}" \
  -e MASTER_PORT="${MASTER_PORT}" \
  "${rjob_env_args[@]}" \
  -- bash -lc "${RUN_CMD}"
