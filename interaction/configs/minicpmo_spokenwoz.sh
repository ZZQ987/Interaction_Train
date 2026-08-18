#!/usr/bin/env bash

# rjob resources
RJOB_NAMESPACE="ailab-llmfrontier"
KUBEBRAIN_CLUSTER_ENTRY="https://h.pjlab.org.cn"
RJOB_TASK_TYPE="normal"
RJOB_GPU=8
RJOB_CPU=128
RJOB_MEMORY=1280000
RJOB_CHARGED_GROUP="llmfrontier_gpu"
RJOB_IMAGE="registry.h.pjlab.org.cn/ailab-p1-p1_gpu/slime:latest"
NODE_COUNT=1
NPROC_PER_NODE=8
LOCAL_NPROC_PER_NODE=8
MASTER_PORT=29646
RJOB_NAME_PREFIX="minicpmo-spokenwoz-duplex-audio"

# Worker runtime
WANDB_MODE="offline"
NCCL_DEBUG="WARN"
NCCL_IB_DISABLE=0
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
PYTORCH_ALLOC_CONF="expandable_segments:True"
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# External environment, model, and data
TRAIN_ENV_PREFIX="/mnt/shared-storage-user/p1-shared/leihaodi/miniconda3/envs/minicpmo45_duplex_flashattn"
PYARROW_SITE_PACKAGES="/mnt/shared-storage-user/p1-shared/leihaodi/miniconda3/envs/specbench/lib/python3.12/site-packages"
HF_HOME="/mnt/shared-storage-gpfs2/p1-shared-2/luoyun/.cache/huggingface"
MODEL="/mnt/shared-storage-gpfs2/p1-shared-2/luoyun/construct_demo/models/openbmb/MiniCPM-o-4_5"
TOKENIZER_MODEL="${MODEL}"
PROCESSOR_MODEL="${MODEL}"
SOURCE_ROOT="/mnt/shared-storage-user/leihaodi/Interaction_Train/data"
TRAIN_DATA="${SOURCE_ROOT}"
EVAL_DATA="${SOURCE_ROOT}"
DATASET_MODE="spokenwoz"
SPOKENWOZ_AUDIO_CHUNK_SECONDS="1"
SPOKENWOZ_SAMPLING_RATE=16000
SPOKENWOZ_TRAIN_PARQUET_PREFIX="train"
SPOKENWOZ_EVAL_PARQUET_PREFIX="dev"

# Output naming; set OUTPUT_DIR externally to resume a particular run.
OUTPUT_NAME_PREFIX="minicpmo_spokenwoz_all_train_dev_duplex_${SPOKENWOZ_AUDIO_CHUNK_SECONDS}s"

# Training
INPUT_SCHEMA="duplex"
MAX_LENGTH=16384
BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
EPOCHS=20
LEARNING_RATE="1e-6"
WEIGHT_DECAY="0.0"
GRADIENT_CHECKPOINTING=1
NO_SAVE_CHECKPOINTS=0

# There are no images in this dataset, but these values remain part of the
# shared launcher interface.
MAX_SLICE_NUMS=1
MAX_IMAGE_PIXELS=0
FORCE_IMAGE_SIZE=0
IMAGE_SCALE_RESOLUTION=0
VISION_BATCH_SIZE=0
ATTN_IMPLEMENTATION="flash_attention_2"

# Generated-trajectory-only settings (unused in DATASET_MODE=spokenwoz).
TRAJECTORY_MAX_TURNS=0
TRAJECTORY_CHUNK_STRIDE=0
EVAL_TRAJECTORY_MAX_TURNS=0
EVAL_TRAJECTORY_CHUNK_STRIDE=0
GENERATED_CHUNK_STRIDE=0
GENERATED_MAX_IMAGES_PER_TURN=0
GENERATED_IMAGE_SELECTION="last"

# SpokenWOZ naturally contains listen and speak units in every dialog.  Do not
# apply the generated-video trajectory filters/samplers.
TRAIN_EXCLUDE_TASK_TYPES=""
EVAL_EXCLUDE_TASK_TYPES=""
TRAIN_SPEAK_SAMPLING_RATIO="-1"
TRAIN_BALANCE_SPEAK_LISTEN_KEEP_DELEGATE=0
TRAIN_LISTEN_TO_SPEAK_RATIO_KEEP_DELEGATE="1.0"
TRAIN_DROP_CHAT_ALL_SILENCE_CHUNKS=0
TRAIN_REQUIRE_PRIOR_INSTRUCTION_FOR_ACTION_CHUNKS=0
COLLAPSE_REPEATED_SPEAK_SEGMENTS=0
DROP_PLACEHOLDER_SPEAK_CHUNKS=0
CLEAN_EVENT_GROUNDING_TEMPLATES=0
LISTEN_WEIGHT="1.0"
SPEAK_WEIGHT="3.0"
SPEAK_BOUNDARY_WEIGHT=0
DELEGATE_WEIGHT="1.0"

# Evaluation, logging, and checkpoints
MAX_TRAIN_BATCHES=0
NO_EVAL=0
EVAL_BEFORE_TRAIN=1
EVAL_EVERY_STEPS=0
EVAL_SAVE_PREDICTIONS_LIMIT=0
# Streaming eval always saves autoregressively generated text; this legacy
# teacher-forced decoding switch is intentionally disabled for SpokenWOZ.
EVAL_SAVE_MODEL_TEXT=0
EVAL_MAX_NEW_SPEAK_TOKENS=128
SPEAK_THRESHOLD_SWEEP=""
# Save only the epoch checkpoint with the highest validation trajectory_acc.
SAVE_EVERY_STEPS=0
SAVE_BEST_TRAJECTORY_CHECKPOINT=1
LOG_EVERY=2
MAX_EVAL_BATCHES=0
MAX_CKPT_LIMIT=1
KEEP_ALIVE=0
