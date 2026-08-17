#!/usr/bin/env bash

# rjob resources
RJOB_NAMESPACE="ailab-llmfrontier"
KUBEBRAIN_CLUSTER_ENTRY="https://h.pjlab.org.cn"
RJOB_TASK_TYPE="normal"
RJOB_GPU=8
RJOB_CPU=128
RJOB_MEMORY=1280000
RJOB_CHARGED_GROUP="llmfrontier_gpu"
RJOB_IMAGE="registry.h.pjlab.org.cn/ailab-llmfrontier/mechanism-merging-env:cu128-v1"
NODE_COUNT=1
NPROC_PER_NODE=8
LOCAL_NPROC_PER_NODE=8
MASTER_PORT=29645
RJOB_NAME_PREFIX="minicpmo-generated-traj-8gpu-t64-e2-event-clean-mediafix"

# Worker runtime
WANDB_MODE="offline"
NCCL_DEBUG="WARN"
NCCL_IB_DISABLE=0
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
PYTORCH_ALLOC_CONF="expandable_segments:True"
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# External environment, model, and data
TRAIN_ENV_PREFIX="/mnt/shared-storage-user/p1-shared/leihaodi/miniconda3/envs/minicpmo45_duplex_flashattn"
HF_HOME="/mnt/shared-storage-gpfs2/p1-shared-2/luoyun/.cache/huggingface"
MODEL="/mnt/shared-storage-gpfs2/p1-shared-2/zhangzhiqiu/interaction-video-train/minicpmo_runs/outputs/minicpmo2_t64_eval320_e2_l110_lr1e6_l2s0p3_sw3_eventClean_fixedImg448_mediaClean_late1_segTiming_20260815_194950/final"
TOKENIZER_MODEL="${MODEL}"
PROCESSOR_MODEL="${MODEL}"
SOURCE_ROOT="/mnt/shared-storage-gpfs2/p1-shared-2/daisong/datasets/JoyAI-VL-SFT/generated"
TRAIN_DATA="/mnt/shared-storage-gpfs2/p1-shared-2/luoyun/datasets/minicpmo_joyai_sft/generated_trajectory_split_event_clean_v1/train"
EVAL_DATA="/mnt/shared-storage-gpfs2/p1-shared-2/daisong/datasets/JoyAI-VL-SFT-DevSet/v1_rule_balanced_3000/sft/joyai_devset_v1.jsonl"

# Output naming; roots are derived from the project directory.
OUTPUT_NAME_PREFIX="minicpmo_eval_only_t320_eventClean_fixedImg448"
# OUTPUT_DIR=
# 手动指定可以实现断点续训

# Training
INPUT_SCHEMA="duplex" # 尝试和duplex的推理message同步，就是unit包括上下文的形式
MAX_LENGTH=40960
BATCH_SIZE=1
EVAL_BATCH_SIZE=2
GRAD_ACCUM_STEPS=8
EPOCHS=1
LEARNING_RATE="1e-6"
WEIGHT_DECAY="0.0"
GRADIENT_CHECKPOINTING=0
NO_SAVE_CHECKPOINTS=1

# Model and image processing
MAX_SLICE_NUMS=1
MAX_IMAGE_PIXELS=0
FORCE_IMAGE_SIZE=448
IMAGE_SCALE_RESOLUTION=0
VISION_BATCH_SIZE=0
ATTN_IMPLEMENTATION="flash_attention_2"

# Generated trajectory
TRAJECTORY_MAX_TURNS=64
TRAJECTORY_CHUNK_STRIDE=48
EVAL_TRAJECTORY_MAX_TURNS=64
EVAL_TRAJECTORY_CHUNK_STRIDE=64
GENERATED_CHUNK_STRIDE=64 # 没用到
GENERATED_MAX_IMAGES_PER_TURN=1
GENERATED_IMAGE_SELECTION="last"

# Filtering and loss weights
TRAIN_EXCLUDE_TASK_TYPES="narration,background,chat"
EVAL_EXCLUDE_TASK_TYPES="narration,background,chat"
TRAIN_SPEAK_SAMPLING_RATIO="-1"
TRAIN_BALANCE_SPEAK_LISTEN_KEEP_DELEGATE=1
TRAIN_LISTEN_TO_SPEAK_RATIO_KEEP_DELEGATE="0.3"
TRAIN_DROP_CHAT_ALL_SILENCE_CHUNKS=1
TRAIN_REQUIRE_PRIOR_INSTRUCTION_FOR_ACTION_CHUNKS=1
COLLAPSE_REPEATED_SPEAK_SEGMENTS=1
DROP_PLACEHOLDER_SPEAK_CHUNKS=1
CLEAN_EVENT_GROUNDING_TEMPLATES=1
LISTEN_WEIGHT="0.4"
SPEAK_WEIGHT="3.0"
SPEAK_BOUNDARY_WEIGHT=0
DELEGATE_WEIGHT="5.0"

# Evaluation, logging, and checkpoints
MAX_TRAIN_BATCHES=0
EVAL_ONLY=1
NO_EVAL=0
EVAL_BEFORE_TRAIN=0
EVAL_EVERY_STEPS=0
EVAL_SAVE_PREDICTIONS_LIMIT=0
SPEAK_THRESHOLD_SWEEP=""
SAVE_EVERY_STEPS=0
LOG_EVERY=2 # 单位是optimizer steps
MAX_EVAL_BATCHES=500
MAX_CKPT_LIMIT=2
KEEP_ALIVE=0
