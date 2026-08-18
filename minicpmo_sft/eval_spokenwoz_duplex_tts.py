#!/usr/bin/env python3
"""Run real duplex SpokenWOZ evaluation with generated text and TTS audio.

The fine-tuning checkpoints in this repository contain the trainable LLM and
the inference-time audio modules, but TTS was disabled while saving them.  This
script therefore loads the complete frozen base model (including TTS), copies
the fine-tuned ``llm.*`` weights from the checkpoint over it, and then performs
one stateful duplex streaming session per ``wav_id``.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from minicpmo_dataset import MiniCPMOSpokenWozDuplexDataset
from spokenwoz_streaming_eval import (
    EpisodeState,
    advance_episode_state,
    build_response_episodes as _shared_build_response_episodes,
    calculate_metrics as _shared_calculate_metrics,
    trajectory_correct as _shared_trajectory_correct,
)


DEFAULT_CHECKPOINT = Path("/mnt/shared-storage-user/leihaodi/interaction-save/ckpt-2100")
DEFAULT_BASE_MODEL = Path(
    "/mnt/shared-storage-gpfs2/p1-shared-2/luoyun/construct_demo/models/"
    "openbmb/MiniCPM-o-4_5"
)
DEFAULT_TEST_PARQUET = Path(
    "/mnt/shared-storage-user/leihaodi/Interaction_Train/data/data/"
    "test-00000-of-00005.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/shared-storage-user/leihaodi/interaction-save/eval-result/ckpt-2100"
)
DEFAULT_REF_AUDIO = DEFAULT_BASE_MODEL / "assets/system_ref_audio.wav"
DEFAULT_PYARROW_SITE_PACKAGES = Path(
    "/mnt/shared-storage-user/p1-shared/leihaodi/miniconda3/envs/"
    "specbench/lib/python3.12/site-packages"
)
OUTPUT_AUDIO_SAMPLE_RATE = 24_000
EVALUATION_SCHEMA = "spokenwoz_duplex_episode_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--test-parquet", type=Path, default=DEFAULT_TEST_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ref-audio", type=Path, default=DEFAULT_REF_AUDIO)
    parser.add_argument(
        "--pyarrow-site-packages",
        type=Path,
        default=DEFAULT_PYARROW_SITE_PACKAGES,
        help="site-packages directory containing pyarrow when it is absent from the model environment",
    )
    parser.add_argument("--audio-chunk-seconds", type=float, default=1.0)
    parser.add_argument("--input-sample-rate", type=int, default=16_000)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--decode-mode", choices=("greedy", "sampling"), default="greedy")
    parser.add_argument("--max-new-speak-tokens-per-chunk", type=int, default=128)
    parser.add_argument("--listen-prob-scale", type=float, default=1.0)
    parser.add_argument("--tts-n-timesteps", type=int, default=10)
    parser.add_argument("--tts-float16", action="store_true")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="number of GPUs for dialog-level data parallelism; 0 detects with nvidia-smi",
    )
    parser.add_argument("--max-dialogs", type=int, default=0, help="0 evaluates every dialog")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="remove prior artifacts in --output-dir and evaluate again from the first dialog",
    )
    parser.add_argument(
        "--gold-only",
        action="store_true",
        help="only inspect data and write gold counts; do not load the model",
    )
    args = parser.parse_args()
    if args.audio_chunk_seconds <= 0:
        parser.error("--audio-chunk-seconds must be positive")
    if args.input_sample_rate <= 0:
        parser.error("--input-sample-rate must be positive")
    if args.max_new_speak_tokens_per_chunk < 2:
        parser.error("--max-new-speak-tokens-per-chunk must be at least 2")
    if args.max_dialogs < 0:
        parser.error("--max-dialogs cannot be negative")
    if args.num_gpus < 0:
        parser.error("--num-gpus cannot be negative")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def absolute_path(path: Path) -> Path:
    """Make a path absolute without resolving user-visible GPFS symlinks."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def configure_logging(
    output_dir: Path,
    resume: bool,
    *,
    logger_name: str = "spokenwoz_duplex_eval",
    log_filename: str = "eval.log",
) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(
        output_dir / log_filename,
        mode="a" if resume else "w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def clear_previous_outputs(output_dir: Path) -> None:
    """Remove only this evaluator's known artifacts from one explicit directory."""
    if output_dir == Path(output_dir.anchor) or len(output_dir.parts) < 4:
        raise ValueError(f"refusing to clear unsafe output directory: {output_dir}")
    for filename in (
        "eval.log",
        "metrics.json",
        "predictions.jsonl",
        "response_episodes.jsonl",
        "progress.json",
        "run_config.json",
        "gold_counts.json",
    ):
        path = output_dir / filename
        if path.is_file() or path.is_symlink():
            path.unlink()
        temporary = output_dir / f"{filename}.tmp"
        if temporary.is_file() or temporary.is_symlink():
            temporary.unlink()
    audio_dir = output_dir / "audio"
    if audio_dir.is_dir():
        shutil.rmtree(audio_dir)
    workers_dir = output_dir / "workers"
    if workers_dir.is_dir():
        shutil.rmtree(workers_dir)


def detect_num_gpus(requested: int) -> tuple[int, int]:
    """Return (GPUs to use, GPUs visible to nvidia-smi)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot auto-detect GPUs with nvidia-smi") from exc
    detected = len([line for line in result.stdout.splitlines() if line.strip()])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        visible_count = len([part for part in visible.split(",") if part.strip()])
        detected = min(detected, visible_count)
    if detected <= 0:
        raise RuntimeError("nvidia-smi did not report any usable GPU")
    selected = requested or detected
    if selected > detected:
        raise ValueError(f"requested {selected} GPUs, but only {detected} are visible")
    return selected, detected


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(data), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def safe_filename(value: str, max_length: int = 120) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    normalized = normalized.strip("._") or "unknown"
    return normalized[:max_length]


def trajectory_correct(records: Sequence[dict[str, Any]]) -> bool:
    """Compare gold and predicted speak starts only while the model is idle."""
    gold_events = sorted(
        int(record["turn_index"])
        for record in records
        if bool(record["action_eval_eligible"]) and record["gold_action"] == "speak"
    )
    pred_events = sorted(
        int(record["turn_index"])
        for record in records
        if bool(record["action_eval_eligible"]) and record["pred_action"] == "speak"
    )
    return gold_events == pred_events


def build_response_episodes(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge all speak/continuation chunks belonging to the same model response."""
    episodes: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        episode_id = record.get("response_episode_id")
        if episode_id is None:
            continue
        episode_id = str(episode_id)
        episode = by_id.get(episode_id)
        if episode is None:
            episode = {
                "evaluation_schema": EVALUATION_SCHEMA,
                "response_episode_id": episode_id,
                "wav_id": str(record["wav_id"]),
                "start_turn_index": int(record["turn_index"]),
                "end_turn_index": int(record["turn_index"]),
                "start_source_turn_index": int(record["source_turn_index"]),
                "end_source_turn_index": int(record["source_turn_index"]),
                "start_gold_action": str(record["gold_action"]),
                "chunk_count": 0,
                "text_chunk_count": 0,
                "continuation_chunk_count": 0,
                "model_text": "",
                "audio_paths": [],
                "tts_audio_samples_24khz": 0,
                "complete": False,
            }
            by_id[episode_id] = episode
            episodes.append(episode)
        episode["end_turn_index"] = int(record["turn_index"])
        episode["end_source_turn_index"] = int(record["source_turn_index"])
        episode["chunk_count"] += 1
        if not bool(record["action_eval_eligible"]):
            episode["continuation_chunk_count"] += 1
        text_chunk = str(record.get("model_text") or "")
        if text_chunk:
            episode["text_chunk_count"] += 1
            episode["model_text"] += text_chunk
        audio_path = record.get("audio_path")
        if audio_path:
            episode["audio_paths"].append(str(audio_path))
        episode["tts_audio_samples_24khz"] += int(
            record.get("tts_audio_samples_24khz", 0)
        )
        if bool(record.get("response_episode_end", False)):
            episode["complete"] = True
    return episodes


def calculate_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    action_records = [record for record in records if bool(record["action_eval_eligible"])]
    continuation_records = [
        record for record in records if not bool(record["action_eval_eligible"])
    ]
    should_speak = sum(record["gold_action"] == "speak" for record in action_records)
    speak_hit = sum(
        record["gold_action"] == "speak" and record["pred_action"] == "speak"
        for record in action_records
    )
    should_listen = sum(record["gold_action"] == "listen" for record in action_records)
    listen_false_speak = sum(
        record["gold_action"] == "listen" and record["pred_action"] == "speak"
        for record in action_records
    )
    by_dialog: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dialog[str(record["wav_id"])].append(record)
    trajectory_correct_count = sum(trajectory_correct(dialog) for dialog in by_dialog.values())
    trajectory_total = len(by_dialog)
    response_episodes = build_response_episodes(records)
    return {
        "evaluation_schema": EVALUATION_SCHEMA,
        "should_speak_points": should_speak,
        "should_speak_predicted_speak": speak_hit,
        "speak_recall": speak_hit / should_speak if should_speak else 0.0,
        "trajectory_correct": trajectory_correct_count,
        "trajectory_total": trajectory_total,
        "trajectory_acc": trajectory_correct_count / trajectory_total if trajectory_total else 0.0,
        "should_listen_points": should_listen,
        "should_listen_predicted_speak": listen_false_speak,
        "false_speak_rate": listen_false_speak / should_listen if should_listen else 0.0,
        "evaluated_points": len(action_records),
        "total_input_points": len(records),
        "action_eval_coverage": len(action_records) / len(records) if records else 0.0,
        "raw_should_speak_points": sum(
            record["gold_action"] == "speak" for record in records
        ),
        "raw_should_listen_points": sum(
            record["gold_action"] == "listen" for record in records
        ),
        "continuation_points": len(continuation_records),
        "continuation_gold_speak_points": sum(
            record["gold_action"] == "speak" for record in continuation_records
        ),
        "continuation_gold_listen_points": sum(
            record["gold_action"] == "listen" for record in continuation_records
        ),
        "predicted_speak_points": sum(
            record["pred_action"] == "speak" for record in action_records
        ),
        "predicted_listen_points": sum(
            record["pred_action"] == "listen" for record in action_records
        ),
        "raw_predicted_speak_chunks": sum(
            record["pred_action"] == "speak" for record in records
        ),
        "raw_predicted_listen_chunks": sum(
            record["pred_action"] == "listen" for record in records
        ),
        "continuation_speak_chunks": sum(
            record["pred_action"] == "speak" for record in continuation_records
        ),
        "response_episode_count": len(response_episodes),
        "complete_response_episodes": sum(
            bool(episode["complete"]) for episode in response_episodes
        ),
        "incomplete_response_episodes": sum(
            not bool(episode["complete"]) for episode in response_episodes
        ),
        "tts_audio_files": sum(bool(record.get("audio_path")) for record in records),
    }


# Keep the standalone test evaluator and training evaluator on the exact same
# episode-v2 metric implementation. The local definitions above remain as a
# readable description of the original standalone schema.
trajectory_correct = _shared_trajectory_correct
build_response_episodes = _shared_build_response_episodes
calculate_metrics = _shared_calculate_metrics


def print_metric_summary(logger: logging.Logger, metrics: dict[str, Any], *, prefix: str) -> None:
    logger.info(
        "%s should_speak=%d speak_at_should_speak=%d speak_recall=%.6f "
        "trajectory_acc=%.6f (%d/%d) should_listen=%d speak_at_should_listen=%d "
        "false_speak_rate=%.6f action_points=%d continuation_points=%d episodes=%d",
        prefix,
        metrics["should_speak_points"],
        metrics["should_speak_predicted_speak"],
        metrics["speak_recall"],
        metrics["trajectory_acc"],
        metrics["trajectory_correct"],
        metrics["trajectory_total"],
        metrics["should_listen_points"],
        metrics["should_listen_predicted_speak"],
        metrics["false_speak_rate"],
        metrics["evaluated_points"],
        metrics["continuation_points"],
        metrics["response_episode_count"],
    )


def load_completed_predictions(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    records: list[dict[str, Any]] = []
    complete_dialogs: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            if record.get("evaluation_schema") != EVALUATION_SCHEMA:
                raise ValueError(
                    f"{path}:{line_number} uses an older evaluation schema; "
                    "resume cannot mix legacy per-chunk action metrics with "
                    f"{EVALUATION_SCHEMA}. Use --overwrite or a new --output-dir."
                )
            records.append(record)
            if record.get("is_dialog_final_unit"):
                complete_dialogs.add(str(record["wav_id"]))

    # Dialogs are appended atomically as groups.  Ignore a possible incomplete
    # trailing group after a killed process so --resume can safely recompute it.
    completed_records = [record for record in records if str(record["wav_id"]) in complete_dialogs]
    return completed_records, complete_dialogs


def append_dialog_predictions(path: Path, records: Sequence[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(json_ready(record), ensure_ascii=False) + "\n" for record in records)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_predictions_atomic(path: Path, records: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def get_parameter_or_buffer(model: Any, name: str) -> Any:
    try:
        return model.get_parameter(name)
    except AttributeError:
        return model.get_buffer(name)


def overlay_finetuned_llm(model: Any, checkpoint: Path, logger: logging.Logger) -> int:
    """Copy checkpoint LLM tensors into the complete base+TTS model in-place."""
    import torch
    from safetensors import safe_open

    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = {
        name: shard
        for name, shard in index["weight_map"].items()
        if str(name).startswith("llm.")
    }
    if not weight_map:
        raise ValueError(f"no llm.* tensors found in {index_path}")

    tensors_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        tensors_by_shard[str(shard)].append(str(name))

    loaded = 0
    with torch.no_grad():
        for shard_name in sorted(tensors_by_shard):
            shard_path = checkpoint / shard_name
            logger.info("overlaying %d LLM tensors from %s", len(tensors_by_shard[shard_name]), shard_path)
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                shard_keys = set(handle.keys())
                for name in sorted(tensors_by_shard[shard_name]):
                    if name not in shard_keys:
                        raise KeyError(f"{name} is indexed in {shard_name} but absent from the shard")
                    target = get_parameter_or_buffer(model, name)
                    source = handle.get_tensor(name)
                    if tuple(target.shape) != tuple(source.shape):
                        raise ValueError(
                            f"shape mismatch for {name}: model={tuple(target.shape)} checkpoint={tuple(source.shape)}"
                        )
                    target.copy_(source.to(dtype=target.dtype, device=target.device))
                    loaded += 1
    if loaded != len(weight_map):
        raise RuntimeError(f"loaded {loaded} LLM tensors, expected {len(weight_map)}")
    logger.info("overlaid all %d fine-tuned LLM tensors", loaded)
    return loaded


def load_duplex_model(
    args: argparse.Namespace,
    logger: logging.Logger,
    *,
    device_index: int = 0,
) -> Any:
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full duplex+TTS evaluation")
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"worker requested cuda:{device_index}, but torch sees only {torch.cuda.device_count()} GPUs"
        )
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    logger.info("loading frozen audio+TTS base model from %s onto %s", args.base_model, device)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "init_vision": False,
        "init_audio": True,
        "init_tts": True,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModel.from_pretrained(str(args.base_model), **load_kwargs)
    overlay_finetuned_llm(model, args.checkpoint, logger)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.checkpoint), trust_remote_code=True, local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(
        str(args.checkpoint), trust_remote_code=True, local_files_only=True
    )
    if not hasattr(processor, "tokenizer") or processor.tokenizer is None:
        processor.tokenizer = tokenizer
    model.processor = processor

    model.eval().to(device)
    logger.info(
        "model ready on %s; allocated=%.3f GiB reserved=%.3f GiB",
        device,
        torch.cuda.memory_allocated(device) / (1024**3),
        torch.cuda.memory_reserved(device) / (1024**3),
    )
    duplex = model.as_duplex(
        device=str(device),
        generate_audio=True,
        enable_float16=args.tts_float16,
        n_timesteps=args.tts_n_timesteps,
        chunk_ms=int(round(args.audio_chunk_seconds * 1000)),
        first_chunk_ms=int(round(args.audio_chunk_seconds * 1000)) + 35,
        sample_rate=args.input_sample_rate,
        force_listen_count=0,
        sliding_window_mode="off",
    )
    return duplex


def save_tts_audio(audio: Any, path: Path) -> int:
    import soundfile as sf

    if audio is None:
        return 0
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        return 0
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
    waveform = np.clip(waveform, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, OUTPUT_AUDIO_SAMPLE_RATE, subtype="PCM_16")
    return int(waveform.size)


def gold_records_for_dialog(dialog: dict[str, Any]) -> list[dict[str, Any]]:
    turns = dialog["turns"]
    records: list[dict[str, Any]] = []
    for turn_number, turn in enumerate(turns):
        records.append(
            {
                "wav_id": dialog["wav_id"],
                "turn_index": int(turn["turn_index"]),
                "source_turn_index": int(turn["source_turn_index"]),
                "source_turn_number": int(turn["source_turn_number"]),
                "audio_chunk_index": int(turn["audio_chunk_index"]),
                "audio_chunk_count": int(turn["audio_chunk_count"]),
                "valid_audio_samples": int(turn["valid_audio_samples"]),
                "padded_audio_samples": int(turn["padded_audio_samples"]),
                "gold_action": str(turn["action"]),
                "pred_action": "not_evaluated",
                "agent_text": str(turn["agent_text"]),
                "model_text": "",
                "domains": turn["domains"],
                "slots": turn["slots"],
                "audio_path": None,
                "is_dialog_final_unit": turn_number == len(turns) - 1,
            }
        )
    return records


def evaluate_dialog(
    duplex: Any,
    dialog: dict[str, Any],
    args: argparse.Namespace,
    audio_dir: Path,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    # Only prompt_wav_path is supplied.  The reference voice conditions TTS but
    # is not inserted as extra audio into the LLM context, matching SFT inputs.
    duplex.prepare(
        prefix_system_prompt="Streaming Omni Conversation.",
        ref_audio=None,
        prompt_wav_path=str(args.ref_audio),
    )
    turns = dialog["turns"]
    records: list[dict[str, Any]] = []
    episode_state = EpisodeState()
    for turn_number, turn in enumerate(turns):
        waveform = np.asarray(turn["user_content"][0], dtype=np.float32)
        prefill = duplex.streaming_prefill(audio_waveform=waveform)
        if not prefill.get("success", False):
            raise RuntimeError(
                f"streaming_prefill failed for {dialog['wav_id']} turn {turn['turn_index']}: "
                f"{prefill.get('reason', 'unknown reason')}"
            )
        stream_state_before = "idle" if bool(duplex.current_turn_ended) else "speaking"
        generation = duplex.streaming_generate(
            prompt_wav_path=str(args.ref_audio),
            max_new_speak_tokens_per_chunk=args.max_new_speak_tokens_per_chunk,
            decode_mode=args.decode_mode,
            listen_prob_scale=args.listen_prob_scale,
        )
        pred_action = "listen" if bool(generation.get("is_listen", True)) else "speak"
        model_text = str(generation.get("text") or "")
        end_of_turn = bool(generation.get("end_of_turn", False))
        episode_state, annotations = advance_episode_state(
            episode_state,
            wav_id=str(dialog["wav_id"]),
            stream_state_before=stream_state_before,
            pred_action=pred_action,
            end_of_turn=end_of_turn,
        )
        action_eval_eligible = bool(annotations["action_eval_eligible"])
        action_decision = str(annotations["action_decision"])
        response_episode_id = annotations["response_episode_id"]
        response_episode_chunk_index = annotations["response_episode_chunk_index"]
        response_episode_start = bool(annotations["response_episode_start"])
        response_episode_end = bool(annotations["response_episode_end"])
        relative_audio_path: str | None = None
        tts_samples = 0
        if pred_action == "speak":
            filename = (
                f"{safe_filename(str(dialog['wav_id']))}"
                f"_source-turn-{int(turn['source_turn_index']):04d}"
                f"_unit-{int(turn['turn_index']):05d}"
                f"_chunk-{int(turn['audio_chunk_index']):03d}.wav"
            )
            audio_path = audio_dir / filename
            tts_samples = save_tts_audio(generation.get("audio_waveform"), audio_path)
            if tts_samples:
                relative_audio_path = str(audio_path.relative_to(args.output_dir))
            logger.info(
                "MODEL_SPEAK wav_id=%s unit=%d source_turn=%d gold=%s decision=%s "
                "episode=%s episode_chunk=%s end_of_turn=%s text=%s audio=%s",
                dialog["wav_id"],
                int(turn["turn_index"]),
                int(turn["source_turn_index"]),
                turn["action"],
                action_decision,
                response_episode_id,
                response_episode_chunk_index,
                end_of_turn,
                json.dumps(model_text, ensure_ascii=False),
                relative_audio_path or "NONE",
            )

        record = {
            "evaluation_schema": EVALUATION_SCHEMA,
            "wav_id": dialog["wav_id"],
            "turn_id": turn["id"],
            "turn_index": int(turn["turn_index"]),
            "source_turn_index": int(turn["source_turn_index"]),
            "source_turn_number": int(turn["source_turn_number"]),
            "audio_chunk_index": int(turn["audio_chunk_index"]),
            "audio_chunk_count": int(turn["audio_chunk_count"]),
            "valid_audio_samples": int(turn["valid_audio_samples"]),
            "padded_audio_samples": int(turn["padded_audio_samples"]),
            "gold_action": str(turn["action"]),
            "pred_action": pred_action,
            "action_decision": action_decision,
            "action_eval_eligible": action_eval_eligible,
            "action_correct": (
                pred_action == str(turn["action"]) if action_eval_eligible else None
            ),
            "stream_state_before": stream_state_before,
            "stream_state_after": (
                "idle" if bool(duplex.current_turn_ended) else "speaking"
            ),
            "response_episode_id": response_episode_id,
            "response_episode_chunk_index": response_episode_chunk_index,
            "response_episode_start": response_episode_start,
            "response_episode_end": response_episode_end,
            "agent_text": str(turn["agent_text"]),
            "model_text": model_text,
            "transcript": str(turn["transcript"]),
            "domains": turn["domains"],
            "slots": turn["slots"],
            "audio_path": relative_audio_path,
            "tts_audio_samples_24khz": tts_samples,
            "end_of_turn": end_of_turn,
            "model_current_time": generation.get("current_time"),
            "n_model_tokens": int(generation.get("n_tokens", 0)),
            "n_tts_tokens": int(generation.get("n_tts_tokens", 0)),
            "cost_prefill_seconds": float(prefill.get("cost_all", 0.0)),
            "cost_generate_seconds": float(generation.get("cost_all", 0.0)),
            "is_dialog_final_unit": turn_number == len(turns) - 1,
        }
        records.append(record)
    return records


def balanced_dialog_shards(
    dataset: Any,
    dialog_indices: Sequence[int],
    worker_count: int,
) -> tuple[list[list[int]], list[int]]:
    """Greedily balance whole dialogs by their number of one-second units."""
    weighted_indices: list[tuple[int, int]] = []
    for dialog_index in dialog_indices:
        weighted_indices.append((len(dataset[dialog_index]["turns"]), dialog_index))
    shards: list[list[int]] = [[] for _ in range(worker_count)]
    loads = [0 for _ in range(worker_count)]
    for unit_count, dialog_index in sorted(weighted_indices, reverse=True):
        worker_id = min(range(worker_count), key=lambda index: loads[index])
        shards[worker_id].append(dialog_index)
        loads[worker_id] += unit_count
    for shard in shards:
        shard.sort()
    return shards, loads


def evaluate_worker(
    worker_id: int,
    device_index: int,
    dialog_indices: Sequence[int],
    args: argparse.Namespace,
    worker_dir: Path,
) -> None:
    logger = configure_logging(
        worker_dir,
        resume=False,
        logger_name=f"spokenwoz_duplex_eval.worker_{worker_id:02d}",
    )
    try:
        logger.info(
            "worker_start worker=%d cuda=%d dialogs=%d indices=%s",
            worker_id,
            device_index,
            len(dialog_indices),
            json.dumps(list(dialog_indices)),
        )
        dataset = MiniCPMOSpokenWozDuplexDataset(
            args.test_parquet,
            audio_chunk_seconds=args.audio_chunk_seconds,
            sampling_rate=args.input_sample_rate,
            pyarrow_site_packages=args.pyarrow_site_packages,
        )
        predictions_path = worker_dir / "predictions.jsonl"
        if predictions_path.exists():
            predictions_path.unlink()
        duplex = load_duplex_model(args, logger, device_index=device_index)
        worker_records: list[dict[str, Any]] = []
        audio_dir = args.output_dir / "audio"
        for worker_position, dialog_index in enumerate(dialog_indices):
            dialog = dataset[dialog_index]
            dialog_started = time.monotonic()
            dialog_records = evaluate_dialog(duplex, dialog, args, audio_dir, logger)
            append_dialog_predictions(predictions_path, dialog_records)
            worker_records.extend(dialog_records)
            atomic_write_json(
                worker_dir / "progress.json",
                {
                    "worker_id": worker_id,
                    "gpu_index": device_index,
                    "completed_dialogs": worker_position + 1,
                    "assigned_dialogs": len(dialog_indices),
                    "last_dialog_index": dialog_index,
                    "last_wav_id": dialog["wav_id"],
                    "completed_points": len(worker_records),
                    "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
                },
            )
            logger.info(
                "dialog_done worker=%d position=%d/%d dataset_index=%d wav_id=%s units=%d elapsed=%.2fs",
                worker_id,
                worker_position + 1,
                len(dialog_indices),
                dialog_index,
                dialog["wav_id"],
                len(dialog_records),
                time.monotonic() - dialog_started,
            )
        worker_metrics = calculate_metrics(worker_records)
        worker_metrics.update(
            {
                "worker_id": worker_id,
                "gpu_index": device_index,
                "assigned_dialogs": len(dialog_indices),
                "completed": True,
            }
        )
        atomic_write_json(worker_dir / "metrics.json", worker_metrics)
        print_metric_summary(logger, worker_metrics, prefix=f"WORKER_{worker_id:02d}_FINAL")
    except Exception:
        logger.exception("worker_failed worker=%d cuda=%d", worker_id, device_index)
        raise


def read_worker_predictions(worker_dir: Path) -> list[dict[str, Any]]:
    path = worker_dir / "predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
    return records


def run_parallel_evaluation(
    args: argparse.Namespace,
    dataset: Any,
    dialog_limit: int,
    predictions_path: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    existing_records, completed_dialogs = (
        load_completed_predictions(predictions_path) if args.resume else ([], set())
    )
    if existing_records:
        logger.info(
            "resume loaded %d predictions for %d completed dialogs",
            len(existing_records),
            len(completed_dialogs),
        )
    remaining_indices = [
        dialog_index
        for dialog_index in range(dialog_limit)
        if dataset.dialog_ids[dialog_index] not in completed_dialogs
    ]
    worker_count = min(args.num_gpus, len(remaining_indices)) if remaining_indices else 0
    workers_root = args.output_dir / "workers"
    if workers_root.is_dir():
        shutil.rmtree(workers_root)
    workers_root.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audio").mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    worker_dirs: list[Path] = []
    if worker_count:
        shards, estimated_loads = balanced_dialog_shards(dataset, remaining_indices, worker_count)
        logger.info(
            "parallel_eval workers=%d remaining_dialogs=%d estimated_unit_loads=%s",
            worker_count,
            len(remaining_indices),
            json.dumps(estimated_loads),
        )
        context = mp.get_context("spawn")
        processes: list[mp.Process] = []
        for worker_id, shard in enumerate(shards):
            worker_dir = workers_root / f"worker-{worker_id:02d}"
            worker_dirs.append(worker_dir)
            process = context.Process(
                target=evaluate_worker,
                args=(worker_id, worker_id, shard, args, worker_dir),
                name=f"spokenwoz-eval-gpu-{worker_id:02d}",
            )
            process.start()
            processes.append(process)
            time.sleep(1.0)

        last_progress_log = 0.0
        while any(process.is_alive() for process in processes):
            failed = [process for process in processes if process.exitcode not in (None, 0)]
            if failed:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    process.join(timeout=30)
                details = ", ".join(f"{p.name}:exit={p.exitcode}" for p in failed)
                raise RuntimeError(f"one or more evaluation workers failed: {details}")
            now = time.monotonic()
            if now - last_progress_log >= 30:
                completed = 0
                points = 0
                per_worker: list[str] = []
                for worker_id, (worker_dir, shard) in enumerate(zip(worker_dirs, shards)):
                    progress_path = worker_dir / "progress.json"
                    worker_completed = 0
                    worker_points = 0
                    if progress_path.is_file():
                        try:
                            with progress_path.open("r", encoding="utf-8") as handle:
                                progress = json.load(handle)
                            worker_completed = int(progress.get("completed_dialogs", 0))
                            worker_points = int(progress.get("completed_points", 0))
                        except (OSError, ValueError, TypeError):
                            pass
                    completed += worker_completed
                    points += worker_points
                    per_worker.append(f"w{worker_id}={worker_completed}/{len(shard)}")
                logger.info(
                    "parallel_progress dialogs=%d/%d points=%d workers=%s",
                    completed,
                    len(remaining_indices),
                    points,
                    ",".join(per_worker),
                )
                last_progress_log = now
            time.sleep(5)
        for process in processes:
            process.join()
        failed = [process for process in processes if process.exitcode != 0]
        if failed:
            details = ", ".join(f"{p.name}:exit={p.exitcode}" for p in failed)
            raise RuntimeError(f"one or more evaluation workers failed: {details}")

    new_records: list[dict[str, Any]] = []
    for worker_dir in worker_dirs:
        new_records.extend(read_worker_predictions(worker_dir))
    all_records = list(existing_records) + new_records
    dialog_order = {wav_id: index for index, wav_id in enumerate(dataset.dialog_ids[:dialog_limit])}
    all_records.sort(
        key=lambda record: (
            dialog_order[str(record["wav_id"])],
            int(record["turn_index"]),
        )
    )
    write_predictions_atomic(predictions_path, all_records)
    response_episodes = build_response_episodes(all_records)
    write_predictions_atomic(
        args.output_dir / "response_episodes.jsonl",
        response_episodes,
    )
    final_metrics = calculate_metrics(all_records)
    final_metrics.update(
        {
            "completed": len({str(record["wav_id"]) for record in all_records}) >= dialog_limit,
            "completed_dialogs": len({str(record["wav_id"]) for record in all_records}),
            "selected_dialogs": dialog_limit,
            "processed_dialogs_this_run": len(remaining_indices),
            "num_gpus": args.num_gpus,
            "elapsed_seconds_this_run": time.monotonic() - started,
            "finished_at": datetime.now(timezone.utc).astimezone().isoformat(),
        }
    )
    atomic_write_json(args.output_dir / "metrics.json", final_metrics)
    atomic_write_json(
        args.output_dir / "progress.json",
        {
            "completed_dialogs": sorted({str(record["wav_id"]) for record in all_records}),
            "completed_dialog_count": final_metrics["completed_dialogs"],
            "selected_dialogs": dialog_limit,
            "completed": final_metrics["completed"],
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        },
    )
    return final_metrics


def write_run_config(args: argparse.Namespace, dataset: Any, output_dir: Path) -> None:
    config = {
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "checkpoint": str(args.checkpoint),
        "base_model": str(args.base_model),
        "test_parquet": str(args.test_parquet),
        "output_dir": str(output_dir),
        "ref_audio": str(args.ref_audio),
        "audio_chunk_seconds": args.audio_chunk_seconds,
        "input_sample_rate": args.input_sample_rate,
        "output_audio_sample_rate": OUTPUT_AUDIO_SAMPLE_RATE,
        "decode_mode": args.decode_mode,
        "max_new_speak_tokens_per_chunk": args.max_new_speak_tokens_per_chunk,
        "listen_prob_scale": args.listen_prob_scale,
        "tts_n_timesteps": args.tts_n_timesteps,
        "tts_float16": args.tts_float16,
        "num_gpus": args.num_gpus,
        "gpu_detection": "nvidia-smi" if args.requested_num_gpus == 0 else "explicit",
        "max_dialogs": args.max_dialogs,
        "dataset_dialogs": len(dataset),
        "dataset_source_rows": dataset.source_row_count,
        "evaluation_schema": EVALUATION_SCHEMA,
        "streaming_action_policy": (
            "advance one real input chunk per duplex step; score listen/speak only while "
            "the model is idle; merge speaking-state chunks into one response episode"
        ),
        "action_metric_denominator": (
            "only input points where stream_state_before is idle; continuation chunks are "
            "reported separately and excluded from speak/listen confusion metrics"
        ),
        "trajectory_acc_definition": (
            "exact equality of gold speak points and predicted response starts among "
            "action-evaluable idle points for each wav_id"
        ),
    }
    atomic_write_json(output_dir / "run_config.json", config)


def require_paths(args: argparse.Namespace) -> None:
    required = [args.test_parquet]
    if not args.gold_only:
        required.extend(
            [
                args.checkpoint / "model.safetensors.index.json",
                args.base_model / "config.json",
                args.ref_audio,
            ]
        )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    args.checkpoint = absolute_path(args.checkpoint)
    args.base_model = absolute_path(args.base_model)
    args.test_parquet = absolute_path(args.test_parquet)
    args.output_dir = absolute_path(args.output_dir)
    args.ref_audio = absolute_path(args.ref_audio)
    require_paths(args)

    args.requested_num_gpus = args.num_gpus
    if not args.gold_only:
        args.num_gpus, detected_gpus = detect_num_gpus(args.num_gpus)
    else:
        detected_gpus = 0

    predictions_path = args.output_dir / "predictions.jsonl"
    if args.overwrite:
        clear_previous_outputs(args.output_dir)
    if predictions_path.exists() and not args.resume:
        raise FileExistsError(
            f"{predictions_path} already exists; pass --resume to continue without overwriting results"
        )
    logger = configure_logging(args.output_dir, args.resume)
    logger.info("checkpoint=%s", args.checkpoint)
    logger.info("test_parquet=%s", args.test_parquet)
    logger.info("output_dir=%s", args.output_dir)
    if not args.gold_only:
        logger.info(
            "num_gpus requested=%d selected=%d detected_by_nvidia_smi=%d",
            args.requested_num_gpus,
            args.num_gpus,
            detected_gpus,
        )

    dataset = MiniCPMOSpokenWozDuplexDataset(
        args.test_parquet,
        audio_chunk_seconds=args.audio_chunk_seconds,
        sampling_rate=args.input_sample_rate,
        pyarrow_site_packages=args.pyarrow_site_packages,
    )
    dialog_limit = len(dataset) if args.max_dialogs == 0 else min(len(dataset), args.max_dialogs)
    write_run_config(args, dataset, args.output_dir)
    logger.info(
        "dataset dialogs=%d source_rows=%d selected_dialogs=%d",
        len(dataset),
        dataset.source_row_count,
        dialog_limit,
    )

    if args.gold_only:
        gold_records: list[dict[str, Any]] = []
        for dialog_index in range(dialog_limit):
            gold_records.extend(gold_records_for_dialog(dataset[dialog_index]))
        gold_metrics = {
            "should_speak_points": sum(r["gold_action"] == "speak" for r in gold_records),
            "should_listen_points": sum(r["gold_action"] == "listen" for r in gold_records),
            "points": len(gold_records),
            "dialogs": dialog_limit,
        }
        atomic_write_json(args.output_dir / "gold_counts.json", gold_metrics)
        logger.info("gold-only counts=%s", json.dumps(gold_metrics, ensure_ascii=False, sort_keys=True))
        return

    final_metrics = run_parallel_evaluation(
        args,
        dataset,
        dialog_limit,
        predictions_path,
        logger,
    )
    print_metric_summary(logger, final_metrics, prefix="FINAL_METRICS")
    logger.info("predictions=%s", predictions_path)
    logger.info("response_episodes=%s", args.output_dir / "response_episodes.jsonl")
    logger.info("tts_audio_dir=%s", args.output_dir / "audio")


if __name__ == "__main__":
    main()
