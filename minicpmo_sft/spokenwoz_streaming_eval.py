"""Shared SpokenWOZ episode-v2 streaming evaluation primitives.

The standalone test evaluator and the in-training evaluator both use this
module so action eligibility, continuation handling, response episodes, and
metric definitions cannot drift apart.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


EVALUATION_SCHEMA = "spokenwoz_duplex_episode_v2"


@dataclass(frozen=True)
class EpisodeState:
    """State needed to group fixed-timeline output chunks into responses."""

    active_episode_id: str | None = None
    active_episode_chunk_index: int = -1
    response_episode_number: int = 0


def advance_episode_state(
    state: EpisodeState,
    *,
    wav_id: str,
    stream_state_before: str,
    pred_action: str,
    end_of_turn: bool,
) -> tuple[EpisodeState, dict[str, Any]]:
    """Advance one episode-v2 action/continuation decision.

    ``stream_state_before`` is authoritative: only an ``idle`` point starts a
    new listen/speak decision. A point reached while ``speaking`` belongs to the
    current response and is excluded from action denominators.
    """

    if stream_state_before not in {"idle", "speaking"}:
        raise ValueError(f"unsupported stream state: {stream_state_before!r}")
    if pred_action not in {"listen", "speak"}:
        raise ValueError(f"unsupported predicted action: {pred_action!r}")

    action_eval_eligible = stream_state_before == "idle"
    active_episode_id = state.active_episode_id
    active_episode_chunk_index = state.active_episode_chunk_index
    response_episode_number = state.response_episode_number
    response_episode_start = False
    response_episode_end = False

    if action_eval_eligible:
        if active_episode_id is not None:
            raise RuntimeError(
                f"idle duplex state still has active response {active_episode_id!r}"
            )
        if pred_action == "speak":
            response_episode_number += 1
            active_episode_id = f"{wav_id}:response-{response_episode_number:04d}"
            active_episode_chunk_index = 0
            response_episode_start = True
    else:
        if active_episode_id is None:
            raise RuntimeError(
                f"duplex state is speaking without an active response episode for {wav_id}"
            )
        active_episode_chunk_index += 1

    response_episode_id = active_episode_id
    response_episode_chunk_index = (
        active_episode_chunk_index if active_episode_id is not None else None
    )
    if active_episode_id is not None and end_of_turn:
        response_episode_end = True

    annotations = {
        "action_eval_eligible": action_eval_eligible,
        "action_decision": pred_action if action_eval_eligible else "continuation",
        "response_episode_id": response_episode_id,
        "response_episode_chunk_index": response_episode_chunk_index,
        "response_episode_start": response_episode_start,
        "response_episode_end": response_episode_end,
    }

    if response_episode_end:
        active_episode_id = None
        active_episode_chunk_index = -1

    return (
        EpisodeState(
            active_episode_id=active_episode_id,
            active_episode_chunk_index=active_episode_chunk_index,
            response_episode_number=response_episode_number,
        ),
        annotations,
    )


def trajectory_correct(records: Sequence[dict[str, Any]]) -> bool:
    """Compare gold speak points with predicted response starts at IDLE points."""

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
    """Merge every text chunk belonging to the same generated response."""

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
    """Calculate the episode-v2 fixed-timeline metrics."""

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
    trajectory_correct_count = sum(
        trajectory_correct(dialog) for dialog in by_dialog.values()
    )
    response_episodes = build_response_episodes(records)
    total_input_points = len(records)
    evaluated_points = len(action_records)
    continuation_points = len(continuation_records)
    return {
        "evaluation_schema": EVALUATION_SCHEMA,
        "raw_gold_points": total_input_points,
        "raw_should_speak_points": sum(
            record["gold_action"] == "speak" for record in records
        ),
        "raw_should_listen_points": sum(
            record["gold_action"] == "listen" for record in records
        ),
        "scorable_idle_points": evaluated_points,
        "evaluated_points": evaluated_points,
        "total_input_points": total_input_points,
        "action_eval_coverage": evaluated_points / total_input_points if records else 0.0,
        "should_speak_points": should_speak,
        "should_speak_predicted_speak": speak_hit,
        "speak_recall": speak_hit / should_speak if should_speak else 0.0,
        "should_listen_points": should_listen,
        "should_listen_predicted_speak": listen_false_speak,
        "false_speak_rate": listen_false_speak / should_listen if should_listen else 0.0,
        "trajectory_correct": trajectory_correct_count,
        "trajectory_total": len(by_dialog),
        "trajectory_acc": (
            trajectory_correct_count / len(by_dialog) if by_dialog else 0.0
        ),
        "continuation_points": continuation_points,
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
        "n_model_tokens": sum(int(record.get("n_model_tokens", 0)) for record in records),
        "n_tts_tokens": sum(int(record.get("n_tts_tokens", 0)) for record in records),
        "tts_audio_files": sum(bool(record.get("audio_path")) for record in records),
    }


AudioWriter = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any]], tuple[str | None, int]
]


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def evaluate_dialog(
    duplex: Any,
    dialog: dict[str, Any],
    *,
    max_new_speak_tokens_per_chunk: int,
    decode_mode: str = "greedy",
    listen_prob_scale: float = 1.0,
    prompt_wav_path: str | None = None,
    audio_writer: AudioWriter | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Run one real fixed-timeline duplex session for one ``wav_id``."""

    duplex.prepare(
        prefix_system_prompt="Streaming Omni Conversation.",
        ref_audio=None,
        prompt_wav_path=prompt_wav_path,
    )
    turns = dialog["turns"]
    records: list[dict[str, Any]] = []
    episode_state = EpisodeState()
    for turn_number, turn in enumerate(turns):
        waveform = np.asarray(turn["user_content"][0], dtype=np.float32)
        prefill = duplex.streaming_prefill(audio_waveform=waveform)
        if not prefill.get("success", False):
            raise RuntimeError(
                f"streaming_prefill failed for {dialog['wav_id']} turn "
                f"{turn['turn_index']}: {prefill.get('reason', 'unknown reason')}"
            )
        stream_state_before = "idle" if bool(duplex.current_turn_ended) else "speaking"
        generation = duplex.streaming_generate(
            prompt_wav_path=prompt_wav_path,
            max_new_speak_tokens_per_chunk=max_new_speak_tokens_per_chunk,
            decode_mode=decode_mode,
            listen_prob_scale=listen_prob_scale,
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

        relative_audio_path: str | None = None
        tts_samples = 0
        if pred_action == "speak" and audio_writer is not None:
            relative_audio_path, tts_samples = audio_writer(dialog, turn, generation)

        if pred_action == "speak" and logger is not None:
            logger.info(
                "MODEL_SPEAK wav_id=%s unit=%d source_turn=%d gold=%s decision=%s "
                "episode=%s episode_chunk=%s end_of_turn=%s text=%s audio=%s",
                dialog["wav_id"],
                int(turn["turn_index"]),
                int(turn["source_turn_index"]),
                turn["action"],
                annotations["action_decision"],
                annotations["response_episode_id"],
                annotations["response_episode_chunk_index"],
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
            **annotations,
            "action_correct": (
                pred_action == str(turn["action"])
                if annotations["action_eval_eligible"]
                else None
            ),
            "stream_state_before": stream_state_before,
            "stream_state_after": (
                "idle" if bool(duplex.current_turn_ended) else "speaking"
            ),
            "agent_text": str(turn.get("agent_text") or ""),
            "model_text": model_text,
            "transcript": str(turn.get("transcript") or ""),
            "domains": turn.get("domains", []),
            "slots": turn.get("slots", {}),
            "audio_path": relative_audio_path,
            "tts_audio_samples_24khz": int(tts_samples),
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


def write_jsonl(path: Any, records: Sequence[dict[str, Any]]) -> None:
    """Atomically write JSONL records to ``path``."""

    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_ready(record), ensure_ascii=False) + "\n")
        handle.flush()
    temporary.replace(path)
