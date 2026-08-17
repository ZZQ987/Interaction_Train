#!/usr/bin/env python3
"""8-GPU FSDP no-LoRA MiniCPM-o text-policy SFT with heldout eval."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from sft_logging import JsonlMetricLogger, setup_logging
from minicpmo_dataset import (
    IGNORE_INDEX,
    MiniCPMOGeneratedTrajectoryDataset,
    MiniCPMOInteractionDataset,
    MiniCPMOTrajectoryDataset,
    MiniCPMODuplexTrajectoryCollator,
    MiniCPMODataCollator,
)


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_rank0() -> bool:
    return _rank() == 0


def _set_default_hf_cache() -> None:
    cache = Path.cwd() / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _setup_dist() -> torch.device:
    torch.cuda.set_device(_local_rank())
    if not dist.is_initialized():
        timeout_seconds = int(os.environ.get("TORCH_DIST_TIMEOUT_SECONDS", "3600"))
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device("cuda", _local_rank()),
            timeout=timedelta(seconds=timeout_seconds),
        )
    return torch.device("cuda", _local_rank())


def _cleanup_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _weighted_ce_loss(logits: torch.Tensor, labels: torch.Tensor, loss_weights: torch.Tensor | None) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    labels = labels.to(logits.device).long()
    flat_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    mask = labels.reshape(-1) != IGNORE_INDEX
    if loss_weights is None:
        return flat_loss[mask].mean()
    weights = loss_weights.to(logits.device).float().reshape(-1)
    return (flat_loss * weights)[mask].sum() / weights[mask].sum().clamp_min(1.0)


def _parse_float_list(value: str | Sequence[float] | None) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return sorted(set(float(part) for part in parts))
    return sorted(set(float(item) for item in value))


def _freeze_non_llm(model: torch.nn.Module) -> tuple[int, int]:
    for param in model.parameters():
        param.requires_grad = False
    if not hasattr(model, "llm"):
        raise AttributeError("model has no .llm module")
    for param in model.llm.parameters():
        param.requires_grad = True
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def _wrap_model_fsdp(model: torch.nn.Module, device: torch.device) -> FSDP:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer})
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
    )


class BinaryRatioDistributedSampler(Sampler[int]):
    """Distributed sampler targeting a fixed positive/negative sample ratio."""

    def __init__(
        self,
        dataset: Any,
        positive_flags: list[bool],
        *,
        positive_ratio: float,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        if len(positive_flags) != len(dataset):
            raise ValueError(f"positive_flags length {len(positive_flags)} != dataset length {len(dataset)}")
        self.dataset = dataset
        self.positive_ratio = min(max(float(positive_ratio), 0.0), 1.0)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.positive_indices = [idx for idx, flag in enumerate(positive_flags) if flag]
        self.negative_indices = [idx for idx, flag in enumerate(positive_flags) if not flag]
        if not self.positive_indices or not self.negative_indices:
            raise ValueError(
                f"need both positive and negative samples, got positive={len(self.positive_indices)} "
                f"negative={len(self.negative_indices)}"
            )
        self.num_samples = int(math.ceil(len(dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Any:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        positive_count = int(round(self.total_size * self.positive_ratio))
        negative_count = self.total_size - positive_count

        positive_tensor = torch.tensor(self.positive_indices, dtype=torch.long)
        negative_tensor = torch.tensor(self.negative_indices, dtype=torch.long)
        sampled_positive = positive_tensor[
            torch.randint(len(positive_tensor), (positive_count,), generator=generator)
        ].tolist()
        sampled_negative = negative_tensor[
            torch.randint(len(negative_tensor), (negative_count,), generator=generator)
        ].tolist()
        indices = sampled_positive + sampled_negative
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[idx] for idx in order]
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class SpeakListenBalancedDistributedSampler(Sampler[int]):
    """Distributed sampler that keeps all delegate chunks and samples speak/listen chunks by ratio."""

    def __init__(
        self,
        dataset: Any,
        action_labels: list[str],
        *,
        listen_to_speak_ratio: float = 1.0,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        if len(action_labels) != len(dataset):
            raise ValueError(f"action_labels length {len(action_labels)} != dataset length {len(dataset)}")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.listen_to_speak_ratio = max(0.0, float(listen_to_speak_ratio))
        self.delegate_indices = [idx for idx, label in enumerate(action_labels) if label == "delegate"]
        self.speak_indices = [idx for idx, label in enumerate(action_labels) if label == "speak"]
        self.listen_indices = [idx for idx, label in enumerate(action_labels) if label not in {"speak", "delegate"}]
        if not self.speak_indices or (self.listen_to_speak_ratio > 0 and not self.listen_indices):
            raise ValueError(
                f"need both speak and listen chunks, got speak={len(self.speak_indices)} "
                f"listen={len(self.listen_indices)} delegate={len(self.delegate_indices)}"
            )
        self.speak_count = len(self.speak_indices)
        self.listen_count = min(len(self.listen_indices), int(round(self.speak_count * self.listen_to_speak_ratio)))
        epoch_size = len(self.delegate_indices) + self.speak_count + self.listen_count
        self.num_samples = int(math.ceil(epoch_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    @staticmethod
    def _shuffled_take(indices: list[int], count: int, generator: torch.Generator) -> list[int]:
        if count <= 0:
            return []
        tensor = torch.tensor(indices, dtype=torch.long)
        if count <= len(tensor):
            order = torch.randperm(len(tensor), generator=generator)[:count]
            return tensor[order].tolist()
        sampled = tensor[torch.randint(len(tensor), (count,), generator=generator)]
        return sampled.tolist()

    def __iter__(self) -> Any:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        sampled_delegate = self._shuffled_take(self.delegate_indices, len(self.delegate_indices), generator)
        sampled_speak = self._shuffled_take(self.speak_indices, self.speak_count, generator)
        sampled_listen = self._shuffled_take(self.listen_indices, self.listen_count, generator)
        indices = sampled_delegate + sampled_speak + sampled_listen
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[idx] for idx in order]
        if len(indices) < self.total_size:
            pad_order = torch.randint(len(indices), (self.total_size - len(indices),), generator=generator).tolist()
            indices.extend(indices[idx] for idx in pad_order)
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def _build_loader(
    dataset: MiniCPMOInteractionDataset,
    collator: Any,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    positive_flags: list[bool] | None = None,
    positive_ratio: float = -1.0,
    action_labels: list[str] | None = None,
    listen_to_speak_ratio: float = 1.0,
) -> tuple[DataLoader, Any]:
    if action_labels is not None:
        sampler = SpeakListenBalancedDistributedSampler(
            dataset,
            action_labels,
            listen_to_speak_ratio=listen_to_speak_ratio,
            num_replicas=_world_size(),
            rank=_rank(),
            seed=seed,
        )
    elif positive_flags is not None and positive_ratio >= 0:
        sampler = BinaryRatioDistributedSampler(
            dataset,
            positive_flags,
            positive_ratio=positive_ratio,
            num_replicas=_world_size(),
            rank=_rank(),
            seed=seed,
        )
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=_world_size(),
            rank=_rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collator,
        pin_memory=True,
    )
    return loader, sampler


@torch.no_grad()
def _score_marker(logits: torch.Tensor, row_idx: int, start_pos: int, marker_ids: list[int]) -> float:
    if not marker_ids or start_pos < 0 or start_pos + len(marker_ids) > logits.shape[1]:
        return float("-inf")
    score = 0.0
    for offset, token_id in enumerate(marker_ids):
        score += float(logits[row_idx, start_pos + offset, token_id].float().item())
    return score / max(1, len(marker_ids))


@torch.no_grad()
def _marker_is_argmax(logits: torch.Tensor, row_idx: int, start_pos: int, marker_ids: list[int]) -> bool:
    if not marker_ids or start_pos < 0 or start_pos + len(marker_ids) > logits.shape[1]:
        return False
    for offset, token_id in enumerate(marker_ids):
        pred_id = int(torch.argmax(logits[row_idx, start_pos + offset]).item())
        if pred_id != int(token_id):
            return False
    return True


def _marker_ids(tokenizer: Any, text: str) -> list[int]:
    token_id = tokenizer.convert_tokens_to_ids(text)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is not None and token_id != unk_id:
        return [int(token_id)]
    try:
        return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
    except TypeError:
        return [int(x) for x in tokenizer.encode(text)]


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def _match_labeled_positions(
    gold_events: Sequence[tuple[int, str]],
    pred_events: Sequence[tuple[int, str]],
    *,
    window: int,
) -> int:
    used = [False] * len(pred_events)
    matched = 0
    for gold_idx, gold_label in sorted(gold_events):
        best_j = -1
        best_dist = window + 1
        for j, (pred_idx, pred_label) in enumerate(pred_events):
            if used[j] or pred_label != gold_label:
                continue
            dist = abs(pred_idx - gold_idx)
            if dist <= window and dist < best_dist:
                best_j = j
                best_dist = dist
        if best_j >= 0:
            used[best_j] = True
            matched += 1
    return matched


def _match_labeled_positions_late_only(
    gold_events: Sequence[tuple[int, str]],
    pred_events: Sequence[tuple[int, str]],
    *,
    late_window: int,
) -> int:
    used = [False] * len(pred_events)
    matched = 0
    for gold_idx, gold_label in sorted(gold_events):
        best_j = -1
        best_delay = late_window + 1
        for j, (pred_idx, pred_label) in enumerate(pred_events):
            if used[j] or pred_label != gold_label:
                continue
            delay = pred_idx - gold_idx
            if 0 <= delay <= late_window and delay < best_delay:
                best_j = j
                best_delay = delay
        if best_j >= 0:
            used[best_j] = True
            matched += 1
    return matched


def _dedup_consecutive_events(events: Sequence[tuple[int, str]]) -> list[tuple[int, str]]:
    """Keep the first event in each consecutive same-label event run."""
    out: list[tuple[int, str]] = []
    last_idx: int | None = None
    last_label: str | None = None
    for idx, label in sorted(events):
        if last_idx is not None and idx == last_idx + 1 and label == last_label:
            last_idx = idx
            continue
        out.append((idx, label))
        last_idx = idx
        last_label = label
    return out


def _event_eval_sequences(records: Sequence[dict[str, Any]]) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    gold_by_segment: dict[tuple[int, str], tuple[int, str]] = {}
    gold_events: list[tuple[int, str]] = []
    pred_events: list[tuple[int, str]] = []
    for record in records:
        idx = int(record["idx"])
        gold = str(record["gold"])
        pred = str(record["pred"])
        if gold != "listen":
            segment_start = record.get("segment_start")
            segment_text = str(record.get("segment_text") or "")
            if segment_start is not None:
                key = (int(segment_start), segment_text)
                gold_by_segment.setdefault(key, (int(segment_start), gold))
            else:
                gold_events.append((idx, gold))
        if pred != "listen":
            pred_events.append((idx, pred))
    gold_events.extend(gold_by_segment.values())
    return sorted(gold_events), _dedup_consecutive_events(pred_events)


def _trajectory_event_match_counts(
    records: Sequence[dict[str, Any]],
    *,
    window: int,
) -> tuple[int, int, int]:
    gold_events, pred_events = _event_eval_sequences(records)
    return (
        len(gold_events),
        len(pred_events),
        _match_labeled_positions(gold_events, pred_events, window=window),
    )


def _trajectory_event_late_match_counts(
    records: Sequence[dict[str, Any]],
    *,
    late_window: int,
) -> tuple[int, int, int]:
    gold_events, pred_events = _event_eval_sequences(records)
    return (
        len(gold_events),
        len(pred_events),
        _match_labeled_positions_late_only(gold_events, pred_events, late_window=late_window),
    )


def _trajectory_transition_match_counts(
    records: Sequence[tuple[int, str, str]],
    *,
    window: int,
) -> tuple[int, int, int]:
    ordered = sorted(records)
    gold_transitions: list[tuple[int, str]] = []
    pred_transitions: list[tuple[int, str]] = []
    for prev, cur in zip(ordered, ordered[1:]):
        cur_idx = cur[0]
        prev_gold, cur_gold = prev[1], cur[1]
        prev_pred, cur_pred = prev[2], cur[2]
        if cur_gold != prev_gold:
            gold_transitions.append((cur_idx, f"{prev_gold}->{cur_gold}"))
        if cur_pred != prev_pred:
            pred_transitions.append((cur_idx, f"{prev_pred}->{cur_pred}"))
    return (
        len(gold_transitions),
        len(pred_transitions),
        _match_labeled_positions(gold_transitions, pred_transitions, window=window),
    )


def _trajectory_transition_late_match_counts(
    records: Sequence[tuple[int, str, str]],
    *,
    late_window: int,
) -> tuple[int, int, int]:
    ordered = sorted(records)
    gold_transitions: list[tuple[int, str]] = []
    pred_transitions: list[tuple[int, str]] = []
    for prev, cur in zip(ordered, ordered[1:]):
        cur_idx = cur[0]
        prev_gold, cur_gold = prev[1], cur[1]
        prev_pred, cur_pred = prev[2], cur[2]
        if cur_gold != prev_gold:
            gold_transitions.append((cur_idx, f"{prev_gold}->{cur_gold}"))
        if cur_pred != prev_pred:
            pred_transitions.append((cur_idx, f"{prev_pred}->{cur_pred}"))
    return (
        len(gold_transitions),
        len(pred_transitions),
        _match_labeled_positions_late_only(gold_transitions, pred_transitions, late_window=late_window),
    )


def _speak_segment_weight(segment_start: int, segment_end: int, pred_idx: int | None) -> float:
    """Score first predicted speak in the valid start/start+1 window."""
    valid_end = min(segment_start + 1, segment_end)
    if pred_idx is None or pred_idx < segment_start or pred_idx > valid_end:
        return 0.0
    if valid_end <= segment_start:
        return 1.0
    relative = (pred_idx - segment_start) / float(valid_end - segment_start)
    return max(0.5, min(1.0, 1.0 - 0.5 * relative))


def _speak_segment_overlap_counts(
    records: Sequence[tuple[int, str, str]],
    segments: Sequence[dict[str, Any]],
    *,
    boundary_window: int = 1,
) -> dict[str, float]:
    """Score predicted speak boundaries against original speak event spans."""
    pred_speak_turns = sorted(idx for idx, _gold, pred in records if pred == "speak")
    available_turns = {idx for idx, _gold, _pred in records}
    out: dict[str, float] = {
        "total": 0.0,
        "hit": 0.0,
        "boundary_pm1_hit": 0.0,
        "overlap_score_sum": 0.0,
        "iou_score_sum": 0.0,
        "pred_inside_pm1": 0.0,
        "pred_outside_pm1": 0.0,
    }
    if not segments:
        out["pred_outside_pm1"] = float(len(pred_speak_turns))
        return out

    normalized_segments = []
    scored_segments = []
    for segment in segments:
        start = int(segment["start"])
        end = int(segment["end"])
        if end < start:
            start, end = end, start
        normalized = {"start": start, "end": end}
        normalized_segments.append(normalized)
        if start in available_turns and end in available_turns:
            scored_segments.append(normalized)
    out["total"] = float(len(scored_segments))

    for pred_idx in pred_speak_turns:
        inside = any(
            segment["start"] - boundary_window <= pred_idx <= segment["end"] + boundary_window
            for segment in normalized_segments
        )
        out["pred_inside_pm1" if inside else "pred_outside_pm1"] += 1.0

    for segment in scored_segments:
        start = int(segment["start"])
        end = int(segment["end"])
        candidates = [
            pred_idx
            for pred_idx in pred_speak_turns
            if start - boundary_window <= pred_idx <= end + boundary_window
        ]
        if not candidates:
            continue
        pred_start = min(candidates)
        pred_end = max(candidates)
        if pred_start < start - boundary_window or pred_end > end + boundary_window:
            continue
        overlap = max(0, min(pred_end, end) - max(pred_start, start) + 1)
        if overlap <= 0:
            continue
        gold_len = max(1, end - start + 1)
        pred_len = max(1, pred_end - pred_start + 1)
        union = max(1, gold_len + pred_len - overlap)
        out["hit"] += 1.0
        out["overlap_score_sum"] += overlap / float(gold_len)
        out["iou_score_sum"] += overlap / float(union)
        if abs(pred_start - start) <= boundary_window and abs(pred_end - end) <= boundary_window:
            out["boundary_pm1_hit"] += 1.0
    return out


def _trajectory_events_correct(records: Sequence[dict[str, Any]], *, late_window: int = 0) -> bool:
    gold_events, pred_events = _event_eval_sequences(records)
    if len(gold_events) != len(pred_events):
        return False
    if late_window <= 0:
        return _match_labeled_positions(gold_events, pred_events, window=0) == len(gold_events)
    return _match_labeled_positions_late_only(gold_events, pred_events, late_window=late_window) == len(gold_events)


def _trajectory_pm1_correct(records: Sequence[tuple[int, str, str]]) -> bool:
    """Return true when non-listen events match with at most one-turn timing drift."""
    gold_events = sorted((idx, gold) for idx, gold, _pred in records if gold != "listen")
    pred_events = sorted((idx, pred) for idx, _gold, pred in records if pred != "listen")
    if len(gold_events) != len(pred_events):
        return False
    used = [False] * len(pred_events)
    for gold_idx, gold_action in gold_events:
        best_j = -1
        best_dist = 2
        for j, (pred_idx, pred_action) in enumerate(pred_events):
            if used[j] or pred_action != gold_action:
                continue
            dist = abs(pred_idx - gold_idx)
            if dist <= 1 and dist < best_dist:
                best_j = j
                best_dist = dist
        if best_j < 0:
            return False
        used[best_j] = True
    return True


def _trajectory_pm2_correct(records: Sequence[tuple[int, str, str]]) -> bool:
    gold_events = [(idx, gold) for idx, gold, _pred in records if gold != "listen"]
    pred_events = [(idx, pred) for idx, _gold, pred in records if pred != "listen"]
    return len(gold_events) == len(pred_events) and _match_labeled_positions(
        gold_events,
        pred_events,
        window=2,
    ) == len(gold_events)


def _trajectory_late1_correct(records: Sequence[tuple[int, str, str]]) -> bool:
    gold_events = [(idx, gold) for idx, gold, _pred in records if gold != "listen"]
    pred_events = [(idx, pred) for idx, _gold, pred in records if pred != "listen"]
    return len(gold_events) == len(pred_events) and _match_labeled_positions_late_only(
        gold_events,
        pred_events,
        late_window=1,
    ) == len(gold_events)


def _derived_action_metrics(counts: Counter | dict[str, int]) -> dict[str, Any]:
    def c(key: str) -> int:
        return int(counts.get(key, 0))

    metrics: dict[str, Any] = {}
    for action in ("listen", "speak", "delegate"):
        gold = c(f"{action}_total")
        pred = c(f"pred_{action}")
        correct = c(f"{action}_correct")
        precision = _safe_rate(correct, pred)
        recall = _safe_rate(correct, gold)
        metrics[f"{action}_precision"] = precision
        metrics[f"{action}_recall"] = recall
        metrics[f"{action}_f1"] = _f1(precision, recall)
        metrics[f"{action}_count_error"] = pred - gold
        metrics[f"{action}_count_ratio"] = _safe_rate(pred, gold)
        metrics[f"{action}_count_abs_error_rate"] = _safe_rate(abs(pred - gold), gold)

    nonlisten_gold = c("speak_total") + c("delegate_total")
    nonlisten_pred = c("pred_speak") + c("pred_delegate")
    nonlisten_correct = c("speak_correct") + c("delegate_correct")
    nonlisten_precision = _safe_rate(nonlisten_correct, nonlisten_pred)
    nonlisten_recall = _safe_rate(nonlisten_correct, nonlisten_gold)
    metrics.update(
        {
            "nonlisten_gold": nonlisten_gold,
            "nonlisten_pred": nonlisten_pred,
            "nonlisten_correct": nonlisten_correct,
            "nonlisten_precision": nonlisten_precision,
            "nonlisten_recall": nonlisten_recall,
            "nonlisten_f1": _f1(nonlisten_precision, nonlisten_recall),
            "nonlisten_count_error": nonlisten_pred - nonlisten_gold,
            "nonlisten_count_ratio": _safe_rate(nonlisten_pred, nonlisten_gold),
            "nonlisten_count_abs_error_rate": _safe_rate(abs(nonlisten_pred - nonlisten_gold), nonlisten_gold),
        }
    )

    for prefix in ("event", "transition"):
        gold = c(f"{prefix}_gold")
        pred = c(f"{prefix}_pred")
        metrics[f"{prefix}_gold"] = gold
        metrics[f"{prefix}_pred"] = pred
        metrics[f"{prefix}_count_error"] = pred - gold
        metrics[f"{prefix}_count_ratio"] = _safe_rate(pred, gold)
        metrics[f"{prefix}_count_abs_error_rate"] = _safe_rate(abs(pred - gold), gold)
        for suffix in ("exact", "late1"):
            matched = c(f"{prefix}_match_{suffix}")
            precision = _safe_rate(matched, pred)
            recall = _safe_rate(matched, gold)
            metrics[f"{prefix}_{suffix}_match"] = matched
            metrics[f"{prefix}_{suffix}_precision"] = precision
            metrics[f"{prefix}_{suffix}_recall"] = recall
            metrics[f"{prefix}_{suffix}_f1"] = _f1(precision, recall)
    return metrics


@torch.no_grad()
def _evaluate(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    max_eval_batches: int,
    *,
    tokenizer: Any,
    prediction_output: Path | None = None,
    prediction_limit: int = 0,
    eval_name: str = "eval",
    speak_threshold_sweep: Sequence[float] = (),
) -> dict[str, Any]:
    model.eval()
    loss_sum = torch.zeros(1, device=device)
    batch_count = torch.zeros(1, device=device)
    action_counts = Counter()
    sweep_thresholds = list(speak_threshold_sweep)
    sweep_counts: dict[float, Counter] = {threshold: Counter() for threshold in sweep_thresholds}
    task_counts: dict[str, Counter] = {}
    local_predictions_written = 0
    pred_handle = None
    local_prediction_output = None
    if prediction_output is not None:
        prediction_output.parent.mkdir(parents=True, exist_ok=True)
        if _is_rank0():
            local_prediction_output = prediction_output
        else:
            local_prediction_output = prediction_output.with_name(
                f"{prediction_output.stem}.rank{_rank():03d}{prediction_output.suffix}"
            )
        pred_handle = local_prediction_output.open("w", encoding="utf-8")

    listen_ids = _marker_ids(tokenizer, "<|listen|>")
    speak_ids = _marker_ids(tokenizer, "<|speak|>")
    delegate_ids = [int(x) for x in tokenizer.encode("<delegate", add_special_tokens=False)]
    for batch_idx, batch in enumerate(loader, start=1):
        labels = batch.pop("labels")
        loss_weights = batch.pop("loss_weights", None)
        action_eval_records = batch.pop("action_eval_records", [])
        batch = _move_to_device(batch, device)
        labels = labels.to(device)
        if loss_weights is not None:
            loss_weights = loss_weights.to(device)
        outputs = model(data=batch, use_cache=False)
        loss = _weighted_ce_loss(outputs.logits, labels, loss_weights)
        loss_sum += loss.detach()
        batch_count += 1
        trajectory_records: dict[str, list[dict[str, Any]]] = {}
        trajectory_segment_records: dict[str, list[tuple[int, str, str]]] = {}
        trajectory_truncated: dict[str, bool] = {}
        trajectory_seen: set[str] = set()
        trajectory_task: dict[str, str] = {}
        trajectory_speak_segments: dict[str, dict[tuple[int, int, str], dict[str, Any]]] = {}
        for record in action_eval_records:
            gold_action = str(record.get("gold_action") or "none")
            if gold_action not in {"listen", "speak", "delegate"}:
                continue
            task_type = str(record.get("task_type") or "unknown")
            task_counter = task_counts.setdefault(task_type, Counter())
            row_idx = int(record.get("row_idx", -1))
            label_pos = int(record.get("label_pos", -1))
            if row_idx < 0 or row_idx >= outputs.logits.shape[0]:
                continue
            listen_score = _score_marker(outputs.logits, row_idx, label_pos, listen_ids)
            speak_score = _score_marker(outputs.logits, row_idx, label_pos, speak_ids)
            if listen_score == float("-inf") and speak_score == float("-inf"):
                continue
            pred_action = "listen" if listen_score >= speak_score else "speak"
            delegate_label_pos = int(record.get("delegate_label_pos", -1))
            delegate_candidate_positions = []
            if delegate_label_pos >= 0:
                delegate_candidate_positions.append(delegate_label_pos)
            else:
                delegate_candidate_positions.extend([label_pos + len(speak_ids), label_pos + len(speak_ids) + 1])
            delegate_predicted = pred_action == "speak" and any(
                _marker_is_argmax(outputs.logits, row_idx, pos, delegate_ids)
                for pos in delegate_candidate_positions
            )
            if delegate_predicted:
                pred_action = "delegate"
            margin = float(speak_score) - float(listen_score)
            for threshold, threshold_counter in sweep_counts.items():
                threshold_pred = "speak" if margin >= threshold else "listen"
                if threshold_pred == "speak" and delegate_predicted:
                    threshold_pred = "delegate"
                threshold_counter["total"] += 1
                threshold_counter[f"{gold_action}_total"] += 1
                threshold_counter[f"gold_{gold_action}"] += 1
                threshold_counter[f"pred_{threshold_pred}"] += 1
                if threshold_pred == gold_action:
                    threshold_counter["correct"] += 1
                    threshold_counter[f"{gold_action}_correct"] += 1
            sample_id = str(record.get("sample_id") or f"batch{batch_idx}:row{row_idx}")
            if sample_id not in trajectory_seen:
                trajectory_seen.add(sample_id)
                trajectory_records[sample_id] = []
                trajectory_segment_records[sample_id] = []
                trajectory_truncated[sample_id] = False
                trajectory_task[sample_id] = task_type
            turn_index = int(record.get("turn_index", len(trajectory_records[sample_id])) or 0)
            original_turn_index = int(record.get("original_turn_index", turn_index) or 0)
            segment_start_raw = record.get("speak_segment_start")
            segment_end_raw = record.get("speak_segment_end")
            trajectory_records[sample_id].append(
                {
                    "idx": original_turn_index,
                    "local_idx": turn_index,
                    "gold": gold_action,
                    "pred": pred_action,
                    "segment_start": int(segment_start_raw) if segment_start_raw is not None else None,
                    "segment_end": int(segment_end_raw) if segment_end_raw is not None else None,
                    "segment_text": record.get("speak_segment_text"),
                }
            )
            trajectory_segment_records[sample_id].append((original_turn_index, gold_action, pred_action))
            if segment_start_raw is not None and segment_end_raw is not None:
                segment_start = int(segment_start_raw)
                segment_end = int(segment_end_raw)
                segment_text = str(record.get("speak_segment_text") or record.get("gold_text") or "")
                segment_key = (segment_start, segment_end, segment_text)
                segment = trajectory_speak_segments.setdefault(sample_id, {}).setdefault(
                    segment_key,
                    {
                        "start": segment_start,
                        "end": segment_end,
                        "text": segment_text,
                        "best_pred_turn": None,
                    },
                )
                valid_segment_end = min(segment_start + 1, segment_end)
                if pred_action == "speak" and segment_start <= original_turn_index <= valid_segment_end:
                    best_pred_turn = segment.get("best_pred_turn")
                    if best_pred_turn is None or original_turn_index < int(best_pred_turn):
                        segment["best_pred_turn"] = original_turn_index
            is_truncated = bool(record.get("truncated"))
            is_sequence_truncated = bool(record.get("sequence_truncated", is_truncated))
            if is_truncated:
                trajectory_truncated[sample_id] = True
            if is_sequence_truncated:
                trajectory_truncated[sample_id] = True
            action_counts["total"] += 1
            action_counts[f"gold_{gold_action}"] += 1
            action_counts[f"pred_{pred_action}"] += 1
            action_counts[f"{gold_action}_total"] += 1
            task_counter["total"] += 1
            task_counter[f"gold_{gold_action}"] += 1
            task_counter[f"pred_{pred_action}"] += 1
            task_counter[f"{gold_action}_total"] += 1
            if is_truncated:
                action_counts["truncated"] += 1
                task_counter["truncated"] += 1
            if pred_action == gold_action:
                action_counts["correct"] += 1
                action_counts[f"{gold_action}_correct"] += 1
                task_counter["correct"] += 1
                task_counter[f"{gold_action}_correct"] += 1
            if pred_handle is not None and (prediction_limit <= 0 or local_predictions_written < prediction_limit):
                pred_handle.write(
                    json.dumps(
                        {
                            "eval_name": eval_name,
                            "batch_index": batch_idx,
                            "sample_id": record.get("sample_id"),
                            "source": record.get("source"),
                            "task_type": record.get("task_type"),
                            "turn_id": record.get("turn_id"),
                            "turn_index": record.get("turn_index"),
                            "local_turn_index": record.get("local_turn_index", record.get("turn_index")),
                            "original_turn_index": record.get("original_turn_index", record.get("turn_index")),
                            "target_time": record.get("target_time"),
                            "gold_action": gold_action,
                            "pred_action": pred_action,
                            "listen_score": listen_score,
                            "speak_score": speak_score,
                            "delegate_label_pos": delegate_label_pos,
                            "delegate_predicted": delegate_predicted,
                            "gold_text": record.get("gold_text"),
                            "speak_segment_start": record.get("speak_segment_start"),
                            "speak_segment_end": record.get("speak_segment_end"),
                            "speak_segment_text": record.get("speak_segment_text"),
                            "truncated": bool(record.get("truncated")),
                            "sequence_truncated": bool(record.get("sequence_truncated", record.get("truncated"))),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                local_predictions_written += 1
        for sample_id in trajectory_seen:
            exact_correct = _trajectory_events_correct(trajectory_records.get(sample_id, []), late_window=0)
            late1_correct = _trajectory_events_correct(trajectory_records.get(sample_id, []), late_window=1)
            event_gold, event_pred, event_match_exact = _trajectory_event_match_counts(
                trajectory_records.get(sample_id, []),
                window=0,
            )
            _event_gold_late1, _event_pred_late1, event_match_late1 = _trajectory_event_late_match_counts(
                trajectory_records.get(sample_id, []),
                late_window=1,
            )
            transition_gold, transition_pred, transition_match_exact = _trajectory_transition_match_counts(
                trajectory_segment_records.get(sample_id, []),
                window=0,
            )
            _transition_gold_late1, _transition_pred_late1, transition_match_late1 = _trajectory_transition_late_match_counts(
                trajectory_segment_records.get(sample_id, []),
                late_window=1,
            )
            speak_segments = trajectory_speak_segments.get(sample_id, {})
            speak_segment_total = len(speak_segments)
            speak_segment_timing_score_sum = 0.0
            segment_values = list(speak_segments.values())
            for segment in segment_values:
                best_pred_turn = segment.get("best_pred_turn")
                speak_segment_timing_score_sum += _speak_segment_weight(
                    int(segment["start"]),
                    int(segment["end"]),
                    int(best_pred_turn) if best_pred_turn is not None else None,
            )
            action_counts["trajectory_total"] += 1
            action_counts["trajectory_all_correct"] += int(exact_correct)
            action_counts["trajectory_late1_all_correct"] += int(late1_correct)
            action_counts["trajectory_truncated"] += int(trajectory_truncated.get(sample_id, False))
            action_counts["speak_segment_total"] += speak_segment_total
            action_counts["speak_segment_timing_score_sum"] += speak_segment_timing_score_sum
            action_counts["event_gold"] += event_gold
            action_counts["event_pred"] += event_pred
            action_counts["event_match_exact"] += event_match_exact
            action_counts["event_match_late1"] += event_match_late1
            action_counts["transition_gold"] += transition_gold
            action_counts["transition_pred"] += transition_pred
            action_counts["transition_match_exact"] += transition_match_exact
            action_counts["transition_match_late1"] += transition_match_late1
            task_type = trajectory_task.get(sample_id, "unknown")
            task_counter = task_counts.setdefault(task_type, Counter())
            task_counter["trajectory_total"] += 1
            task_counter["trajectory_all_correct"] += int(exact_correct)
            task_counter["trajectory_late1_all_correct"] += int(late1_correct)
            task_counter["trajectory_truncated"] += int(trajectory_truncated.get(sample_id, False))
            task_counter["speak_segment_total"] += speak_segment_total
            task_counter["speak_segment_timing_score_sum"] += speak_segment_timing_score_sum
            task_counter["event_gold"] += event_gold
            task_counter["event_pred"] += event_pred
            task_counter["event_match_exact"] += event_match_exact
            task_counter["event_match_late1"] += event_match_late1
            task_counter["transition_gold"] += transition_gold
            task_counter["transition_pred"] += transition_pred
            task_counter["transition_match_exact"] += transition_match_exact
            task_counter["transition_match_late1"] += transition_match_late1
        if max_eval_batches > 0 and batch_idx >= max_eval_batches:
            break
    if pred_handle is not None:
        pred_handle.close()
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
    count_keys = [
        "total",
        "correct",
        "listen_total",
        "listen_correct",
        "speak_total",
        "speak_correct",
        "delegate_total",
        "delegate_correct",
        "gold_listen",
        "gold_speak",
        "gold_delegate",
        "pred_listen",
        "pred_speak",
        "pred_delegate",
        "truncated",
        "trajectory_total",
        "trajectory_all_correct",
        "trajectory_late1_all_correct",
        "trajectory_truncated",
        "speak_segment_total",
        "event_gold",
        "event_pred",
        "event_match_exact",
        "event_match_late1",
        "transition_gold",
        "transition_pred",
        "transition_match_exact",
        "transition_match_late1",
    ]
    count_tensor = torch.tensor([float(action_counts[key]) for key in count_keys], device=device)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    reduced_counts = {key: int(count_tensor[idx].item()) for idx, key in enumerate(count_keys)}
    float_count_keys = [
        "speak_segment_timing_score_sum",
    ]
    float_count_tensor = torch.tensor([float(action_counts[key]) for key in float_count_keys], device=device)
    dist.all_reduce(float_count_tensor, op=dist.ReduceOp.SUM)
    reduced_float_counts = {
        key: float(float_count_tensor[idx].item()) for idx, key in enumerate(float_count_keys)
    }
    sweep_metrics: dict[str, dict[str, Any]] = {}
    best_sweep: dict[str, Any] = {}
    if sweep_thresholds:
        sweep_count_keys = [
            "total",
            "correct",
            "listen_total",
            "listen_correct",
            "speak_total",
            "speak_correct",
            "delegate_total",
            "delegate_correct",
            "gold_listen",
            "gold_speak",
            "gold_delegate",
            "pred_listen",
            "pred_speak",
            "pred_delegate",
        ]
        flat_sweep_counts = torch.tensor(
            [
                float(sweep_counts[threshold][key])
                for threshold in sweep_thresholds
                for key in sweep_count_keys
            ],
            device=device,
        )
        dist.all_reduce(flat_sweep_counts, op=dist.ReduceOp.SUM)
        for threshold_idx, threshold in enumerate(sweep_thresholds):
            offset = threshold_idx * len(sweep_count_keys)
            reduced_sweep = {
                key: int(flat_sweep_counts[offset + key_idx].item())
                for key_idx, key in enumerate(sweep_count_keys)
            }
            threshold_metrics = {
                "threshold": threshold,
                "action_accuracy": _safe_rate(reduced_sweep["correct"], reduced_sweep["total"]),
                "gold_speak": reduced_sweep["gold_speak"],
                "pred_speak": reduced_sweep["pred_speak"],
                "speak_accuracy": _safe_rate(reduced_sweep["speak_correct"], reduced_sweep["speak_total"]),
                **_derived_action_metrics(reduced_sweep),
            }
            sweep_metrics[f"{threshold:g}"] = threshold_metrics
            if not best_sweep or threshold_metrics["speak_f1"] > best_sweep.get("speak_f1", -1.0):
                best_sweep = threshold_metrics
    gathered_task_counts: list[dict[str, dict[str, int]] | None] = [None for _ in range(_world_size())]
    dist.all_gather_object(gathered_task_counts, {task: dict(counts) for task, counts in task_counts.items()})
    reduced_task_counts: dict[str, Counter] = {}
    for rank_payload in gathered_task_counts:
        if not rank_payload:
            continue
        for task, counts in rank_payload.items():
            reduced_task_counts.setdefault(task, Counter()).update(counts)
    task_metrics = {
        task: {
            "action_total": int(counts["total"]),
            "action_correct": int(counts["correct"]),
            "action_accuracy": _safe_rate(int(counts["correct"]), int(counts["total"])),
            "listen_total": int(counts["listen_total"]),
            "listen_correct": int(counts["listen_correct"]),
            "listen_accuracy": _safe_rate(int(counts["listen_correct"]), int(counts["listen_total"])),
            "speak_total": int(counts["speak_total"]),
            "speak_correct": int(counts["speak_correct"]),
            "speak_accuracy": _safe_rate(int(counts["speak_correct"]), int(counts["speak_total"])),
            "delegate_total": int(counts["delegate_total"]),
            "delegate_correct": int(counts["delegate_correct"]),
            "delegate_accuracy": _safe_rate(int(counts["delegate_correct"]), int(counts["delegate_total"])),
            "gold_listen": int(counts["gold_listen"]),
            "gold_speak": int(counts["gold_speak"]),
            "gold_delegate": int(counts["gold_delegate"]),
            "pred_listen": int(counts["pred_listen"]),
            "pred_speak": int(counts["pred_speak"]),
            "pred_delegate": int(counts["pred_delegate"]),
            "truncated_total": int(counts["truncated"]),
            "truncation_rate": _safe_rate(int(counts["truncated"]), int(counts["total"])),
            "trajectory_total": int(counts["trajectory_total"]),
            "trajectory_all_correct": int(counts["trajectory_all_correct"]),
            "trajectory_late1_all_correct": int(counts["trajectory_late1_all_correct"]),
            "trajectory_truncated": int(counts["trajectory_truncated"]),
            "trajectory_truncation_rate": _safe_rate(
                int(counts["trajectory_truncated"]),
                int(counts["trajectory_total"]),
            ),
            "trajectory_accuracy": _safe_rate(
                int(counts["trajectory_all_correct"]),
                int(counts["trajectory_total"]),
            ),
            "trajectory_late1_accuracy": _safe_rate(
                int(counts["trajectory_late1_all_correct"]),
                int(counts["trajectory_total"]),
            ),
            "speak_segment_total": int(counts["speak_segment_total"]),
            "speak_segment_timing_score": _safe_rate(
                float(counts["speak_segment_timing_score_sum"]),
                int(counts["speak_segment_total"]),
            ),
            **_derived_action_metrics(counts),
        }
        for task, counts in sorted(reduced_task_counts.items())
    }
    total_actions = max(1, reduced_counts["total"])
    listen_total = max(1, reduced_counts["listen_total"])
    speak_total = max(1, reduced_counts["speak_total"])
    delegate_total = max(1, reduced_counts["delegate_total"])
    trajectory_total = max(1, reduced_counts["trajectory_total"])
    speak_segment_total = max(1, reduced_counts["speak_segment_total"])
    derived_metrics = _derived_action_metrics(reduced_counts)
    model.train()
    result = {
        "loss": float((loss_sum / batch_count.clamp_min(1)).cpu()),
        "batches_all_ranks": int(batch_count.item()),
        "action_total": reduced_counts["total"],
        "action_correct": reduced_counts["correct"],
        "action_accuracy": reduced_counts["correct"] / total_actions,
        "listen_total": reduced_counts["listen_total"],
        "listen_correct": reduced_counts["listen_correct"],
        "listen_accuracy": reduced_counts["listen_correct"] / listen_total,
        "speak_total": reduced_counts["speak_total"],
        "speak_correct": reduced_counts["speak_correct"],
        "speak_accuracy": reduced_counts["speak_correct"] / speak_total,
        "delegate_total": reduced_counts["delegate_total"],
        "delegate_correct": reduced_counts["delegate_correct"],
        "delegate_accuracy": reduced_counts["delegate_correct"] / delegate_total,
        "gold_listen": reduced_counts["gold_listen"],
        "gold_speak": reduced_counts["gold_speak"],
        "gold_delegate": reduced_counts["gold_delegate"],
        "pred_listen": reduced_counts["pred_listen"],
        "pred_speak": reduced_counts["pred_speak"],
        "pred_delegate": reduced_counts["pred_delegate"],
        "truncated_total": reduced_counts["truncated"],
        "truncation_rate": reduced_counts["truncated"] / total_actions,
        "trajectory_total": reduced_counts["trajectory_total"],
        "trajectory_all_correct": reduced_counts["trajectory_all_correct"],
        "trajectory_truncated": reduced_counts["trajectory_truncated"],
        "trajectory_truncation_rate": reduced_counts["trajectory_truncated"] / trajectory_total,
        "trajectory_accuracy": reduced_counts["trajectory_all_correct"] / trajectory_total,
        "trajectory_late1_all_correct": reduced_counts["trajectory_late1_all_correct"],
        "trajectory_late1_accuracy": reduced_counts["trajectory_late1_all_correct"] / trajectory_total,
        "speak_segment_total": reduced_counts["speak_segment_total"],
        "speak_segment_timing_score_sum": reduced_float_counts["speak_segment_timing_score_sum"],
        "speak_segment_timing_score": reduced_float_counts["speak_segment_timing_score_sum"] / speak_segment_total,
        **derived_metrics,
        "task_metrics": task_metrics,
        "trajectory_accuracy_by_task": {
            task: metrics["trajectory_accuracy"] for task, metrics in task_metrics.items()
        },
        "trajectory_late1_accuracy_by_task": {
            task: metrics["trajectory_late1_accuracy"] for task, metrics in task_metrics.items()
        },
        "speak_segment_timing_score_by_task": {
            task: metrics["speak_segment_timing_score"] for task, metrics in task_metrics.items()
        },
        "prediction_output": str(prediction_output) if prediction_output is not None and _is_rank0() else None,
        "prediction_output_pattern": (
            str(prediction_output.with_name(f"{prediction_output.stem}.rank*.jsonl"))
            if prediction_output is not None and _is_rank0()
            else None
        ),
        "prediction_records_written_local": local_predictions_written,
    }
    if sweep_metrics:
        result["speak_threshold_sweep"] = sweep_metrics
        result["speak_threshold_best"] = best_sweep.get("threshold")
        result["speak_threshold_best_f1"] = best_sweep.get("speak_f1")
        result["speak_threshold_best_precision"] = best_sweep.get("speak_precision")
        result["speak_threshold_best_recall"] = best_sweep.get("speak_recall")
        result["speak_threshold_best_pred_speak"] = best_sweep.get("pred_speak")
    return result


def _save_checkpoint(
    *,
    model: Any,
    processor: Any,
    tokenizer: Any,
    output_dir: Path,
    epoch: int,
    state: dict[str, Any],
    logger: Any,
    source_model_path: str | os.PathLike[str],
    checkpoint_name: str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = model.state_dict()
    optimizer_state = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True) if optimizer is not None else None
    ckpt_dir = output_dir / (checkpoint_name or f"epoch_{epoch:03d}")
    resume_dir = ckpt_dir / "resume"
    if _is_rank0():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Keep the checkpoint root independently loadable. Everything that is
        # useful only for continuing training lives under resume/ and can be
        # removed as one unit.
        if resume_dir.exists():
            shutil.rmtree(resume_dir)
        for legacy_name in ("trainer_state.json", "optimizer.pt", "checkpoint_complete"):
            (ckpt_dir / legacy_name).unlink(missing_ok=True)
        for legacy_rng_path in ckpt_dir.glob("rng_state_rank*.pt"):
            legacy_rng_path.unlink()
        if optimizer is not None:
            resume_dir.mkdir(parents=True, exist_ok=True)
        model.module.save_pretrained(ckpt_dir, state_dict=state_dict)
        processor.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        source_dir = Path(source_model_path)
        for filename in ("modeling_minicpmo.py", "utils.py"):
            source_file = source_dir / filename
            if source_file.exists():
                shutil.copy2(source_file, ckpt_dir / filename)
        if optimizer_state is not None:
            with (resume_dir / "trainer_state.json").open("w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False, indent=2, sort_keys=True)
            torch.save(optimizer_state, resume_dir / "optimizer.pt")
    dist.barrier()
    if optimizer is not None:
        rng_state = {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(torch.cuda.current_device()),
        }
        torch.save(rng_state, resume_dir / f"rng_state_rank{_rank():05d}.pt")
    dist.barrier()
    if _is_rank0() and optimizer is not None:
        (resume_dir / "checkpoint_complete").write_text("ok\n", encoding="utf-8")
    if _is_rank0():
        logger.info(
            "checkpoint_saved epoch=%d step=%s path=%s optimizer=%s",
            epoch,
            state.get("global_step_rank_local"),
            ckpt_dir,
            optimizer is not None,
        )
    dist.barrier()


def _latest_resume_checkpoint(checkpoint_root: Path) -> Path | None:
    """Return the highest complete ckpt-N containing all resume artifacts."""
    if not checkpoint_root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.iterdir():
        match = re.fullmatch(r"ckpt-(\d+)", path.name)
        if not path.is_dir() or match is None:
            continue
        resume_dir = path / "resume"
        required = (
            resume_dir / "checkpoint_complete",
            resume_dir / "trainer_state.json",
            resume_dir / "optimizer.pt",
        )
        if all(item.is_file() for item in required):
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _load_resume_state(checkpoint_dir: Path, device: torch.device) -> dict[str, Any]:
    resume_dir = checkpoint_dir / "resume"
    state_path = resume_dir / "trainer_state.json"
    with state_path.open("r", encoding="utf-8") as fh:
        state = json.load(fh)
    rng_path = resume_dir / f"rng_state_rank{_rank():05d}.pt"
    state["_rng_path"] = str(rng_path)
    return state


def _resume_config(args: argparse.Namespace) -> dict[str, Any]:
    """Configuration fields that must stay fixed for an exact mid-epoch resume."""
    keys = (
        "model",
        "train_data",
        "input_schema",
        "batch_size",
        "grad_accum_steps",
        "learning_rate",
        "weight_decay",
        "max_train_batches",
        "max_length",
        "num_frames",
        "window_seconds",
        "max_slice_nums",
        "max_image_pixels",
        "force_image_size",
        "image_scale_resolution",
        "listen_weight",
        "speak_weight",
        "speak_boundary_weight",
        "delegate_weight",
        "seed",
        "trajectory_mode",
        "generated_trajectory_mode",
        "trajectory_max_turns",
        "trajectory_chunk_stride",
        "trajectory_drop_single_turn",
        "generated_max_images_per_turn",
        "generated_image_selection",
        "train_exclude_task_types",
        "train_speak_sampling_ratio",
        "train_balance_speak_listen_keep_delegate",
        "train_listen_to_speak_ratio_keep_delegate",
        "train_drop_chat_all_silence_chunks",
        "train_require_prior_instruction_for_action_chunks",
        "collapse_repeated_speak_segments",
        "drop_placeholder_speak_chunks",
        "clean_event_grounding_templates",
    )
    return {key: getattr(args, key) for key in keys}


def _validate_resume_state(state: dict[str, Any], args: argparse.Namespace, train_batches: int) -> None:
    expected = {
        "world_size": _world_size(),
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "max_length": args.max_length,
        "input_schema": args.input_schema,
        "batches_per_epoch": train_batches,
        "train_data": str(args.train_data),
        "model": str(args.model),
        "seed": args.seed,
    }
    mismatches = []
    for key, current in expected.items():
        if key in state and state[key] != current:
            mismatches.append(f"{key}: checkpoint={state[key]!r} current={current!r}")
    saved_config = state.get("resume_config")
    current_config = _resume_config(args)
    if isinstance(saved_config, dict):
        for key, current in current_config.items():
            if saved_config.get(key) != current:
                mismatches.append(f"{key}: checkpoint={saved_config.get(key)!r} current={current!r}")
    if mismatches:
        raise ValueError("resume configuration mismatch: " + "; ".join(mismatches))


def _restore_optimizer_and_rng(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    state: dict[str, Any],
    device: torch.device,
) -> None:
    if not Path(state["_rng_path"]).is_file():
        raise FileNotFoundError(f"missing per-rank RNG state: {state['_rng_path']}")
    full_optimizer_state = torch.load(checkpoint_dir / "resume" / "optimizer.pt", map_location="cpu", weights_only=False) if _is_rank0() else None
    sharded_optimizer_state = FSDP.scatter_full_optim_state_dict(full_optimizer_state, model, optim=optimizer)
    optimizer.load_state_dict(sharded_optimizer_state)
    rng_state = torch.load(state["_rng_path"], map_location="cpu", weights_only=False)
    random.setstate(rng_state["python"])
    torch.set_rng_state(rng_state["torch_cpu"])
    torch.cuda.set_rng_state(rng_state["torch_cuda"], device=device)


def _prune_intermediate_checkpoints(checkpoint_root: Path, max_ckpt_limit: int, logger: Any) -> None:
    """Keep only the newest numbered ckpt-* intermediate checkpoints."""
    if max_ckpt_limit <= 0:
        return
    checkpoint_dirs: list[tuple[int, Path]] = []
    for path in checkpoint_root.iterdir():
        match = re.fullmatch(r"ckpt-(\d+)", path.name)
        if path.is_dir() and match:
            checkpoint_dirs.append((int(match.group(1)), path))
    checkpoint_dirs.sort(key=lambda item: item[0])
    for _, path in checkpoint_dirs[:-max_ckpt_limit]:
        shutil.rmtree(path)
        logger.info("checkpoint_pruned path=%s max_ckpt_limit=%d", path, max_ckpt_limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="construct_demo/models/openbmb/MiniCPM-o-4_5")
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--processor-model", default=None)
    parser.add_argument("--train-data", default="datasets/minicpmo_sft/with_frames_heldout/train.jsonl")
    parser.add_argument("--eval-data", default="datasets/minicpmo_sft/with_frames_heldout/heldout.jsonl")
    parser.add_argument("--output-dir", default="outputs/minicpmo_text_policy_8gpu_16k_10epoch")
    parser.add_argument("--log-dir", default="datasets/minicpmo_sft/logs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument("--max-image-pixels", type=int, default=0)
    parser.add_argument("--force-image-size", type=int, default=0, help="Pad-resize every image to this square size when > 0.")
    parser.add_argument("--image-scale-resolution", type=int, default=0, help="Override MiniCPM-o image_processor.scale_resolution when > 0.")
    parser.add_argument("--vision-batch-size", type=int, default=0, help="Override MiniCPM-o config.vision_batch_size when > 0.")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="",
        help="Override the Transformers attention implementation used when loading MiniCPM-o.",
    )
    parser.add_argument("--listen-weight", type=float, default=0.4)
    parser.add_argument("--speak-weight", type=float, default=2.0)
    parser.add_argument(
        "--speak-boundary-weight",
        type=float,
        default=0.0,
        help="When >0, use this token loss weight for speak labels at event segment boundaries.",
    )
    parser.add_argument("--delegate-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--log-every", type=int, default=50, help="Log training progress every N optimizer steps.")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--no-eval", action="store_true", help="Disable initial, periodic, and end-of-epoch evaluation.")
    parser.add_argument("--eval-only", action="store_true", help="Run one evaluation pass and exit without training or saving a checkpoint.")
    parser.add_argument("--eval-every-steps", type=int, default=0, help="Run action-accuracy eval every N optimizer steps when > 0.")
    parser.add_argument("--eval-before-train", action="store_true", help="Run one eval pass before the first training batch.")
    parser.add_argument("--eval-save-predictions-limit", type=int, default=0, help="Max rank0 eval predictions to save per eval; <=0 saves all rank0 predictions.")
    parser.add_argument(
        "--speak-threshold-sweep",
        default="",
        help="Comma-separated speak_score-listen_score thresholds to report during eval.",
    )
    parser.add_argument("--save-every-steps", type=int, default=0, help="Save checkpoints every N optimizer steps when > 0.")
    parser.add_argument(
        "--max-ckpt-limit",
        type=int,
        default=3,
        help="Keep at most this many ckpt-* intermediate checkpoints; 0 keeps all. The final checkpoint is never pruned.",
    )
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trajectory-mode", action="store_true", help="Group point samples from the same event into multi-turn trajectories.")
    parser.add_argument("--generated-trajectory-mode", action="store_true", help="Load generated messages/images rows as multi-turn trajectories.")
    parser.add_argument(
        "--input-schema",
        choices=("chatml", "duplex"),
        default="chatml",
        help="Serialize trajectory training inputs as ChatML messages or MiniCPMO duplex <unit> streams.",
    )
    parser.add_argument("--trajectory-max-turns", type=int, default=0, help="Keep only the last N turns in each trajectory; 0 keeps all turns.")
    parser.add_argument("--trajectory-chunk-stride", type=int, default=0, help="Stride for generated trajectory chunks when loading unsplit generated data.")
    parser.add_argument("--eval-trajectory-max-turns", type=int, default=-1, help="Eval-only trajectory max turns; <0 uses --trajectory-max-turns.")
    parser.add_argument("--eval-trajectory-chunk-stride", type=int, default=-1, help="Eval-only generated trajectory chunk stride; <0 uses --trajectory-chunk-stride.")
    parser.add_argument("--trajectory-drop-single-turn", action="store_true", help="Drop singleton trajectories when --trajectory-mode is enabled.")
    parser.add_argument("--generated-max-images-per-turn", type=int, default=0, help="Keep at most N generated <image> frames per user turn; 0 keeps all.")
    parser.add_argument("--generated-image-selection", choices=("first", "last", "uniform"), default="last")
    parser.add_argument("--train-exclude-task-types", default="", help="Comma-separated task_type values to exclude from the train dataset.")
    parser.add_argument("--eval-exclude-task-types", default="", help="Comma-separated task_type values to exclude from the eval dataset.")
    parser.add_argument(
        "--train-speak-sampling-ratio",
        type=float,
        default=-1.0,
        help="When >=0 in generated trajectory mode, sample this fraction of train chunks from chunks containing speak/delegate.",
    )
    parser.add_argument(
        "--train-balance-speak-listen-keep-delegate",
        action="store_true",
        help="In generated trajectory mode, keep all delegate chunks and sample equal numbers of speak/listen chunks per epoch.",
    )
    parser.add_argument(
        "--train-listen-to-speak-ratio-keep-delegate",
        type=float,
        default=1.0,
        help="With --train-balance-speak-listen-keep-delegate, sample this many listen chunks per speak chunk.",
    )
    parser.add_argument(
        "--train-drop-chat-all-silence-chunks",
        action="store_true",
        help="In generated trajectory mode, drop train chat chunks that contain only listen actions.",
    )
    parser.add_argument(
        "--train-require-prior-instruction-for-action-chunks",
        action="store_true",
        help="In generated trajectory mode, drop train chunks where speak/delegate appears before any prior user instruction.",
    )
    parser.add_argument(
        "--collapse-repeated-speak-segments",
        action="store_true",
        help="In generated trajectory mode, keep only the first turn of consecutive identical speak responses; later repeated turns become listen.",
    )
    parser.add_argument(
        "--drop-placeholder-speak-chunks",
        action="store_true",
        help="In generated trajectory mode, drop chunks containing placeholder speak text such as [phrase] or [xxx].",
    )
    parser.add_argument(
        "--clean-event-grounding-templates",
        action="store_true",
        help="In generated trajectory mode, rewrite cleanable event-grounding template speaks like Report: ... is still happening.",
    )
    args = parser.parse_args()
    if args.max_ckpt_limit < 0:
        parser.error("--max-ckpt-limit must be >= 0")
    if args.eval_only and args.no_eval:
        parser.error("--eval-only cannot be combined with --no-eval")

    _set_default_hf_cache()
    device = _setup_dist()
    random.seed(args.seed + _rank())
    torch.manual_seed(args.seed + _rank())
    torch.cuda.reset_peak_memory_stats(device)
    speak_threshold_sweep = _parse_float_list(args.speak_threshold_sweep)

    output_dir = Path(args.output_dir)
    resume_payload: list[str | None] = [None]
    if _is_rank0() and not args.eval_only:
        latest_checkpoint = _latest_resume_checkpoint(output_dir / "ckpt")
        resume_payload[0] = str(latest_checkpoint) if latest_checkpoint is not None else None
    dist.broadcast_object_list(resume_payload, src=0)
    resume_checkpoint = Path(resume_payload[0]) if resume_payload[0] is not None else None
    resume_state = _load_resume_state(resume_checkpoint, device) if resume_checkpoint is not None else None
    if _is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "args.json").open("w", encoding="utf-8") as fh:
            json.dump(vars(args), fh, ensure_ascii=False, indent=2, sort_keys=True)
    dist.barrier()

    logger = setup_logging(log_dir=args.log_dir, name="text_policy_fsdp")
    metrics = JsonlMetricLogger(output_dir / "metrics.jsonl")
    logger.info("hf_home=%s", os.environ.get("HF_HOME"))
    logger.info("model=%s", args.model)
    logger.info("tokenizer_model=%s processor_model=%s", args.tokenizer_model or args.model, args.processor_model or args.model)
    logger.info("no_lora=true init_tts=false trainable_scope=llm fsdp=true world_size=%d max_length=%d", _world_size(), args.max_length)
    logger.info("train_data=%s eval_data=%s output_dir=%s", args.train_data, args.eval_data, output_dir)
    logger.info("eval_enabled=%s", not args.no_eval)
    logger.info("auto_resume checkpoint=%s", resume_checkpoint or "none")
    logger.info("torch=%s device=%s dtype=%s", torch.__version__, device, torch.bfloat16)

    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    model_load_path = resume_checkpoint or Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model or model_load_path, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.processor_model or model_load_path, trust_remote_code=True, local_files_only=True)
    if not hasattr(processor, "tokenizer") or processor.tokenizer is None:
        processor.tokenizer = tokenizer
    if args.image_scale_resolution > 0 and hasattr(processor, "image_processor"):
        processor.image_processor.scale_resolution = args.image_scale_resolution
        logger.info("image_scale_resolution=%d", args.image_scale_resolution)

    if args.generated_trajectory_mode and args.trajectory_mode:
        raise ValueError("Use only one of --generated-trajectory-mode or --trajectory-mode")
    if args.input_schema == "duplex" and not (args.generated_trajectory_mode or args.trajectory_mode):
        raise ValueError("--input-schema duplex requires --generated-trajectory-mode or --trajectory-mode")

    if args.generated_trajectory_mode:
        dataset_cls = MiniCPMOGeneratedTrajectoryDataset
    elif args.trajectory_mode:
        dataset_cls = MiniCPMOTrajectoryDataset
    else:
        dataset_cls = MiniCPMOInteractionDataset
    eval_trajectory_max_turns = (
        args.trajectory_max_turns if args.eval_trajectory_max_turns < 0 else args.eval_trajectory_max_turns
    )
    eval_trajectory_chunk_stride = (
        args.trajectory_chunk_stride if args.eval_trajectory_chunk_stride < 0 else args.eval_trajectory_chunk_stride
    )

    dataset_kwargs_base = {
        "num_frames": args.num_frames,
        "window_seconds": args.window_seconds,
        "strict_media": True,
    }
    train_dataset_kwargs = dict(dataset_kwargs_base)
    eval_dataset_kwargs = dict(dataset_kwargs_base)
    if args.generated_trajectory_mode:
        train_dataset_kwargs.update(
            {
                "max_turns": args.trajectory_max_turns,
                "chunk_stride": args.trajectory_chunk_stride,
                "max_images_per_turn": args.generated_max_images_per_turn,
                "image_selection": args.generated_image_selection,
                "drop_chat_all_silence_chunks": args.train_drop_chat_all_silence_chunks,
                "require_prior_instruction_for_action_chunks": args.train_require_prior_instruction_for_action_chunks,
                "collapse_repeated_speak_segments": args.collapse_repeated_speak_segments,
                "drop_placeholder_speak_chunks": args.drop_placeholder_speak_chunks,
                "clean_event_grounding_templates": args.clean_event_grounding_templates,
                "exclude_task_types": args.train_exclude_task_types,
            }
        )
        eval_dataset_kwargs.update(
            {
                "max_turns": eval_trajectory_max_turns,
                "chunk_stride": eval_trajectory_chunk_stride,
                "max_images_per_turn": args.generated_max_images_per_turn,
                "image_selection": args.generated_image_selection,
                "collapse_repeated_speak_segments": args.collapse_repeated_speak_segments,
                "drop_placeholder_speak_chunks": args.drop_placeholder_speak_chunks,
                "clean_event_grounding_templates": args.clean_event_grounding_templates,
                "exclude_task_types": args.eval_exclude_task_types,
            }
        )
    elif args.trajectory_mode:
        train_dataset_kwargs.update(
            {
                "max_turns": args.trajectory_max_turns,
                "include_single_turn": not args.trajectory_drop_single_turn,
            }
        )
        eval_dataset_kwargs.update(
            {
                "max_turns": eval_trajectory_max_turns,
                "include_single_turn": not args.trajectory_drop_single_turn,
            }
        )
    train_dataset = None if args.eval_only else dataset_cls(args.train_data, **train_dataset_kwargs)
    eval_dataset = dataset_cls(args.eval_data, **eval_dataset_kwargs)
    if args.generated_trajectory_mode and _is_rank0() and train_dataset is not None and hasattr(train_dataset, "filter_summary"):
        logger.info("train_generated_chunk_filter_summary=%s", json.dumps(train_dataset.filter_summary, sort_keys=True))
        logger.info(
            "generated_task_filters train_exclude=%s eval_exclude=%s eval_rows_after_filter=%d",
            args.train_exclude_task_types,
            args.eval_exclude_task_types,
            len(eval_dataset),
        )
    train_speak_flags = None
    train_action_labels = None
    if not args.eval_only and args.train_balance_speak_listen_keep_delegate:
        if args.train_speak_sampling_ratio >= 0:
            raise ValueError("Use only one of --train-speak-sampling-ratio or --train-balance-speak-listen-keep-delegate")
        if not args.generated_trajectory_mode or not hasattr(train_dataset, "chunk_action_labels"):
            raise ValueError("--train-balance-speak-listen-keep-delegate currently requires --generated-trajectory-mode")
        train_action_labels = train_dataset.chunk_action_labels()
        delegate_chunks = sum(1 for label in train_action_labels if label == "delegate")
        speak_chunks = sum(1 for label in train_action_labels if label == "speak")
        listen_chunks = len(train_action_labels) - delegate_chunks - speak_chunks
        sampled_listen_chunks = min(
            listen_chunks,
            int(round(speak_chunks * max(0.0, args.train_listen_to_speak_ratio_keep_delegate))),
        )
        logger.info(
            "train_balance_speak_listen_keep_delegate=true delegate_chunks=%d speak_chunks=%d listen_chunks=%d listen_to_speak_ratio=%.6f sampled_listen_chunks=%d sampled_epoch_chunks=%d",
            delegate_chunks,
            speak_chunks,
            listen_chunks,
            args.train_listen_to_speak_ratio_keep_delegate,
            sampled_listen_chunks,
            delegate_chunks + speak_chunks + sampled_listen_chunks,
        )
    if not args.eval_only and args.train_speak_sampling_ratio >= 0:
        if not args.generated_trajectory_mode or not hasattr(train_dataset, "speak_chunk_flags"):
            raise ValueError("--train-speak-sampling-ratio currently requires --generated-trajectory-mode")
        train_speak_flags = train_dataset.speak_chunk_flags()
        speak_chunks = sum(1 for flag in train_speak_flags if flag)
        listen_only_chunks = len(train_speak_flags) - speak_chunks
        logger.info(
            "train_speak_sampling_ratio=%.4f speak_chunks=%d listen_only_chunks=%d natural_speak_chunk_ratio=%.6f",
            args.train_speak_sampling_ratio,
            speak_chunks,
            listen_only_chunks,
            speak_chunks / max(1, len(train_speak_flags)),
        )
    collator_cls = MiniCPMODuplexTrajectoryCollator if args.input_schema == "duplex" else MiniCPMODataCollator
    collator = collator_cls(
        processor=processor,
        max_length=args.max_length,
        max_slice_nums=args.max_slice_nums,
        max_image_pixels=args.max_image_pixels,
        force_image_size=args.force_image_size,
        listen_weight=args.listen_weight,
        speak_weight=args.speak_weight,
        speak_boundary_weight=args.speak_boundary_weight,
        delegate_weight=args.delegate_weight,
    )
    eval_collator = collator_cls(
        processor=processor,
        max_length=args.max_length,
        max_slice_nums=args.max_slice_nums,
        max_image_pixels=args.max_image_pixels,
        force_image_size=args.force_image_size,
        listen_weight=args.listen_weight,
        speak_weight=args.speak_weight,
        speak_boundary_weight=args.speak_boundary_weight,
        delegate_weight=args.delegate_weight,
    )
    logger.info("input_schema=%s collator=%s", args.input_schema, collator_cls.__name__)

    model_load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "init_vision": True,
        "init_audio": True,
        "init_tts": False,
    }
    if args.attn_implementation:
        model_load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModel.from_pretrained(model_load_path, **model_load_kwargs)
    logger.info(
        "attn_implementation requested=%s model=%s vision=%s llm=%s",
        args.attn_implementation or "auto",
        getattr(model.config, "_attn_implementation", "unknown"),
        getattr(getattr(model.config, "vision_config", None), "_attn_implementation", "unknown"),
        getattr(getattr(model, "llm", None), "config", None)._attn_implementation
        if getattr(getattr(model, "llm", None), "config", None) is not None
        else "unknown",
    )
    if args.vision_batch_size > 0 and hasattr(model, "config"):
        model.config.vision_batch_size = args.vision_batch_size
        logger.info("vision_batch_size=%d", args.vision_batch_size)
    if args.gradient_checkpointing:
        target = model.llm if hasattr(model, "llm") else model
        if hasattr(target, "gradient_checkpointing_enable"):
            target.gradient_checkpointing_enable()
        if hasattr(target, "config"):
            target.config.use_cache = False
        logger.info("gradient_checkpointing=true")
    total, trainable = _freeze_non_llm(model)
    model.to(device)
    model = _wrap_model_fsdp(model, device)
    model.train()
    logger.info(
        "model_loaded total_params=%d trainable_params=%d train_rows=%s eval_rows=%d num_frames=%d eval_only=%s",
        total,
        trainable,
        len(train_dataset) if train_dataset is not None else "skipped",
        len(eval_dataset),
        args.num_frames,
        args.eval_only,
    )
    if args.generated_trajectory_mode:
        logger.info(
            "generated_trajectory_mode=true train_max_turns=%d train_chunk_stride=%d eval_max_turns=%d eval_chunk_stride=%d max_images_per_turn=%d image_selection=%s max_image_pixels=%d force_image_size=%d",
            args.trajectory_max_turns,
            args.trajectory_chunk_stride,
            eval_trajectory_max_turns,
            eval_trajectory_chunk_stride,
            args.generated_max_images_per_turn,
            args.generated_image_selection,
            args.max_image_pixels,
            args.force_image_size,
        )
        logger.info(
            "train_chunk_filters drop_chat_all_silence=%s require_prior_instruction_for_action=%s",
            args.train_drop_chat_all_silence_chunks,
            args.train_require_prior_instruction_for_action_chunks,
        )
        logger.info(
            "generated_label_normalization collapse_repeated_speak_segments=%s drop_placeholder_speak_chunks=%s clean_event_grounding_templates=%s",
            args.collapse_repeated_speak_segments,
            args.drop_placeholder_speak_chunks,
            args.clean_event_grounding_templates,
        )
    elif args.trajectory_mode:
        logger.info(
            "trajectory_mode=true train_max_turns=%d eval_max_turns=%d include_single_turn=%s",
            args.trajectory_max_turns,
            eval_trajectory_max_turns,
            not args.trajectory_drop_single_turn,
        )

    if args.eval_only:
        eval_loader, eval_sampler = _build_loader(
            eval_dataset,
            eval_collator,
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=args.seed,
        )
        eval_sampler.set_epoch(0)
        eval_metrics = _evaluate(
            model,
            eval_loader,
            device,
            args.max_eval_batches,
            tokenizer=tokenizer,
            prediction_output=output_dir / "eval_predictions" / "eval_only.jsonl",
            prediction_limit=args.eval_save_predictions_limit,
            eval_name="eval_only",
            speak_threshold_sweep=speak_threshold_sweep,
        )
        if _is_rank0():
            with (output_dir / "eval_only_metrics.json").open("w", encoding="utf-8") as fh:
                json.dump(eval_metrics, fh, ensure_ascii=False, indent=2, sort_keys=True)
            metrics.log(event="eval_only", step=0, **eval_metrics)
            logger.info(
                "eval_only_complete loss=%.6f action_acc=%.6f trajectory_acc=%.6f trajectory_late1_acc=%.6f "
                "listen_acc=%.6f speak_acc=%.6f delegate_acc=%.6f action_total=%d trajectory_total=%d pred_file=%s metrics_file=%s",
                eval_metrics["loss"],
                eval_metrics["action_accuracy"],
                eval_metrics["trajectory_accuracy"],
                eval_metrics["trajectory_late1_accuracy"],
                eval_metrics["listen_accuracy"],
                eval_metrics["speak_accuracy"],
                eval_metrics["delegate_accuracy"],
                eval_metrics["action_total"],
                eval_metrics["trajectory_total"],
                eval_metrics["prediction_output"],
                output_dir / "eval_only_metrics.json",
            )
        dist.barrier()
        _cleanup_dist()
        return

    assert train_dataset is not None
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    global_step = int(resume_state.get("global_step_rank_local", 0)) if resume_state is not None else 0
    resume_epoch = int(resume_state.get("epoch", 1)) if resume_state is not None else 1
    resume_batch = int(resume_state.get("batch", 0)) if resume_state is not None else 0
    if resume_state is not None and resume_batch >= int(resume_state.get("batches_per_epoch", resume_batch + 1)):
        resume_epoch += 1
        resume_batch = 0
    total_global_steps = 0
    did_initial_eval = resume_state is not None
    training_progress_start: float | None = None
    resume_restored = False
    resume_completed_batches_base = 0

    for epoch in range(resume_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loader, train_sampler = _build_loader(
            train_dataset,
            collator,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
            positive_flags=train_speak_flags,
            positive_ratio=args.train_speak_sampling_ratio,
            action_labels=train_action_labels,
            listen_to_speak_ratio=args.train_listen_to_speak_ratio_keep_delegate,
        )
        eval_loader, eval_sampler = _build_loader(
            eval_dataset,
            eval_collator,
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=args.seed,
        )
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        train_batches_per_epoch = len(train_loader)
        if args.max_train_batches > 0:
            train_batches_per_epoch = min(train_batches_per_epoch, args.max_train_batches)
        if resume_state is not None and not resume_restored:
            _validate_resume_state(resume_state, args, train_batches_per_epoch)
            _restore_optimizer_and_rng(model, optimizer, resume_checkpoint, resume_state, device)
            resume_restored = True
            resume_completed_batches_base = (epoch - 1) * train_batches_per_epoch + resume_batch
            logger.info(
                "resume_loaded checkpoint=%s epoch=%d completed_batch=%d/%d global_step=%d",
                resume_checkpoint,
                epoch,
                resume_batch,
                train_batches_per_epoch,
                global_step,
            )
        if total_global_steps == 0:
            optimizer_steps_per_epoch = train_batches_per_epoch // max(1, args.grad_accum_steps)
            if train_batches_per_epoch == len(train_loader) and train_batches_per_epoch % max(1, args.grad_accum_steps):
                optimizer_steps_per_epoch += 1
            total_global_steps = optimizer_steps_per_epoch * args.epochs
            if _is_rank0():
                logger.info(
                    "training_steps batches_per_epoch=%d optimizer_steps_per_epoch=%d total_global_steps=%d",
                    train_batches_per_epoch,
                    optimizer_steps_per_epoch,
                    total_global_steps,
                )
        if not args.no_eval and args.eval_before_train and not did_initial_eval:
            eval_metrics = _evaluate(
                model,
                eval_loader,
                device,
                args.max_eval_batches,
                tokenizer=tokenizer,
                prediction_output=output_dir / "eval_predictions" / "step_00000000.jsonl",
                prediction_limit=args.eval_save_predictions_limit,
                eval_name="step_0",
                speak_threshold_sweep=speak_threshold_sweep,
            )
            if _is_rank0():
                logger.info(
                    "initial_eval step=0 loss=%.6f action_acc=%.6f trajectory_acc=%.6f trajectory_late1_acc=%.6f listen_acc=%.6f speak_acc=%.6f delegate_acc=%.6f speak_precision=%.6f delegate_precision=%.6f speak_seg_timing=%.6f event_late1_f1=%.6f transition_late1_f1=%.6f trunc_rate=%.6f traj_trunc_rate=%.6f action_total=%d truncated_total=%d trajectory_total=%d trajectory_truncated=%d listen_total=%d speak_total=%d delegate_total=%d task_traj_acc=%s task_traj_late1_acc=%s pred_file=%s",
                    eval_metrics["loss"],
                    eval_metrics["action_accuracy"],
                    eval_metrics["trajectory_accuracy"],
                    eval_metrics["trajectory_late1_accuracy"],
                    eval_metrics["listen_accuracy"],
                    eval_metrics["speak_accuracy"],
                    eval_metrics["delegate_accuracy"],
                    eval_metrics["speak_precision"],
                    eval_metrics["delegate_precision"],
                    eval_metrics["speak_segment_timing_score"],
                    eval_metrics["event_late1_f1"],
                    eval_metrics["transition_late1_f1"],
                    eval_metrics["truncation_rate"],
                    eval_metrics["trajectory_truncation_rate"],
                    eval_metrics["action_total"],
                    eval_metrics["truncated_total"],
                    eval_metrics["trajectory_total"],
                    eval_metrics["trajectory_truncated"],
                    eval_metrics["listen_total"],
                    eval_metrics["speak_total"],
                    eval_metrics["delegate_total"],
                    json.dumps(eval_metrics.get("trajectory_accuracy_by_task", {}), sort_keys=True),
                    json.dumps(eval_metrics.get("trajectory_late1_accuracy_by_task", {}), sort_keys=True),
                    eval_metrics["prediction_output"],
                )
                metrics.log(event="initial_eval", step=0, **eval_metrics)
            did_initial_eval = True
        if training_progress_start is None:
            training_progress_start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        local_loss_sum = 0.0
        local_batches = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            if epoch == resume_epoch and batch_idx <= resume_batch:
                continue
            labels = batch.pop("labels")
            loss_weights = batch.pop("loss_weights", None)
            batch.pop("action_eval_records", None)
            batch = _move_to_device(batch, device)
            labels = labels.to(device)
            if loss_weights is not None:
                loss_weights = loss_weights.to(device)

            outputs = model(data=batch, use_cache=False)
            loss = _weighted_ce_loss(outputs.logits, labels, loss_weights)
            (loss / max(1, args.grad_accum_steps)).backward()
            local_loss_sum += float(loss.detach().cpu())
            local_batches += 1

            should_step = batch_idx % max(1, args.grad_accum_steps) == 0 or batch_idx == train_batches_per_epoch
            grad_norm = math.nan
            if should_step:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), max_norm=1.0)
                grad_norm = float(grad_norm_tensor.detach().cpu())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if _is_rank0() and should_step and global_step > 0 and global_step % max(1, args.log_every) == 0:
                peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
                completed_train_batches = (epoch - 1) * train_batches_per_epoch + min(
                    batch_idx, train_batches_per_epoch
                )
                total_train_batches = train_batches_per_epoch * args.epochs
                progress_elapsed = time.monotonic() - training_progress_start
                completed_since_resume = completed_train_batches - resume_completed_batches_base
                seconds_per_batch = progress_elapsed / max(1, completed_since_resume)
                eta_seconds = seconds_per_batch * max(0, total_train_batches - completed_train_batches)
                progress_percent = 100.0 * completed_train_batches / max(1, total_train_batches)
                logger.info(
                    "epoch=%d batch=%d/%d global_step=%d/%d progress=%.2f%% elapsed=%s eta=%s sec_per_batch=%.2f local_loss_avg=%.6f last_loss=%.6f grad_norm=%.6f seq_len=%d peak_alloc_gb=%.3f",
                    epoch,
                    batch_idx,
                    train_batches_per_epoch,
                    global_step,
                    total_global_steps,
                    progress_percent,
                    str(timedelta(seconds=int(progress_elapsed))),
                    str(timedelta(seconds=int(eta_seconds))),
                    seconds_per_batch,
                    local_loss_sum / max(1, local_batches),
                    float(loss.detach().cpu()),
                    grad_norm,
                    int(outputs.logits.shape[1]),
                    peak_alloc,
                )
            if not args.no_eval and should_step and args.eval_every_steps > 0 and global_step > 0 and global_step % args.eval_every_steps == 0:
                eval_metrics = _evaluate(
                    model,
                    eval_loader,
                    device,
                    args.max_eval_batches,
                    tokenizer=tokenizer,
                    prediction_output=output_dir / "eval_predictions" / f"step_{global_step:08d}.jsonl",
                    prediction_limit=args.eval_save_predictions_limit,
                    eval_name=f"step_{global_step}",
                    speak_threshold_sweep=speak_threshold_sweep,
                )
                if _is_rank0():
                    logger.info(
                        "periodic_eval step=%d loss=%.6f action_acc=%.6f trajectory_acc=%.6f trajectory_late1_acc=%.6f listen_acc=%.6f speak_acc=%.6f delegate_acc=%.6f speak_precision=%.6f delegate_precision=%.6f speak_seg_timing=%.6f event_late1_f1=%.6f transition_late1_f1=%.6f trunc_rate=%.6f traj_trunc_rate=%.6f action_total=%d truncated_total=%d trajectory_total=%d trajectory_truncated=%d listen_total=%d speak_total=%d delegate_total=%d task_traj_acc=%s task_traj_late1_acc=%s pred_file=%s",
                        global_step,
                        eval_metrics["loss"],
                        eval_metrics["action_accuracy"],
                        eval_metrics["trajectory_accuracy"],
                        eval_metrics["trajectory_late1_accuracy"],
                        eval_metrics["listen_accuracy"],
                        eval_metrics["speak_accuracy"],
                        eval_metrics["delegate_accuracy"],
                        eval_metrics["speak_precision"],
                        eval_metrics["delegate_precision"],
                        eval_metrics["speak_segment_timing_score"],
                        eval_metrics["event_late1_f1"],
                        eval_metrics["transition_late1_f1"],
                        eval_metrics["truncation_rate"],
                        eval_metrics["trajectory_truncation_rate"],
                        eval_metrics["action_total"],
                        eval_metrics["truncated_total"],
                        eval_metrics["trajectory_total"],
                        eval_metrics["trajectory_truncated"],
                        eval_metrics["listen_total"],
                        eval_metrics["speak_total"],
                        eval_metrics["delegate_total"],
                        json.dumps(eval_metrics.get("trajectory_accuracy_by_task", {}), sort_keys=True),
                        json.dumps(eval_metrics.get("trajectory_late1_accuracy_by_task", {}), sort_keys=True),
                        eval_metrics["prediction_output"],
                    )
                    metrics.log(event="periodic_eval", step=global_step, **eval_metrics)
            if should_step and args.save_every_steps > 0 and global_step > 0 and global_step % args.save_every_steps == 0:
                if args.no_save_checkpoints:
                    dist.barrier()
                    if _is_rank0():
                        logger.info("checkpoint_skipped step=%d no_save_checkpoints=true", global_step)
                else:
                    checkpoint_root = output_dir / "ckpt"
                    checkpoint_state = {
                        "epoch": epoch,
                        "epochs": args.epochs,
                        "batch": batch_idx,
                        "batches_per_epoch": train_batches_per_epoch,
                        "global_step_rank_local": global_step,
                        "batch_size": args.batch_size,
                        "grad_accum_steps": args.grad_accum_steps,
                        "train_data": str(args.train_data),
                        "model": str(args.model),
                        "seed": args.seed,
                        "no_lora": True,
                        "init_tts": False,
                        "trainable_scope": "llm",
                        "world_size": _world_size(),
                        "max_length": args.max_length,
                        "input_schema": args.input_schema,
                        "num_frames": args.num_frames,
                        "save_every_steps": args.save_every_steps,
                        "resume_config": _resume_config(args),
                    }
                    _save_checkpoint(
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        output_dir=checkpoint_root,
                        epoch=epoch,
                        state=checkpoint_state,
                        logger=logger,
                        source_model_path=args.model,
                        checkpoint_name=f"ckpt-{global_step}",
                        optimizer=optimizer,
                    )
                    if _is_rank0():
                        checkpoint_path = checkpoint_root / f"ckpt-{global_step}"
                        metrics.log(event="checkpoint_saved", **checkpoint_state, path=str(checkpoint_path))
                        _prune_intermediate_checkpoints(checkpoint_root, args.max_ckpt_limit, logger)
                    dist.barrier()
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

        loss_pair = torch.tensor([local_loss_sum, float(local_batches)], device=device)
        dist.all_reduce(loss_pair, op=dist.ReduceOp.SUM)
        train_loss = float((loss_pair[0] / loss_pair[1].clamp_min(1)).cpu())
        eval_metrics = None
        if not args.no_eval:
            eval_metrics = _evaluate(
                model,
                eval_loader,
                device,
                args.max_eval_batches,
                tokenizer=tokenizer,
                prediction_output=output_dir / "eval_predictions" / f"epoch_{epoch:03d}.jsonl",
                prediction_limit=args.eval_save_predictions_limit,
                eval_name=f"epoch_{epoch}",
                speak_threshold_sweep=speak_threshold_sweep,
            )
        elapsed = time.time() - epoch_start
        peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        state = {
            "epoch": epoch,
            "epochs": args.epochs,
            "global_step_rank_local": global_step,
            "train_loss": train_loss,
            "eval_enabled": not args.no_eval,
            "elapsed_seconds": elapsed,
            "no_lora": True,
            "init_tts": False,
            "trainable_scope": "llm",
            "world_size": _world_size(),
            "max_length": args.max_length,
            "input_schema": args.input_schema,
            "num_frames": args.num_frames,
            "save_every_steps": args.save_every_steps,
        }
        if eval_metrics is not None:
            state.update(
                {
                    "eval_loss": eval_metrics["loss"],
                    "eval_batches_all_ranks": eval_metrics["batches_all_ranks"],
                    "eval_action_accuracy": eval_metrics["action_accuracy"],
                    "eval_action_total": eval_metrics["action_total"],
                    "eval_action_correct": eval_metrics["action_correct"],
                    "eval_truncated_total": eval_metrics["truncated_total"],
                    "eval_truncation_rate": eval_metrics["truncation_rate"],
                    "eval_listen_accuracy": eval_metrics["listen_accuracy"],
                    "eval_listen_total": eval_metrics["listen_total"],
                    "eval_speak_accuracy": eval_metrics["speak_accuracy"],
                    "eval_speak_total": eval_metrics["speak_total"],
                    "eval_delegate_accuracy": eval_metrics["delegate_accuracy"],
                    "eval_delegate_total": eval_metrics["delegate_total"],
                    "eval_trajectory_accuracy": eval_metrics["trajectory_accuracy"],
                    "eval_trajectory_late1_accuracy": eval_metrics["trajectory_late1_accuracy"],
                    "eval_trajectory_total": eval_metrics["trajectory_total"],
                    "eval_trajectory_all_correct": eval_metrics["trajectory_all_correct"],
                    "eval_trajectory_late1_all_correct": eval_metrics["trajectory_late1_all_correct"],
                    "eval_trajectory_truncated": eval_metrics["trajectory_truncated"],
                    "eval_trajectory_truncation_rate": eval_metrics["trajectory_truncation_rate"],
                    "eval_speak_precision": eval_metrics["speak_precision"],
                    "eval_delegate_precision": eval_metrics["delegate_precision"],
                    "eval_speak_segment_timing_score": eval_metrics["speak_segment_timing_score"],
                    "eval_event_late1_f1": eval_metrics["event_late1_f1"],
                    "eval_transition_late1_f1": eval_metrics["transition_late1_f1"],
                    "eval_task_metrics": eval_metrics.get("task_metrics", {}),
                    "eval_trajectory_accuracy_by_task": eval_metrics.get("trajectory_accuracy_by_task", {}),
                    "eval_trajectory_late1_accuracy_by_task": eval_metrics.get("trajectory_late1_accuracy_by_task", {}),
                    "eval_speak_segment_timing_score_by_task": eval_metrics.get("speak_segment_timing_score_by_task", {}),
                }
            )
        if _is_rank0():
            with (output_dir / "last_state.json").open("w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False, indent=2, sort_keys=True)
            if eval_metrics is None:
                logger.info(
                    "epoch_done epoch=%d train_loss=%.6f eval_disabled=true elapsed=%.1fs peak_alloc_gb=%.3f peak_reserved_gb=%.3f",
                    epoch,
                    train_loss,
                    elapsed,
                    peak_alloc,
                    peak_reserved,
                )
            else:
                logger.info(
                    "epoch_done epoch=%d train_loss=%.6f eval_loss=%.6f action_acc=%.6f trajectory_acc=%.6f trajectory_late1_acc=%.6f listen_acc=%.6f speak_acc=%.6f delegate_acc=%.6f speak_precision=%.6f delegate_precision=%.6f speak_seg_timing=%.6f event_late1_f1=%.6f transition_late1_f1=%.6f trunc_rate=%.6f traj_trunc_rate=%.6f task_traj_acc=%s task_traj_late1_acc=%s eval_batches=%d elapsed=%.1fs peak_alloc_gb=%.3f peak_reserved_gb=%.3f",
                    epoch,
                    train_loss,
                    eval_metrics["loss"],
                    eval_metrics["action_accuracy"],
                    eval_metrics["trajectory_accuracy"],
                    eval_metrics["trajectory_late1_accuracy"],
                    eval_metrics["listen_accuracy"],
                    eval_metrics["speak_accuracy"],
                    eval_metrics["delegate_accuracy"],
                    eval_metrics["speak_precision"],
                    eval_metrics["delegate_precision"],
                    eval_metrics["speak_segment_timing_score"],
                    eval_metrics["event_late1_f1"],
                    eval_metrics["transition_late1_f1"],
                    eval_metrics["truncation_rate"],
                    eval_metrics["trajectory_truncation_rate"],
                    json.dumps(eval_metrics.get("trajectory_accuracy_by_task", {}), sort_keys=True),
                    json.dumps(eval_metrics.get("trajectory_late1_accuracy_by_task", {}), sort_keys=True),
                    eval_metrics["batches_all_ranks"],
                    elapsed,
                    peak_alloc,
                    peak_reserved,
                )
            metrics.log(event="epoch_done", **state, peak_alloc_gb=peak_alloc, peak_reserved_gb=peak_reserved)
        if epoch == args.epochs:
            if args.no_save_checkpoints:
                dist.barrier()
                logger.info("final_checkpoint_skipped no_save_checkpoints=true")
            else:
                _save_checkpoint(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    epoch=epoch,
                    state=state,
                    logger=logger,
                    source_model_path=args.model,
                    checkpoint_name="final",
                )

    logger.info("training_complete epochs=%d output_dir=%s", args.epochs, output_dir)
    _cleanup_dist()


if __name__ == "__main__":
    main()
