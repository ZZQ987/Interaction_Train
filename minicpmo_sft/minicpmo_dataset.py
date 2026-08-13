"""Dataset utilities for MiniCPM-o interaction SFT.

The classes here are intentionally independent from the OpenSQZ CookBook
dataset.py because interaction samples are video/timestamp based, while
the CookBook official dataset loader currently targets image conversations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tarfile
import time
from bisect import bisect_right
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_PATTERN = "<image>./</image>"
AUDIO_PATTERN = "<audio>./</audio>"
IGNORE_INDEX = -100
GENERATED_TIME_LINE_RE = re.compile(r"<\s*[0-9]+(?:\.[0-9]+)?\s*seconds\s*>")
GENERATED_MEDIA_MARKER_RE = re.compile(r"</?\s*(?:image|audio|video)\s*>", re.IGNORECASE)


def open_rgb_image_with_retry(path: str | os.PathLike[str], attempts: int = 3, delay_seconds: float = 0.5) -> Image.Image:
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                time.sleep(delay_seconds * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def resize_image_to_max_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    if max_pixels <= 0:
        return image
    width, height = image.size
    pixels = width * height
    if pixels <= max_pixels or width <= 0 or height <= 0:
        return image
    scale = math.sqrt(max_pixels / float(pixels))
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def resize_image_to_fixed_square(image: Image.Image, size: int) -> Image.Image:
    if size <= 0:
        return image
    if image.size == (size, size):
        return image
    width, height = image.size
    if width <= 0 or height <= 0:
        return image.resize((size, size), Image.Resampling.LANCZOS)
    scale = min(size / float(width), size / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def read_json_or_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    """Read either a JSON array or JSONL annotation file."""
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array or JSONL rows")
    return data


def parse_task_type_set(value: str | Sequence[str] | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = []
        for item in value:
            parts.extend(str(item).split(","))
    return {part.strip() for part in parts if part.strip()}


def discover_annotation_files(path: str | os.PathLike[str]) -> List[Path]:
    """Return JSONL annotation files for either one file or a sharded directory."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.rglob("*.jsonl") if not p.name.endswith(".bad.jsonl"))
    if not files:
        raise FileNotFoundError(f"no jsonl files found under {path}")
    return files


def jsonl_line_offsets(path: Path) -> List[int]:
    offsets: List[int] = []
    with path.open("rb") as fh:
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
    return offsets


def read_jsonl_at_offset(path: str | os.PathLike[str], offset: int) -> Dict[str, Any]:
    path = Path(path)
    with path.open("rb") as fh:
        fh.seek(offset)
        line = fh.readline()
    if not line.strip():
        raise ValueError(f"empty jsonl row at {path}:{offset}")
    try:
        return json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        preview = line[:160].decode("utf-8", errors="replace")
        raise ValueError(f"invalid jsonl row at {path}:{offset}: {preview!r}") from exc


def clean_user_text(text: str) -> str:
    """Remove the annotation-only <video> marker before native image frames are inserted."""
    return re.sub(r"^\s*<video>\s*\n?", "", text, count=1)


def discover_generated_shard_files(path: str | os.PathLike[str]) -> List[Path]:
    """Return completed generated-format JSONL shard files."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.rglob("*.jsonl") if not p.name.endswith(".tmp"))
    if not files:
        raise FileNotFoundError(f"no generated jsonl files found under {path}")
    return files


def discover_generated_event_shard_files(path: str | os.PathLike[str]) -> List[Path]:
    """Return generated-format event_grounding shards without scanning frame trees."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(path.glob("*/shards/event_grounding/*.jsonl"))
    files = [file_path for file_path in files if not file_path.name.endswith(".tmp")]
    if not files:
        raise FileNotFoundError(f"no event_grounding jsonl files found under {path}")
    return files


def generated_manifest_ranges(num_turns: int, max_turns: int, stride: int) -> List[Tuple[int, int]]:
    if num_turns <= 0:
        return []
    if max_turns <= 0 or num_turns <= max_turns:
        return [(0, num_turns)]
    if stride <= 0:
        return [(max(0, num_turns - max_turns), num_turns)]
    ranges = {(start, min(start + max_turns, num_turns)) for start in range(0, num_turns, stride)}
    ranges.add((max(0, num_turns - max_turns), num_turns))
    return sorted(ranges)


def normalize_generated_assistant_content(text: str) -> str:
    """Map generated labels onto the interaction action tokens."""
    stripped = text.strip()
    if stripped.startswith("</silence>"):
        tail = strip_generated_media_markers(stripped[len("</silence>") :]).lstrip()
        return "<|listen|>" + (tail if tail else "")
    if stripped.startswith("</response>"):
        tail = strip_generated_media_markers(stripped[len("</response>") :]).lstrip()
        return "<|speak|>" + (tail if not tail.startswith(" ") else tail)
    if stripped.startswith("</delegation>"):
        tail = strip_generated_media_markers(stripped[len("</delegation>") :]).lstrip()
        return "<|speak|><delegate>" + tail + ("</delegate>" if "</delegate>" not in tail else "")
    return text


def strip_generated_media_markers(text: Any) -> str:
    return GENERATED_MEDIA_MARKER_RE.sub(" ", str(text or ""))


def discover_generated_source_shards(path: str | os.PathLike[str]) -> List[Path]:
    """Return generated shards, skipping ops/canary/tmp-style trees."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)

    files: List[Path] = []
    for dataset_dir in sorted(p for p in path.iterdir() if p.is_dir()):
        name = dataset_dir.name
        if name.startswith("_") or "invalid-canary" in name:
            continue
        shard_dir = dataset_dir / "shards"
        if not shard_dir.is_dir():
            continue
        files.extend(
            sorted(
                p
                for p in shard_dir.rglob("*.jsonl")
                if not p.name.endswith(".tmp") and ".bad" not in p.name
            )
        )
    if not files:
        raise FileNotFoundError(f"no generated source shards found under {path}")
    return files


def parse_timestamp(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def trajectory_key_from_sample_id(sample_id: Any) -> str:
    """Group split point samples back into one interaction trajectory."""
    text = str(sample_id)
    for pattern in (
        r"^(?P<base>.+)_listen_\d+$",
        r"^(?P<base>.+)_speak_\d+$",
        r"^(?P<base>.+)_speak$",
        r"^(?P<base>.+)_delegate$",
    ):
        match = re.match(pattern, text)
        if match:
            return match.group("base")
    return text


def _action_sort_rank(action: Any) -> int:
    return {"listen": 0, "speak": 1, "delegate": 1}.get(str(action), 2)


def cache_key(*parts: str) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def cached_archive_member_path(
    sample: Dict[str, Any],
    cache_dir: str | os.PathLike[str],
) -> Optional[Path]:
    archive = sample.get("video_archive")
    member = sample.get("video_member")
    if not archive or not member:
        return None
    suffix = Path(member).suffix or ".mp4"
    return Path(cache_dir) / f"{cache_key(str(Path(archive)), member)}{suffix}"


def video_cache_status(sample: Dict[str, Any], cache_dir: str | os.PathLike[str]) -> str:
    """Return direct/cache_hit/cache_miss/missing for logging and debugging."""
    if sample.get("frame_paths"):
        return "frames"
    if sample.get("video"):
        return "direct"
    path = cached_archive_member_path(sample, cache_dir)
    if path is None:
        return "missing"
    return "cache_hit" if path.exists() else "cache_miss"


@contextmanager
def resolved_video_path(
    sample: Dict[str, Any],
    cache_dir: str | os.PathLike[str],
    keep_extracted: bool = True,
) -> Iterator[Path]:
    """Yield a local video path for either direct files or tar/tgz members.

    Archive members are extracted into cache_dir. This does not modify the
    original archives.
    """
    if sample.get("video"):
        path = Path(sample["video"])
        if not path.exists():
            raise FileNotFoundError(path)
        yield path
        return

    archive = sample.get("video_archive")
    member = sample.get("video_member")
    if not archive or not member:
        raise FileNotFoundError(f"sample {sample.get('id')} has no video or archive member")

    archive_path = Path(archive)
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cached_archive_member_path(sample, cache_dir)
    if out_path is None:
        raise FileNotFoundError(f"sample {sample.get('id')} has no archive member")
    existed_before = out_path.exists()
    if not existed_before:
        with tarfile.open(archive_path, "r:*") as tar:
            try:
                src = tar.extractfile(member)
            except KeyError as exc:
                raise FileNotFoundError(f"{member} not found in {archive_path}") from exc
            if src is None:
                raise FileNotFoundError(f"{member} is not a regular file in {archive_path}")
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            with tmp_path.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            tmp_path.replace(out_path)

    try:
        yield out_path
    finally:
        if not keep_extracted and not existed_before:
            try:
                out_path.unlink()
            except FileNotFoundError:
                pass


def sample_frame_indices(
    frame_count: int,
    fps: float,
    target_time: float,
    window_seconds: float,
    num_frames: int,
) -> List[int]:
    if frame_count <= 0:
        return []
    if fps <= 0 or math.isnan(fps):
        fps = 1.0
    end_sec = max(target_time, 0.0)
    start_sec = max(0.0, end_sec - max(window_seconds, 0.0))
    start_idx = max(0, min(frame_count - 1, int(round(start_sec * fps))))
    end_idx = max(start_idx, min(frame_count - 1, int(round(end_sec * fps))))
    if num_frames <= 1 or start_idx == end_idx:
        return [end_idx]
    return sorted({int(round(x)) for x in np.linspace(start_idx, end_idx, num_frames)})


def load_video_frames(
    video_path: str | os.PathLike[str],
    target_time: float,
    window_seconds: float = 8.0,
    num_frames: int = 4,
) -> List[Image.Image]:
    """Load uniformly sampled RGB PIL frames from the target-time window."""
    try:
        return load_video_frames_decord(video_path, target_time, window_seconds, num_frames)
    except ImportError:
        return load_video_frames_cv2(video_path, target_time, window_seconds, num_frames)


def load_video_frames_decord(
    video_path: str | os.PathLike[str],
    target_time: float,
    window_seconds: float = 8.0,
    num_frames: int = 4,
) -> List[Image.Image]:
    """Load frames with decord, preferred in the MiniCPM conda environment."""
    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0))
    frame_count = len(vr)
    if frame_count <= 0:
        return []
    fps = float(vr.get_avg_fps() or 1.0)
    indices = sample_frame_indices(frame_count, fps, target_time, window_seconds, num_frames)
    if not indices:
        return []
    batch = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(frame) for frame in batch]


def load_video_frames_cv2(
    video_path: str | os.PathLike[str],
    target_time: float,
    window_seconds: float = 8.0,
    num_frames: int = 4,
) -> List[Image.Image]:
    """Load frames with OpenCV as a fallback."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        indices = sample_frame_indices(frame_count, fps, target_time, window_seconds, num_frames)
        frames: List[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        return frames
    finally:
        cap.release()


def row_to_minicpmo_interaction_feature(
    row: Dict[str, Any],
    *,
    fallback_id: str,
    video_cache_dir: Path,
    window_seconds: float,
    num_frames: int,
    use_audio: bool,
    keep_extracted: bool,
    strict_media: bool,
) -> Dict[str, Any]:
    conversations = row["conversations"]
    user_text = clean_user_text(conversations[0]["content"])
    assistant_text = conversations[1]["content"]
    target_time = parse_timestamp(row.get("time", {}).get("target", 0))

    frames: List[Image.Image] = []
    audio_segments: List[np.ndarray] = []

    try:
        frame_paths = row.get("frame_paths") or []
        if frame_paths:
            frames = [open_rgb_image_with_retry(path) for path in frame_paths[:num_frames]]
        else:
            with resolved_video_path(row, video_cache_dir, keep_extracted) as video_path:
                frames = load_video_frames(
                    video_path,
                    target_time=target_time,
                    window_seconds=window_seconds,
                    num_frames=num_frames,
                )
                if use_audio:
                    raise NotImplementedError(
                        "Audio extraction is intentionally not enabled in this wrapper yet; "
                        "install ffmpeg/librosa and add an audio extractor before setting use_audio=True."
                    )
    except Exception:
        if strict_media:
            raise

    content: List[Any] = []
    content.extend(frames)
    content.extend(audio_segments)
    content.append(user_text)

    return {
        "id": row.get("id", fallback_id),
        "action": row.get("action"),
        "source": row.get("source"),
        "task_type": row.get("task_type"),
        "target_time": target_time,
        "user_content": content,
        "assistant_content": assistant_text,
        "raw": row,
    }


def generated_assistant_to_minicpmo(content: Any) -> str:
    """Map generated tags onto the MiniCPM-o policy tokens."""
    text = str(content or "").strip()
    if not text:
        return "<|listen|>"
    if text.startswith("<|listen|>") or text.startswith("<|speak|>"):
        prefix = "<|listen|>" if text.startswith("<|listen|>") else "<|speak|>"
        return prefix + strip_generated_media_markers(text[len(prefix) :])
    if text.startswith("</silence>"):
        return "<|listen|>"

    if "</delegation>" in text:
        before, delegated = text.split("</delegation>", 1)
        before = strip_generated_media_markers(before.replace("</response>", "", 1)).strip()
        delegated = strip_generated_media_markers(delegated).strip()
        spoken = f"<|speak|>{before}" if before else "<|speak|>"
        return f"{spoken}\n<delegate>{delegated}</delegate>" if delegated else spoken

    if text.startswith("</response>"):
        return "<|speak|>" + strip_generated_media_markers(text[len("</response>") :]).lstrip()
    if text.startswith("</delegation>"):
        delegated = strip_generated_media_markers(text[len("</delegation>") :]).strip()
        return f"<|speak|><delegate>{delegated}</delegate>"
    return "<|speak|>" + strip_generated_media_markers(text)


def collapse_repeated_speak_segments_in_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the first turn of consecutive identical speak responses."""
    messages = row.get("messages") or []
    collapsed_messages = [dict(message) for message in messages]
    assistant_turns: List[Tuple[int, str, str]] = []
    pending_user = False

    for message_idx, copied in enumerate(collapsed_messages):
        role = copied.get("role")
        if role == "user":
            pending_user = True
            continue
        if role == "assistant" and pending_user:
            normalized = generated_assistant_to_minicpmo(copied.get("content", "")).strip()
            assistant_turns.append((message_idx, _generated_action_from_minicpmo_text(normalized), normalized))
            pending_user = False

    start = 0
    cur_action = ""
    cur_text = ""
    run_len = 0
    for idx, (_message_idx, action, text) in enumerate(assistant_turns + [(-1, "", "")]):
        if action == cur_action and text == cur_text:
            run_len += 1
            continue
        if cur_action == "speak" and run_len > 1:
            for message_idx, _action, _text in assistant_turns[start + 1 : idx]:
                collapsed_messages[message_idx]["content"] = "</silence>"
        start = idx
        cur_action = action
        cur_text = text
        run_len = 1

    out = dict(row)
    out["messages"] = collapsed_messages
    return out


def _generated_action_from_minicpmo_text(text: str) -> str:
    if text.startswith("<|listen|>"):
        return "listen"
    if "<delegate>" in text:
        return "delegate"
    if text.startswith("<|speak|>"):
        return "speak"
    return "none"


def generated_speak_segment_metadata(row: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Map original turn indices to consecutive identical speak segment metadata."""
    seq: List[Tuple[str, str]] = []
    pending_user = False
    for message in row.get("messages") or []:
        role = message.get("role")
        if role == "user":
            pending_user = True
        elif role == "assistant" and pending_user:
            normalized = generated_assistant_to_minicpmo(message.get("content", "")).strip()
            seq.append((_generated_action_from_minicpmo_text(normalized), normalized))
            pending_user = False

    metadata: Dict[int, Dict[str, Any]] = {}
    start = 0
    cur_action = ""
    cur_text = ""
    run_len = 0
    for idx, (action, text) in enumerate(seq + [("", "")]):
        if action == cur_action and text == cur_text:
            run_len += 1
            continue
        if cur_action == "speak" and run_len > 0:
            end = idx - 1
            for turn_idx in range(start, end + 1):
                metadata[turn_idx] = {
                    "speak_segment_start": start,
                    "speak_segment_end": end,
                    "speak_segment_text": cur_text,
                }
        start = idx
        cur_action = action
        cur_text = text
        run_len = 1
    return metadata


def _select_generated_image_indices(count: int, max_images: int, selection: str) -> set[int]:
    if count <= 0:
        return set()
    if max_images <= 0 or max_images >= count:
        return set(range(count))
    if selection == "first":
        return set(range(max_images))
    if selection == "uniform":
        return {int(round(x)) for x in np.linspace(0, count - 1, max_images)}
    return set(range(count - max_images, count))


def generated_user_to_minicpmo_content(
    content: Any,
    image_paths: Sequence[str],
    image_cursor: int,
    *,
    max_images_per_turn: int = 0,
    image_selection: str = "last",
    strict_media: bool = True,
) -> Tuple[List[Any], int]:
    """Replace generated <image> markers with PIL frames while preserving turn order."""
    text = str(content or "")
    parts = re.split(r"(<image>)", text)
    marker_count = sum(1 for part in parts if part == "<image>")
    keep_marker_indices = _select_generated_image_indices(marker_count, max_images_per_turn, image_selection)
    out: List[Any] = []
    marker_idx = 0

    for part in parts:
        if part != "<image>":
            if part:
                out.append(part)
            continue

        path_idx = image_cursor + marker_idx
        if marker_idx in keep_marker_indices:
            try:
                out.append(open_rgb_image_with_retry(image_paths[path_idx]))
            except Exception:
                if strict_media:
                    raise
        marker_idx += 1

    return out, image_cursor + marker_count


def generated_turn_count(row: Dict[str, Any]) -> int:
    messages = row.get("messages") or []
    return sum(1 for message in messages if message.get("role") == "assistant")


def generated_user_content_has_instruction(content: Any) -> bool:
    text = str(content or "")
    text = text.replace("<image>", "")
    text = text.replace("<video>", "")
    text = GENERATED_TIME_LINE_RE.sub("", text)
    return bool(text.strip())


PLACEHOLDER_SPEAK_RE = re.compile(r"^\[[A-Za-z0-9 _-]{1,64}\]$")
GENERATED_RESIDUE_RE = re.compile(
    r"(?:->|Original:|Input:|After removing|removed\.?|Left with|Convert to|present tense|"
    r"finite present tense|fix verb|I'll go with|Actually|Show the moment|Find the segment|"
    r"Locate the moment)",
    re.IGNORECASE,
)
EVENT_TEMPLATE_PREFIX_RE = re.compile(r"^(?:Report|Alert|Notice|Update)\s*[:：]\s*", re.IGNORECASE)
EVENT_TEMPLATE_SUFFIX_RE = re.compile(
    r"\s*(?:is|are|was|were)?\s*(?:still\s+)?(?:ongoing|happening|in progress|continuing|visible|present)\s*[.。!！]?\s*$",
    re.IGNORECASE,
)
QUOTED_TEXT_RE = re.compile(r"[\"“”']([^\"“”']{2,240})[\"“”']")


def generated_speak_text_is_placeholder(content: Any) -> bool:
    text = generated_assistant_to_minicpmo(content).strip()
    if _generated_action_from_minicpmo_text(text) != "speak":
        return False
    text = text[len("<|speak|>") :].strip()
    text = re.sub(r"</?response>", "", text).strip()
    text = text.strip().strip('"“”\'‘’。.!?！？：:，,;；')
    return PLACEHOLDER_SPEAK_RE.match(text) is not None


def _generated_speak_payload(content: Any) -> str:
    text = generated_assistant_to_minicpmo(content).strip()
    if not text.startswith("<|speak|>"):
        return ""
    return text[len("<|speak|>") :].strip()


def generated_text_has_template_residue(text: Any) -> bool:
    return GENERATED_RESIDUE_RE.search(str(text or "")) is not None


def _clean_event_phrase(text: Any) -> str:
    phrase = strip_generated_media_markers(text)
    phrase = GENERATED_TIME_LINE_RE.sub(" ", phrase).strip()
    phrase = EVENT_TEMPLATE_PREFIX_RE.sub("", phrase).strip()
    phrase = phrase.strip().strip('"“”\'‘’')
    phrase = EVENT_TEMPLATE_SUFFIX_RE.sub("", phrase).strip()
    phrase = phrase.strip().strip('"“”\'‘’。.!?！？：:，,;；')
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase


def generated_event_phrase_from_assistant(content: Any) -> str:
    return _clean_event_phrase(_generated_speak_payload(content))


def generated_event_phrase_from_instruction(content: Any) -> str:
    text = strip_generated_media_markers(content)
    candidates = [_clean_event_phrase(match.group(1)) for match in QUOTED_TEXT_RE.finditer(text)]
    candidates = [candidate for candidate in candidates if candidate]
    if candidates:
        return max(candidates, key=len)
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and GENERATED_TIME_LINE_RE.sub("", line).strip()
    ]
    return _clean_event_phrase(lines[-1] if lines else text)


def generated_event_phrase_is_clean(phrase: str) -> bool:
    if not phrase:
        return False
    if generated_text_has_template_residue(phrase):
        return False
    stripped = phrase.strip().strip('"“”\'‘’。.!?！？：:，,;；')
    if GENERATED_MEDIA_MARKER_RE.search(stripped) or GENERATED_TIME_LINE_RE.search(stripped):
        return False
    if PLACEHOLDER_SPEAK_RE.match(stripped):
        return False
    if len(stripped) < 4:
        return False
    ascii_words = re.findall(r"[A-Za-z]+", stripped)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", stripped)
    if len(ascii_words) + len(cjk_chars) < 1:
        return False
    return True


def generated_event_grounding_dirty_reasons(row: Dict[str, Any]) -> List[str]:
    if str(row.get("task_type") or "") != "event_grounding":
        return []
    reasons: set[str] = set()
    pending_user: Optional[Dict[str, Any]] = None
    speak_count = 0
    clean_speak_count = 0
    for message in row.get("messages") or []:
        role = message.get("role")
        if role == "user":
            pending_user = message
            if generated_text_has_template_residue(message.get("content", "")):
                reasons.add("instruction_generation_residue")
            continue
        if role != "assistant" or pending_user is None:
            continue
        if _generated_action_from_assistant(message.get("content", "")) == "speak":
            speak_count += 1
            if generated_speak_text_is_placeholder(message.get("content", "")):
                reasons.add("placeholder_speak")
            if generated_text_has_template_residue(message.get("content", "")):
                reasons.add("assistant_generation_residue")
            phrase = generated_event_phrase_from_assistant(message.get("content", ""))
            if not generated_event_phrase_is_clean(phrase):
                phrase = generated_event_phrase_from_instruction(pending_user.get("content", ""))
            if generated_event_phrase_is_clean(phrase):
                clean_speak_count += 1
            else:
                reasons.add("unclean_event_phrase")
        pending_user = None
    if speak_count > 0 and clean_speak_count == 0:
        reasons.add("no_clean_speak")
    return sorted(reasons)


def clean_generated_event_grounding_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if str(row.get("task_type") or "") != "event_grounding":
        return row
    cleaned_messages = [dict(message) for message in (row.get("messages") or [])]
    pending_user: Optional[Dict[str, Any]] = None
    for message in cleaned_messages:
        role = message.get("role")
        if role == "user":
            pending_user = message
            continue
        if role != "assistant" or pending_user is None:
            continue
        if _generated_action_from_assistant(message.get("content", "")) == "speak":
            phrase = generated_event_phrase_from_assistant(message.get("content", ""))
            if not generated_event_phrase_is_clean(phrase):
                phrase = generated_event_phrase_from_instruction(pending_user.get("content", ""))
            if generated_event_phrase_is_clean(phrase):
                message["content"] = f"<|speak|>{phrase}."
        pending_user = None
    out = dict(row)
    out["messages"] = cleaned_messages
    return out


def generated_turn_metadata(row: Dict[str, Any]) -> Tuple[List[str], List[bool], List[bool]]:
    actions: List[str] = []
    user_has_instruction: List[bool] = []
    placeholder_speak: List[bool] = []
    pending_user_instruction = False
    pending_user = False
    for message in row.get("messages") or []:
        role = message.get("role")
        if role == "user":
            pending_user_instruction = generated_user_content_has_instruction(message.get("content", ""))
            pending_user = True
        elif role == "assistant" and pending_user:
            action = _generated_action_from_assistant(message.get("content", ""))
            actions.append(action)
            user_has_instruction.append(pending_user_instruction)
            placeholder_speak.append(action == "speak" and generated_speak_text_is_placeholder(message.get("content", "")))
            pending_user = False
            pending_user_instruction = False
    return actions, user_has_instruction, placeholder_speak


def generated_turn_actions_and_instruction_flags(row: Dict[str, Any]) -> Tuple[List[str], List[bool]]:
    actions, user_has_instruction, _placeholder_speak = generated_turn_metadata(row)
    return actions, user_has_instruction


def generated_timestamp_from_user_content(content: Any) -> float:
    match = re.search(r"<\s*([0-9]+(?:\.[0-9]+)?)\s*seconds\s*>", str(content or ""))
    if not match:
        return 0.0
    return parse_timestamp(match.group(1))


def generated_row_to_minicpmo_trajectory_feature(
    row: Dict[str, Any],
    *,
    fallback_id: str,
    turn_start: int = 0,
    turn_end: int = 0,
    instruction: str = "",
    max_images_per_turn: int = 0,
    max_images_per_sample: int = 0,
    image_selection: str = "last",
    strict_media: bool = True,
    speak_segment_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    messages = row.get("messages") or []
    image_paths = list(row.get("images") or [])
    turns: List[Dict[str, Any]] = []
    pending_user: Optional[Dict[str, Any]] = None
    image_cursor = 0
    turn_idx = 0
    per_turn_keep_indices: Dict[int, set[int]] = {}
    kept_markers_before_turn: Dict[int, int] = {}
    kept_marker_total = 0

    scan_turn_idx = 0
    for message in messages:
        role = message.get("role")
        if role == "user":
            include_turn = scan_turn_idx >= max(0, turn_start) and (turn_end <= 0 or scan_turn_idx < turn_end)
            if include_turn:
                marker_count = str(message.get("content", "") or "").count("<image>")
                keep_indices = _select_generated_image_indices(marker_count, max_images_per_turn, image_selection)
                per_turn_keep_indices[scan_turn_idx] = keep_indices
                kept_markers_before_turn[scan_turn_idx] = kept_marker_total
                kept_marker_total += len(keep_indices)
        elif role == "assistant":
            scan_turn_idx += 1
    sample_keep_indices = _select_generated_image_indices(kept_marker_total, max_images_per_sample, image_selection)

    for message in messages:
        role = message.get("role")
        if role == "user":
            raw_user_content = message.get("content", "")
            include_turn = turn_idx >= max(0, turn_start) and (turn_end <= 0 or turn_idx < turn_end)
            if include_turn:
                parts = re.split(r"(<image>)", str(raw_user_content or ""))
                keep_marker_indices = per_turn_keep_indices.get(turn_idx, set())
                kept_before_turn = kept_markers_before_turn.get(turn_idx, 0)
                user_content = []
                marker_idx = 0
                selected_marker_idx = 0
                for part in parts:
                    if part != "<image>":
                        if part:
                            user_content.append(part)
                        continue
                    path_idx = image_cursor + marker_idx
                    if marker_idx in keep_marker_indices:
                        sample_marker_idx = kept_before_turn + selected_marker_idx
                        selected_marker_idx += 1
                        if sample_marker_idx in sample_keep_indices:
                            try:
                                user_content.append(open_rgb_image_with_retry(image_paths[path_idx]))
                            except Exception:
                                if strict_media:
                                    raise
                    marker_idx += 1
                image_cursor += marker_idx
            else:
                image_cursor += str(raw_user_content or "").count("<image>")
                user_content = []
            pending_user = {"content": user_content, "raw": message}
        elif role == "assistant" and pending_user is not None:
            if turn_idx >= max(0, turn_start) and (turn_end <= 0 or turn_idx < turn_end):
                turn = {
                    "id": f"{fallback_id}:turn{turn_idx:04d}",
                    "turn_index": turn_idx,
                    "user_content": pending_user["content"],
                    "assistant_content": generated_assistant_to_minicpmo(message.get("content", "")),
                    "target_time": generated_timestamp_from_user_content(pending_user["raw"].get("content", "")),
                    "action": _generated_action_from_assistant(message.get("content", "")),
                    "preserve_user_text": True,
                    "raw": {"user": pending_user["raw"], "assistant": message},
                }
                if speak_segment_metadata and turn_idx in speak_segment_metadata:
                    turn.update(speak_segment_metadata[turn_idx])
                turns.append(turn)
            pending_user = None
            turn_idx += 1

    if not turns:
        raise ValueError(f"generated row {fallback_id} produced no turns")

    if instruction:
        first_text = "".join(part for part in turns[0]["user_content"] if isinstance(part, str))
        if instruction not in first_text:
            turns[0]["user_content"].insert(0, instruction.rstrip() + "\n")

    base_id = row.get("annotation_id") or row.get("id") or fallback_id
    effective_turn_end = turn_end or turn_idx
    return {
        "id": f"{base_id}:turns{turn_start:04d}-{effective_turn_end:04d}",
        "turns": turns,
        "source": row.get("source"),
        "task_type": row.get("task_type"),
        "video_name": row.get("video_name"),
        "annotation_id": row.get("annotation_id"),
        "generated_format": True,
        "turn_start": turn_start,
        "turn_end": effective_turn_end,
    }


def _generated_action_from_assistant(content: Any) -> str:
    text = generated_assistant_to_minicpmo(content).strip()
    return _generated_action_from_minicpmo_text(text)


class MiniCPMOInteractionDataset:
    """Load MiniCPM-o interaction samples and video frames."""

    def __init__(
        self,
        annotation_path: str | os.PathLike[str],
        *,
        video_cache_dir: str | os.PathLike[str] | None = None,
        window_seconds: float = 8.0,
        num_frames: int = 4,
        use_audio: bool = False,
        keep_extracted: bool = True,
        strict_media: bool = True,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.video_cache_dir = Path(video_cache_dir) if video_cache_dir else self.annotation_path.parent / "video_cache"
        self.window_seconds = window_seconds
        self.num_frames = num_frames
        self.use_audio = use_audio
        self.keep_extracted = keep_extracted
        self.strict_media = strict_media
        self.rows: List[Dict[str, Any]] = []
        self._lazy_paths: List[Path] = []
        self._lazy_offsets: List[List[int]] = []
        self._lazy_cumulative: List[int] = []

        if self.annotation_path.is_file():
            self.rows = read_json_or_jsonl(self.annotation_path)
        else:
            total = 0
            self._lazy_paths = discover_annotation_files(self.annotation_path)
            for path in self._lazy_paths:
                offsets = jsonl_line_offsets(path)
                self._lazy_offsets.append(offsets)
                total += len(offsets)
                self._lazy_cumulative.append(total)

    def __len__(self) -> int:
        if self.rows:
            return len(self.rows)
        return self._lazy_cumulative[-1] if self._lazy_cumulative else 0

    def get_row(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.rows:
            return self.rows[index]
        shard_idx = bisect_right(self._lazy_cumulative, index)
        prev_total = self._lazy_cumulative[shard_idx - 1] if shard_idx else 0
        row_idx = index - prev_total
        path = self._lazy_paths[shard_idx]
        offset = self._lazy_offsets[shard_idx][row_idx]
        with path.open("rb") as fh:
            fh.seek(offset)
            line = fh.readline()
        return json.loads(line.decode("utf-8"))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.get_row(index)
        return row_to_minicpmo_interaction_feature(
            row,
            fallback_id=str(index),
            video_cache_dir=self.video_cache_dir,
            window_seconds=self.window_seconds,
            num_frames=self.num_frames,
            use_audio=self.use_audio,
            keep_extracted=self.keep_extracted,
            strict_media=self.strict_media,
        )


class MiniCPMOTrajectoryDataset:
    """Group pointwise samples into ordered interaction trajectories."""

    def __init__(
        self,
        annotation_path: str | os.PathLike[str],
        *,
        video_cache_dir: str | os.PathLike[str] | None = None,
        window_seconds: float = 8.0,
        num_frames: int = 4,
        use_audio: bool = False,
        keep_extracted: bool = True,
        strict_media: bool = True,
        max_turns: int = 0,
        include_single_turn: bool = True,
        require_listen_and_speak: bool = False,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.video_cache_dir = Path(video_cache_dir) if video_cache_dir else self.annotation_path.parent / "video_cache"
        self.window_seconds = window_seconds
        self.num_frames = num_frames
        self.use_audio = use_audio
        self.keep_extracted = keep_extracted
        self.strict_media = strict_media
        self.max_turns = max_turns
        self.include_single_turn = include_single_turn
        self.require_listen_and_speak = require_listen_and_speak
        self._groups: List[List[Tuple[Path, int]]] = []
        self._build_index()

    def _build_index(self) -> None:
        grouped: Dict[str, List[Tuple[Tuple[float, int, str], str, Path, int]]] = {}
        for path in discover_annotation_files(self.annotation_path):
            with path.open("rb") as fh:
                while True:
                    offset = fh.tell()
                    line = fh.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    row = json.loads(line.decode("utf-8"))
                    sample_id = row.get("id", f"{path}:{offset}")
                    key = trajectory_key_from_sample_id(sample_id)
                    target_time = parse_timestamp(row.get("time", {}).get("target", 0))
                    action = str(row.get("action", ""))
                    sort_key = (target_time, _action_sort_rank(action), str(sample_id))
                    grouped.setdefault(key, []).append((sort_key, action, path, offset))

        for key in sorted(grouped):
            items = sorted(grouped[key], key=lambda item: item[0])
            if not self.include_single_turn and len(items) <= 1:
                continue
            actions = {action for _sort_key, action, _path, _offset in items}
            if self.require_listen_and_speak and not {"listen", "speak"}.issubset(actions):
                continue
            if self.max_turns > 0:
                items = items[-self.max_turns :]
            self._groups.append([(path, offset) for _sort_key, _action, path, offset in items])

    def __len__(self) -> int:
        return len(self._groups)

    @staticmethod
    def _read_row(path: Path, offset: int) -> Dict[str, Any]:
        with path.open("rb") as fh:
            fh.seek(offset)
            return json.loads(fh.readline().decode("utf-8"))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        refs = self._groups[index]
        turns = [
            row_to_minicpmo_interaction_feature(
                self._read_row(path, offset),
                fallback_id=f"{index}:{turn_idx}",
                video_cache_dir=self.video_cache_dir,
                window_seconds=self.window_seconds,
                num_frames=self.num_frames,
                use_audio=self.use_audio,
                keep_extracted=self.keep_extracted,
                strict_media=self.strict_media,
            )
            for turn_idx, (path, offset) in enumerate(refs)
        ]
        return {
            "id": trajectory_key_from_sample_id(turns[0]["id"]) if turns else str(index),
            "turns": turns,
            "source": turns[0].get("source") if turns else None,
            "task_type": turns[0].get("task_type") if turns else None,
        }


class MiniCPMOGeneratedTrajectoryDataset:
    """Load generated messages/images rows as MiniCPM-o multi-turn trajectories."""

    def __init__(
        self,
        annotation_path: str | os.PathLike[str],
        *,
        max_turns: int = 0,
        chunk_stride: int = 0,
        max_images_per_turn: int = 0,
        max_images_per_sample: int = 0,
        image_selection: str = "last",
        strict_media: bool = True,
        drop_chat_all_silence_chunks: bool = False,
        require_prior_instruction_for_action_chunks: bool = False,
        collapse_repeated_speak_segments: bool = False,
        drop_placeholder_speak_chunks: bool = False,
        clean_event_grounding_templates: bool = False,
        exclude_task_types: str | Sequence[str] | None = None,
        **_unused: Any,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.max_turns = max_turns
        self.chunk_stride = chunk_stride
        self.max_images_per_turn = max_images_per_turn
        self.max_images_per_sample = max_images_per_sample
        self.image_selection = image_selection
        self.strict_media = strict_media
        self.drop_chat_all_silence_chunks = drop_chat_all_silence_chunks
        self.require_prior_instruction_for_action_chunks = require_prior_instruction_for_action_chunks
        self.collapse_repeated_speak_segments = collapse_repeated_speak_segments
        self.drop_placeholder_speak_chunks = drop_placeholder_speak_chunks
        self.clean_event_grounding_templates = clean_event_grounding_templates
        self.exclude_task_types = parse_task_type_set(exclude_task_types)
        self.filter_summary: Dict[str, int] = {
            "refs_before_filter": 0,
            "refs_after_filter": 0,
            "dropped_invalid_jsonl_ref": 0,
            "dropped_excluded_task_type": 0,
            "dropped_chat_all_silence_chunks": 0,
            "dropped_action_without_prior_instruction_chunks": 0,
            "dropped_placeholder_speak_chunks": 0,
        }
        self.refs: List[Dict[str, Any]] = []
        self._invalid_ref_keys: set[Tuple[str, int]] = set()
        self._build_index()

    def _should_collapse_ref(self, ref: Dict[str, Any]) -> bool:
        return self.collapse_repeated_speak_segments or bool(ref.get("collapse_repeated_speak_segments"))

    def _should_clean_event_ref(self, ref: Dict[str, Any]) -> bool:
        return self.clean_event_grounding_templates or bool(ref.get("clean_event_grounding_templates"))

    def _prepare_row_for_ref(self, ref: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
        if self._should_clean_event_ref(ref):
            row = clean_generated_event_grounding_row(row)
        return row

    def _build_index(self) -> None:
        manifest_path = self.annotation_path / "manifest.jsonl" if self.annotation_path.is_dir() else self.annotation_path
        if manifest_path.name == "manifest.jsonl" and manifest_path.exists():
            for ref in read_json_or_jsonl(manifest_path):
                self.refs.extend(self._expand_manifest_ref(ref))
            self._filter_refs()
            return

        if self.annotation_path.is_file():
            with self.annotation_path.open("rb") as fh:
                first = fh.readline()
            if first:
                first_row = json.loads(first.decode("utf-8"))
                if "path" in first_row and "offset" in first_row:
                    for ref in read_json_or_jsonl(self.annotation_path):
                        self.refs.extend(self._expand_manifest_ref(ref))
                    self._filter_refs()
                    return
                if self.exclude_task_types:
                    for row in read_json_or_jsonl(self.annotation_path):
                        ref = {
                            "row": row,
                            "task_type": row.get("task_type"),
                            "source": row.get("source"),
                            "num_turns": generated_turn_count(row),
                            "instruction": row.get("instruction", ""),
                        }
                        self.refs.extend(self._expand_manifest_ref(ref))
                    self._filter_refs()
                    return

        for path in discover_generated_source_shards(self.annotation_path):
            offsets = jsonl_line_offsets(path)
            for offset in offsets:
                self.refs.append({"path": str(path), "offset": offset})

        self._filter_refs()

    def _expand_manifest_ref(self, ref: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "turn_start" in ref or "turn_end" in ref or self.max_turns <= 0:
            return [ref]
        num_turns = int(ref.get("num_turns", 0) or 0)
        if num_turns <= 0:
            return [ref]
        expanded: List[Dict[str, Any]] = []
        for chunk_idx, (turn_start, turn_end) in enumerate(
            generated_manifest_ranges(num_turns, self.max_turns, self.chunk_stride)
        ):
            item = dict(ref)
            item["turn_start"] = turn_start
            item["turn_end"] = turn_end
            item["chunk_index"] = chunk_idx
            expanded.append(item)
        return expanded

    @staticmethod
    def _ref_turn_bounds(ref: Dict[str, Any], total_turns: int, max_turns: int) -> Tuple[int, int]:
        turn_start = int(ref.get("turn_start", 0) or 0)
        turn_end = int(ref.get("turn_end", 0) or 0)
        if turn_end <= 0:
            turn_end = total_turns
            if max_turns > 0:
                turn_start = max(0, turn_end - max_turns)
        elif max_turns > 0 and turn_end - turn_start > max_turns:
            turn_start = max(turn_start, turn_end - max_turns)
        return max(0, turn_start), max(0, turn_end)

    @staticmethod
    def _dominant_chunk_action(actions: Sequence[str], turn_start: int, turn_end: int) -> str:
        chunk_actions = actions[turn_start:turn_end]
        if any(action == "delegate" for action in chunk_actions):
            return "delegate"
        if any(action == "speak" for action in chunk_actions):
            return "speak"
        return "listen"

    @staticmethod
    def _chunk_actions_have_prior_instruction(
        actions: Sequence[str],
        user_has_instruction: Sequence[bool],
        turn_start: int,
        turn_end: int,
    ) -> bool:
        seen_instruction = any(user_has_instruction[:turn_start])
        for idx in range(turn_start, min(turn_end, len(actions), len(user_has_instruction))):
            seen_instruction = seen_instruction or user_has_instruction[idx]
            if actions[idx] in {"speak", "delegate"} and not seen_instruction:
                return False
        return True

    def _turn_metadata_by_ref(self) -> Dict[Tuple[str, int], Tuple[List[str], List[bool], List[bool]]]:
        metadata: Dict[Tuple[str, int], Tuple[List[str], List[bool], List[bool]]] = {}
        path_to_offsets: Dict[str, set[int]] = {}
        clean_by_key: Dict[Tuple[str, int], bool] = {}
        collapse_by_key: Dict[Tuple[str, int], bool] = {}
        for ref in self.refs:
            if "row" in ref:
                original_row = ref["row"]
                row = self._prepare_row_for_ref(ref, original_row)
                if self._should_collapse_ref(ref):
                    row = collapse_repeated_speak_segments_in_row(row)
                metadata[("", id(original_row))] = generated_turn_metadata(row)
                continue
            path = str(ref.get("path", ""))
            offset = int(ref.get("offset", 0) or 0)
            path_to_offsets.setdefault(path, set()).add(offset)
            clean_by_key[(path, offset)] = clean_by_key.get((path, offset), False) or self._should_clean_event_ref(ref)
            collapse_by_key[(path, offset)] = collapse_by_key.get((path, offset), False) or self._should_collapse_ref(ref)

        for path, offsets in path_to_offsets.items():
            with Path(path).open("rb") as fh:
                for offset in sorted(offsets):
                    fh.seek(offset)
                    line = fh.readline()
                    if not line.strip():
                        self._invalid_ref_keys.add((path, offset))
                        continue
                    try:
                        row = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        self._invalid_ref_keys.add((path, offset))
                        continue
                    if clean_by_key.get((path, offset), self.clean_event_grounding_templates):
                        row = clean_generated_event_grounding_row(row)
                    if collapse_by_key.get((path, offset), self.collapse_repeated_speak_segments):
                        row = collapse_repeated_speak_segments_in_row(row)
                    metadata[(path, offset)] = generated_turn_metadata(row)
        return metadata

    def _filter_refs(self) -> None:
        self.filter_summary["refs_before_filter"] = len(self.refs)
        if not (
            self.exclude_task_types
            or self.drop_chat_all_silence_chunks
            or self.require_prior_instruction_for_action_chunks
            or self.drop_placeholder_speak_chunks
        ):
            self.filter_summary["refs_after_filter"] = len(self.refs)
            return

        metadata: Dict[Tuple[str, int], Tuple[List[str], List[bool], List[bool]]] = {}
        if (
            self.drop_chat_all_silence_chunks
            or self.require_prior_instruction_for_action_chunks
            or self.drop_placeholder_speak_chunks
        ):
            metadata = self._turn_metadata_by_ref()
        kept: List[Dict[str, Any]] = []
        for ref in self.refs:
            if "row" not in ref:
                key = (str(ref.get("path", "")), int(ref.get("offset", 0) or 0))
                if key in self._invalid_ref_keys:
                    self.filter_summary["dropped_invalid_jsonl_ref"] += 1
                    continue
            if str(ref.get("task_type") or "") in self.exclude_task_types:
                self.filter_summary["dropped_excluded_task_type"] += 1
                continue
            if not (self.drop_chat_all_silence_chunks or self.require_prior_instruction_for_action_chunks):
                if not self.drop_placeholder_speak_chunks:
                    kept.append(ref)
                    continue
            if "row" in ref:
                actions, user_has_instruction, placeholder_speak = metadata[("", id(ref["row"]))]
            else:
                key = (str(ref.get("path", "")), int(ref.get("offset", 0) or 0))
                actions, user_has_instruction, placeholder_speak = metadata[key]
            turn_start, turn_end = self._ref_turn_bounds(ref, len(actions), self.max_turns)
            label = self._dominant_chunk_action(actions, turn_start, turn_end)

            if (
                self.drop_chat_all_silence_chunks
                and str(ref.get("task_type")) == "chat"
                and label == "listen"
            ):
                self.filter_summary["dropped_chat_all_silence_chunks"] += 1
                continue
            if (
                self.require_prior_instruction_for_action_chunks
                and not self._chunk_actions_have_prior_instruction(actions, user_has_instruction, turn_start, turn_end)
            ):
                self.filter_summary["dropped_action_without_prior_instruction_chunks"] += 1
                continue
            if (
                self.drop_placeholder_speak_chunks
                and any(placeholder_speak[turn_start : min(turn_end, len(placeholder_speak))])
            ):
                self.filter_summary["dropped_placeholder_speak_chunks"] += 1
                continue
            kept.append(ref)
        self.refs = kept
        self.filter_summary["refs_after_filter"] = len(self.refs)

    def __len__(self) -> int:
        return len(self.refs)

    @staticmethod
    def _read_ref(ref: Dict[str, Any]) -> Dict[str, Any]:
        if "row" in ref:
            return ref["row"]
        return read_jsonl_at_offset(ref["path"], int(ref["offset"]))

    def speak_chunk_flags(self) -> List[bool]:
        """Return whether each expanded trajectory chunk contains speak/delegate."""
        return [label in {"speak", "delegate"} for label in self.chunk_action_labels()]

    def chunk_action_labels(self) -> List[str]:
        """Return each expanded chunk's dominant action: delegate, speak, or listen."""
        actions_cache: Dict[Tuple[str, int], List[str]] = {}
        path_to_offsets: Dict[str, set[int]] = {}
        clean_by_key: Dict[Tuple[str, int], bool] = {}
        collapse_by_key: Dict[Tuple[str, int], bool] = {}
        for ref in self.refs:
            if "row" in ref:
                original_row = ref["row"]
                row = self._prepare_row_for_ref(ref, original_row)
                if self._should_collapse_ref(ref):
                    row = collapse_repeated_speak_segments_in_row(row)
                key = ("", id(original_row))
                actions_cache[key] = [
                    _generated_action_from_assistant(message.get("content", ""))
                    for message in row.get("messages", [])
                    if message.get("role") == "assistant"
                ]
                continue
            path = str(ref.get("path", ""))
            offset = int(ref.get("offset", 0) or 0)
            path_to_offsets.setdefault(path, set()).add(offset)
            clean_by_key[(path, offset)] = clean_by_key.get((path, offset), False) or self._should_clean_event_ref(ref)
            collapse_by_key[(path, offset)] = collapse_by_key.get((path, offset), False) or self._should_collapse_ref(ref)

        for path, offsets in path_to_offsets.items():
            with Path(path).open("rb") as fh:
                for offset in sorted(offsets):
                    fh.seek(offset)
                    line = fh.readline()
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if clean_by_key.get((path, offset), self.clean_event_grounding_templates):
                        row = clean_generated_event_grounding_row(row)
                    if collapse_by_key.get((path, offset), self.collapse_repeated_speak_segments):
                        row = collapse_repeated_speak_segments_in_row(row)
                    actions_cache[(path, offset)] = [
                        _generated_action_from_assistant(message.get("content", ""))
                        for message in row.get("messages", [])
                        if message.get("role") == "assistant"
                    ]

        labels: List[str] = []
        for ref in self.refs:
            if "row" in ref:
                actions = actions_cache[("", id(ref["row"]))]
            else:
                key = (str(ref.get("path", "")), int(ref.get("offset", 0) or 0))
                if key not in actions_cache:
                    labels.append("listen")
                    continue
                actions = actions_cache[key]

            turn_start = int(ref.get("turn_start", 0) or 0)
            turn_end = int(ref.get("turn_end", 0) or 0)
            turn_start, turn_end = self._ref_turn_bounds(ref, len(actions), self.max_turns)
            labels.append(self._dominant_chunk_action(actions, turn_start, turn_end))
        return labels

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ref = self.refs[index]
        row = self._read_ref(ref)
        row = self._prepare_row_for_ref(ref, row)
        speak_segments = generated_speak_segment_metadata(row)
        if self._should_collapse_ref(ref):
            row = collapse_repeated_speak_segments_in_row(row)
        turn_start = int(ref.get("turn_start", 0) or 0)
        turn_end = int(ref.get("turn_end", 0) or 0)
        if turn_end <= 0 or (self.max_turns > 0 and turn_end - turn_start > self.max_turns):
            total_turns = int(ref.get("num_turns", 0) or generated_turn_count(row))
            turn_start, turn_end = self._ref_turn_bounds(ref, total_turns, self.max_turns)
        fallback_id = f"{ref.get('path', self.annotation_path)}:{ref.get('offset', index)}"
        return generated_row_to_minicpmo_trajectory_feature(
            row,
            fallback_id=fallback_id,
            turn_start=turn_start,
            turn_end=turn_end,
            instruction=str(ref.get("instruction") or ""),
            max_images_per_turn=self.max_images_per_turn,
            max_images_per_sample=self.max_images_per_sample,
            image_selection=self.image_selection,
            strict_media=self.strict_media,
            speak_segment_metadata=speak_segments,
        )


def split_content_for_processor(content: Sequence[Any]) -> Tuple[str, List[Image.Image], List[np.ndarray], List[int]]:
    images: List[Image.Image] = []
    audios: List[np.ndarray] = []
    audio_parts: List[int] = []
    text_parts: List[str] = []
    for idx, item in enumerate(content):
        if isinstance(item, Image.Image):
            images.append(item)
            text_parts.append(IMAGE_PATTERN)
        elif isinstance(item, np.ndarray):
            audios.append(item)
            audio_parts.append(idx)
            text_parts.append(AUDIO_PATTERN)
        elif isinstance(item, str):
            text_parts.append(item)
        else:
            raise TypeError(f"Unsupported content item: {type(item)!r}")
    return "".join(text_parts), images, audios, audio_parts


def _convert_keep_tokens(processor: Any, input_str: str, max_length: Optional[int] = None) -> Tuple[Any, Any, Any, Any]:
    """Processor._convert without the built-in <|listen|> removal.

    Returns input_ids, image_bounds, audio_bounds, spk_bounds as torch tensors.
    """
    import torch

    tokenizer = processor.tokenizer
    ids = tokenizer.encode(input_str)
    if max_length is not None:
        ids = ids[:max_length]
    input_ids = torch.tensor(ids, dtype=torch.int32)

    # MiniCPM-o expects one image_bound span per image. Including slice_start/end
    # can mis-pair nested slice markers with outer image markers when many frames
    # are concatenated, producing variable bound lengths such as 64/80/232.
    start_cond = input_ids == tokenizer.im_start_id
    end_cond = input_ids == tokenizer.im_end_id
    image_start_idx = torch.where(start_cond)[0] + 1
    image_end_idx = torch.where(end_cond)[0]
    valid_image_nums = min(len(image_start_idx), len(image_end_idx))
    if valid_image_nums:
        image_bounds = torch.hstack(
            [image_start_idx[:valid_image_nums].unsqueeze(-1), image_end_idx[:valid_image_nums].unsqueeze(-1)]
        )
    else:
        image_bounds = torch.empty((0, 2), dtype=torch.long)

    audio_start_idx = torch.where(input_ids == tokenizer.audio_start_id)[0]
    audio_end_idx = torch.where(input_ids == tokenizer.audio_end_id)[0]
    if len(audio_start_idx) != len(audio_end_idx):
        raise ValueError("audio start token count does not match audio end token count")
    audio_bounds = (
        torch.hstack([(audio_start_idx + 1).unsqueeze(-1), audio_end_idx.unsqueeze(-1)])
        if len(audio_start_idx)
        else torch.empty((0, 2), dtype=torch.long)
    )

    spk_start_idx = torch.where(input_ids == tokenizer.spk_start_id)[0]
    spk_end_idx = torch.where(input_ids == tokenizer.spk_end_id)[0]
    if len(spk_start_idx) != len(spk_end_idx):
        raise ValueError("speaker start token count does not match speaker end token count")
    spk_bounds = (
        torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])
        if len(spk_start_idx)
        else torch.empty((0, 2), dtype=torch.long)
    )

    return input_ids, image_bounds, audio_bounds, spk_bounds


def convert_omni_to_inputs_keep_listen(
    processor: Any,
    images: Optional[Dict[str, Any]],
    audio_phs: Optional[List[List[str]]],
    texts: Sequence[str],
    *,
    max_slice_nums: Optional[int] = None,
    use_image_id: bool = False,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Build MiniCPM-o model inputs while preserving <|listen|> target tokens."""
    import torch

    split_pattern = f"({re.escape(IMAGE_PATTERN)}|{re.escape(AUDIO_PATTERN)})"
    bs = len(texts)
    if images is not None:
        pixel_values = images["pixel_values"]
        image_sizes = images["image_sizes"]
        tgt_sizes = images["tgt_sizes"]
    else:
        pixel_values, image_sizes, tgt_sizes = [[]] * bs, [[]] * bs, [[]] * bs

    input_ids_list = []
    image_bounds_list = []
    audio_bounds_list = []
    spk_bounds_list = []

    for index, text in enumerate(texts):
        chunks = re.split(split_pattern, text)
        image_tags = re.findall(re.escape(IMAGE_PATTERN), text)
        audio_tags = re.findall(re.escape(AUDIO_PATTERN), text)
        if len(image_tags) != len(image_sizes[index]):
            raise ValueError(f"image placeholders/images mismatch at batch index {index}")
        if audio_phs is not None and len(audio_tags) != len(audio_phs[index]):
            raise ValueError(f"audio placeholders/audios mismatch at batch index {index}")

        image_id = 0
        audio_id = 0
        for chunk_idx, chunk in enumerate(chunks):
            if chunk == IMAGE_PATTERN:
                chunks[chunk_idx] = processor.image_processor.get_slice_image_placeholder(
                    image_sizes[index][image_id],
                    image_id,
                    max_slice_nums,
                    use_image_id,
                )
                image_id += 1
            elif chunk == AUDIO_PATTERN:
                chunks[chunk_idx] = audio_phs[index][audio_id]
                audio_id += 1

        input_ids, image_bounds, audio_bounds, spk_bounds = _convert_keep_tokens(
            processor, "".join(chunks), max_length
        )
        if len(image_bounds) != len(image_sizes[index]):
            raise ValueError(
                f"image bound count mismatch at batch index {index}: "
                f"bounds={len(image_bounds)} placeholders={len(image_tags)} images={len(image_sizes[index])}"
            )
        input_ids_list.append(input_ids)
        image_bounds_list.append(image_bounds)
        audio_bounds_list.append(audio_bounds)
        spk_bounds_list.append(spk_bounds)

    padded_input_ids, padding_lengths = processor.pad(input_ids_list, padding_side="left")
    attention_mask = torch.ones_like(padded_input_ids, dtype=torch.bool)
    for i, length in enumerate(padding_lengths):
        if length:
            image_bounds_list[i] = image_bounds_list[i] + length
            audio_bounds_list[i] = audio_bounds_list[i] + length
            spk_bounds_list[i] = spk_bounds_list[i] + length
            attention_mask[i, :length] = False

    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_sizes": image_sizes,
        "image_bound": image_bounds_list,
        "tgt_sizes": tgt_sizes,
        "audio_bounds": audio_bounds_list,
        "spk_bounds": spk_bounds_list,
    }


class MiniCPMODataCollator:
    """Collate interaction samples into MiniCPM-o SFT tensors.

    The returned batch matches MiniCPM-o forward(data=...) plus `labels`. When
    `return_loss_weights=True`, the batch also includes token-level
    `loss_weights` for a trainer that supports weighted CE.
    """

    def __init__(
        self,
        processor: Any,
        *,
        max_length: Optional[int] = 4096,
        max_slice_nums: int = 1,
        use_image_id: bool = False,
        stream_input: bool = False,
        sampling_rate: int = 16000,
        listen_weight: float = 0.4,
        speak_weight: float = 2.0,
        speak_boundary_weight: float = 0.0,
        delegate_weight: float = 2.0,
        max_image_pixels: int = 0,
        force_image_size: int = 0,
        return_loss_weights: bool = True,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.max_slice_nums = max_slice_nums
        self.use_image_id = use_image_id
        self.stream_input = stream_input
        self.sampling_rate = sampling_rate
        self.listen_weight = listen_weight
        self.speak_weight = speak_weight
        self.speak_boundary_weight = speak_boundary_weight
        self.delegate_weight = delegate_weight
        self.max_image_pixels = max_image_pixels
        self.force_image_size = force_image_size
        self.return_loss_weights = return_loss_weights

    def _resize_images(self, images_list: Sequence[List[Image.Image]]) -> List[List[Image.Image]]:
        if self.force_image_size > 0:
            return [
                [resize_image_to_fixed_square(image, self.force_image_size) for image in images]
                for images in images_list
            ]
        if self.max_image_pixels <= 0:
            return [list(images) for images in images_list]
        return [
            [resize_image_to_max_pixels(image, self.max_image_pixels) for image in images]
            for images in images_list
        ]

    def _texts_and_media(self, features: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[str], List[List[Image.Image]], List[List[np.ndarray]], Optional[List[List[int]]]]:
        prompt_texts = []
        full_texts = []
        images_list = []
        audios_list = []
        audio_parts_list = []

        for feature in features:
            user_text, images, audios, audio_parts = split_content_for_processor(feature["user_content"])
            prompt_msg = [{"role": "user", "content": user_text}]
            full_msgs = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": feature["assistant_content"]},
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_msg,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = self.tokenizer.apply_chat_template(
                full_msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
            images_list.append(images)
            audios_list.append(audios)
            audio_parts_list.append(audio_parts)

        if not any(audio_parts_list):
            audio_parts = None
        else:
            audio_parts = audio_parts_list
        return prompt_texts, full_texts, images_list, audios_list, audio_parts

    def _texts_to_inputs(
        self,
        texts: Sequence[str],
        images_list: Sequence[List[Image.Image]],
        *,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        images_list = self._resize_images(images_list)
        image_inputs = self.processor.process_image(
            images=list(images_list),
            do_pad=True,
            max_slice_nums=self.max_slice_nums,
            return_tensors="pt",
        )
        full_inputs = convert_omni_to_inputs_keep_listen(
            self.processor,
            image_inputs,
            None,
            texts,
            max_slice_nums=self.max_slice_nums,
            use_image_id=self.use_image_id,
            max_length=self.max_length if max_length is None else max_length,
        )
        audio_features, audio_feature_lens, _audio_phs = self.processor.audio_feature_extract(
            [[] for _ in texts],
            None,
            self.stream_input,
            self.sampling_rate,
        )
        full_inputs["audio_features"] = audio_features
        full_inputs["audio_feature_lens"] = audio_feature_lens
        return full_inputs

    def _text_to_input_length(
        self,
        text: str,
        image_sizes: Sequence[Any],
        *,
        max_length: Optional[int] = None,
    ) -> int:
        split_pattern = f"({re.escape(IMAGE_PATTERN)}|{re.escape(AUDIO_PATTERN)})"
        chunks = re.split(split_pattern, text)
        image_id = 0
        for chunk_idx, chunk in enumerate(chunks):
            if chunk == IMAGE_PATTERN:
                chunks[chunk_idx] = self.processor.image_processor.get_slice_image_placeholder(
                    image_sizes[image_id],
                    image_id,
                    self.max_slice_nums,
                    self.use_image_id,
                )
                image_id += 1
            elif chunk == AUDIO_PATTERN:
                raise NotImplementedError("Trajectory length calculation currently supports image/text turns only.")
        ids = self.tokenizer.encode("".join(chunks))
        if max_length is not None:
            ids = ids[:max_length]
        return len(ids)

    def _trajectory_messages_and_spans(
        self,
        feature: Dict[str, Any],
    ) -> Tuple[List[Dict[str, str]], List[List[Image.Image]], List[Tuple[str, List[Image.Image], str, List[Image.Image]]]]:
        messages: List[Dict[str, str]] = []
        images_by_user_turn: List[List[Image.Image]] = []

        for turn in feature["turns"]:
            user_text, images, audios, _audio_parts = split_content_for_processor(turn["user_content"])
            if audios:
                raise NotImplementedError("Trajectory MiniCPM-o collator currently supports image/text turns only.")
            messages.append({"role": "user", "content": user_text})
            images_by_user_turn.append(images)
            messages.append({"role": "assistant", "content": turn["assistant_content"]})

        spans: List[Tuple[str, List[Image.Image], str, List[Image.Image]]] = []
        template_owner = self.tokenizer
        for assistant_idx in range(1, len(messages), 2):
            prefix_messages = messages[:assistant_idx]
            target_messages = messages[: assistant_idx + 1]
            prefix_text = template_owner.apply_chat_template(
                prefix_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            target_text = template_owner.apply_chat_template(
                target_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            user_turn_count = (assistant_idx + 1) // 2
            span_images = [image for group in images_by_user_turn[:user_turn_count] for image in group]
            spans.append((prefix_text, span_images, target_text, span_images))

        return messages, images_by_user_turn, spans

    def _trajectory_texts_and_media(
        self,
        features: Sequence[Dict[str, Any]],
    ) -> Tuple[List[str], List[List[Image.Image]], List[List[Tuple[str, List[Image.Image], str, List[Image.Image]]]]]:
        full_texts: List[str] = []
        full_images_list: List[List[Image.Image]] = []
        span_specs: List[List[Tuple[str, List[Image.Image], str, List[Image.Image]]]] = []

        for feature in features:
            messages, images_by_user_turn, spans = self._trajectory_messages_and_spans(feature)
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            full_texts.append(full_text)
            full_images_list.append([image for group in images_by_user_turn for image in group])
            span_specs.append(spans)

        return full_texts, full_images_list, span_specs

    def _assistant_span_lengths(
        self,
        span_specs: Sequence[Tuple[str, List[Image.Image], str, List[Image.Image]]],
        image_sizes: Sequence[Any],
    ) -> List[Tuple[int, int]]:
        lengths: List[Tuple[int, int]] = []
        for prefix_text, prefix_images, target_text, target_images in span_specs:
            prefix_len = self._text_to_input_length(prefix_text, image_sizes[: len(prefix_images)])
            target_len = self._text_to_input_length(target_text, image_sizes[: len(target_images)])
            lengths.append((prefix_len, target_len))
        return lengths

    def _collate_trajectory_features(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        full_texts, full_images_list, span_specs = self._trajectory_texts_and_media(features)
        full_images_list = self._resize_images(full_images_list)
        full_inputs = self._texts_to_inputs(full_texts, full_images_list)

        input_ids = full_inputs["input_ids"]
        labels = torch.full_like(input_ids, IGNORE_INDEX, dtype=torch.int64)
        loss_weights = torch.ones_like(input_ids, dtype=torch.float32)
        action_eval_records: List[Dict[str, Any]] = []
        full_unpadded_lens = full_inputs["attention_mask"].sum(dim=1).tolist()
        seq_len = input_ids.shape[1]

        listen_id = self.tokenizer.convert_tokens_to_ids("<|listen|>")
        speak_id = self.tokenizer.convert_tokens_to_ids("<|speak|>")
        delegate_ids = self.tokenizer.encode("<delegate", add_special_tokens=False)

        for row_idx, spans in enumerate(span_specs):
            boundary_speak_positions: List[int] = []
            pad = seq_len - int(full_unpadded_lens[row_idx])
            image_sizes = full_inputs["image_sizes"][row_idx]
            raw_full_len = self._text_to_input_length(full_texts[row_idx], image_sizes)
            sequence_truncated = self.max_length is not None and raw_full_len > int(self.max_length)
            for turn_idx, (prefix_len, target_len) in enumerate(self._assistant_span_lengths(spans, image_sizes)):
                start = pad + max(int(prefix_len) - 1, 0)
                end = pad + min(int(target_len), int(full_unpadded_lens[row_idx])) - 1
                if start < end:
                    labels[row_idx, start:end] = input_ids[row_idx, start + 1 : end + 1].long()
                    turn = features[row_idx]["turns"][turn_idx]
                    action = str(turn.get("action") or "")
                    delegate_label_pos = -1
                    if action == "delegate" and delegate_ids:
                        target_labels = labels[row_idx, start:end].tolist()
                        for offset in range(0, len(target_labels) - len(delegate_ids) + 1):
                            if target_labels[offset : offset + len(delegate_ids)] == delegate_ids:
                                delegate_label_pos = start + offset
                                break
                    original_turn_index = turn.get("turn_index", turn_idx)
                    segment_start = turn.get("speak_segment_start")
                    segment_end = turn.get("speak_segment_end")
                    if action == "speak" and self.speak_boundary_weight > 0:
                        try:
                            original_turn_int = int(original_turn_index)
                            segment_start_int = int(segment_start)
                            segment_end_int = int(segment_end)
                        except (TypeError, ValueError):
                            pass
                        else:
                            valid_segment_end = min(segment_start_int + 1, segment_end_int)
                            if segment_start_int <= original_turn_int <= valid_segment_end:
                                boundary_speak_positions.append(start)
                    action_eval_records.append(
                        {
                            "row_idx": row_idx,
                            "label_pos": start,
                            "delegate_label_pos": delegate_label_pos,
                            "sample_id": features[row_idx].get("id"),
                            "source": features[row_idx].get("source"),
                            "task_type": features[row_idx].get("task_type"),
                            "turn_id": turn.get("id"),
                            "turn_index": turn_idx,
                            "local_turn_index": turn_idx,
                            "original_turn_index": original_turn_index,
                            "target_time": turn.get("target_time"),
                            "gold_action": action,
                            "gold_text": turn.get("assistant_content"),
                            "speak_segment_start": turn.get("speak_segment_start"),
                            "speak_segment_end": turn.get("speak_segment_end"),
                            "speak_segment_text": turn.get("speak_segment_text"),
                            "truncated": self.max_length is not None and int(target_len) > int(self.max_length),
                            "sequence_truncated": sequence_truncated,
                        }
                    )

            for pos in range(seq_len):
                target_id = int(labels[row_idx, pos].item())
                if target_id == listen_id:
                    loss_weights[row_idx, pos] = self.listen_weight
                elif target_id == speak_id:
                    loss_weights[row_idx, pos] = self.speak_weight

            for pos in boundary_speak_positions:
                if 0 <= pos < seq_len and int(labels[row_idx, pos].item()) == speak_id:
                    loss_weights[row_idx, pos] = self.speak_boundary_weight

            if delegate_ids:
                unpadded = input_ids[row_idx, pad : pad + int(full_unpadded_lens[row_idx])].tolist()
                for offset in range(0, len(unpadded) - len(delegate_ids) + 1):
                    if unpadded[offset : offset + len(delegate_ids)] == delegate_ids:
                        label_pos = pad + offset - 1
                        if 0 <= label_pos < seq_len and labels[row_idx, label_pos] != IGNORE_INDEX:
                            loss_weights[row_idx, label_pos] = self.delegate_weight

        full_inputs["labels"] = labels
        full_inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand_as(input_ids).clone()
        full_inputs["action_eval_records"] = action_eval_records
        if self.return_loss_weights:
            full_inputs["loss_weights"] = loss_weights
        return full_inputs

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        if features and "turns" in features[0]:
            return self._collate_trajectory_features(features)

        prompt_texts, full_texts, images_list, audios_list, audio_parts = self._texts_and_media(features)
        images_list = self._resize_images(images_list)

        image_inputs = self.processor.process_image(
            images=images_list,
            do_pad=True,
            max_slice_nums=self.max_slice_nums,
            return_tensors="pt",
        )
        audio_features, audio_feature_lens, audio_phs = self.processor.audio_feature_extract(
            audios_list,
            audio_parts,
            self.stream_input,
            self.sampling_rate,
        )

        full_inputs = convert_omni_to_inputs_keep_listen(
            self.processor,
            image_inputs,
            audio_phs,
            full_texts,
            max_slice_nums=self.max_slice_nums,
            use_image_id=self.use_image_id,
            max_length=self.max_length,
        )
        prompt_inputs = convert_omni_to_inputs_keep_listen(
            self.processor,
            image_inputs,
            audio_phs,
            prompt_texts,
            max_slice_nums=self.max_slice_nums,
            use_image_id=self.use_image_id,
            max_length=self.max_length,
        )

        full_inputs["audio_features"] = audio_features
        full_inputs["audio_feature_lens"] = audio_feature_lens

        input_ids = full_inputs["input_ids"]
        labels = torch.full_like(input_ids, IGNORE_INDEX, dtype=torch.int64)
        loss_weights = torch.ones_like(input_ids, dtype=torch.float32)
        action_eval_records: List[Dict[str, Any]] = []

        full_unpadded_lens = full_inputs["attention_mask"].sum(dim=1).tolist()
        prompt_unpadded_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        seq_len = input_ids.shape[1]

        listen_id = self.tokenizer.convert_tokens_to_ids("<|listen|>")
        speak_id = self.tokenizer.convert_tokens_to_ids("<|speak|>")
        delegate_ids = self.tokenizer.encode("<delegate", add_special_tokens=False)

        for row_idx, (full_len, prompt_len) in enumerate(zip(full_unpadded_lens, prompt_unpadded_lens)):
            pad = seq_len - int(full_len)
            prompt_len = int(prompt_len)
            start = pad + max(prompt_len - 1, 0)
            end = pad + int(full_len) - 1
            if start < end:
                labels[row_idx, start:end] = input_ids[row_idx, start + 1 : end + 1].long()
                action = str(features[row_idx].get("action") or "")
                delegate_label_pos = -1
                if action == "delegate" and delegate_ids:
                    target_labels = labels[row_idx, start:end].tolist()
                    for offset in range(0, len(target_labels) - len(delegate_ids) + 1):
                        if target_labels[offset : offset + len(delegate_ids)] == delegate_ids:
                            delegate_label_pos = start + offset
                            break
                action_eval_records.append(
                    {
                        "row_idx": row_idx,
                        "label_pos": start,
                        "delegate_label_pos": delegate_label_pos,
                        "sample_id": features[row_idx].get("id"),
                        "source": features[row_idx].get("source"),
                        "task_type": features[row_idx].get("task_type"),
                        "turn_id": features[row_idx].get("id"),
                        "turn_index": 0,
                        "target_time": features[row_idx].get("target_time"),
                        "gold_action": action,
                        "gold_text": features[row_idx].get("assistant_content"),
                        "truncated": end >= pad + int(full_len) - 1,
                    }
                )

            for pos in range(start, end):
                target_id = int(labels[row_idx, pos].item())
                if target_id == listen_id:
                    loss_weights[row_idx, pos] = self.listen_weight
                elif target_id == speak_id:
                    loss_weights[row_idx, pos] = self.speak_weight

            if delegate_ids:
                unpadded = input_ids[row_idx, pad : pad + int(full_len)].tolist()
                for offset in range(0, len(unpadded) - len(delegate_ids) + 1):
                    if unpadded[offset : offset + len(delegate_ids)] == delegate_ids:
                        label_pos = pad + offset - 1
                        if 0 <= label_pos < seq_len and labels[row_idx, label_pos] != IGNORE_INDEX:
                            loss_weights[row_idx, label_pos] = self.delegate_weight

        full_inputs["labels"] = labels
        full_inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand_as(input_ids).clone()
        full_inputs["action_eval_records"] = action_eval_records
        if self.return_loss_weights:
            full_inputs["loss_weights"] = loss_weights

        return full_inputs


class MiniCPMODuplexTrajectoryCollator(MiniCPMODataCollator):
    """Collate trajectories using MiniCPMO's full-duplex streaming token schema.

    Unlike :class:`MiniCPMODataCollator`, this collator does not render a
    sequence of ChatML user/assistant messages.  Each trajectory turn becomes a
    duplex unit whose context is supplied by the streaming prefill side and
    whose action/text is generated by the model::

        <|im_start|>system\nStreaming Omni Conversation.<|im_end|>
        <unit><image>...</image>optional text<|listen|></unit>
        <unit><image>...</image><|speak|>response<|turn_eos|><|chunk_eos|></unit>

    Labels follow this training script's existing *pre-shifted* convention:
    the logit immediately before an action predicts ``<|listen|>`` or
    ``<|speak|>``, and speak text through the chunk terminator is supervised.
    Wrapper/prefill tokens, including ``<unit>`` and ``</unit>``, are ignored.
    """

    def __init__(
        self,
        processor: Any,
        *,
        system_prompt: str = "Streaming Omni Conversation.",
        strip_timestamps: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(processor, **kwargs)
        self.system_prompt = str(system_prompt)
        self.strip_timestamps = bool(strip_timestamps)

        self.unit_id = self._required_single_token("<unit>")
        self.unit_end_id = self._required_single_token("</unit>")
        self.listen_id = self._required_single_token("<|listen|>")
        self.speak_id = self._required_single_token("<|speak|>")
        self.turn_eos_id = self._required_single_token("<|turn_eos|>")
        self.chunk_eos_id = self._required_single_token("<|chunk_eos|>")
        self.delegate_ids = [int(x) for x in self.tokenizer.encode("<delegate", add_special_tokens=False)]

    def _required_single_token(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if token_id is None or token_id == unk_id:
            raise ValueError(f"Duplex special token is missing from tokenizer: {token}")
        encoded = [int(x) for x in self.tokenizer.encode(token, add_special_tokens=False)]
        if encoded != [int(token_id)]:
            raise ValueError(f"Duplex marker must encode as one token: {token} -> {encoded}")
        return int(token_id)

    def _duplex_system_prefix(self) -> str:
        # This is the exact text layout used by MiniCPMODuplex.prepare().
        return f"<|im_start|>system\n{self.system_prompt}<|im_end|>"

    def _duplex_unit_text(self, turn: Dict[str, Any]) -> Tuple[str, List[Image.Image]]:
        user_text, images, audios, _audio_parts = split_content_for_processor(turn["user_content"])
        if audios:
            raise NotImplementedError(
                "Duplex trajectory collation currently supports image/text units only; "
                "streaming audio needs chunk-exact feature extraction matching MiniCPMODuplex."
            )

        # streaming_prefill feeds modalities in image -> audio -> text order.
        user_text = user_text.replace(IMAGE_PATTERN, "").replace(AUDIO_PATTERN, "")
        if self.strip_timestamps:
            user_text = GENERATED_TIME_LINE_RE.sub("", user_text)
        user_text = user_text.strip()
        prefill_text = IMAGE_PATTERN * len(images)
        if user_text:
            prefill_text += user_text

        assistant_text = str(turn.get("assistant_content") or "").strip()
        action = str(turn.get("action") or "")
        if action == "listen":
            target_text = "<|listen|>"
        elif action in {"speak", "delegate"}:
            if not assistant_text.startswith("<|speak|>"):
                assistant_text = "<|speak|>" + assistant_text
            target_text = assistant_text
            if not target_text.endswith("<|chunk_eos|>"):
                if not target_text.endswith("<|turn_eos|>"):
                    target_text += "<|turn_eos|>"
                target_text += "<|chunk_eos|>"
        else:
            raise ValueError(f"Unsupported duplex action {action!r} in turn {turn.get('id')!r}")

        return f"<unit>{prefill_text}{target_text}</unit>", images

    def _duplex_texts_and_media(
        self,
        features: Sequence[Dict[str, Any]],
    ) -> Tuple[List[str], List[List[Image.Image]]]:
        texts: List[str] = []
        images_list: List[List[Image.Image]] = []
        system_prefix = self._duplex_system_prefix()
        for feature in features:
            if "turns" not in feature:
                raise ValueError("MiniCPMODuplexTrajectoryCollator requires trajectory features with 'turns'")
            unit_texts: List[str] = []
            sample_images: List[Image.Image] = []
            for turn in feature["turns"]:
                unit_text, unit_images = self._duplex_unit_text(turn)
                unit_texts.append(unit_text)
                sample_images.extend(unit_images)
            texts.append(system_prefix + "".join(unit_texts))
            images_list.append(sample_images)
        return texts, images_list

    @staticmethod
    def _find_token(ids: Sequence[int], token_id: int, start: int, end: int) -> int:
        for pos in range(max(0, start), min(len(ids), end)):
            if int(ids[pos]) == token_id:
                return pos
        return -1

    @staticmethod
    def _find_subsequence(ids: Sequence[int], needle: Sequence[int], start: int, end: int) -> int:
        if not needle:
            return -1
        stop = min(len(ids), end) - len(needle) + 1
        for pos in range(max(0, start), max(max(0, start), stop)):
            if [int(x) for x in ids[pos : pos + len(needle)]] == [int(x) for x in needle]:
                return pos
        return -1

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        full_texts, full_images_list = self._duplex_texts_and_media(features)
        full_inputs = self._texts_to_inputs(full_texts, full_images_list)
        input_ids = full_inputs["input_ids"]
        labels = torch.full_like(input_ids, IGNORE_INDEX, dtype=torch.int64)
        loss_weights = torch.ones_like(input_ids, dtype=torch.float32)
        action_eval_records: List[Dict[str, Any]] = []

        unpadded_lens = [int(x) for x in full_inputs["attention_mask"].sum(dim=1).tolist()]
        seq_len = int(input_ids.shape[1])

        for row_idx, feature in enumerate(features):
            unpadded_len = unpadded_lens[row_idx]
            pad = seq_len - unpadded_len
            ids = [int(x) for x in input_ids[row_idx, pad : pad + unpadded_len].tolist()]
            image_sizes = full_inputs["image_sizes"][row_idx]
            raw_full_len = self._text_to_input_length(full_texts[row_idx], image_sizes)
            sequence_truncated = self.max_length is not None and raw_full_len > int(self.max_length)
            search_pos = 0

            for local_turn_idx, turn in enumerate(feature["turns"]):
                unit_start = self._find_token(ids, self.unit_id, search_pos, len(ids))
                if unit_start < 0:
                    break  # right truncation removed this and all later units
                unit_end = self._find_token(ids, self.unit_end_id, unit_start + 1, len(ids))
                scan_end = unit_end if unit_end >= 0 else len(ids)
                action = str(turn.get("action") or "")
                expected_action_id = self.listen_id if action == "listen" else self.speak_id
                action_input_pos = self._find_token(ids, expected_action_id, unit_start + 1, scan_end)
                if action_input_pos < 0:
                    break  # action itself was truncated
                if action_input_pos == 0:
                    raise ValueError("Duplex action cannot be the first token in a sequence")

                target_complete = True
                if action == "listen":
                    target_input_end = action_input_pos + 1
                else:
                    turn_eos_pos = self._find_token(ids, self.turn_eos_id, action_input_pos + 1, scan_end)
                    chunk_eos_pos = self._find_token(ids, self.chunk_eos_id, action_input_pos + 1, scan_end)
                    if turn_eos_pos < 0 or chunk_eos_pos != turn_eos_pos + 1:
                        target_input_end = scan_end
                        target_complete = False
                    else:
                        target_input_end = chunk_eos_pos + 1

                # Pre-shift labels because train_text_policy_fsdp._weighted_ce_loss
                # compares logits and labels at the same positions without shifting.
                label_start = action_input_pos - 1
                label_end = target_input_end - 1
                labels[row_idx, pad + label_start : pad + label_end] = input_ids[
                    row_idx, pad + action_input_pos : pad + target_input_end
                ].long()

                action_label_pos = pad + label_start
                if action == "listen":
                    loss_weights[row_idx, action_label_pos] = self.listen_weight
                else:
                    loss_weights[row_idx, action_label_pos] = self.speak_weight

                delegate_label_pos = -1
                if action == "delegate" and self.delegate_ids:
                    delegate_input_pos = self._find_subsequence(
                        ids, self.delegate_ids, action_input_pos + 1, target_input_end
                    )
                    if delegate_input_pos >= 1:
                        delegate_label_pos = pad + delegate_input_pos - 1
                        loss_weights[row_idx, delegate_label_pos] = self.delegate_weight

                original_turn_index = turn.get("turn_index", local_turn_idx)
                segment_start = turn.get("speak_segment_start")
                segment_end = turn.get("speak_segment_end")
                if action == "speak" and self.speak_boundary_weight > 0:
                    try:
                        original_turn_int = int(original_turn_index)
                        segment_start_int = int(segment_start)
                        segment_end_int = int(segment_end)
                    except (TypeError, ValueError):
                        pass
                    else:
                        valid_segment_end = min(segment_start_int + 1, segment_end_int)
                        if segment_start_int <= original_turn_int <= valid_segment_end:
                            loss_weights[row_idx, action_label_pos] = self.speak_boundary_weight

                action_eval_records.append(
                    {
                        "row_idx": row_idx,
                        "label_pos": action_label_pos,
                        "delegate_label_pos": delegate_label_pos,
                        "sample_id": feature.get("id"),
                        "source": feature.get("source"),
                        "task_type": feature.get("task_type"),
                        "turn_id": turn.get("id"),
                        "turn_index": local_turn_idx,
                        "local_turn_index": local_turn_idx,
                        "original_turn_index": original_turn_index,
                        "target_time": turn.get("target_time"),
                        "gold_action": action,
                        "gold_text": turn.get("assistant_content"),
                        "speak_segment_start": segment_start,
                        "speak_segment_end": segment_end,
                        "speak_segment_text": turn.get("speak_segment_text"),
                        "truncated": not target_complete,
                        "sequence_truncated": sequence_truncated,
                    }
                )
                search_pos = unit_end + 1 if unit_end >= 0 else len(ids)

        full_inputs["labels"] = labels
        full_inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand_as(input_ids).clone()
        full_inputs["action_eval_records"] = action_eval_records
        if self.return_loss_weights:
            full_inputs["loss_weights"] = loss_weights
        return full_inputs


class WeightedMiniCPMOTrainerMixin:
    """Mixin for CookBook-style CPMTrainer to consume `loss_weights`.

    Use this as a small reference when patching the trainer:

    class MiniCPMOTrainer(WeightedMiniCPMOTrainerMixin, CPMTrainer):
        pass
    """

    def compute_weighted_ce_loss(self, logits: Any, labels: Any, loss_weights: Any) -> Any:
        import torch
        import torch.nn as nn

        vocab_size = logits.shape[-1]
        raw_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")(
            logits.view(-1, vocab_size).contiguous(),
            labels.view(-1).long().to(logits.device).contiguous(),
        )
        weights = loss_weights.view(-1).to(logits.device)
        mask = labels.view(-1).to(logits.device) != IGNORE_INDEX
        weighted = raw_loss * weights
        return weighted[mask].sum() / weights[mask].sum().clamp_min(1.0)
