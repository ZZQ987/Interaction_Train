"""Logging helpers for MiniCPM-o SFT runs."""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def is_rank0() -> bool:
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    return str(rank) == "0"


def run_id(prefix: str = "minicpmo") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    host = socket.gethostname().split(".")[0]
    return f"{prefix}_{stamp}_{host}_pid{os.getpid()}"


def setup_logging(
    *,
    log_dir: str | os.PathLike[str] = "datasets/minicpmo_sft/logs",
    name: str = "minicpmo",
    level: str = "INFO",
    filename: Optional[str] = None,
) -> logging.Logger:
    """Configure rank-aware console + file logging.

    Returns a standard `logging.Logger`. Only rank 0 writes console/file logs by
    default; nonzero ranks get a NullHandler to avoid noisy duplicate logs.
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not is_rank0():
        logger.addHandler(logging.NullHandler())
        return logger

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = run_id(name) + ".log"
    log_path = log_dir / filename

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.info("log_file=%s", log_path)
    install_exception_hook(logger)
    log_environment(logger)
    return logger


def install_exception_hook(logger: logging.Logger) -> None:
    def _hook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.error("unhandled_exception=%s", exc, exc_info=(exc_type, exc, tb))

    sys.excepthook = _hook


def command_output(cmd: list[str], timeout: int = 10) -> str:
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=timeout)
    except Exception as exc:
        return f"<failed: {exc}>"
    out = (proc.stdout or proc.stderr or "").strip()
    if proc.returncode != 0:
        return f"<failed rc={proc.returncode}: {out[:1000]}>"
    return out[:4000]


def log_environment(logger: logging.Logger) -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix and "/envs/" in sys.executable:
        conda_prefix = sys.executable.split("/bin/python", 1)[0]
    logger.info("python=%s", sys.executable)
    logger.info("argv=%s", " ".join(sys.argv))
    logger.info("cwd=%s", os.getcwd())
    logger.info("host=%s platform=%s", socket.gethostname(), platform.platform())
    logger.info("conda_prefix=%s", conda_prefix)
    logger.info("cuda_visible_devices=%s", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    logger.info("rank=%s local_rank=%s world_size=%s", os.environ.get("RANK", "0"), os.environ.get("LOCAL_RANK", "0"), os.environ.get("WORLD_SIZE", "1"))

    git_root = command_output(["git", "rev-parse", "--show-toplevel"])
    if git_root and not git_root.startswith("<failed"):
        logger.info("git_root=%s", git_root)
        logger.info("git_head=%s", command_output(["git", "rev-parse", "--short", "HEAD"]))
        status = command_output(["git", "status", "--short"])
        if status:
            logger.info("git_status=\n%s", status)

    logger.info("nvidia_smi=\n%s", command_output(["nvidia-smi"], timeout=15))


class JsonlMetricLogger:
    """Append structured metrics/events to a JSONL sidecar file."""

    def __init__(self, path: str | os.PathLike[str], *, enabled: Optional[bool] = None) -> None:
        self.enabled = is_rank0() if enabled is None else enabled
        self.path = Path(path)
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **payload: Any) -> None:
        if not self.enabled:
            return
        payload.setdefault("time", time.time())
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def log_dataset_sample(logger: logging.Logger, sample: Dict[str, Any], *, prefix: str = "sample") -> None:
    raw = sample.get("raw", sample)
    media = raw.get("media", {})
    logger.info(
        "%s id=%s action=%s task=%s media=%s video=%s member=%s",
        prefix,
        sample.get("id", raw.get("id")),
        sample.get("action", raw.get("action")),
        sample.get("task_type", raw.get("task_type")),
        media.get("kind"),
        raw.get("video") or raw.get("video_archive"),
        raw.get("video_member", ""),
    )


def log_exception(logger: logging.Logger, message: str, exc: BaseException) -> None:
    logger.error("%s: %s\n%s", message, exc, "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
