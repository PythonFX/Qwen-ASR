from __future__ import annotations

import gc
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


def run_cmd(cmd: Sequence[str]) -> None:
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("找不到 ffmpeg。请先安装：brew install ffmpeg")

    if not shutil.which("ffprobe"):
        raise RuntimeError("找不到 ffprobe。ffmpeg 安装后通常会自带 ffprobe。")


def cleanup_after_inference() -> None:
    """
    尽量释放每个 chunk 推理后的临时内存。

    MLX 多线程并发推理时，即使共享同一个模型对象，
    推理图、中间张量和 cache 仍可能叠加，导致 unified memory 暴涨。
    """
    gc.collect()

    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def is_valid_file(path: Path, min_size: int = 1) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size >= min_size


def exists_and_not_empty_text(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False

    try:
        return bool(path.read_text(encoding="utf-8").strip())
    except Exception:
        return False


def remove_files(patterns: list[str], work_dir: Path) -> None:
    for pattern in patterns:
        for p in work_dir.glob(pattern):
            print(f"Remove: {p.name}")
            p.unlink()
