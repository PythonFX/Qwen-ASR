from __future__ import annotations

import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from utils import is_valid_file, run_cmd


def video_stem_paths(video_path: Path) -> Tuple[Path, Path]:
    """
    返回：
    - 全量 wav 路径
    - 工作目录路径
    """
    base = video_path.with_suffix("")
    full_wav = base.with_name(base.name + "_16k_mono.wav")
    work_dir = base.with_name(base.name + ".srt_work")
    return full_wav, work_dir


def extract_audio_if_needed(video_path: Path, wav_path: Path) -> None:
    """
    从视频提取 16kHz mono wav。
    如果同名中间 wav 已存在，则跳过。
    """
    if is_valid_file(wav_path, min_size=1024):
        print(f"Reuse existing wav: {wav_path}")
        return

    print(f"Extract audio to wav: {wav_path}")
    run_cmd([
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        str(wav_path),
    ])


def get_audio_duration(audio_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    text = result.stdout.strip()
    if not text:
        raise RuntimeError(f"ffprobe 没有返回音频时长: {audio_path}")

    return float(text)


def get_chunk_count(audio_path: Path, chunk_seconds: int) -> int:
    duration = get_audio_duration(audio_path)
    return int(math.ceil(duration / chunk_seconds))


def chunk_wav_path(work_dir: Path, index: int) -> Path:
    return work_dir / f"chunk_{index:04d}.wav"


def chunk_txt_path(work_dir: Path, index: int) -> Path:
    return work_dir / f"chunk_{index:04d}.txt"


def chunk_tokens_path(work_dir: Path, index: int) -> Path:
    return work_dir / f"chunk_{index:04d}.tokens.json"


def chunk_srt_path(work_dir: Path, index: int) -> Path:
    return work_dir / f"chunk_{index:04d}.srt"


def part_srt_path(work_dir: Path, start_index: int, end_index: int) -> Path:
    return work_dir / f"part_{start_index:04d}_{end_index:04d}.srt"


def final_tokens_path(work_dir: Path) -> Path:
    return work_dir / "all.tokens.json"


def split_audio_if_needed(
    audio_path: Path,
    work_dir: Path,
    chunk_seconds: int = 280,
) -> List[Tuple[int, Path, float]]:
    """
    把音频切成若干段，返回 [(chunk_index, chunk_path, chunk_start_offset), ...]。

    chunk wav 会保存在：
      视频名.srt_work/chunk_0000.wav

    已存在的 chunk wav 会复用；缺失的 chunk wav 会用 5 个 worker 并发生成。
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    duration = get_audio_duration(audio_path)
    chunk_count = get_chunk_count(audio_path, chunk_seconds)

    chunks: List[Tuple[int, Path, float]] = []
    missing_chunks: List[Tuple[int, Path, float]] = []

    for index in range(chunk_count):
        start = float(index * chunk_seconds)
        if start >= duration:
            break

        wav_path = chunk_wav_path(work_dir, index)
        chunks.append((index, wav_path, start))

        if is_valid_file(wav_path, min_size=1024):
            print(f"Reuse existing chunk wav: {wav_path.name}")
        else:
            missing_chunks.append((index, wav_path, start))

    if not missing_chunks:
        return chunks

    def chunk_creation_workers() -> int:
        return 5

    workers = min(chunk_creation_workers(), len(missing_chunks))
    print(f"Create {len(missing_chunks)} missing chunk wav files with workers={workers}")

    def create_chunk_wav(chunk_info: Tuple[int, Path, float]) -> int:
        index, wav_path, start = chunk_info
        print(f"Create chunk wav: {wav_path.name}, offset={start:.2f}s")
        run_cmd([
            "ffmpeg",
            "-y",
            "-ss", f"{start:.3f}",
            "-t", str(chunk_seconds),
            "-i", str(audio_path),
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            str(wav_path),
        ])
        return index

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(create_chunk_wav, chunk_info) for chunk_info in missing_chunks]
        for future in as_completed(futures):
            finished_index = future.result()
            print(f"Finished creating chunk wav: chunk_{finished_index:04d}.wav")

    return chunks


def create_single_chunk_wav_if_needed(
    audio_path: Path,
    work_dir: Path,
    chunk_index: int,
    chunk_seconds: int,
    force: bool = False,
) -> Tuple[int, Path, float]:
    """
    只生成指定 chunk 的 wav，用于 --only-chunk-srt 调试。
    chunk_index 是 0-based。
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    duration = get_audio_duration(audio_path)
    start = float(chunk_index * chunk_seconds)
    if start >= duration:
        raise ValueError(f"chunk_index={chunk_index} 超出音频时长范围，duration={duration:.2f}s")

    wav_path = chunk_wav_path(work_dir, chunk_index)

    if force and wav_path.exists():
        print(f"Remove existing chunk wav because --force-chunks is set: {wav_path.name}")
        wav_path.unlink()

    if is_valid_file(wav_path, min_size=1024):
        print(f"Reuse existing chunk wav: {wav_path.name}")
    else:
        print(f"Create only target chunk wav: {wav_path.name}, offset={start:.2f}s")
        run_cmd([
            "ffmpeg",
            "-y",
            "-ss", f"{start:.3f}",
            "-t", str(chunk_seconds),
            "-i", str(audio_path),
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            str(wav_path),
        ])

    return chunk_index, wav_path, start


def split_audio_vad_aware(
    audio_path: Path,
    work_dir: Path,
    speech_chunks: List[Tuple[float, float]],
    pad_seconds: float = 1.0,
) -> List[Tuple[int, Path, float]]:
    """
    基于 VAD 检测的语音边界切分音频。

    speech_chunks 是 merge_segments_into_chunks 的输出：
      [(start_seconds, end_seconds), ...]

    pad_seconds: 每个 chunk 前后额外扩展的秒数，给 ASR 提供更多声学上下文。
                 实际提取范围是 [start-pad, end+pad]，offset 对应调整，
                 不影响最终 token 时间戳精度（clip_tokens_to_speech 会裁剪）。

    返回格式与 split_audio_if_needed 一致：
      [(chunk_index, chunk_path, chunk_start_offset), ...]
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    total_duration = get_audio_duration(audio_path)

    chunks: List[Tuple[int, Path, float]] = []
    missing_chunks: List[Tuple[int, int, Path, float, float]] = []

    for index, (start, end) in enumerate(speech_chunks):
        padded_start = max(0.0, start - pad_seconds)
        padded_end = min(total_duration, end + pad_seconds)
        duration = padded_end - padded_start
        wav_path = chunk_wav_path(work_dir, index)
        chunks.append((index, wav_path, padded_start))

        if is_valid_file(wav_path, min_size=1024):
            print(f"Reuse existing chunk wav: {wav_path.name}")
        else:
            missing_chunks.append((index, index, wav_path, padded_start, duration))

    if not missing_chunks:
        return chunks

    print(f"Create {len(missing_chunks)} VAD-aware chunk wav files (pad={pad_seconds}s)")

    def create_chunk_wav(info: Tuple[int, int, Path, float, float]) -> int:
        _, idx, wav_path, start, duration = info
        print(f"Create chunk wav: {wav_path.name}, offset={start:.2f}s, duration={duration:.2f}s")
        run_cmd([
            "ffmpeg",
            "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(audio_path),
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            str(wav_path),
        ])
        return idx

    with ThreadPoolExecutor(max_workers=min(5, len(missing_chunks))) as executor:
        futures = [executor.submit(create_chunk_wav, info) for info in missing_chunks]
        for future in as_completed(futures):
            finished_index = future.result()
            print(f"Finished creating chunk wav: chunk_{finished_index:04d}.wav")

    return chunks
