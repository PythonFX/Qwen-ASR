#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用本地 MLX 格式 Qwen3-ASR + Qwen3-ForcedAligner，从视频生成断句合理的 SRT 字幕。

依赖：
  pip install -U mlx-audio
  brew install ffmpeg

示例：
  python qwen3_video_to_srt.py input.mp4 -o output.srt --language Japanese

语言建议：
  中文：Chinese
  英文：English
  日文：Japanese
  韩文：Korean
  粤语：Cantonese
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


ASR_MODEL_PATH = "/Users/vincent/Projects/Model/Qwen3-ASR-1.7B-8bit"
ALIGN_MODEL_PATH = "/Users/vincent/Projects/Model/Qwen3-ForcedAligner-0.6B-8bit"


@dataclass
class AlignToken:
    text: str
    start: float
    end: float


@dataclass
class Subtitle:
    start: float
    end: float
    text: str


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


def video_stem_paths(video_path: Path) -> Tuple[Path, Path]:
    """
    返回：
    - 全量 wav 路径
    - 工作目录路径
    """
    base = video_path.with_suffix("")
    full_wav = base.with_name(base.name + ".qwen3_16k_mono.wav")
    work_dir = base.with_name(base.name + ".qwen3_srt_work")
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
      视频名.qwen3_srt_work/chunk_0000.wav

    如果 chunk wav 已存在，则跳过该 chunk 的切分。
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    duration = get_audio_duration(audio_path)
    chunk_count = get_chunk_count(audio_path, chunk_seconds)

    chunks: List[Tuple[int, Path, float]] = []

    for index in range(chunk_count):
        start = index * chunk_seconds
        if start >= duration:
            break

        wav_path = chunk_wav_path(work_dir, index)

        if is_valid_file(wav_path, min_size=1024):
            print(f"Reuse existing chunk wav: {wav_path.name}")
        else:
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

        chunks.append((index, wav_path, float(start)))

    return chunks


def normalize_transcript(text: str) -> str:
    """
    轻微清理 ASR 文本，避免把奇怪空白带进 aligner。
    不做激进改写，否则可能影响 forced alignment。
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_text_from_asr_result(result) -> str:
    """
    从 ASR 返回结果中安全提取文本。

    关键修复：
    - 如果 result.text 存在但为空字符串，应该返回空字符串。
    - 不要因为 text 为空，就 fallback 到 str(result)。
    - 否则会把 STTOutput(text='', segments=...) 这种 repr 当成 transcript。
    """
    if result is None:
        return ""

    if isinstance(result, str):
        return normalize_transcript(result)

    if isinstance(result, dict):
        return normalize_transcript(result.get("text") or "")

    if hasattr(result, "text"):
        return normalize_transcript(getattr(result, "text") or "")

    print(f"Warning: unknown ASR result type: {type(result)}")
    return ""


def transcribe_chunk(asr_model, chunk_path: Path, language: str) -> str:
    result = asr_model.generate(str(chunk_path), language=language)
    return extract_text_from_asr_result(result)


def is_effective_transcript(text: str) -> bool:
    """
    判断 transcript 是否包含真正可对齐的内容。

    只有空白、句号、逗号、感叹号等标点时，认为无效。
    例如：
      ""
      "。"
      "..."
      "！？"
    都会返回 False。
    """
    text = text.strip()
    if not text:
        return False

    stripped = re.sub(
        r"[\s。．\.，,、！？!?；;：:「」『』（）()\[\]【】《》〈〉…\-—_~〜\"'“”‘’]+",
        "",
        text,
    )
    return bool(stripped)


def iter_align_items(align_result) -> Iterable:
    """
    兼容几种可能的返回结构：
    - ForcedAlignResult(items=[...])
    - list[ForcedAlignItem]
    - dict{"items": [...]}
    """
    if align_result is None:
        return []

    if hasattr(align_result, "items") and not callable(getattr(align_result, "items")):
        return getattr(align_result, "items")

    if isinstance(align_result, dict) and "items" in align_result:
        return align_result["items"]

    if isinstance(align_result, list):
        return align_result

    return align_result


def get_item_attr(item, *names, default=None):
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
        if isinstance(item, dict) and name in item:
            return item[name]
    return default


def align_chunk(align_model, chunk_path: Path, transcript: str, language: str, offset: float) -> List[AlignToken]:
    if not is_effective_transcript(transcript):
        return []

    result = align_model.generate(
        audio=str(chunk_path),
        text=transcript,
        language=language,
    )

    tokens: List[AlignToken] = []
    for item in iter_align_items(result):
        text = get_item_attr(item, "text", "word", default="")
        start = get_item_attr(item, "start_time", "start", default=None)
        end = get_item_attr(item, "end_time", "end", default=None)

        if text is None or start is None or end is None:
            continue

        text = str(text).strip()
        if not text:
            continue

        try:
            start_f = float(start) + offset
            end_f = float(end) + offset
        except Exception:
            continue

        if end_f <= start_f:
            continue

        tokens.append(AlignToken(text=text, start=start_f, end=end_f))

    return tokens


def save_tokens(tokens: List[AlignToken], path: Path) -> None:
    path.write_text(
        json.dumps([asdict(t) for t in tokens], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_tokens(path: Path) -> List[AlignToken]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError(f"tokens json 不是 list: {path}")

    tokens: List[AlignToken] = []
    for item in data:
        tokens.append(
            AlignToken(
                text=str(item["text"]),
                start=float(item["start"]),
                end=float(item["end"]),
            )
        )

    return tokens


def load_chunk_tokens_if_valid(path: Path) -> Optional[List[AlignToken]]:
    if not is_valid_file(path, min_size=2):
        return None

    try:
        return load_tokens(path)
    except Exception as e:
        print(f"Failed to load cached tokens, will regenerate: {path.name}, error={e}")
        return None


def process_chunk_if_needed(
    asr_model,
    align_model,
    chunk_index: int,
    chunk_path: Path,
    offset: float,
    language: str,
    work_dir: Path,
) -> Tuple[str, List[AlignToken]]:
    """
    如果 chunk_XXXX.tokens.json 已存在，则跳过 ASR + ForcedAligner。
    否则执行：
    - ASR，写 chunk_XXXX.txt
    - ForcedAligner，写 chunk_XXXX.tokens.json
    """
    txt_path = chunk_txt_path(work_dir, chunk_index)
    tokens_path = chunk_tokens_path(work_dir, chunk_index)

    cached_tokens = load_chunk_tokens_if_valid(tokens_path)
    if cached_tokens is not None:
        print(f"Reuse cached ASR+align result: {tokens_path.name}")
        transcript = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
        return transcript, cached_tokens

    print(f"\nASR chunk {chunk_index:04d}: {chunk_path.name}, offset={offset:.2f}s")
    transcript = transcribe_chunk(asr_model, chunk_path, language)
    txt_path.write_text(transcript, encoding="utf-8")

    print("Transcript:")
    print(transcript[:500] + ("..." if len(transcript) > 500 else ""))

    if not is_effective_transcript(transcript):
        print(f"No effective transcript, skip align. transcript={transcript!r}")
        save_tokens([], tokens_path)
        return transcript, []

    print(f"Forced alignment chunk {chunk_index:04d}")
    tokens = align_chunk(align_model, chunk_path, transcript, language, offset=offset)
    print(f"Aligned tokens: {len(tokens)}")

    save_tokens(tokens, tokens_path)
    return transcript, tokens


def is_cjk_text(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", s))


def join_tokens_text(tokens: List[AlignToken]) -> str:
    """
    中文/日文/韩文通常不加空格；英文等语言保留词间空格。
    """
    if not tokens:
        return ""

    raw = "".join(t.text for t in tokens)
    cjk = is_cjk_text(raw)

    if cjk:
        text = "".join(t.text for t in tokens)
        text = re.sub(r"\s+", "", text)
    else:
        text = " ".join(t.text for t in tokens)
        text = re.sub(r"\s+([,.!?;:%，。！？；：])", r"\1", text)
        text = re.sub(r"([¿¡])\s+", r"\1", text)

    return text.strip()


def token_ends_sentence(token_text: str) -> bool:
    return bool(re.search(r"[。！？!?；;…]\s*$", token_text))


def token_has_soft_punctuation(token_text: str) -> bool:
    return bool(re.search(r"[，,、：:]\s*$", token_text))


def build_subtitles(
    tokens: List[AlignToken],
    max_chars: int = 42,
    min_chars: int = 8,
    max_duration: float = 6.5,
    min_duration: float = 0.7,
    pause_threshold: float = 0.65,
) -> List[Subtitle]:
    """
    根据 token 时间戳重新断句：
    - 优先在句末标点处断
    - 其次在明显停顿处断
    - 再其次按最大字符数/最大时长断
    - 尽量避免太短字幕
    """
    subtitles: List[Subtitle] = []
    buf: List[AlignToken] = []

    def flush():
        nonlocal buf
        if not buf:
            return

        text = join_tokens_text(buf)
        if not text:
            buf = []
            return

        subtitles.append(Subtitle(
            start=buf[0].start,
            end=buf[-1].end,
            text=text,
        ))
        buf = []

    for i, tok in enumerate(tokens):
        if not tok.text.strip():
            continue

        buf.append(tok)
        text_now = join_tokens_text(buf)
        dur_now = buf[-1].end - buf[0].start

        next_tok: Optional[AlignToken] = tokens[i + 1] if i + 1 < len(tokens) else None
        next_pause = (next_tok.start - tok.end) if next_tok else 999.0

        enough_text = len(text_now) >= min_chars
        too_long_text = len(text_now) >= max_chars
        too_long_time = dur_now >= max_duration
        good_sentence_end = token_ends_sentence(tok.text) and enough_text
        good_pause = next_pause >= pause_threshold and enough_text
        soft_punct_cut = token_has_soft_punctuation(tok.text) and len(text_now) >= int(max_chars * 0.75)

        should_flush = (
            good_sentence_end
            or good_pause
            or too_long_text
            or too_long_time
            or soft_punct_cut
            or next_tok is None
        )

        if should_flush:
            if dur_now < min_duration and next_tok is not None:
                continue
            flush()

    flush()

    return merge_too_short_subtitles(subtitles)


def merge_too_short_subtitles(
    subtitles: List[Subtitle],
    min_chars: int = 4,
    min_duration: float = 0.5,
    max_merged_chars: int = 52,
    max_merged_duration: float = 7.5,
) -> List[Subtitle]:
    """
    合并过短字幕，避免 SRT 阅读体验太碎。
    """
    if not subtitles:
        return []

    merged: List[Subtitle] = []

    for sub in subtitles:
        if not merged:
            merged.append(sub)
            continue

        too_short = len(sub.text) < min_chars or (sub.end - sub.start) < min_duration
        prev = merged[-1]
        combined_text = combine_subtitle_text(prev.text, sub.text)
        combined_duration = sub.end - prev.start

        if too_short and len(combined_text) <= max_merged_chars and combined_duration <= max_merged_duration:
            merged[-1] = Subtitle(
                start=prev.start,
                end=sub.end,
                text=combined_text,
            )
        else:
            merged.append(sub)

    return merged


def combine_subtitle_text(a: str, b: str) -> str:
    if not a:
        return b
    if not b:
        return a

    if is_cjk_text(a + b):
        return a + b
    return a + " " + b


def format_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0

    millis = int(round(seconds * 1000))
    h = millis // 3_600_000
    millis %= 3_600_000
    m = millis // 60_000
    millis %= 60_000
    s = millis // 1000
    ms = millis % 1000

    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(subtitles: List[Subtitle], output_path: Path) -> None:
    lines: List[str] = []

    for i, sub in enumerate(subtitles, start=1):
        lines.append(str(i))
        lines.append(f"{format_srt_time(sub.start)} --> {format_srt_time(sub.end)}")
        lines.append(sub.text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_srt_atomic(subtitles: List[Subtitle], output_path: Path) -> None:
    """
    原子写入 SRT：先写临时文件，再 replace 成目标文件。
    这样外部检查到目标 srt 存在时，可以认为它已经完整写入。
    """
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    write_srt(subtitles, tmp_path)
    tmp_path.replace(output_path)


def write_srt_with_index_offset(
    subtitles: List[Subtitle],
    output_path: Path,
    start_index: int = 1,
) -> int:
    """
    写 part srt 时使用。
    返回下一个字幕序号。
    """
    lines: List[str] = []
    index = start_index

    for sub in subtitles:
        lines.append(str(index))
        lines.append(f"{format_srt_time(sub.start)} --> {format_srt_time(sub.end)}")
        lines.append(sub.text)
        lines.append("")
        index += 1

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return index


def build_and_write_part_srt_if_needed(
    work_dir: Path,
    chunk_start: int,
    chunk_end: int,
    max_chars: int,
    max_duration: float,
    pause_threshold: float,
) -> None:
    """
    每 N 个 chunk 生成一个中间 part srt。

    如果 part srt 已存在，并且对应 token 文件都存在，则跳过。
    """
    token_paths: List[Path] = []

    for i in range(chunk_start, chunk_end + 1):
        p = chunk_tokens_path(work_dir, i)
        if not p.exists():
            print(f"Part srt not ready, missing tokens: {p.name}")
            return
        token_paths.append(p)

    p = part_srt_path(work_dir, chunk_start, chunk_end)

    if is_valid_file(p, min_size=1):
        print(f"Reuse existing part srt: {p.name}")
        return

    print(f"Write part srt: {p.name}")

    tokens: List[AlignToken] = []
    for token_path in token_paths:
        tokens.extend(load_tokens(token_path))

    tokens.sort(key=lambda x: (x.start, x.end))

    subtitles = build_subtitles(
        tokens,
        max_chars=max_chars,
        max_duration=max_duration,
        pause_threshold=pause_threshold,
    )

    write_srt_with_index_offset(subtitles, p, start_index=1)


def write_chunk_srt_if_needed(
    work_dir: Path,
    chunk_index: int,
    tokens: List[AlignToken],
    max_chars: int,
    max_duration: float,
    pause_threshold: float,
) -> Path:
    """
    每个 chunk 完成后写一个 chunk 级 SRT 作为完成标记。
    下一个 chunk 只有在这个文件完整写入后才会启动。
    """
    srt_path = chunk_srt_path(work_dir, chunk_index)

    if is_valid_file(srt_path, min_size=0):
        print(f"Reuse existing chunk srt: {srt_path.name}")
        return srt_path

    subtitles = build_subtitles(
        tokens,
        max_chars=max_chars,
        max_duration=max_duration,
        pause_threshold=pause_threshold,
    )
    write_srt_atomic(subtitles, srt_path)
    print(f"Write chunk srt: {srt_path.name}")
    return srt_path


def maybe_write_recent_part_srt(
    work_dir: Path,
    finished_chunk_index: int,
    chunk_count: int,
    part_size: int,
    max_chars: int,
    max_duration: float,
    pause_threshold: float,
) -> None:
    """
    每完成 part_size 个 chunks，就生成一次中间 srt。
    最后一组不足 part_size 时，在末尾也会生成。
    """
    is_group_boundary = (finished_chunk_index + 1) % part_size == 0
    is_last_chunk = finished_chunk_index == chunk_count - 1

    if not is_group_boundary and not is_last_chunk:
        return

    chunk_start = (finished_chunk_index // part_size) * part_size
    chunk_end = min(chunk_start + part_size - 1, chunk_count - 1)

    build_and_write_part_srt_if_needed(
        work_dir=work_dir,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        max_chars=max_chars,
        max_duration=max_duration,
        pause_threshold=pause_threshold,
    )


def load_all_cached_tokens(work_dir: Path, chunk_count: int) -> List[AlignToken]:
    tokens: List[AlignToken] = []

    for i in range(chunk_count):
        p = chunk_tokens_path(work_dir, i)
        if not p.exists():
            print(f"Warning: missing token cache: {p.name}")
            continue
        tokens.extend(load_tokens(p))

    tokens.sort(key=lambda x: (x.start, x.end))
    return tokens


def write_all_transcripts(work_dir: Path, chunk_count: int, output_path: Path) -> None:
    """
    把所有 chunk txt 合并成一个 transcript txt。
    """
    parts: List[str] = []

    for i in range(chunk_count):
        p = chunk_txt_path(work_dir, i)
        if exists_and_not_empty_text(p):
            text = p.read_text(encoding="utf-8").strip()
            if text:
                parts.append(text)

    output_path.write_text("\n".join(parts), encoding="utf-8")


def exists_and_not_empty_text(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False

    try:
        return bool(path.read_text(encoding="utf-8").strip())
    except Exception:
        return False


def remove_files(patterns: List[str], work_dir: Path) -> None:
    for pattern in patterns:
        for p in work_dir.glob(pattern):
            print(f"Remove: {p.name}")
            p.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="输入视频路径，例如 input.mp4")
    parser.add_argument("-o", "--output", help="输出 SRT 路径，默认和视频同名 .srt")
    parser.add_argument("--language", default="Japanese", help="语言，例如 Chinese / English / Japanese / Cantonese")
    parser.add_argument("--asr-model", default=ASR_MODEL_PATH)
    parser.add_argument("--align-model", default=ALIGN_MODEL_PATH)
    parser.add_argument("--chunk-seconds", type=int, default=280)
    parser.add_argument("--part-chunks", type=int, default=10, help="每多少个 chunks 生成一个中间 part srt")
    parser.add_argument("--workers", type=int, default=2, help="worker 数量，只支持 1 或 2。默认 2；workers=2 时每个线程单独加载一套模型")
    parser.add_argument("--max-chars", type=int, default=42)
    parser.add_argument("--max-duration", type=float, default=6.5)
    parser.add_argument("--pause-threshold", type=float, default=0.65)

    parser.add_argument("--force-wav", action="store_true", help="强制重新生成全量 wav")
    parser.add_argument("--force-chunks", action="store_true", help="强制重新切分 chunk wav")
    parser.add_argument("--force-asr", action="store_true", help="强制重新执行 ASR + ForcedAligner")
    parser.add_argument("--force-parts", action="store_true", help="强制重新生成 part srt")
    parser.add_argument("--parallel-first-two-only", action="store_true", help="只并行处理前两个 chunk，然后退出（用于调试）")

    args = parser.parse_args()

    if args.chunk_seconds <= 0:
        raise ValueError("--chunk-seconds 必须大于 0")

    if args.part_chunks <= 0:
        raise ValueError("--part-chunks 必须大于 0")

    if args.workers not in (1, 2):
        raise ValueError("--workers 只支持 1 或 2。workers=2 时每个线程会单独加载一套模型。")

    require_ffmpeg()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    output_path = Path(args.output).expanduser().resolve() if args.output else video_path.with_suffix(".srt")

    full_wav, work_dir = video_stem_paths(video_path)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.force_wav and full_wav.exists():
        print(f"Remove existing wav because --force-wav is set: {full_wav}")
        full_wav.unlink()

    if args.force_chunks:
        remove_files(["chunk_*.wav"], work_dir)

    if args.force_asr:
        remove_files(["chunk_*.txt", "chunk_*.tokens.json", "chunk_*.srt", "all.tokens.json", "part_*.srt"], work_dir)

    if args.force_parts:
        remove_files(["chunk_*.srt", "part_*.srt"], work_dir)

    extract_audio_if_needed(video_path, full_wav)

    chunks = split_audio_if_needed(
        full_wav,
        work_dir,
        chunk_seconds=args.chunk_seconds,
    )

    chunk_count = len(chunks)

    if chunk_count == 0:
        raise RuntimeError("没有生成任何 chunk，请检查输入视频或音频。")

    from mlx_audio.stt import load

    if args.parallel_first_two_only:
        print(f"Loading ASR model: {args.asr_model}")
        asr_model = load(args.asr_model)

        print(f"Loading ForcedAligner model: {args.align_model}")
        align_model = load(args.align_model)
        print("DEBUG MODE: --parallel-first-two-only is enabled. Only chunk 0000 and 0001 will run sequentially, then the program exits.")
        print("Note: MLX GPU inference is thread-bound here; Python thread parallelism is disabled to avoid `There is no Stream(gpu, 1) in current thread`.")
        test_chunks = chunks[:2]
        if len(test_chunks) < 2:
            raise RuntimeError("Not enough chunks for --parallel-first-two-only")

        for done_count, (chunk_index, chunk_path, offset) in enumerate(test_chunks, start=1):
            print(f"\nDEBUG [{done_count}/2] start chunk={chunk_index:04d}, offset={offset:.2f}s")
            srt_path = chunk_srt_path(work_dir, chunk_index)
            tokens_path = chunk_tokens_path(work_dir, chunk_index)

            try:
                if is_valid_file(srt_path, min_size=0) and is_valid_file(tokens_path, min_size=2):
                    print(f"DEBUG reuse completed chunk task: {srt_path.name}")
                else:
                    _, tokens = process_chunk_if_needed(
                        asr_model=asr_model,
                        align_model=align_model,
                        chunk_index=chunk_index,
                        chunk_path=chunk_path,
                        offset=offset,
                        language=args.language,
                        work_dir=work_dir,
                    )
                    write_chunk_srt_if_needed(
                        work_dir=work_dir,
                        chunk_index=chunk_index,
                        tokens=tokens,
                        max_chars=args.max_chars,
                        max_duration=args.max_duration,
                        pause_threshold=args.pause_threshold,
                    )

                if not is_valid_file(srt_path, min_size=0):
                    raise RuntimeError(f"chunk srt was not written successfully: {srt_path}")

                print(f"DEBUG [{done_count}/2] finished chunk={chunk_index:04d}; marker={srt_path.name}")
            finally:
                cleanup_after_inference()

        cleanup_after_inference()
        print("DEBUG MODE DONE: first two chunks finished sequentially. Exit now.")
        return

    if args.workers == 1:
        print(f"Loading ASR model: {args.asr_model}")
        asr_model = load(args.asr_model)

        print(f"Loading ForcedAligner model: {args.align_model}")
        align_model = load(args.align_model)

        print("Process chunks sequentially with workers=1")

        completed_chunks = set()

        for done_count, (chunk_index, chunk_path, offset) in enumerate(chunks, start=1):
            print(f"\n[{done_count}/{chunk_count}] start chunk={chunk_index:04d}, offset={offset:.2f}s")

            srt_path = chunk_srt_path(work_dir, chunk_index)
            tokens_path = chunk_tokens_path(work_dir, chunk_index)

            if is_valid_file(srt_path, min_size=0) and is_valid_file(tokens_path, min_size=2):
                print(f"Reuse completed chunk task: {srt_path.name}")
                completed_chunks.add(chunk_index)
            else:
                try:
                    _, tokens = process_chunk_if_needed(
                        asr_model=asr_model,
                        align_model=align_model,
                        chunk_index=chunk_index,
                        chunk_path=chunk_path,
                        offset=offset,
                        language=args.language,
                        work_dir=work_dir,
                    )

                    write_chunk_srt_if_needed(
                        work_dir=work_dir,
                        chunk_index=chunk_index,
                        tokens=tokens,
                        max_chars=args.max_chars,
                        max_duration=args.max_duration,
                        pause_threshold=args.pause_threshold,
                    )

                    if not is_valid_file(srt_path, min_size=0):
                        raise RuntimeError(f"chunk srt was not written successfully: {srt_path}")

                    completed_chunks.add(chunk_index)
                finally:
                    cleanup_after_inference()

            print(f"[{done_count}/{chunk_count}] finished chunk={chunk_index:04d}; completion_marker={srt_path.name}")

            group_start = (chunk_index // args.part_chunks) * args.part_chunks
            group_end = min(group_start + args.part_chunks - 1, chunk_count - 1)
            group_done = all(i in completed_chunks for i in range(group_start, group_end + 1))

            if group_done:
                build_and_write_part_srt_if_needed(
                    work_dir=work_dir,
                    chunk_start=group_start,
                    chunk_end=group_end,
                    max_chars=args.max_chars,
                    max_duration=args.max_duration,
                    pause_threshold=args.pause_threshold,
                )

    else:
        print("Process chunks with workers=2. Each worker thread loads its own ASR and ForcedAligner models.")
        print("Note: this avoids sharing MLX GPU streams across threads, but roughly doubles model memory usage.")

        worker_chunks = [chunks[0::2], chunks[1::2]]

        def run_worker(worker_id: int, assigned_chunks: List[Tuple[int, Path, float]]) -> List[int]:
            print(f"Worker {worker_id}: loading ASR model: {args.asr_model}")
            worker_asr_model = load(args.asr_model)

            print(f"Worker {worker_id}: loading ForcedAligner model: {args.align_model}")
            worker_align_model = load(args.align_model)

            finished: List[int] = []
            try:
                for local_pos, (chunk_index, chunk_path, offset) in enumerate(assigned_chunks, start=1):
                    print(
                        f"\nWorker {worker_id} [{local_pos}/{len(assigned_chunks)}] "
                        f"start chunk={chunk_index:04d}, offset={offset:.2f}s"
                    )

                    srt_path = chunk_srt_path(work_dir, chunk_index)
                    tokens_path = chunk_tokens_path(work_dir, chunk_index)

                    try:
                        if is_valid_file(srt_path, min_size=0) and is_valid_file(tokens_path, min_size=2):
                            print(f"Worker {worker_id}: reuse completed chunk task: {srt_path.name}")
                        else:
                            _, tokens = process_chunk_if_needed(
                                asr_model=worker_asr_model,
                                align_model=worker_align_model,
                                chunk_index=chunk_index,
                                chunk_path=chunk_path,
                                offset=offset,
                                language=args.language,
                                work_dir=work_dir,
                            )

                            write_chunk_srt_if_needed(
                                work_dir=work_dir,
                                chunk_index=chunk_index,
                                tokens=tokens,
                                max_chars=args.max_chars,
                                max_duration=args.max_duration,
                                pause_threshold=args.pause_threshold,
                            )

                        if not is_valid_file(srt_path, min_size=0):
                            raise RuntimeError(f"chunk srt was not written successfully: {srt_path}")

                        finished.append(chunk_index)
                        print(f"Worker {worker_id}: finished chunk={chunk_index:04d}; completion_marker={srt_path.name}")
                    finally:
                        cleanup_after_inference()
            finally:
                del worker_asr_model
                del worker_align_model
                cleanup_after_inference()

            return finished

        completed_chunks = set()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_worker, worker_id, assigned)
                for worker_id, assigned in enumerate(worker_chunks)
                if assigned
            ]

            for future in as_completed(futures):
                completed_chunks.update(future.result())

        for group_start in range(0, chunk_count, args.part_chunks):
            group_end = min(group_start + args.part_chunks - 1, chunk_count - 1)
            group_done = all(
                is_valid_file(chunk_srt_path(work_dir, i), min_size=0)
                and is_valid_file(chunk_tokens_path(work_dir, i), min_size=2)
                for i in range(group_start, group_end + 1)
            )
            if group_done:
                build_and_write_part_srt_if_needed(
                    work_dir=work_dir,
                    chunk_start=group_start,
                    chunk_end=group_end,
                    max_chars=args.max_chars,
                    max_duration=args.max_duration,
                    pause_threshold=args.pause_threshold,
                )

    cleanup_after_inference()
    all_tokens = load_all_cached_tokens(work_dir, chunk_count)
    save_tokens(all_tokens, final_tokens_path(work_dir))

    subtitles = build_subtitles(
        all_tokens,
        max_chars=args.max_chars,
        max_duration=args.max_duration,
        pause_threshold=args.pause_threshold,
    )

    write_srt(subtitles, output_path)

    transcript_path = output_path.with_suffix(".txt")
    write_all_transcripts(work_dir, chunk_count, transcript_path)

    print("\nDone.")
    print(f"Full wav: {full_wav}")
    print(f"Work dir: {work_dir}")
    print(f"SRT: {output_path}")
    print(f"Transcript: {transcript_path}")
    print(f"All tokens: {final_tokens_path(work_dir)}")


if __name__ == "__main__":
    main()