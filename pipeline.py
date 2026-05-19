from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from config import AlignToken
from utils import is_valid_file, exists_and_not_empty_text
from audio import chunk_tokens_path, chunk_txt_path, chunk_srt_path, part_srt_path
from asr import load_chunk_tokens_if_valid, save_tokens
from subtitle import build_subtitles, write_srt_atomic, write_srt_with_index_offset, format_srt_time


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
        loaded = load_chunk_tokens_if_valid(token_path)
        if loaded is not None:
            tokens.extend(loaded)

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

        loaded = load_chunk_tokens_if_valid(p)
        if loaded is not None:
            tokens.extend(loaded)

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


_SRT_ENTRY_RE = re.compile(
    r"(\d+)\s*\n"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n"
    r"((?:.*\n)*?)"
    r"(?=\n|\Z)",
)


def _parse_srt_time(s: str) -> float:
    h, m, rest = s.split(":", 2)
    s_part, ms = rest.split(",", 1)
    return int(h) * 3600 + int(m) * 60 + int(s_part) + int(ms) / 1000.0


def _parse_srt_entries(text: str) -> List[Tuple[float, float, str]]:
    entries: List[Tuple[float, float, str]] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        if len(lines) < 3:
            continue
        ts_line = lines[1].strip()
        m = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", ts_line
        )
        if not m:
            continue
        start = _parse_srt_time(m.group(1))
        end = _parse_srt_time(m.group(2))
        subtitle_text = "\n".join(lines[2:]).strip()
        if subtitle_text:
            entries.append((start, end, subtitle_text))
    return entries


def merge_chunk_srts_to_final(
    work_dir: Path,
    chunk_count: int,
    output_path: Path,
) -> None:
    """
    将所有 chunk SRT 直接拼接为最终 SRT（不重新断句）。
    按 chunk 顺序读取，按开始时间排序，重新编号后写入。
    """
    all_entries: List[Tuple[float, float, str]] = []

    for i in range(chunk_count):
        srt_path = chunk_srt_path(work_dir, i)
        if not srt_path.exists():
            print(f"Warning: missing chunk srt: {srt_path.name}")
            continue

        text = srt_path.read_text(encoding="utf-8")
        entries = _parse_srt_entries(text)
        all_entries.extend(entries)

    # Sort by start time (safety measure, chunks should already be in order)
    all_entries.sort(key=lambda e: (e[0], e[1]))

    # Write final SRT with sequential numbering
    lines: List[str] = []
    for idx, (start, end, sub_text) in enumerate(all_entries, start=1):
        lines.append(str(idx))
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(sub_text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Merged {len(all_entries)} subtitle entries from {chunk_count} chunk SRTs")
