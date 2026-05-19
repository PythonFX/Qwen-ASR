from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from config import AlignToken, Subtitle


def is_cjk_text(s: str) -> bool:
    return bool(re.search(r"[一-鿿぀-ヿ가-힯]", s))


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
    from asr import fix_abnormal_token_durations

    subtitles: List[Subtitle] = []
    buf: List[AlignToken] = []
    tokens = fix_abnormal_token_durations(tokens)

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
        good_sentence_end = token_ends_sentence(tok.text)
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
            if not good_sentence_end and dur_now < min_duration and next_tok is not None:
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
    max_gap: float = 1.0,
) -> List[Subtitle]:
    """
    合并过短字幕，避免 SRT 阅读体验太碎。
    只在两条字幕间隔 <= max_gap 秒时才合并。
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
        gap = sub.start - prev.end

        if too_short and len(combined_text) <= max_merged_chars and combined_duration <= max_merged_duration and gap <= max_gap:
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
