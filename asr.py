from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from config import AlignToken, SpeechSegment
from utils import is_valid_file
from audio import chunk_txt_path, chunk_tokens_path

_align_lock = threading.Lock()


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
        r"[\s。．\.，,、！？!?；;：:「」『』（）()\[\]【】《》〈〉…\-—_~〜\"'""'']+",
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

    with _align_lock:
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
        tokens = load_tokens(path)
        txt_path = path.with_name(path.name.replace(".tokens.json", ".txt"))
        if txt_path.exists():
            transcript = txt_path.read_text(encoding="utf-8")
            tokens = apply_transcript_punctuation_to_tokens(tokens, transcript)
        return tokens
    except Exception as e:
        print(f"Failed to load cached tokens, will regenerate: {path.name}, error={e}")
        return None


def collect_inter_token_punctuation(text: str) -> str:
    return re.sub(r"[^。．\.，,、！？!?；;：:…]", "", text)


def apply_transcript_punctuation_to_tokens(tokens: List[AlignToken], transcript: str) -> List[AlignToken]:
    if not tokens or not transcript:
        return tokens

    cursor = 0
    fixed_tokens: List[AlignToken] = []

    for idx, tok in enumerate(tokens):
        token_text = tok.text.strip()
        if not token_text:
            fixed_tokens.append(tok)
            continue

        found_at = transcript.find(token_text, cursor)
        if found_at < 0:
            fixed_tokens.append(tok)
            continue

        token_end = found_at + len(token_text)
        next_found_at = -1

        if idx + 1 < len(tokens):
            next_text = tokens[idx + 1].text.strip()
            if next_text:
                next_found_at = transcript.find(next_text, token_end)

        gap_end = next_found_at if next_found_at >= 0 else token_end
        punct = collect_inter_token_punctuation(transcript[token_end:gap_end])

        if punct and not tok.text.endswith(punct):
            fixed_tokens.append(AlignToken(text=tok.text + punct, start=tok.start, end=tok.end))
        else:
            fixed_tokens.append(tok)

        cursor = token_end

    return fixed_tokens


def fix_abnormal_token_durations(tokens: List[AlignToken]) -> List[AlignToken]:
    """
    修正 forced aligner 因噪声导致的时间戳异常：
    当 1-2 字 token 的时长超过 4 秒时，将其起始时间修正为 end - 0.5s/字。
    """
    fixed = []
    for tok in tokens:
        char_count = len(tok.text.strip())
        duration = tok.end - tok.start
        if 0 < char_count <= 2 and duration > 4.0:
            new_start = tok.end - 0.5 * char_count
            fixed.append(AlignToken(text=tok.text, start=new_start, end=tok.end))
        else:
            fixed.append(tok)
    return fixed


def clip_tokens_to_speech(
    tokens: List[AlignToken],
    speech_regions: List[SpeechSegment],
) -> List[AlignToken]:
    """
    将 token 时间戳裁剪到 VAD 检测的语音区域内。

    对于每个 token：
    - 如果 token.start 落在静音区域，移到下一个语音区域的起始
    - 如果 token.end 落在静音区域，移到上一个语音区域的结束
    - 如果 token 完全不在任何语音区域内，丢弃
    """
    if not tokens or not speech_regions:
        return tokens

    clipped = []
    for tok in tokens:
        best_start = tok.start
        best_end = tok.end

        if tok.start < speech_regions[0].start:
            best_start = speech_regions[0].start
        if tok.end > speech_regions[-1].end:
            best_end = speech_regions[-1].end

        for i, region in enumerate(speech_regions):
            if region.start <= tok.start <= region.end:
                best_start = tok.start
                break
            if i + 1 < len(speech_regions):
                next_region = speech_regions[i + 1]
                if region.end < tok.start < next_region.start:
                    best_start = next_region.start
                    break

        for i in range(len(speech_regions) - 1, -1, -1):
            region = speech_regions[i]
            if region.start <= tok.end <= region.end:
                best_end = tok.end
                break
            if i > 0:
                prev_region = speech_regions[i - 1]
                if prev_region.end < tok.end < region.start:
                    best_end = prev_region.end
                    break

        if best_end <= best_start:
            continue

        if best_start != tok.start or best_end != tok.end:
            clipped.append(AlignToken(text=tok.text, start=best_start, end=best_end))
        else:
            clipped.append(tok)

    return clipped


def process_chunk_if_needed(
    asr_model,
    align_model,
    chunk_index: int,
    chunk_path: Path,
    offset: float,
    language: str,
    work_dir: Path,
    speech_regions: Optional[List[SpeechSegment]] = None,
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
    tokens = apply_transcript_punctuation_to_tokens(tokens, transcript)

    if speech_regions:
        before_count = len(tokens)
        tokens = clip_tokens_to_speech(tokens, speech_regions)
        clipped_count = before_count - len(tokens)
        if clipped_count > 0:
            print(f"Clipped {clipped_count} tokens outside speech regions")

    print(f"Aligned tokens: {len(tokens)}")

    save_tokens(tokens, tokens_path)
    return transcript, tokens
