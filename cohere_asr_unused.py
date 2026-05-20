#!/usr/bin/env python3
"""
cohere_ctc_forced_alignment_subtitle.py

Generate SRT/VTT subtitles from video using:

    Cohere Transcribe MLX  -> transcript text
    Torchaudio MMS_FA CTC  -> forced alignment timestamps
    Gap-based segmenter    -> subtitle boundaries

Why this exists:
- Cohere Transcribe MLX currently may not expose word/char timestamps.
- This script therefore does NOT try to guess timestamps from fixed text length.
- It uses an external CTC aligner to align Cohere's transcript back to the audio.

Default Cohere MLX model:
    /Users/vincent/Projects/Model/cohere-transcribe-03-2026-mlx-8bit/mlx-int8

Install dependencies, roughly:
    pip install numpy soundfile mlx-speech torch torchaudio uroman

System dependency:
    brew install ffmpeg

Basic usage:
    python cohere_ctc_forced_alignment_subtitle.py \
      --video input.mp4 \
      --output input.srt \
      --language zh

Useful tuning:
    python cohere_ctc_forced_alignment_subtitle.py \
      --video input.mp4 \
      --output input.srt \
      --language zh \
      --chunk-seconds 30 \
      --gap-threshold 0.45 \
      --strong-gap-threshold 0.85 \
      --max-chars 28

Notes:
- The first run will download Torchaudio's MMS_FA alignment model, about 1GB+.
- MMS_FA expects normalized romanized text. This script uses uroman automatically.
- For Chinese, each CJK character is aligned as one unit when possible.
- For English-like text, words are aligned as units.
- Long videos are processed chunk by chunk to keep memory usage manageable.
"""

from __future__ import annotations

import argparse
import functools
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from mlx_speech.generation import CohereAsrModel


TARGET_SR = 16000
DEFAULT_MODEL_PATH = "/Users/vincent/Projects/Model/cohere-transcribe-03-2026-mlx-8bit/mlx-int8"


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class AlignmentUnit:
    """
    One transcript unit to align.

    display:
        Original text displayed in subtitles, such as "你" or "hello".

    norm:
        Romanized normalized token passed to MMS_FA, such as "ni" or "hello".

    suffix:
        Punctuation attached after this unit, such as "，" or ".".
    """

    display: str
    norm: str
    suffix: str = ""


@dataclass
class TimedUnit:
    text: str
    start: float
    end: float
    score: float = 0.0


@dataclass
class SubtitleSegment:
    text: str
    start: float
    end: float


# ---------------------------------------------------------------------
# Command / audio helpers
# ---------------------------------------------------------------------

def run_cmd(cmd: Sequence[str]) -> None:
    subprocess.run(list(cmd), check=True)


def extract_audio(video_path: Path, wav_path: Path) -> None:
    """
    Extract mono 16 kHz WAV from video.
    """
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", str(TARGET_SR),
            "-f", "wav",
            str(wav_path),
        ]
    )


def load_wav_16k(wav_path: Path) -> np.ndarray:
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != TARGET_SR:
        old_len = len(audio)
        new_len = int(round(old_len * TARGET_SR / sr))
        audio = np.interp(
            np.linspace(0, old_len - 1, new_len),
            np.arange(old_len),
            audio,
        ).astype(np.float32)

    return audio.astype(np.float32)


def audio_to_chunks(
    audio: np.ndarray,
    *,
    chunk_seconds: float,
    sample_rate: int = TARGET_SR,
) -> Iterable[Tuple[int, int, np.ndarray]]:
    """
    Yield (start_sample, end_sample, chunk_audio).

    If chunk_seconds <= 0, yield the full audio once.
    """
    if chunk_seconds <= 0:
        yield 0, len(audio), audio
        return

    chunk_size = int(chunk_seconds * sample_rate)
    if chunk_size <= 0:
        raise ValueError("chunk_seconds must be > 0 or exactly 0 for full-audio mode")

    start = 0
    while start < len(audio):
        end = min(start + chunk_size, len(audio))
        chunk = audio[start:end]

        if len(chunk) >= int(0.25 * sample_rate):
            yield start, end, chunk

        if end >= len(audio):
            break
        start = end


# ---------------------------------------------------------------------
# Cohere ASR
# ---------------------------------------------------------------------

def result_to_text(result: Any) -> str:
    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text.strip()

    if isinstance(result, dict):
        text = result.get("text")
        if isinstance(text, str):
            return text.strip()

    return str(result).strip()


def transcribe_chunk(
    model: CohereAsrModel,
    chunk_audio: np.ndarray,
    *,
    language: str,
) -> str:
    result = model.transcribe(
        chunk_audio,
        sample_rate=TARGET_SR,
        language=language,
    )
    return result_to_text(result)


# ---------------------------------------------------------------------
# Romanization / transcript unitization
# ---------------------------------------------------------------------

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
LATIN_WORD_CHAR_RE = re.compile(r"[A-Za-z0-9']")
PUNCT_RE = re.compile(r"[，。！？；：、,.!?;:…]")
SPACE_RE = re.compile(r"\s+")


def is_cjk_char(ch: str) -> bool:
    return bool(CJK_RE.fullmatch(ch))


def is_word_char(ch: str) -> bool:
    return bool(LATIN_WORD_CHAR_RE.fullmatch(ch))


def is_punct(ch: str) -> bool:
    return bool(PUNCT_RE.fullmatch(ch))


class Uromanizer:
    """
    Romanize text using the `uroman` package.

    This class tries the Python API first. If the installed package exposes a
    different API, it falls back to `python -m uroman`.
    """

    def __init__(self, language: Optional[str] = None):
        self.language = language
        self._api = None

        try:
            import uroman  # type: ignore

            # Newer uroman package commonly exposes Uroman.
            if hasattr(uroman, "Uroman"):
                self._api = uroman.Uroman()
            elif hasattr(uroman, "uroman"):
                self._api = uroman
            else:
                self._api = None
        except Exception:
            self._api = None

    @functools.lru_cache(maxsize=100_000)
    def romanize(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""

        # ASCII words do not need uroman.
        if re.fullmatch(r"[A-Za-z']+", text):
            return text.lower()

        if self._api is not None:
            for method_name in ("romanize_string", "romanize", "uromanize_string", "uromanize"):
                method = getattr(self._api, method_name, None)
                if method is None:
                    continue

                try:
                    # Some APIs accept lcode/language, some do not.
                    try:
                        return str(method(text, lcode=self.language)).strip()
                    except TypeError:
                        return str(method(text)).strip()
                except Exception:
                    pass

        # CLI fallback. Slower but robust.
        cmd = [sys.executable, "-m", "uroman", text]
        if self.language:
            cmd = [sys.executable, "-m", "uroman", text, "-l", self.language]

        try:
            p = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            return p.stdout.strip()
        except Exception:
            # Last resort: return original text; normalization may remove it.
            return text


def normalize_romanized_for_mms(text: str) -> str:
    """
    Normalize romanized text to MMS_FA's token alphabet.

    MMS_FA labels are basically lowercase Latin letters plus apostrophe.
    For forced alignment, transcript should be normalized before tokenization.
    """
    text = text.lower()
    text = text.replace("’", "'")
    text = text.replace("`", "'")
    text = re.sub(r"[^a-z' ]", " ", text)
    text = SPACE_RE.sub(" ", text).strip()

    # One AlignmentUnit must be one MMS transcript item.
    # If uroman returns multiple words for a single display unit,
    # remove spaces so the item remains one unit.
    text = text.replace(" ", "")
    return text


def append_suffix_to_previous(units: List[AlignmentUnit], suffix: str) -> None:
    if units:
        units[-1].suffix += suffix


def build_alignment_units(
    transcript: str,
    *,
    romanizer: Uromanizer,
) -> List[AlignmentUnit]:
    """
    Convert Cohere transcript into alignable units.

    Chinese/Japanese/Korean:
        one CJK char -> one alignment unit

    English-like:
        one word -> one alignment unit

    Punctuation:
        attached to the previous display unit, not aligned directly.
    """
    transcript = transcript.strip()
    units: List[AlignmentUnit] = []
    word_buf: List[str] = []

    def flush_word() -> None:
        nonlocal word_buf
        if not word_buf:
            return

        word = "".join(word_buf)
        word_buf = []

        romanized = romanizer.romanize(word)
        norm = normalize_romanized_for_mms(romanized)

        if norm:
            units.append(AlignmentUnit(display=word, norm=norm))
        else:
            append_suffix_to_previous(units, word)

    for ch in transcript:
        if is_word_char(ch):
            word_buf.append(ch)
            continue

        flush_word()

        if is_cjk_char(ch):
            romanized = romanizer.romanize(ch)
            norm = normalize_romanized_for_mms(romanized)

            if norm:
                units.append(AlignmentUnit(display=ch, norm=norm))
            else:
                append_suffix_to_previous(units, ch)
            continue

        if is_punct(ch):
            append_suffix_to_previous(units, ch)
            continue

        if ch.isspace():
            continue

        # Other symbols: preserve visually by attaching to the previous unit.
        append_suffix_to_previous(units, ch)

    flush_word()

    return units


# ---------------------------------------------------------------------
# CTC forced alignment through Torchaudio MMS_FA
# ---------------------------------------------------------------------

class MMSForcedAligner:
    """
    Thin wrapper around torchaudio.pipelines.MMS_FA.

    The expensive acoustic model is loaded once and reused for all chunks.
    """

    def __init__(self, *, device: str = "cpu"):
        import torch
        import torchaudio

        self.torch = torch
        self.torchaudio = torchaudio

        try:
            from torchaudio.pipelines import MMS_FA as bundle
        except Exception as exc:
            raise RuntimeError(
                "Could not import torchaudio.pipelines.MMS_FA. "
                "Install a torchaudio version that still includes MMS_FA, for example "
                "a 2.8-era build. Note that Torchaudio docs mark this API as deprecated "
                "in newer versions."
            ) from exc

        self.bundle = bundle
        self.sample_rate = int(bundle.sample_rate)

        self.device = torch.device(device)
        self.model = bundle.get_model().to(self.device)
        self.model.eval()

        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()

    def align(
        self,
        audio: np.ndarray,
        units: Sequence[AlignmentUnit],
        *,
        offset_seconds: float,
    ) -> List[TimedUnit]:
        """
        Align units to audio chunk and return absolute-timestamp TimedUnit objects.
        """
        if not units:
            return []

        transcript_items = [u.norm for u in units if u.norm]
        filtered_units = [u for u in units if u.norm]

        if not transcript_items:
            return []

        torch = self.torch

        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        if self.sample_rate != TARGET_SR:
            waveform = self.torchaudio.functional.resample(
                waveform,
                TARGET_SR,
                self.sample_rate,
            )

        waveform = waveform.to(self.device)

        with torch.inference_mode():
            emission, _ = self.model(waveform)
            tokenized = self.tokenizer(transcript_items)
            token_spans = self.aligner(emission[0], tokenized)

        # Convert frame spans to seconds.
        # Torchaudio tutorial uses:
        # ratio = waveform.size(1) / emission.size(1) / sample_rate
        ratio = waveform.size(1) / emission.size(1) / self.sample_rate

        timed: List[TimedUnit] = []

        for unit, spans in zip(filtered_units, token_spans):
            if not spans:
                continue

            start = float(spans[0].start) * ratio + offset_seconds
            end = float(spans[-1].end) * ratio + offset_seconds

            score = 0.0
            total_len = 0
            for span in spans:
                span_len = max(1, int(span.end) - int(span.start))
                score += float(span.score) * span_len
                total_len += span_len
            if total_len > 0:
                score /= total_len

            text = unit.display + unit.suffix
            timed.append(
                TimedUnit(
                    text=text,
                    start=start,
                    end=end,
                    score=score,
                )
            )

        return timed


# ---------------------------------------------------------------------
# Subtitle segmentation
# ---------------------------------------------------------------------

STRONG_PUNCT_END_RE = re.compile(r"[。！？；.!?;]$")
WEAK_PUNCT_END_RE = re.compile(r"[，、,:：]$")


def clean_subtitle_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = SPACE_RE.sub(" ", text).strip()

    # Remove spaces between CJK characters.
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)

    # Remove space before punctuation.
    text = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", text)

    return text


def visual_len(text: str) -> int:
    return len(clean_subtitle_text(text))


def join_timed_units(units: Sequence[TimedUnit]) -> str:
    if not units:
        return ""

    # CJK units were single characters, English units were words.
    # We can join by inspecting neighboring units.
    out = ""

    for unit in units:
        piece = unit.text
        if not piece:
            continue

        if not out:
            out = piece
            continue

        last = out[-1]
        first = piece[0]

        # No space around CJK or punctuation.
        if is_cjk_char(last) or is_cjk_char(first) or is_punct(first):
            out += piece
        else:
            out += " " + piece

    return clean_subtitle_text(out)


def should_split(
    *,
    text: str,
    start: float,
    current: TimedUnit,
    nxt: Optional[TimedUnit],
    gap_threshold: float,
    strong_gap_threshold: float,
    punctuation_gap_threshold: float,
    max_chars: int,
    max_duration: float,
    min_chars_before_gap_split: int,
    min_duration_before_gap_split: float,
) -> bool:
    text = clean_subtitle_text(text)
    length = visual_len(text)
    duration = max(0.0, current.end - start)

    if length >= max_chars:
        return True

    if duration >= max_duration:
        return True

    if nxt is None:
        return True

    gap = max(0.0, nxt.start - current.end)

    if STRONG_PUNCT_END_RE.search(text):
        if length >= 4 or gap >= punctuation_gap_threshold:
            return True

    if WEAK_PUNCT_END_RE.search(text):
        if gap >= punctuation_gap_threshold and length >= 6:
            return True

    if gap >= strong_gap_threshold:
        return True

    if (
        gap >= gap_threshold
        and length >= min_chars_before_gap_split
        and duration >= min_duration_before_gap_split
    ):
        return True

    return False


def build_subtitles_from_timed_units(
    timed_units: Sequence[TimedUnit],
    *,
    gap_threshold: float,
    strong_gap_threshold: float,
    punctuation_gap_threshold: float,
    max_chars: int,
    min_chars: int,
    max_duration: float,
    min_duration: float,
    min_chars_before_gap_split: int,
    min_duration_before_gap_split: float,
) -> List[SubtitleSegment]:
    timed_units = sorted(timed_units, key=lambda x: (x.start, x.end))
    timed_units = [u for u in timed_units if u.text.strip()]

    segments: List[SubtitleSegment] = []
    buf: List[TimedUnit] = []

    for i, unit in enumerate(timed_units):
        buf.append(unit)

        nxt = timed_units[i + 1] if i + 1 < len(timed_units) else None
        text = join_timed_units(buf)
        split = should_split(
            text=text,
            start=buf[0].start,
            current=unit,
            nxt=nxt,
            gap_threshold=gap_threshold,
            strong_gap_threshold=strong_gap_threshold,
            punctuation_gap_threshold=punctuation_gap_threshold,
            max_chars=max_chars,
            max_duration=max_duration,
            min_chars_before_gap_split=min_chars_before_gap_split,
            min_duration_before_gap_split=min_duration_before_gap_split,
        )

        if split:
            seg = make_subtitle_segment(buf, min_duration=min_duration)
            if seg:
                segments.append(seg)
            buf = []

    if buf:
        seg = make_subtitle_segment(buf, min_duration=min_duration)
        if seg:
            segments.append(seg)

    return postprocess_segments(
        segments,
        min_chars=min_chars,
        min_duration=min_duration,
        max_chars=max_chars,
    )


def make_subtitle_segment(
    units: Sequence[TimedUnit],
    *,
    min_duration: float,
) -> Optional[SubtitleSegment]:
    if not units:
        return None

    text = join_timed_units(units)
    if not text:
        return None

    start = float(units[0].start)
    end = float(units[-1].end)

    if end <= start:
        end = start + min_duration

    return SubtitleSegment(text=text, start=start, end=end)


def postprocess_segments(
    segments: Sequence[SubtitleSegment],
    *,
    min_chars: int,
    min_duration: float,
    max_chars: int,
) -> List[SubtitleSegment]:
    """
    Merge very tiny subtitle fragments and ensure monotonic timestamps.
    """
    if not segments:
        return []

    merged: List[SubtitleSegment] = []
    i = 0

    while i < len(segments):
        seg = segments[i]
        length = visual_len(seg.text)
        duration = seg.end - seg.start

        if i + 1 < len(segments) and length < min_chars and duration < min_duration * 1.5:
            nxt = segments[i + 1]
            combined_text = clean_subtitle_text(seg.text + nxt.text)

            if visual_len(combined_text) <= max_chars + 8:
                merged.append(
                    SubtitleSegment(
                        text=combined_text,
                        start=seg.start,
                        end=nxt.end,
                    )
                )
                i += 2
                continue

        merged.append(seg)
        i += 1

    monotonic: List[SubtitleSegment] = []
    last_end = 0.0

    for seg in merged:
        start = max(seg.start, last_end)
        end = max(seg.end, start + 0.05)

        monotonic.append(
            SubtitleSegment(
                text=clean_subtitle_text(seg.text),
                start=start,
                end=end,
            )
        )
        last_end = end

    return monotonic


# ---------------------------------------------------------------------
# Subtitle writers
# ---------------------------------------------------------------------

def srt_timestamp(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def vtt_timestamp(seconds: float) -> str:
    return srt_timestamp(seconds).replace(",", ".")


def write_srt(segments: Sequence[SubtitleSegment], output_path: Path) -> None:
    lines: List[str] = []

    idx = 1
    for seg in segments:
        text = clean_subtitle_text(seg.text)
        if not text:
            continue

        lines.append(str(idx))
        lines.append(f"{srt_timestamp(seg.start)} --> {srt_timestamp(seg.end)}")
        lines.append(text)
        lines.append("")
        idx += 1

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: Sequence[SubtitleSegment], output_path: Path) -> None:
    lines = ["WEBVTT", ""]

    for seg in segments:
        text = clean_subtitle_text(seg.text)
        if not text:
            continue

        lines.append(f"{vtt_timestamp(seg.start)} --> {vtt_timestamp(seg.end)}")
        lines.append(text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_transcript_debug(
    chunks: Sequence[Tuple[float, float, str]],
    path: Path,
) -> None:
    lines = []
    for start, end, text in chunks:
        lines.append(f"[{start:.3f} - {end:.3f}]")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def process_video(args: argparse.Namespace) -> None:
    video_path = Path(args.video).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not model_path.exists() and not args.transcript:
        raise FileNotFoundError(f"Cohere model path not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Video: {video_path}")
    print(f"Output: {output_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "audio_16k.wav"

        print("Extracting 16 kHz mono audio with ffmpeg...")
        extract_audio(video_path, wav_path)

        print("Loading audio...")
        audio = load_wav_16k(wav_path)
        total_seconds = len(audio) / TARGET_SR
        print(f"Audio duration: {total_seconds:.2f}s")

        if args.transcript:
            transcript_text = Path(args.transcript).expanduser().read_text(encoding="utf-8")
            chunks = [(0, len(audio), audio, transcript_text)]
            print(f"Using provided transcript file: {args.transcript}")
            cohere_model = None
        else:
            print(f"Loading Cohere MLX model: {model_path}")
            cohere_model = CohereAsrModel.from_path(str(model_path))
            chunks = []

            for start_sample, end_sample, chunk_audio in audio_to_chunks(
                audio,
                chunk_seconds=args.chunk_seconds,
                sample_rate=TARGET_SR,
            ):
                chunk_start = start_sample / TARGET_SR
                chunk_end = end_sample / TARGET_SR
                print(f"Transcribing chunk: {chunk_start:.2f}s - {chunk_end:.2f}s")

                text = transcribe_chunk(
                    cohere_model,
                    chunk_audio,
                    language=args.language,
                ).strip()

                if text:
                    print(f"  transcript chars: {len(text)}")
                    chunks.append((start_sample, end_sample, chunk_audio, text))
                else:
                    print("  empty transcript, skipped")

        if not chunks:
            raise RuntimeError("No transcript chunks were produced.")

        if args.debug_transcript:
            debug_path = output_path.with_suffix(".transcript.txt")
            write_transcript_debug(
                [
                    (s / TARGET_SR, e / TARGET_SR, text)
                    for s, e, _audio, text in chunks
                ],
                debug_path,
            )
            print(f"Wrote debug transcript: {debug_path}")

        print("Loading MMS_FA CTC forced aligner...")
        aligner = MMSForcedAligner(device=args.device)
        romanizer = Uromanizer(language=args.uroman_language or args.language)

        all_timed_units: List[TimedUnit] = []

        for start_sample, end_sample, chunk_audio, transcript_text in chunks:
            chunk_start_sec = start_sample / TARGET_SR
            chunk_end_sec = end_sample / TARGET_SR

            print(f"Preparing alignment units: {chunk_start_sec:.2f}s - {chunk_end_sec:.2f}s")
            units = build_alignment_units(transcript_text, romanizer=romanizer)

            if not units:
                print("  no alignable units, skipped")
                continue

            print(f"  alignable units: {len(units)}")

            try:
                timed = aligner.align(
                    chunk_audio,
                    units,
                    offset_seconds=chunk_start_sec,
                )
            except Exception as exc:
                print(f"  alignment failed for chunk {chunk_start_sec:.2f}-{chunk_end_sec:.2f}s: {exc}")
                if args.strict:
                    raise
                continue

            print(f"  aligned units: {len(timed)}")
            all_timed_units.extend(timed)

    if not all_timed_units:
        raise RuntimeError(
            "No timed units were produced. "
            "Try a shorter chunk size, check the transcript language, or inspect --debug-transcript output."
        )

    if args.debug_units:
        units_path = output_path.with_suffix(".timed_units.tsv")
        lines = ["start\tend\tscore\ttext"]
        for u in all_timed_units:
            lines.append(f"{u.start:.3f}\t{u.end:.3f}\t{u.score:.4f}\t{u.text}")
        units_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote debug timed units: {units_path}")

    print("Building subtitle segments from actual aligned token gaps...")
    segments = build_subtitles_from_timed_units(
        all_timed_units,
        gap_threshold=args.gap_threshold,
        strong_gap_threshold=args.strong_gap_threshold,
        punctuation_gap_threshold=args.punctuation_gap_threshold,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        min_chars_before_gap_split=args.min_chars_before_gap_split,
        min_duration_before_gap_split=args.min_duration_before_gap_split,
    )

    print(f"Subtitle segments: {len(segments)}")

    if output_path.suffix.lower() == ".vtt":
        write_vtt(segments, output_path)
    else:
        write_srt(segments, output_path)

    print(f"Done: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate subtitles with Cohere Transcribe MLX + CTC forced alignment. "
            "Cohere provides transcript text; Torchaudio MMS_FA provides timestamps."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Input video path, for example input.mp4",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output subtitle path, for example input.srt or input.vtt",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Cohere MLX model path. Default: {DEFAULT_MODEL_PATH}",
    )

    parser.add_argument(
        "--language",
        default="zh",
        help="Language code passed to Cohere ASR and uroman. Default: zh",
    )

    parser.add_argument(
        "--uroman-language",
        default=None,
        help=(
            "Optional uroman language code. Defaults to --language. "
            "For Chinese, zh usually works; uroman may also accept language-specific ISO codes."
        ),
    )

    parser.add_argument(
        "--transcript",
        default=None,
        help=(
            "Optional transcript file. If provided, skip Cohere ASR and only run forced alignment. "
            "Useful for debugging alignment."
        ),
    )

    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help=(
            "Audio chunk size for Cohere transcription and CTC alignment. "
            "Use 0 to process the full audio as one chunk. Default: 30.0"
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Torch device for MMS_FA alignment model. Default: cpu. "
            "You may try mps on Apple Silicon, but cpu is usually more compatible."
        ),
    )

    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.50,
        help="Normal inter-unit gap in seconds that can trigger subtitle split. Default: 0.50",
    )

    parser.add_argument(
        "--strong-gap-threshold",
        type=float,
        default=0.85,
        help="Large inter-unit gap in seconds that almost always triggers split. Default: 0.85",
    )

    parser.add_argument(
        "--punctuation-gap-threshold",
        type=float,
        default=0.25,
        help="Gap after weak punctuation that can trigger split. Default: 0.25",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=28,
        help="Maximum subtitle display length before forced split. Default: 28",
    )

    parser.add_argument(
        "--min-chars",
        type=int,
        default=6,
        help="Very short subtitle fragments below this may be merged. Default: 6",
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=6.0,
        help="Maximum subtitle duration in seconds before forced split. Default: 6.0",
    )

    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.60,
        help="Minimum subtitle duration used for cleanup. Default: 0.60",
    )

    parser.add_argument(
        "--min-chars-before-gap-split",
        type=int,
        default=8,
        help="Avoid normal gap splitting if current subtitle is shorter than this. Default: 8",
    )

    parser.add_argument(
        "--min-duration-before-gap-split",
        type=float,
        default=0.80,
        help="Avoid normal gap splitting if current subtitle is shorter than this duration. Default: 0.80",
    )

    parser.add_argument(
        "--debug-transcript",
        action="store_true",
        help="Write chunk-level Cohere transcript to output.transcript.txt",
    )

    parser.add_argument(
        "--debug-units",
        action="store_true",
        help="Write aligned units to output.timed_units.tsv",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if one chunk cannot be aligned. Default is to skip failed chunks.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
