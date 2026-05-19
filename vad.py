from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from config import SpeechSegment
from utils import is_valid_file


def load_vad_model():
    from silero_vad import load_silero_vad
    return load_silero_vad()


def detect_speech_segments(
    audio_path: Path,
    vad_model,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 300,
    speech_pad_ms: int = 300,
) -> List[SpeechSegment]:
    """Run Silero VAD on full audio, return speech segments in seconds."""
    from silero_vad import read_audio, get_speech_timestamps

    wav = read_audio(str(audio_path), sampling_rate=16000)
    timestamps = get_speech_timestamps(
        wav,
        vad_model,
        threshold=threshold,
        sampling_rate=16000,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )

    return [
        SpeechSegment(start=ts["start"], end=ts["end"])
        for ts in timestamps
    ]


def merge_segments_into_chunks(
    segments: List[SpeechSegment],
    max_duration: float = 180.0,
) -> List[Tuple[float, float]]:
    """
    Merge adjacent speech segments into chunks respecting max_duration.

    Each chunk spans from its first segment's start to its last segment's end.
    Chunks never cut through speech — boundaries always fall on inter-segment silence.

    Returns list of (start_seconds, end_seconds) tuples.
    """
    if not segments:
        return []

    chunks: List[Tuple[float, float]] = []
    chunk_start = segments[0].start
    chunk_end = segments[0].end

    for seg in segments[1:]:
        if seg.end - chunk_start <= max_duration:
            chunk_end = seg.end
        else:
            chunks.append((chunk_start, chunk_end))
            chunk_start = seg.start
            chunk_end = seg.end

    chunks.append((chunk_start, chunk_end))
    return chunks


def vad_cache_path(work_dir: Path) -> Path:
    return work_dir / "vad_segments.json"


def save_vad_segments(segments: List[SpeechSegment], path: Path) -> None:
    data = [{"start": s.start, "end": s.end} for s in segments]
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_vad_segments_if_valid(path: Path) -> Optional[List[SpeechSegment]]:
    if not is_valid_file(path, min_size=2):
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return None
        return [SpeechSegment(start=item["start"], end=item["end"]) for item in data]
    except Exception as e:
        print(f"Failed to load cached VAD segments: {path.name}, error={e}")
        return None
