from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

from config import AlignToken, ASREngineType


class BaseASREngine(ABC):
    """ASR engine interface. Each engine encapsulates its own model loading, transcription, and alignment logic."""

    @property
    @abstractmethod
    def engine_type(self) -> ASREngineType:
        ...

    @abstractmethod
    def transcribe_and_align(
        self,
        chunk_path: Path,
        language: str,
        offset: float,
    ) -> Tuple[str, List[AlignToken]]:
        """Return (transcript_text, aligned_tokens) for the given audio chunk."""
        ...


class QwenEngine(BaseASREngine):
    """Qwen3-ASR + Qwen3-ForcedAligner (two-step: ASR then forced alignment)."""

    def __init__(self, asr_model_path: str, align_model_path: str):
        from mlx_audio.stt import load

        print(f"Loading Qwen ASR model: {asr_model_path}")
        self._asr_model = load(asr_model_path)
        print(f"Loading Qwen ForcedAligner model: {align_model_path}")
        self._align_model = load(align_model_path)

    @property
    def engine_type(self) -> ASREngineType:
        return ASREngineType.QWEN

    def transcribe_and_align(
        self,
        chunk_path: Path,
        language: str,
        offset: float,
    ) -> Tuple[str, List[AlignToken]]:
        from asr import transcribe_chunk, align_chunk, apply_transcript_punctuation_to_tokens

        transcript = transcribe_chunk(self._asr_model, chunk_path, language)
        if not transcript.strip():
            return transcript, []

        tokens = align_chunk(self._align_model, chunk_path, transcript, language, offset=offset)
        tokens = apply_transcript_punctuation_to_tokens(tokens, transcript)
        return transcript, tokens


class ParakeetEngine(BaseASREngine):
    """Parakeet models via parakeet-mlx.

    When *align_model_path* is provided, Parakeet is used only for
    transcription and Qwen ForcedAligner handles timestamp alignment.
    Otherwise Parakeet's built-in word-level timestamps are used.
    """

    def __init__(self, model_path: str, align_model_path: str | None = None):
        from parakeet_mlx import from_pretrained

        print(f"Loading Parakeet model: {model_path}")
        self._model = from_pretrained(model_path)
        self._align_model = None

        if align_model_path:
            from mlx_audio.stt import load as mlx_load

            print(f"Loading Qwen ForcedAligner for Parakeet: {align_model_path}")
            self._align_model = mlx_load(align_model_path)

    @property
    def engine_type(self) -> ASREngineType:
        return ASREngineType.PARAKEET

    def transcribe_and_align(
        self,
        chunk_path: Path,
        language: str,
        offset: float,
    ) -> Tuple[str, List[AlignToken]]:
        result = self._model.transcribe(str(chunk_path))

        transcript = result.text or ""
        if not transcript.strip():
            return transcript, []

        # Use Qwen ForcedAligner when available
        if self._align_model is not None:
            from asr import align_chunk, apply_transcript_punctuation_to_tokens

            tokens = align_chunk(
                self._align_model, chunk_path, transcript, language, offset=offset,
            )
            tokens = apply_transcript_punctuation_to_tokens(tokens, transcript)
            return transcript, tokens

        # Fallback: Parakeet built-in timestamps
        tokens: List[AlignToken] = []
        for tok in result.tokens:
            end = tok.end if tok.end > 0 else tok.start + tok.duration
            if end <= tok.start:
                continue
            tokens.append(AlignToken(
                text=tok.text,
                start=tok.start + offset,
                end=end + offset,
            ))
        return transcript, tokens


_ENGINES = {
    ASREngineType.QWEN: QwenEngine,
    ASREngineType.PARAKEET: ParakeetEngine,
}

_engine_lock = threading.Lock()


def create_engine(
    engine_type: ASREngineType,
    asr_model_path: str | None = None,
    align_model_path: str | None = None,
    parakeet_model_path: str | None = None,
) -> BaseASREngine:
    """Factory: lazily create the requested ASR engine."""
    with _engine_lock:
        cls = _ENGINES.get(engine_type)
        if cls is None:
            raise ValueError(f"Unknown ASR engine type: {engine_type}")

        if engine_type == ASREngineType.QWEN:
            return cls(asr_model_path, align_model_path)
        elif engine_type == ASREngineType.PARAKEET:
            return cls(parakeet_model_path, align_model_path=align_model_path)
        else:
            raise ValueError(f"Unsupported engine: {engine_type}")
