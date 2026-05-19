from __future__ import annotations

from dataclasses import dataclass


ASR_MODEL_PATH = "/Users/vincent/Projects/Model/Qwen3-ASR-1.7B-8bit"
ALIGN_MODEL_PATH = "/Users/vincent/Projects/Model/Qwen3-ForcedAligner-0.6B-8bit"


@dataclass
class AlignToken:
    text: str
    start: float
    end: float


@dataclass
class SpeechSegment:
    start: float
    end: float


@dataclass
class Subtitle:
    start: float
    end: float
    text: str
