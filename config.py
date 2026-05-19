from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ASREngineType(str, Enum):
    QWEN = "qwen"
    PARAKEET = "parakeet"


ASR_MODEL_PATH = "/Users/vincent/Projects/Model/Qwen3-ASR-1.7B-8bit"
ALIGN_MODEL_PATH = "/Users/vincent/Projects/Model/Qwen3-ForcedAligner-0.6B-8bit"
PARAKEET_MODEL_PATH = "/Users/vincent/Projects/Model/parakeet-tdt_ctc-0.6b-ja"


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
