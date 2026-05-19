#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Business logic for ASR SRT generation, decoupled from any UI framework.

UI layers (PyQt, CLI, etc.) call ASRService methods and receive progress
via callbacks.  No Qt / GUI dependency lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from config import (
    ALIGN_MODEL_PATH,
    ASR_MODEL_PATH,
    ASREngineType,
    PARAKEET_MODEL_PATH,
    SpeechSegment,
)
from utils import cleanup_after_inference, is_valid_file, require_ffmpeg
from asr_engine import create_engine
from audio import (
    chunk_srt_path,
    chunk_tokens_path,
    extract_audio_if_needed,
    final_tokens_path,
    split_audio_if_needed,
    split_audio_vad_aware,
    video_stem_paths,
)
from asr import process_chunk_if_needed, save_tokens
from pipeline import (
    build_and_write_part_srt_if_needed,
    load_all_cached_tokens,
    merge_chunk_srts_to_final,
    write_all_transcripts,
    write_chunk_srt_if_needed,
)


VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mkv", ".mov", ".wmv",
    ".flv", ".ts", ".m2ts", ".webm",
}


class ASRService:
    """Pure business logic for SRT generation."""

    def __init__(self) -> None:
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def find_video_by_code(self, root_dir: str, code: str) -> Tuple[Path, Path]:
        """Find the video file in a subfolder matching *code* under *root_dir*.

        Matching: strip hyphens, upper-case both, check folder name contains
        the code.

        Returns (video_path, srt_output_path).
        """
        root = Path(root_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"根目录不存在: {root}")

        code_norm = code.strip().upper().replace("-", "")
        if not code_norm:
            raise ValueError("番号不能为空")

        matched: Optional[Path] = None
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            if code_norm in entry.name.upper().replace("-", ""):
                matched = entry
                break

        if matched is None:
            raise FileNotFoundError(
                f"在 {root} 下未找到包含 '{code}' 的文件夹"
            )

        videos = sorted(
            f
            for f in matched.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )

        if not videos:
            raise FileNotFoundError(
                f"文件夹 {matched.name} 中未找到视频文件"
            )
        if len(videos) > 1:
            raise RuntimeError(
                f"文件夹 {matched.name} 中找到多个视频: "
                f"{[v.name for v in videos]}"
            )

        video = videos[0]
        if "-C." in video.name:
            raise ValueError(
                f"视频 {video.name} 自带字幕（文件名包含 \"-C.\"），无需生成"
            )

        return video, video.with_suffix(".srt")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def generate_srt(
        self,
        video_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        *,
        language: str = "Japanese",
        engine_type: ASREngineType = ASREngineType.PARAKEET,
        workers: int = 1,
        chunk_seconds: int = 60,
        part_chunks: int = 10,
        max_chars: int = 42,
        max_duration: float = 6.5,
        pause_threshold: float = 0.65,
        no_vad: bool = False,
        sub_display_delay: float = 0.5,
    ) -> None:
        """Run the full ASR -> SRT pipeline with progress reporting.

        *progress_callback(stage, percent)*: percent is 0 -- 100.
        """
        self._cancel_requested = False
        _cb = progress_callback or (lambda _s, _p: None)

        require_ffmpeg()
        video_path = Path(video_path).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()

        if not video_path.exists():
            raise FileNotFoundError(video_path)

        full_wav, work_dir = video_stem_paths(video_path)
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Extract audio  (0-5 %)
        _cb("提取音频", 2)
        extract_audio_if_needed(video_path, full_wav)

        # 2. VAD  (5-15 %)
        all_speech_segments: List[SpeechSegment] = []
        speech_chunks: List[Tuple[float, float]] = []

        if not no_vad:
            from vad import (
                detect_speech_segments,
                load_vad_model,
                load_vad_segments_if_valid,
                merge_segments_into_chunks,
                save_vad_segments,
                vad_cache_path,
            )

            vad_cache = vad_cache_path(work_dir)
            cached = load_vad_segments_if_valid(vad_cache)

            if cached is not None:
                all_speech_segments = cached
            else:
                _cb("语音检测", 8)
                vad_model = load_vad_model()
                all_speech_segments = detect_speech_segments(
                    full_wav, vad_model,
                )
                save_vad_segments(all_speech_segments, vad_cache)

            speech_chunks = merge_segments_into_chunks(
                all_speech_segments,
                max_duration=float(chunk_seconds),
            )

        # 3. Chunking  (15-20 %)
        _cb("切分音频", 18)

        if no_vad:
            chunks = split_audio_if_needed(
                full_wav, work_dir, chunk_seconds=chunk_seconds,
            )
        else:
            chunks = split_audio_vad_aware(
                full_wav, work_dir, speech_chunks,
            )

        # Per-chunk speech regions
        chunk_speech_map: dict = {}
        if not no_vad:
            for idx, (cs, ce) in enumerate(speech_chunks):
                chunk_speech_map[idx] = [
                    SpeechSegment(start=max(s.start, cs), end=min(s.end, ce))
                    for s in all_speech_segments
                    if s.end > cs and s.start < ce
                ]

        chunk_count = len(chunks)
        if chunk_count == 0:
            raise RuntimeError("没有生成任何 chunk，请检查输入视频或音频。")

        # Quick check: already fully processed?
        all_done = all(
            is_valid_file(chunk_srt_path(work_dir, i), min_size=0)
            and is_valid_file(chunk_tokens_path(work_dir, i), min_size=2)
            for i in range(chunk_count)
        )

        # 4. Process chunks  (20-95 %)
        if not all_done:
            engine = None

            def _ensure_engine():
                nonlocal engine
                if engine is None:
                    engine = create_engine(
                        engine_type,
                        asr_model_path=ASR_MODEL_PATH,
                        align_model_path=ALIGN_MODEL_PATH,
                        parakeet_model_path=PARAKEET_MODEL_PATH,
                    )

            completed: set = set()

            for done_count, (ci, cpath, offset) in enumerate(chunks):
                if self._cancel_requested:
                    _cb("已取消", 0)
                    return

                pct = 20 + (done_count / chunk_count) * 75
                _cb(f"处理字幕 [{done_count + 1}/{chunk_count}]", pct)

                srt_p = chunk_srt_path(work_dir, ci)
                tok_p = chunk_tokens_path(work_dir, ci)

                if is_valid_file(srt_p, min_size=0) and is_valid_file(
                    tok_p, min_size=2
                ):
                    completed.add(ci)
                else:
                    _ensure_engine()
                    try:
                        _, tokens = process_chunk_if_needed(
                            engine=engine,
                            chunk_index=ci,
                            chunk_path=cpath,
                            offset=offset,
                            language=language,
                            work_dir=work_dir,
                            speech_regions=chunk_speech_map.get(ci),
                        )
                        write_chunk_srt_if_needed(
                            work_dir=work_dir,
                            chunk_index=ci,
                            tokens=tokens,
                            max_chars=max_chars,
                            max_duration=max_duration,
                            pause_threshold=pause_threshold,
                        )
                        completed.add(ci)
                    finally:
                        cleanup_after_inference()

                # Write part-SRT when a group is complete
                gs = (ci // part_chunks) * part_chunks
                ge = min(gs + part_chunks - 1, chunk_count - 1)
                if all(i in completed for i in range(gs, ge + 1)):
                    build_and_write_part_srt_if_needed(
                        work_dir=work_dir,
                        chunk_start=gs,
                        chunk_end=ge,
                        max_chars=max_chars,
                        max_duration=max_duration,
                        pause_threshold=pause_threshold,
                    )

        # 5. Merge  (95-100 %)
        _cb("合并字幕", 96)

        all_tokens_path = final_tokens_path(work_dir)
        if not is_valid_file(all_tokens_path, min_size=2):
            all_tokens = load_all_cached_tokens(work_dir, chunk_count)
            save_tokens(all_tokens, all_tokens_path)

        if not is_valid_file(output_path, min_size=1):
            merge_chunk_srts_to_final(
                work_dir,
                chunk_count,
                output_path,
                sub_display_delay=sub_display_delay,
            )

        transcript_path = output_path.with_suffix(".txt")
        write_all_transcripts(work_dir, chunk_count, transcript_path)

        # 6. Cleanup work dir (keep mono wav)
        if work_dir.exists():
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

        _cb("完成", 100)
