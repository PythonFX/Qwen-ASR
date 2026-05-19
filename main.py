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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from config import ASR_MODEL_PATH, ALIGN_MODEL_PATH, SpeechSegment
from utils import require_ffmpeg, cleanup_after_inference, is_valid_file, remove_files
from audio import (
    video_stem_paths,
    extract_audio_if_needed,
    get_chunk_count,
    split_audio_if_needed,
    split_audio_vad_aware,
    create_single_chunk_wav_if_needed,
    chunk_srt_path,
    chunk_tokens_path,
    final_tokens_path,
)
from asr import process_chunk_if_needed, save_tokens
from subtitle import build_subtitles, write_srt
from pipeline import (
    write_chunk_srt_if_needed,
    build_and_write_part_srt_if_needed,
    load_all_cached_tokens,
    write_all_transcripts,
    merge_chunk_srts_to_final,
)


def _remove_vad_cache(work_dir: Path) -> None:
    from vad import vad_cache_path
    cache = vad_cache_path(work_dir)
    if cache.exists():
        print(f"Remove VAD cache: {cache.name}")
        cache.unlink()


def _get_speech_regions_for_chunk(
    all_segments: List[SpeechSegment],
    chunk_start: float,
    chunk_end: float,
) -> List[SpeechSegment]:
    return [
        SpeechSegment(
            start=max(seg.start, chunk_start),
            end=min(seg.end, chunk_end),
        )
        for seg in all_segments
        if seg.end > chunk_start and seg.start < chunk_end
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="输入视频路径，例如 input.mp4")
    parser.add_argument("-o", "--output", help="输出 SRT 路径，默认和视频同名 .srt")
    parser.add_argument("--language", default="Japanese", help="语言，例如 Chinese / English / Japanese / Cantonese")
    parser.add_argument("--asr-model", default=ASR_MODEL_PATH)
    parser.add_argument("--align-model", default=ALIGN_MODEL_PATH)
    parser.add_argument("--chunk-seconds", type=int, default=60)
    parser.add_argument("--part-chunks", type=int, default=10, help="每多少个 chunks 生成一个中间 part srt")
    parser.add_argument("--workers", type=int, default=2, help="worker 数量，只支持 1 或 2。默认 2；workers=2 时每个线程单独加载一套模型")
    parser.add_argument("--max-chars", type=int, default=42)
    parser.add_argument("--max-duration", type=float, default=6.5)
    parser.add_argument("--pause-threshold", type=float, default=0.65)

    parser.add_argument("--force-wav", action="store_true", help="强制重新生成全量 wav")
    parser.add_argument("--force-chunks", action="store_true", help="强制重新切分 chunk wav")
    parser.add_argument("--force-asr", action="store_true", help="强制重新执行 ASR + ForcedAligner")
    parser.add_argument("--force-parts", action="store_true", help="强制重新生成 part srt")
    parser.add_argument("--force-srt", action="store_true", help="强制重新生成所有 srt（不重新跑 ASR，仅从已有 tokens 重新断句）")
    parser.add_argument("--parallel-first-two-only", action="store_true", help="只并行处理前两个 chunk，然后退出（用于调试）")
    parser.add_argument("--only-chunk-srt", nargs="?", const=1, type=int, help="只生成第 i 个 chunk 的 SRT；i 从 1 开始。不传 i 时默认 1")

    parser.add_argument("--no-vad", action="store_true", help="禁用 VAD 感知切分，使用固定间隔切分")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Silero VAD 语音概率阈值（默认 0.5）")
    parser.add_argument("--vad-min-speech-ms", type=int, default=250, help="最短语音片段 ms（默认 250）")
    parser.add_argument("--vad-min-silence-ms", type=int, default=300, help="最短静音间隔 ms（默认 300）")
    parser.add_argument("--vad-speech-pad-ms", type=int, default=300, help="VAD 语音段前后 padding ms（默认 300）")
    parser.add_argument("--vad-chunk-pad", type=float, default=1.0, help="VAD chunk 前后额外扩展秒数，给 ASR 更多上下文（默认 1.0）")
    parser.add_argument("--max-chunks", type=int, default=0, help="最多处理前 N 个 chunk，0 表示全部（用于调试）")
    parser.add_argument("--sub-display-delay", type=float, default=0.5, help="字幕结束时间向后延长秒数（默认 0.5）")

    args = parser.parse_args()

    if args.chunk_seconds <= 0:
        raise ValueError("--chunk-seconds 必须大于 0")

    if args.part_chunks <= 0:
        raise ValueError("--part-chunks 必须大于 0")

    if args.workers not in (1, 2):
        raise ValueError("--workers 只支持 1 或 2。workers=2 时每个线程会单独加载一套模型。")

    if args.only_chunk_srt is not None and args.only_chunk_srt < 1:
        raise ValueError("--only-chunk-srt 的 i 必须大于等于 1")

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
        _remove_vad_cache(work_dir)

    if args.force_asr:
        remove_files(["chunk_*.txt", "chunk_*.tokens.json", "chunk_*.srt", "all.tokens.json", "part_*.srt"], work_dir)
        _remove_vad_cache(work_dir)

    if args.force_parts:
        remove_files(["chunk_*.srt", "part_*.srt"], work_dir)

    if args.force_srt:
        remove_files(["chunk_*.srt", "part_*.srt"], work_dir)

    extract_audio_if_needed(video_path, full_wav)

    from mlx_audio.stt import load

    # --- VAD pipeline ---
    all_speech_segments: List[SpeechSegment] = []
    if not args.no_vad:
        from vad import (
            load_vad_model,
            detect_speech_segments,
            merge_segments_into_chunks,
            vad_cache_path,
            save_vad_segments,
            load_vad_segments_if_valid,
        )

        vad_cache = vad_cache_path(work_dir)
        cached_segments = load_vad_segments_if_valid(vad_cache)

        if cached_segments is not None:
            print(f"Reuse cached VAD segments: {vad_cache.name} ({len(cached_segments)} segments)")
            all_speech_segments = cached_segments
        else:
            print("Loading Silero VAD model...")
            vad_model = load_vad_model()

            print(f"Running VAD on full audio: {full_wav.name}")
            all_speech_segments = detect_speech_segments(
                full_wav, vad_model,
                threshold=args.vad_threshold,
                min_speech_duration_ms=args.vad_min_speech_ms,
                min_silence_duration_ms=args.vad_min_silence_ms,
                speech_pad_ms=args.vad_speech_pad_ms,
            )
            print(f"VAD detected {len(all_speech_segments)} speech segments")

            save_vad_segments(all_speech_segments, vad_cache)
            print(f"Saved VAD segments cache: {vad_cache.name}")

    if args.only_chunk_srt is not None:
        target_pos = args.only_chunk_srt - 1

        if args.no_vad:
            total_chunks = get_chunk_count(full_wav, args.chunk_seconds)
        else:
            speech_chunks = merge_segments_into_chunks(
                all_speech_segments,
                max_duration=float(args.chunk_seconds),
            )
            total_chunks = len(speech_chunks)

        if target_pos >= total_chunks:
            raise ValueError(f"--only-chunk-srt={args.only_chunk_srt} 超出范围；当前共有 {total_chunks} 个 chunks")

        if args.no_vad:
            chunk_index, chunk_path, offset = create_single_chunk_wav_if_needed(
                audio_path=full_wav,
                work_dir=work_dir,
                chunk_index=target_pos,
                chunk_seconds=args.chunk_seconds,
                force=args.force_chunks,
            )
            speech_regions = None
        else:
            chunk_start, chunk_end = speech_chunks[target_pos]
            speech_chunks_single = [(chunk_start, chunk_end)]
            result = split_audio_vad_aware(full_wav, work_dir, speech_chunks_single, pad_seconds=args.vad_chunk_pad)
            chunk_index, chunk_path, offset = result[0]
            speech_regions = _get_speech_regions_for_chunk(
                all_speech_segments, chunk_start, chunk_end,
            )

        print(
            f"ONLY CHUNK SRT MODE: generate only chunk {args.only_chunk_srt} -> {chunk_srt_path(work_dir, chunk_index)}")

        print(f"Loading ASR model: {args.asr_model}")
        asr_model = load(args.asr_model)

        print(f"Loading ForcedAligner model: {args.align_model}")
        align_model = load(args.align_model)

        try:
            _, tokens = process_chunk_if_needed(
                asr_model=asr_model,
                align_model=align_model,
                chunk_index=chunk_index,
                chunk_path=chunk_path,
                offset=offset,
                language=args.language,
                work_dir=work_dir,
                speech_regions=speech_regions,
            )
            srt_path = write_chunk_srt_if_needed(
                work_dir=work_dir,
                chunk_index=chunk_index,
                tokens=tokens,
                max_chars=args.max_chars,
                max_duration=args.max_duration,
                pause_threshold=args.pause_threshold,
            )
        finally:
            cleanup_after_inference()

        print(f"ONLY CHUNK SRT DONE: {srt_path}")
        return

    # --- Chunk audio ---
    if args.no_vad:
        chunks = split_audio_if_needed(
            full_wav,
            work_dir,
            chunk_seconds=args.chunk_seconds,
        )
    else:
        speech_chunks = merge_segments_into_chunks(
            all_speech_segments,
            max_duration=float(args.chunk_seconds),
        )
        print(f"VAD-aware chunking: {len(speech_chunks)} chunks")
        chunks = split_audio_vad_aware(full_wav, work_dir, speech_chunks, pad_seconds=args.vad_chunk_pad)

    # Build per-chunk speech regions for VAD mode
    if not args.no_vad:
        chunk_speech_map: dict = {}
        for idx, (chunk_start, chunk_end) in enumerate(speech_chunks):
            chunk_speech_map[idx] = _get_speech_regions_for_chunk(
                all_speech_segments, chunk_start, chunk_end,
            )
    else:
        chunk_speech_map = {}

    chunk_count = len(chunks)

    if args.max_chunks > 0 and args.max_chunks < chunk_count:
        print(f"--max-chunks: limiting to {args.max_chunks}/{chunk_count} chunks")
        chunks = chunks[:args.max_chunks]
        chunk_count = args.max_chunks

    if chunk_count == 0:
        raise RuntimeError("没有生成任何 chunk，请检查输入视频或音频。")

    # Check if all chunks are already fully processed
    all_chunks_done = all(
        is_valid_file(chunk_srt_path(work_dir, i), min_size=0)
        and is_valid_file(chunk_tokens_path(work_dir, i), min_size=2)
        for i in range(chunk_count)
    )

    if all_chunks_done:
        print("All chunks already processed, skip model loading and chunk processing")
    elif args.parallel_first_two_only:
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
                        speech_regions=chunk_speech_map.get(chunk_index),
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
        asr_model = None
        align_model = None

        def _ensure_models():
            nonlocal asr_model, align_model
            if asr_model is not None:
                return
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
                _ensure_models()
                try:
                    _, tokens = process_chunk_if_needed(
                        asr_model=asr_model,
                        align_model=align_model,
                        chunk_index=chunk_index,
                        chunk_path=chunk_path,
                        offset=offset,
                        language=args.language,
                        work_dir=work_dir,
                        speech_regions=chunk_speech_map.get(chunk_index),
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

        worker_chunks: List[List[Tuple[int, Path, float]]] = [chunks[0::2], chunks[1::2]]

        def run_worker(worker_id: int, assigned_chunks: List[Tuple[int, Path, float]]) -> List[int]:
            worker_asr_model = None
            worker_align_model = None

            def _ensure_worker_models():
                nonlocal worker_asr_model, worker_align_model
                if worker_asr_model is not None:
                    return
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
                            _ensure_worker_models()
                            _, tokens = process_chunk_if_needed(
                                asr_model=worker_asr_model,
                                align_model=worker_align_model,
                                chunk_index=chunk_index,
                                chunk_path=chunk_path,
                                offset=offset,
                                language=args.language,
                                work_dir=work_dir,
                                speech_regions=chunk_speech_map.get(chunk_index),
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
                if worker_asr_model is not None:
                    del worker_asr_model
                if worker_align_model is not None:
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

    # Merge tokens cache
    all_tokens_path = final_tokens_path(work_dir)
    if is_valid_file(all_tokens_path, min_size=2):
        print(f"Reuse cached all tokens: {all_tokens_path.name}")
    else:
        all_tokens = load_all_cached_tokens(work_dir, chunk_count)
        save_tokens(all_tokens, all_tokens_path)

    # Final SRT cache
    if is_valid_file(output_path, min_size=1):
        print(f"Reuse existing final SRT: {output_path}")
    else:
        # 新策略：直接拼接 chunk SRT，不重新断句
        merge_chunk_srts_to_final(work_dir, chunk_count, output_path, sub_display_delay=args.sub_display_delay)

    # 旧策略：从所有 token 重新断句（已弃用，保留供回撤）
    # subtitles = build_subtitles(
    #     all_tokens,
    #     max_chars=args.max_chars,
    #     max_duration=args.max_duration,
    #     pause_threshold=args.pause_threshold,
    # )
    # write_srt(subtitles, output_path)

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
