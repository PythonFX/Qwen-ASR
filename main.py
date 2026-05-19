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

from config import ASR_MODEL_PATH, ALIGN_MODEL_PATH
from utils import require_ffmpeg, cleanup_after_inference, is_valid_file, remove_files
from audio import (
    video_stem_paths,
    extract_audio_if_needed,
    get_chunk_count,
    split_audio_if_needed,
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
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="输入视频路径，例如 input.mp4")
    parser.add_argument("-o", "--output", help="输出 SRT 路径，默认和视频同名 .srt")
    parser.add_argument("--language", default="Japanese", help="语言，例如 Chinese / English / Japanese / Cantonese")
    parser.add_argument("--asr-model", default=ASR_MODEL_PATH)
    parser.add_argument("--align-model", default=ALIGN_MODEL_PATH)
    parser.add_argument("--chunk-seconds", type=int, default=180)
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

    if args.force_asr:
        remove_files(["chunk_*.txt", "chunk_*.tokens.json", "chunk_*.srt", "all.tokens.json", "part_*.srt"], work_dir)

    if args.force_parts:
        remove_files(["chunk_*.srt", "part_*.srt"], work_dir)

    if args.force_srt:
        remove_files(["chunk_*.srt", "part_*.srt"], work_dir)

    extract_audio_if_needed(video_path, full_wav)

    from mlx_audio.stt import load

    if args.only_chunk_srt is not None:
        target_pos = args.only_chunk_srt - 1
        total_chunks = get_chunk_count(full_wav, args.chunk_seconds)

        if target_pos >= total_chunks:
            raise ValueError(f"--only-chunk-srt={args.only_chunk_srt} 超出范围；当前共有 {total_chunks} 个 chunks")

        chunk_index, chunk_path, offset = create_single_chunk_wav_if_needed(
            audio_path=full_wav,
            work_dir=work_dir,
            chunk_index=target_pos,
            chunk_seconds=args.chunk_seconds,
            force=args.force_chunks,
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

    chunks = split_audio_if_needed(
        full_wav,
        work_dir,
        chunk_seconds=args.chunk_seconds,
    )

    chunk_count = len(chunks)

    if chunk_count == 0:
        raise RuntimeError("没有生成任何 chunk，请检查输入视频或音频。")

    if args.parallel_first_two_only:
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
                try:
                    _, tokens = process_chunk_if_needed(
                        asr_model=asr_model,
                        align_model=align_model,
                        chunk_index=chunk_index,
                        chunk_path=chunk_path,
                        offset=offset,
                        language=args.language,
                        work_dir=work_dir,
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
                            _, tokens = process_chunk_if_needed(
                                asr_model=worker_asr_model,
                                align_model=worker_align_model,
                                chunk_index=chunk_index,
                                chunk_path=chunk_path,
                                offset=offset,
                                language=args.language,
                                work_dir=work_dir,
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
                del worker_asr_model
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
    all_tokens = load_all_cached_tokens(work_dir, chunk_count)
    save_tokens(all_tokens, final_tokens_path(work_dir))

    subtitles = build_subtitles(
        all_tokens,
        max_chars=args.max_chars,
        max_duration=args.max_duration,
        pause_threshold=args.pause_threshold,
    )

    write_srt(subtitles, output_path)

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
