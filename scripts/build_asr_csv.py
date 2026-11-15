#!/usr/bin/env python3
"""Build a resumable Vietnamese ASR CSV with faster-whisper."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe media recursively into video_id,start,end,text CSV rows."
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("asr.csv"))
    parser.add_argument("--prefixes", default="", help="Optional comma-separated video prefixes, e.g. L01,L02")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device-count", type=int, default=0, help="GPU workers; <=0 auto-detects with nvidia-smi -L")
    parser.add_argument("--pilot-files", type=int, default=0, help="Limit the sorted input list for timing")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--worker-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def discover_media(input_root: Path, prefixes: tuple[str, ...]) -> list[Path]:
    """Find media, preferring an audio file when the same relative stem has video too."""
    selected: dict[str, Path] = {}
    for path in sorted(input_root.rglob("*"), key=lambda item: str(item).lower()):
        if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        if prefixes and not path.stem.upper().startswith(prefixes):
            continue
        relative_stem = str(path.relative_to(input_root).with_suffix("")).lower()
        previous = selected.get(relative_stem)
        if previous is None or (path.suffix.lower() in AUDIO_EXTENSIONS and previous.suffix.lower() in VIDEO_EXTENSIONS):
            selected[relative_stem] = path
    return sorted(selected.values(), key=lambda item: str(item.relative_to(input_root)).lower())


def detect_device_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            count = sum(1 for line in result.stdout.splitlines() if line.strip().lower().startswith("gpu "))
            if count:
                return count
    except OSError:
        pass
    return 1


def state_dir_for(output: Path) -> Path:
    return output.parent / f".{output.stem}_asr_state"


def read_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
            completed.add(str(value["path"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return completed


def extract_audio(video: Path, destination: Path) -> None:
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}: {video}")


def decode_audio_16k(media: Path):
    """Decode any audio/video file to 16kHz mono float32 numpy array via ffmpeg (avoids PyAV metadata_errors bugs)."""
    import numpy as np

    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(media), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1",
    ]
    result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed with exit code {result.returncode} ({err}): {media}")
    if not result.stdout:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_segments(model: object, pipeline: object | None, audio: object, args: argparse.Namespace):
    options = {
        "beam_size": args.beam_size,
        "language": "vi",
        "vad_filter": True,
        "vad_parameters": {
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        },
        "condition_on_previous_text": False,
    }
    if pipeline is not None:
        try:
            segments, _ = pipeline.transcribe(audio, batch_size=args.batch_size, **options)
            return segments
        except (AttributeError, TypeError, NotImplementedError) as exc:
            print(f"BatchedInferencePipeline unsupported ({exc}); using WhisperModel.transcribe", file=sys.stderr)
    segments, _ = model.transcribe(audio, **options)
    return segments


def run_worker(args: argparse.Namespace) -> int:
    # Imported only in worker mode so --help and parent orchestration need no Whisper install.
    from faster_whisper import WhisperModel

    try:
        from faster_whisper import BatchedInferencePipeline
    except ImportError:
        BatchedInferencePipeline = None

    assert args.worker_id is not None and args.num_workers is not None
    input_root = args.input_root.resolve()
    prefixes = tuple(part.strip().upper() for part in args.prefixes.split(",") if part.strip())
    files = discover_media(input_root, prefixes)
    if args.pilot_files > 0:
        files = files[: args.pilot_files]
    files = [path for index, path in enumerate(files) if index % args.num_workers == args.worker_id]

    state_dir = state_dir_for(args.output.resolve())
    state_dir.mkdir(parents=True, exist_ok=True)
    completed_path = state_dir / f"worker_{args.worker_id}.completed.jsonl"
    shard_path = state_dir / f"worker_{args.worker_id}.csv"
    if not args.resume:
        completed_path.unlink(missing_ok=True)
        shard_path.unlink(missing_ok=True)
    completed = read_completed(completed_path)

    model_name = args.model
    try:
        model = WhisperModel(model_name, device="cuda", device_index=0, compute_type=args.compute_type)
    except ValueError as exc:
        message = str(exc).lower()
        unsupported_name = model_name == "large-v3-turbo" and any(
            marker in message
            for marker in ("unsupported", "not supported", "invalid model", "model size", "expected one of")
        )
        if not unsupported_name:
            raise RuntimeError(f"Unable to load faster-whisper model {model_name!r}: {exc}") from exc
        model_name = "medium"
        print(
            f"Model {args.model!r} is unsupported by this faster-whisper version ({exc}); "
            f"falling back to {model_name!r}.",
            file=sys.stderr,
        )
        model = WhisperModel(model_name, device="cuda", device_index=0, compute_type=args.compute_type)

    pipeline = None
    if BatchedInferencePipeline is not None:
        try:
            pipeline = BatchedInferencePipeline(model=model)
        except (AttributeError, TypeError, NotImplementedError) as exc:
            print(f"BatchedInferencePipeline unsupported ({exc}); using WhisperModel.transcribe", file=sys.stderr)

    shard_exists = shard_path.exists() and shard_path.stat().st_size > 0
    with shard_path.open("a", encoding="utf-8", newline="") as shard_handle, completed_path.open(
        "a", encoding="utf-8", newline=""
    ) as completed_handle, tempfile.TemporaryDirectory(prefix=f"spatial_kis_asr_w{args.worker_id}_") as temp_name:
        writer = csv.writer(shard_handle)
        if not shard_exists:
            writer.writerow(["video_id", "start", "end", "text"])
            shard_handle.flush()
        temp_dir = Path(temp_name)
        consecutive_failures = 0
        succeeded_count = 0
        for sequence, source in enumerate(files, 1):
            relative_key = source.relative_to(input_root).as_posix()
            if relative_key in completed:
                succeeded_count += 1
                continue
            transcribe_path = source
            temporary_wav: Path | None = None
            try:
                audio_pcm = decode_audio_16k(source)
                if len(audio_pcm) == 0:
                    print(f"worker {args.worker_id}: {sequence}/{len(files)} {relative_key} (empty audio)", flush=True)
                    completed_handle.write(json.dumps({"path": relative_key}, ensure_ascii=False) + "\n")
                    completed_handle.flush()
                    succeeded_count += 1
                    consecutive_failures = 0
                    continue
                segments = transcribe_segments(model, pipeline, audio_pcm, args)
                output_rows: list[list[str]] = []
                for segment in segments:
                    text = str(segment.text).strip()
                    if text:
                        output_rows.append(
                            [source.stem, f"{float(segment.start):.3f}", f"{float(segment.end):.3f}", text]
                        )
                writer.writerows(output_rows)
                shard_handle.flush()
                os.fsync(shard_handle.fileno())
                completed_handle.write(json.dumps({"path": relative_key}, ensure_ascii=False) + "\n")
                completed_handle.flush()
                os.fsync(completed_handle.fileno())
                succeeded_count += 1
                consecutive_failures = 0
                print(f"worker {args.worker_id}: {sequence}/{len(files)} {relative_key}", flush=True)
            except Exception as exc:
                consecutive_failures += 1
                print(
                    f"WARNING worker {args.worker_id}: skipping {relative_key}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if succeeded_count == 0 and consecutive_failures >= 5:
                    raise RuntimeError(
                        f"Aborting worker {args.worker_id}: first {consecutive_failures} files all failed ({exc})"
                    ) from exc
            finally:
                if temporary_wav is not None:
                    temporary_wav.unlink(missing_ok=True)
    return 0


def merge_shards(output: Path, state_dir: Path, num_workers: int) -> None:
    rows: dict[tuple[str, str, str, str], tuple[str, float, float, str]] = {}
    for worker_id in range(num_workers):
        shard = state_dir / f"worker_{worker_id}.csv"
        if not shard.exists():
            continue
        with shard.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    parsed = (row["video_id"], float(row["start"]), float(row["end"]), row["text"])
                except (KeyError, TypeError, ValueError):
                    continue
                key = (parsed[0], f"{parsed[1]:.3f}", f"{parsed[2]:.3f}", parsed[3])
                rows[key] = parsed
    ordered = sorted(rows.values(), key=lambda row: (row[0], row[1]))
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_name(f".{output.name}.tmp")
    with temp_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "start", "end", "text"])
        for video_id, start, end, text in ordered:
            writer.writerow([video_id, f"{start:.3f}", f"{end:.3f}", text])
    os.replace(temp_output, output)


def run_parent(args: argparse.Namespace) -> int:
    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Input root does not exist or is not a directory: {input_root}")
    output = args.output.resolve()
    num_workers = args.device_count if args.device_count > 0 else detect_device_count()
    num_workers = max(1, num_workers)
    state_dir = state_dir_for(output)
    if not args.resume and state_dir.exists():
        for stale_file in state_dir.glob("worker_*.csv"):
            stale_file.unlink()
        for stale_file in state_dir.glob("worker_*.completed.jsonl"):
            stale_file.unlink()
    state_dir.mkdir(parents=True, exist_ok=True)

    processes: list[subprocess.Popen] = []
    for worker_id in range(num_workers):
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--input-root", str(input_root), "--output", str(output),
            "--prefixes", args.prefixes, "--model", args.model,
            "--compute-type", args.compute_type, "--beam-size", str(args.beam_size),
            "--batch-size", str(args.batch_size), "--device-count", str(num_workers),
            "--pilot-files", str(args.pilot_files), "--worker-id", str(worker_id),
            "--num-workers", str(num_workers), "--resume" if args.resume else "--no-resume",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(worker_id)
        processes.append(subprocess.Popen(command, env=environment))
    failures = [process.wait() for process in processes]
    if any(code != 0 for code in failures):
        raise SystemExit(f"ASR worker failure(s): {failures}")
    merge_shards(output, state_dir, num_workers)
    print(f"Wrote {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_id is not None:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
