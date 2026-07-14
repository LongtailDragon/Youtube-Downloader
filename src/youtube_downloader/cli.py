from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL


DEFAULT_MODEL_NAME = "faster-whisper-base.en"


def default_model_dirs() -> tuple[Path, ...]:
    dirs = [
        Path.home() / "Models",
        Path.home() / "models",
    ]
    env_dir = os.environ.get("YTDL_MODEL_DIR")
    if env_dir:
        dirs.insert(0, Path(env_dir))

    if os.name == "nt":
        dirs.append(Path("C:/models"))
    else:
        dirs.extend([Path("/opt/models"), Path("/usr/local/share/models")])

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in dirs:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return tuple(unique)


DEFAULT_MODEL_DIRS = default_model_dirs()
DEFAULT_OPENAI_WHISPER_MODEL = "base.en"
DEFAULT_OPENAI_WHISPER_MODEL_DIR = Path.home() / "Models" / "openai-whisper"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_WHISPERX_MODEL = "small"

_WHISPERX_ASR_MODELS: dict[tuple[str, str, str], object] = {}
_WHISPERX_ALIGN_MODELS: dict[tuple[str, str], tuple[object, object]] = {}
_WHISPERX_DIARIZATION_MODELS: dict[tuple[str, str], object] = {}


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class DownloadedMedia:
    source_path: Path
    title: str
    video_id: str
    webpage_url: str


@dataclass(frozen=True)
class PlaylistInfo:
    source_url: str
    title: str
    video_urls: list[str]


def normalize_argv(argv: list[str] | None) -> list[str] | None:
    if argv is None:
        return None
    normalized = list(argv)
    for idx in range(len(normalized) - 1):
        if normalized[idx].lower() == "format":
            normalized[idx] = "--format"
            break
    return normalized


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = normalize_argv(argv)
    parser = argparse.ArgumentParser(
        prog="ytdl-local",
        description=(
            "Download a YouTube URL and optionally convert it to MKV, MP3, "
            "or transcribe it to TXT using local tools."
        ),
    )
    parser.add_argument("url", help="YouTube URL to download")
    parser.add_argument(
        "-f",
        "--format",
        dest="formats",
        action="append",
        choices=("original", "mkv", "mp3", "txt", "txt-diarize", "summary"),
        default=None,
        help="Output format. Repeat for multiple outputs. Default: original",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("downloads"),
        help="Directory for downloaded/generated files. Default: downloads",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional local faster-whisper model path/name for --format txt. "
            "By default TXT uses the global whisper command with the local base.en model. "
            "If --model or YTDL_WHISPER_MODEL is set, faster-whisper is used instead."
        ),
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language hint for transcription, such as en.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Transcription device for faster-whisper. Default: auto",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help=(
            "faster-whisper compute type, e.g. int8, float16, int8_float16. "
            "Default lets faster-whisper choose."
        ),
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help=(
            "Enable speaker diarization for transcript output using WhisperX. "
            "Equivalent to requesting --format txt-diarize."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help=(
            "Hugging Face token used by WhisperX diarization (pyannote). "
            "Defaults to HF_TOKEN or HUGGINGFACE_TOKEN."
        ),
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Optional minimum number of speakers for diarization.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Optional maximum number of speakers for diarization.",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Download audio only before conversion/transcription.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep temporary source media when only converted outputs were requested.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print machine-readable JSON result.",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("YTD_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        help=(
            "Local Ollama model used for --format summary. "
            f"Default: YTD_OLLAMA_MODEL or {DEFAULT_OLLAMA_MODEL}."
        ),
    )
    return parser.parse_args(argv)


def require_command(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise ToolError(f"Required command not found on PATH: {name}")
    return exe


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise ToolError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def safe_stem(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in " ._-()[]" else "_" for ch in value).strip()
    cleaned = " ".join(cleaned.split())
    return cleaned[:150] or "youtube-video"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for idx in range(1, 1000):
        candidate = parent / f"{stem} ({idx}){suffix}"
        if not candidate.exists():
            return candidate
    raise ToolError(f"Could not find unused filename for {path}")


def find_downloaded_file(info: dict, before: set[Path], download_dir: Path) -> Path:
    requested = info.get("requested_downloads") or []
    for item in requested:
        filepath = item.get("filepath") or item.get("filename")
        if filepath and Path(filepath).exists():
            return Path(filepath).resolve()

    direct = info.get("filepath") or info.get("_filename")
    if direct and Path(direct).exists():
        return Path(direct).resolve()

    after = {p.resolve() for p in download_dir.glob("**/*") if p.is_file()}
    created = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if created:
        return created[0]

    raise ToolError("Download finished, but the downloaded file could not be located.")


def download_youtube(url: str, output_dir: Path, audio_only: bool) -> DownloadedMedia:
    output_dir.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in output_dir.glob("**/*") if p.is_file()}
    fmt = "bestaudio/best" if audio_only else "bv*+ba/best"
    ydl_opts = {
        "format": fmt,
        "outtmpl": str(output_dir / "%(title).150B [%(id)s].%(ext)s"),
        "restrictfilenames": False,
        "windowsfilenames": True,
        "merge_output_format": "mkv",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    source = find_downloaded_file(info, before, output_dir)
    return DownloadedMedia(
        source_path=source,
        title=info.get("title") or source.stem,
        video_id=info.get("id") or "unknown-id",
        webpage_url=info.get("webpage_url") or url,
    )


def is_playlist_url(url: str) -> bool:
    parsed = urlparse(url)
    if "youtube.com" not in parsed.netloc and "youtu.be" not in parsed.netloc:
        return False
    params = parse_qs(parsed.query)
    return bool(params.get("list"))


def extract_playlist_info(url: str) -> PlaylistInfo | None:
    if not is_playlist_url(url):
        return None

    ydl_opts = {
        "extract_flat": True,
        "skip_download": True,
        "noplaylist": False,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if not isinstance(info, dict) or info.get("_type") != "playlist":
        return None

    entries = info.get("entries") or []
    video_urls: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_url = entry.get("url") or entry.get("webpage_url")
        if isinstance(entry_url, str) and entry_url.startswith("http"):
            video_urls.append(entry_url)
            continue
        video_id = entry.get("id")
        if video_id:
            video_urls.append(f"https://www.youtube.com/watch?v={video_id}")

    if not video_urls:
        return None

    return PlaylistInfo(
        source_url=url,
        title=info.get("title") or "Untitled Playlist",
        video_urls=video_urls,
    )


def convert_to_mkv(source: Path, output_dir: Path, base_stem: str) -> Path:
    require_command("ffmpeg")
    target = unique_path(output_dir / f"{base_stem}.mkv")
    if source.suffix.lower() == ".mkv":
        if source.resolve() == target.resolve():
            return source
        shutil.copy2(source, target)
        return target
    run(["ffmpeg", "-hide_banner", "-y", "-i", str(source), "-map", "0", "-c", "copy", str(target)])
    return target


def convert_to_mp3(source: Path, output_dir: Path, base_stem: str) -> Path:
    require_command("ffmpeg")
    target = unique_path(output_dir / f"{base_stem}.mp3")
    run([
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(target),
    ])
    return target


def extract_temp_wav(source: Path) -> Path:
    require_command("ffmpeg")
    tmp_dir = Path(tempfile.mkdtemp(prefix="ytdl-local-audio-"))
    wav = tmp_dir / "audio.wav"
    run([
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav),
    ])
    return wav


def resolve_whisper_model(model: str | None) -> str:
    if model:
        return model

    env_model = os.environ.get("YTDL_WHISPER_MODEL")
    if env_model:
        return env_model

    model_dirs = list(DEFAULT_MODEL_DIRS)
    env_dir = os.environ.get("YTDL_MODEL_DIR")
    if env_dir:
        env_path = Path(env_dir)
        model_dirs = [env_path] + [root for root in model_dirs if root != env_path]

    for model_root in model_dirs:
        candidate = model_root / DEFAULT_MODEL_NAME
        if candidate.exists():
            return str(candidate)

    searched = ", ".join(str(root / DEFAULT_MODEL_NAME) for root in model_dirs)
    raise ToolError(
        "No local faster-whisper model was found. Install the default model at "
        f"{model_dirs[0] / DEFAULT_MODEL_NAME}, set YTDL_WHISPER_MODEL, "
        "or pass --model explicitly. Searched: "
        f"{searched}"
    )


def select_transcription_backend(model: str | None) -> str:
    if model or os.environ.get("YTDL_WHISPER_MODEL"):
        return "faster-whisper"
    if shutil.which("whisper"):
        return "whisper-cli"
    try:
        resolve_whisper_model(None)
    except ToolError as exc:
        raise ToolError(
            "No global whisper command was found, and no faster-whisper model was configured. "
            "Install global Whisper with: uv tool install openai-whisper. "
            "Then pre-download a local model, or set YTDL_WHISPER_MODEL / pass --model. "
            f"Details: {exc}"
        ) from exc
    return "faster-whisper"


def transcribe_with_whisper_cli(
    source: Path,
    output_dir: Path,
    base_stem: str,
    language: str | None,
    device: str,
) -> Path:
    whisper_exe = require_command("whisper")
    target = unique_path(output_dir / f"{base_stem}.txt")
    wav = extract_temp_wav(source)
    whisper_output_dir = Path(tempfile.mkdtemp(prefix="ytdl-local-whisper-output-"))
    try:
        cmd = [
            whisper_exe,
            str(wav),
            "--model",
            DEFAULT_OPENAI_WHISPER_MODEL,
            "--model_dir",
            str(DEFAULT_OPENAI_WHISPER_MODEL_DIR),
            "--output_dir",
            str(whisper_output_dir),
            "--output_format",
            "txt",
            "--verbose",
            "False",
            "--fp16",
            "False",
        ]
        if device != "auto":
            cmd.extend(["--device", device])
        if language:
            cmd.extend(["--language", language])
        run(cmd)
        generated = whisper_output_dir / f"{wav.stem}.txt"
        if not generated.exists():
            txt_outputs = sorted(whisper_output_dir.glob("*.txt"))
            if not txt_outputs:
                raise ToolError("Global whisper command finished, but no TXT output was created.")
            generated = txt_outputs[0]
        shutil.copy2(generated, target)
    finally:
        shutil.rmtree(wav.parent, ignore_errors=True)
        shutil.rmtree(whisper_output_dir, ignore_errors=True)
    return target


def transcribe_with_faster_whisper(
    source: Path,
    output_dir: Path,
    base_stem: str,
    model: str | None,
    language: str | None,
    device: str,
    compute_type: str,
) -> Path:
    model = resolve_whisper_model(model)

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - install issue only
        raise ToolError("faster-whisper is not installed. Run: uv sync") from exc

    target = unique_path(output_dir / f"{base_stem}.txt")
    wav = extract_temp_wav(source)
    try:
        kwargs = {"device": device}
        if compute_type != "default":
            kwargs["compute_type"] = compute_type
        whisper = WhisperModel(model, **kwargs)
        segments, info = whisper.transcribe(str(wav), language=language, vad_filter=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"Title: {base_stem}\n")
            handle.write(f"Detected language: {info.language} ({info.language_probability:.2f})\n")
            handle.write("\n")
            for segment in segments:
                start = format_timestamp(segment.start)
                end = format_timestamp(segment.end)
                handle.write(f"[{start} --> {end}] {segment.text.strip()}\n")
    finally:
        shutil.rmtree(wav.parent, ignore_errors=True)
    return target


def resolve_device_for_whisperx(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def select_whisperx_compute_type(device: str, compute_type: str) -> str:
    if compute_type != "default":
        return compute_type

    preferred = (
        ("float16", "int8_float16", "int8_float32", "float32", "int8")
        if device == "cuda"
        else ("int8", "int8_float32", "int16", "float32")
    )

    try:
        import ctranslate2

        supported = ctranslate2.get_supported_compute_types(device)
    except Exception:
        return "float16" if device == "cuda" else "int8"

    for candidate in preferred:
        if candidate in supported:
            return candidate

    return "float32"


def get_whisperx_asr_model(whisperx, model: str, device: str, compute_type: str):
    cache_key = (model, device, compute_type)
    if cache_key not in _WHISPERX_ASR_MODELS:
        _WHISPERX_ASR_MODELS[cache_key] = whisperx.load_model(model, device, compute_type=compute_type)
    return _WHISPERX_ASR_MODELS[cache_key]


def get_whisperx_align_model(whisperx, language: str, device: str):
    cache_key = (language, device)
    if cache_key not in _WHISPERX_ALIGN_MODELS:
        _WHISPERX_ALIGN_MODELS[cache_key] = whisperx.load_align_model(language_code=language, device=device)
    return _WHISPERX_ALIGN_MODELS[cache_key]


def get_whisperx_diarization_model(diarization_pipeline, hf_token: str, device: str):
    cache_key = (hf_token, device)
    if cache_key not in _WHISPERX_DIARIZATION_MODELS:
        _WHISPERX_DIARIZATION_MODELS[cache_key] = diarization_pipeline(token=hf_token, device=device)
    return _WHISPERX_DIARIZATION_MODELS[cache_key]


def transcribe_with_whisperx_diarization(
    source: Path,
    output_dir: Path,
    base_stem: str,
    model: str | None,
    language: str | None,
    device: str,
    compute_type: str,
    hf_token: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> Path:
    if not hf_token:
        raise ToolError(
            "WhisperX diarization requires a Hugging Face token. "
            "Set HF_TOKEN (or HUGGINGFACE_TOKEN), or pass --hf-token."
        )

    try:
        import whisperx
        from whisperx.diarize import DiarizationPipeline
    except ImportError as exc:  # pragma: no cover - install issue only
        raise ToolError(
            "WhisperX is not installed. Install diarization extras with: uv sync --extra diarization"
        ) from exc

    target = unique_path(output_dir / f"{base_stem}.txt")
    whisperx_device = resolve_device_for_whisperx(device)
    whisperx_model = model or DEFAULT_WHISPERX_MODEL
    whisperx_compute_type = select_whisperx_compute_type(whisperx_device, compute_type)

    audio = whisperx.load_audio(str(source))

    asr_model = get_whisperx_asr_model(whisperx, whisperx_model, whisperx_device, whisperx_compute_type)
    result = asr_model.transcribe(audio, language=language)

    align_model, metadata = get_whisperx_align_model(whisperx, result["language"], whisperx_device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, whisperx_device, return_char_alignments=False)

    diarize_model = get_whisperx_diarization_model(DiarizationPipeline, hf_token, whisperx_device)
    diarize_kwargs: dict[str, int] = {}
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    diarize_segments = diarize_model(audio, **diarize_kwargs)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"Title: {base_stem}\n")
        handle.write(f"Detected language: {result.get('language', 'unknown')}\n")
        handle.write("Diarization: WhisperX\n")
        if min_speakers is not None or max_speakers is not None:
            handle.write(f"Speaker bounds: min={min_speakers}, max={max_speakers}\n")
        handle.write("\n")
        for segment in result.get("segments", []):
            start = format_timestamp(float(segment.get("start", 0.0)))
            end = format_timestamp(float(segment.get("end", 0.0)))
            speaker = segment.get("speaker") or "UNKNOWN"
            text = str(segment.get("text", "")).strip()
            if text:
                handle.write(f"[{start} --> {end}] {speaker}: {text}\n")
    return target


def transcribe_to_txt(
    source: Path,
    output_dir: Path,
    base_stem: str,
    model: str | None,
    language: str | None,
    device: str,
    compute_type: str,
    diarize: bool,
    hf_token: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> Path:
    if diarize:
        return transcribe_with_whisperx_diarization(
            source,
            output_dir,
            base_stem,
            model,
            language,
            device,
            compute_type,
            hf_token,
            min_speakers,
            max_speakers,
        )
    backend = select_transcription_backend(model)
    if backend == "whisper-cli":
        return transcribe_with_whisper_cli(source, output_dir, base_stem, language, device)
    return transcribe_with_faster_whisper(source, output_dir, base_stem, model, language, device, compute_type)


def build_summary_prompt(transcript: str) -> str:
    return (
        "You are summarizing a YouTube transcript for a busy user.\n"
        "Write a clear, useful summary in this format:\n"
        "1. One-paragraph overview\n"
        "2. Key points as bullets\n"
        "3. Action items or takeaways, if any\n\n"
        "Transcript:\n"
        f"{transcript}"
    )


def stream_ollama_summary(transcript_path: Path, model: str) -> None:
    require_command("ollama")
    transcript = transcript_path.read_text(encoding="utf-8", errors="replace").strip()
    if not transcript:
        raise ToolError(f"Transcript is empty, so there is nothing to summarize: {transcript_path}")

    payload = {
        "model": model,
        "prompt": build_summary_prompt(transcript),
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"\nSummary from local Ollama model ({model}):\n")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    print(token, end="", flush=True)
                if chunk.get("done"):
                    break
    except urllib.error.URLError as exc:
        raise ToolError(
            "Could not reach Ollama at http://127.0.0.1:11434. "
            "Start Ollama and make sure the requested model is installed."
        ) from exc
    print()


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def should_remove_source(formats: Iterable[str], keep_intermediate: bool) -> bool:
    requested = set(formats)
    return not keep_intermediate and "original" not in requested and bool(requested & {"mkv", "mp3", "txt", "summary"})


def normalize_formats_and_diarize(formats: list[str] | None, diarize: bool) -> tuple[list[str], bool]:
    requested = formats or ["original"]
    diarize_requested = diarize or "txt-diarize" in requested
    normalized = ["txt" if item == "txt-diarize" else item for item in requested]
    return normalized, diarize_requested


def validate_args(args: argparse.Namespace) -> None:
    if args.min_speakers is not None and args.min_speakers < 1:
        raise ToolError("--min-speakers must be >= 1")
    if args.max_speakers is not None and args.max_speakers < 1:
        raise ToolError("--max-speakers must be >= 1")
    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        raise ToolError("--min-speakers cannot be greater than --max-speakers")

    normalized_formats, diarize_requested = normalize_formats_and_diarize(args.formats, args.diarize)
    if diarize_requested and not any(fmt in {"txt", "summary"} for fmt in normalized_formats):
        raise ToolError("Diarization requires transcript output. Use --format txt, txt-diarize, or summary.")


def build_single_output(args: argparse.Namespace, url: str) -> dict:
    formats, diarize_requested = normalize_formats_and_diarize(args.formats, args.diarize)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    needs_audio_only = args.audio_only or set(formats).issubset({"mp3", "txt", "summary"})
    media = download_youtube(url, args.output_dir, audio_only=needs_audio_only)
    base_stem = safe_stem(f"{media.title} [{media.video_id}]")

    outputs: dict[str, str] = {}
    if "original" in formats:
        outputs["original"] = str(media.source_path)
    if "mkv" in formats:
        outputs["mkv"] = str(convert_to_mkv(media.source_path, args.output_dir, base_stem))
    if "mp3" in formats:
        outputs["mp3"] = str(convert_to_mp3(media.source_path, args.output_dir, base_stem))
    if "txt" in formats or "summary" in formats:
        txt_path = transcribe_to_txt(
            media.source_path,
            args.output_dir,
            base_stem,
            args.model,
            args.language,
            args.device,
            args.compute_type,
            diarize_requested,
            args.hf_token,
            args.min_speakers,
            args.max_speakers,
        )
        outputs["txt"] = str(txt_path)
        if "summary" in formats:
            stream_ollama_summary(txt_path, args.ollama_model)
            outputs["summary"] = "streamed-to-terminal"

    if should_remove_source(formats, args.keep_intermediate):
        try:
            media.source_path.unlink()
        except OSError:
            pass

    return {
        "title": media.title,
        "video_id": media.video_id,
        "url": media.webpage_url,
        "outputs": outputs,
    }


def build_outputs(args: argparse.Namespace) -> dict:
    validate_args(args)
    playlist = extract_playlist_info(args.url)
    if playlist is None:
        return build_single_output(args, args.url)

    items: list[dict] = []
    total = len(playlist.video_urls)
    for idx, video_url in enumerate(playlist.video_urls, start=1):
        print(f"[{idx}/{total}] Processing: {video_url}")
        item_result = build_single_output(args, video_url)
        items.append(item_result)

    return {
        "playlist": {
            "title": playlist.title,
            "url": playlist.source_url,
            "count": total,
        },
        "items": items,
    }


def print_command_help() -> None:
    print("YouTube Downloader (ytd)")
    print()
    print("Commands:")
    print('  ytd URL                         Download best available video/audio')
    print('  ytd URL --format mkv            Download and convert/remux to MKV')
    print('  ytd URL --format mp3            Download and convert audio to MP3')
    print('  ytd URL --format txt            Download and transcribe locally to TXT')
    print('  ytd URL --format txt-diarize    Download and transcribe with speaker diarization')
    print('  ytd URL --format txt --diarize  Same as: ytd URL --format txt-diarize')
    print('  ytd URL format summary          Save TXT transcript, then stream local Ollama summary')
    print('  ytd URL --format summary        Same as: ytd URL format summary')
    print('  ytd URL --format mkv --format mp3 --format txt')
    print('                                  Create multiple outputs')
    print('  ytd help                        Show this command list')
    print('  ytd --help                      Show detailed options')
    print()
    print("Useful options:")
    print('  --output-dir PATH               Save files somewhere else')
    print('  --language en                   Hint transcription language')
    print('  --device cpu|cuda|auto          Transcription device')
    print('  --diarize                       Enable WhisperX speaker diarization')
    print('  --keep-intermediate             Keep downloaded source media')


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv == ["help"]:
        print_command_help()
        return 0

    args = parse_args(argv)
    try:
        result = build_outputs(args)
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.print_json:
        print(json.dumps(result, indent=2))
    else:
        if "items" in result:
            playlist = result["playlist"]
            print("Done.")
            print(f"Playlist: {playlist['title']}")
            print(f"Videos processed: {playlist['count']}")
            for item in result["items"]:
                print(f"Title: {item['title']}")
                for label, path in item["outputs"].items():
                    print(f"{label}: {path}")
        else:
            print("Done.")
            print(f"Title: {result['title']}")
            for label, path in result["outputs"].items():
                print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
