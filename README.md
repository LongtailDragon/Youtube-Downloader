# Youtube Downloader

Local command-line tool for downloading a YouTube URL, converting it to MKV or MP3, or transcribing audio to TXT using local tools only.

## Requirements

- Python 3.10+
- uv
- ffmpeg available on PATH
- global `whisper` command for default TXT transcription

This project uses open-source local tooling:

- `yt-dlp` for YouTube downloads
- `ffmpeg` for MKV/MP3 conversion
- global OpenAI Whisper CLI for default local transcription
- optional `faster-whisper` if you pass `--model` or set `YTDL_WHISPER_MODEL`

No cloud transcription API is used.

Current local default installed on this PC:

- Whisper command: `C:\Users\Admin\.local\bin\whisper`
- OpenAI Whisper model: `C:\Users\Admin\Models\openai-whisper\base.en.pt`
- Optional faster-whisper model: `C:\Users\Admin\Models\faster-whisper-base.en`

## Install

From this folder:

```bash
uv sync
```

Global Whisper install, if needed on another machine:

```bash
uv tool install openai-whisper
```

Pre-download the default local model, if needed on another machine:

```bash
whisper C:\path\to\short-test-audio.wav --model base.en --model_dir C:\Users\Admin\Models\openai-whisper --output_format txt --fp16 False --device cpu
```

## Fresh install notes for another system

These notes are for future me / another assistant reinstalling this tool on a new Windows machine.

### 1. Install system dependencies

Install these first and make sure each command is available on PATH:

- Python 3.10 or newer
- `uv`
- `ffmpeg`
- `ollama`, only needed for `format summary`
- Git, if cloning from GitHub

Quick checks:

```bash
python --version
uv --version
ffmpeg -version
ollama --version
```

On Windows, practical install options are:

```powershell
winget install --id astral-sh.uv
winget install --id Gyan.FFmpeg
winget install --id Ollama.Ollama
```

If the machine does not already have Python, install Python 3.11+ from python.org or winget.

### 2. Clone or copy this repo

Expected path on this PC is:

```text
C:\wamp64\www\Youtube-Downloader
```

But the repo can live anywhere. From the repo folder, install the project dependencies:

```bash
uv sync
```

This installs the Python dependencies from `pyproject.toml`, including:

- `yt-dlp`
- `faster-whisper`
- dev dependency: `pytest`

### 3. Install the global `ytd` command

From the repo folder:

```bash
uv tool install --editable . --force
```

Verify it works from outside the repo:

```bash
ytd help
```

If `ytd` is not found, add uv's tool bin directory to PATH. On this Windows PC it is:

```text
C:\Users\Admin\.local\bin
```

On another Windows user account, replace `Admin` with that username.

### 4. Install local Whisper transcription

The default TXT transcription path uses the global OpenAI Whisper CLI.

Install it:

```bash
uv tool install openai-whisper
```

Verify:

```bash
whisper --help
```

Pre-download the default `base.en` model so transcription works locally without needing a model path. Create or use a tiny local audio file and run:

```bash
whisper C:\path\to\short-test-audio.wav --model base.en --model_dir C:\Users\Admin\Models\openai-whisper --output_format txt --fp16 False --device cpu
```

Expected model file after download:

```text
C:\Users\Admin\Models\openai-whisper\base.en.pt
```

On another Windows user account, replace `Admin` with that username. If changing this location, also update `DEFAULT_OPENAI_WHISPER_MODEL_DIR` in `src/youtube_downloader/cli.py`.

### 5. Optional faster-whisper fallback

The project can use faster-whisper when `--model` is passed or `YTDL_WHISPER_MODEL` is set.

Optional default location used on this PC:

```text
C:\Users\Admin\Models\faster-whisper-base.en
```

If needed, download with Python / Hugging Face Hub from the repo venv:

```bash
uv run python -c "from huggingface_hub import snapshot_download; from pathlib import Path; p=Path.home()/'Models'/'faster-whisper-base.en'; p.parent.mkdir(parents=True, exist_ok=True); snapshot_download('Systran/faster-whisper-base.en', local_dir=str(p))"
```

Then either pass it explicitly:

```bash
ytd "YOUTUBE_URL" --format txt --model C:\Users\Admin\Models\faster-whisper-base.en
```

Or set:

```bash
set YTDL_WHISPER_MODEL=C:\Users\Admin\Models\faster-whisper-base.en
```

### 6. Install Ollama model for summaries

Summary mode requires local Ollama running at:

```text
http://127.0.0.1:11434
```

Install the default model:

```bash
ollama pull llama3.1:8b
```

Optional models that have worked on this PC:

```bash
ollama pull qwen3:8b
ollama pull qwen2.5-coder:7b
```

Verify Ollama has models:

```bash
ollama list
```

### 7. Verify the installation

From the repo folder:

```bash
uv run pytest tests/test_cli_defaults.py -q
uv run python -m py_compile src/youtube_downloader/cli.py
```

From any folder:

```bash
ytd help
ytd --help
```

Smoke-test MP3 download:

```bash
ytd "https://www.youtube.com/watch?v=jNQXAC9IVRw" --format mp3 --output-dir C:\Temp\ytd-test
```

Smoke-test TXT transcription:

```bash
ytd "https://www.youtube.com/watch?v=jNQXAC9IVRw" --format txt --language en --device cpu --output-dir C:\Temp\ytd-test
```

Smoke-test Ollama summary:

```bash
ytd "https://www.youtube.com/watch?v=jNQXAC9IVRw" format summary --language en --device cpu --output-dir C:\Temp\ytd-test --ollama-model llama3.1:8b
```

### 8. Known dependency warning

`yt-dlp` may warn:

```text
No supported JavaScript runtime could be found
```

The tool can still work, but some YouTube videos/formats may fail in the future. If that happens, install a JavaScript runtime supported by yt-dlp, such as Deno or Node, and make sure it is on PATH.


## Usage

You can run the tool from any folder with the global `ytd` command.

Show the short command list:

```bash
ytd help
```

Download best video/audio as MP4/WebM/MKV-compatible source:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID"
```

Download and convert/remux to MKV:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" --format mkv
```

Download/convert audio to MP3:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" --format mp3
```

Download audio and transcribe to TXT locally with the global Whisper install:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" --format txt
```

Download audio, save a TXT transcript, then stream a local Ollama summary to the terminal:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" format summary
```

Equivalent long-form option:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" --format summary
```

The default summary model is `llama3.1:8b`. Override it with:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" format summary --ollama-model qwen3:8b
```

You can combine outputs:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" --format mkv --format mp3 --format txt
```

The old project-local command still works from this folder:

```bash
uv run ytdl-local "https://www.youtube.com/watch?v=VIDEO_ID" --format mp3
```

Or use the Windows wrapper:

```bat
run-ytdl-local.bat "https://www.youtube.com/watch?v=VIDEO_ID" --format mp3
```

## Options

```text
--output-dir downloads          Where files are saved
--format original|mkv|mp3|txt   Output type; can be repeated
--format summary                Save TXT transcript, then stream Ollama summary
--model PATH_OR_NAME            Optional faster-whisper model path/name; overrides global Whisper CLI
--language en                   Optional transcription language hint
--device cpu|cuda|auto          Transcription device. For global Whisper, auto omits the device flag.
--compute-type int8             faster-whisper-only compute type
--ollama-model llama3.1:8b       Local Ollama model for summaries
--keep-intermediate             Keep temporary downloaded media after conversion
--audio-only                    Download audio only before conversion/transcription
```

## Notes

- YouTube downloading itself requires internet access.
- Conversion and transcription run locally.
- TXT transcription now works without `--model` by using the globally installed `whisper` command and the locally cached `base.en` model.
- If you set `YTDL_WHISPER_MODEL` or pass `--model`, the tool uses faster-whisper instead.
- Summary mode requires Ollama running locally at `http://127.0.0.1:11434` and the requested model installed.
