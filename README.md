# YouTube Downloader

Portable command-line tool for downloading YouTube videos, converting media, creating local transcripts, and generating local summaries.

No cloud transcription APIs are required. Downloading still needs internet access, but conversion/transcription/summarization run on your machine.

## Features

- Download best available source media via `yt-dlp`
- Convert/remux to `mkv`
- Convert audio to `mp3`
- Transcribe to `txt` using either:
	- global OpenAI Whisper CLI (default path), or
	- `faster-whisper` when `--model` or `YTDL_WHISPER_MODEL` is set
- Stream local transcript summaries through Ollama (`--format summary`)

## Requirements

- Python 3.10+
- `uv`
- `ffmpeg` on `PATH`
- Optional: `ollama` on `PATH` for summary mode
- Optional: global `whisper` command for default TXT mode

### Prerequisite Installation

Install prerequisites before running `uv sync`.

Windows (PowerShell, via winget):

```powershell
winget install --id Python.Python.3.12 -e
winget install --id astral-sh.uv -e
winget install --id Gyan.FFmpeg -e
winget install --id Ollama.Ollama -e
```

macOS (Homebrew):

```bash
brew install python uv ffmpeg
brew install --cask ollama
```

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y python3 python3-venv ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install Whisper CLI (optional, needed for default TXT backend):

```bash
uv tool install openai-whisper
```

Quick checks:

```bash
python --version
uv --version
ffmpeg -version
ollama --version
whisper --help
```

## Install

1. Clone this repository.
2. Install project dependencies:

```bash
uv sync
```

3. Optional but recommended: install the global `ytd` command:

```bash
uv tool install --editable . --force
```

4. Verify CLI install:

```bash
ytd help
ytd --help
```

## Optional Local AI Setup

### Whisper CLI (default TXT backend)

Install:

```bash
uv tool install openai-whisper
```

Pre-download the `base.en` model to a local model directory (example uses a placeholder path):

```bash
whisper /path/to/sample-audio.wav --model base.en --model_dir /path/to/models/openai-whisper --output_format txt --fp16 False --device cpu
```

Default model directory in code:

- `Path.home() / "Models" / "openai-whisper"`

If you want a different default, update `DEFAULT_OPENAI_WHISPER_MODEL_DIR` in `src/youtube_downloader/cli.py`.

### faster-whisper (optional backend)

When you pass `--model` or set `YTDL_WHISPER_MODEL`, the app uses `faster-whisper`.

Example model download:

```bash
uv run python -c "from huggingface_hub import snapshot_download; from pathlib import Path; p=Path.home()/'Models'/'faster-whisper-base.en'; p.parent.mkdir(parents=True, exist_ok=True); snapshot_download('Systran/faster-whisper-base.en', local_dir=str(p))"
```

Use it directly:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" --format txt --model /path/to/models/faster-whisper-base.en
```

Or set an environment variable:

Windows (PowerShell):

```powershell
$env:YTDL_WHISPER_MODEL = "C:\path\to\models\faster-whisper-base.en"
```

macOS/Linux:

```bash
export YTDL_WHISPER_MODEL="/path/to/models/faster-whisper-base.en"
```

### Ollama (summary mode)

Install a model:

```bash
ollama pull llama3.1:8b
```

Then run summary mode with either syntax:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" format summary
ytd "https://www.youtube.com/watch?v=VIDEO_ID" --format summary
```

## Usage

Show short help:

```bash
ytd help
```

Download best source media:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID"
```

Convert/remux to MKV:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" --format mkv
```

Convert to MP3:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" --format mp3
```

Transcribe to TXT:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" --format txt --language en --device cpu
```

Generate transcript + local summary:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" format summary --ollama-model llama3.1:8b
```

Produce multiple outputs in one run:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" --format mkv --format mp3 --format txt
```

Local project command without global install:

```bash
uv run ytdl-local "https://www.youtube.com/watch?v=VIDEO_ID" --format mp3
```

Windows wrapper script:

```bat
run-ytdl-local.bat "https://www.youtube.com/watch?v=VIDEO_ID" --format mp3
```

macOS/Linux wrapper script:

```bash
sh run-ytdl-local.sh "https://www.youtube.com/watch?v=VIDEO_ID" --format mp3
```

## CLI Options

```text
--output-dir downloads                 Output directory (default: downloads)
--format original|mkv|mp3|txt|summary Output format; can be repeated
--model PATH_OR_NAME                  faster-whisper model path/name
--language en                         Optional language hint
--device cpu|cuda|auto                Transcription device
--compute-type default|int8|float16   faster-whisper compute type
--ollama-model llama3.1:8b            Local Ollama model for summary mode
--keep-intermediate                   Keep downloaded source media
--audio-only                          Download audio-only source
--print-json                          Print machine-readable result
```

## Configuration

Environment variables:

- `YTDL_WHISPER_MODEL`: default `faster-whisper` model path/name
- `YTDL_MODEL_DIR`: preferred directory containing `faster-whisper-base.en`
- `YTD_OLLAMA_MODEL`: default Ollama model for summary mode

## Verification

Run tests:

```bash
uv run pytest -q
```

Syntax check:

```bash
uv run python -m py_compile src/youtube_downloader/cli.py
```

CI runs these checks on Windows, macOS, and Linux for Python 3.10-3.12:

- `.github/workflows/ci.yml`

## Troubleshooting

`No supported JavaScript runtime could be found` from `yt-dlp`:

- Install a supported runtime such as Node.js or Deno.
- Ensure it is available on `PATH`.

`whisper` command not found:

- Install with `uv tool install openai-whisper`.
- Confirm your tool bin directory is on `PATH`.

Ollama connection errors:

- Start Ollama locally.
- Check `ollama list` and verify your chosen model is installed.

## Contributing

Contributions are welcome. Please read `CONTRIBUTING.md` for development setup, testing, and pull request guidelines.

Recommended local checks before commit:

```bash
uv sync --group dev
uv run pre-commit install
uv run pre-commit run --all-files
```

## Security

To report a vulnerability, please follow `SECURITY.md` and avoid opening a public issue for sensitive findings.



## License

This project is licensed under the MIT License. See `LICENSE`.
