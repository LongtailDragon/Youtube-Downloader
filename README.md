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
- Optional speaker diarization (`--format txt-diarize` or `--diarize`) using WhisperX
- Stream local transcript summaries through Ollama (`--format summary`)

## Requirements

- Python 3.10+
- `uv`
- `ffmpeg` on `PATH`
- Optional: `ollama` on `PATH` for summary mode
- Optional: global `whisper` command for default TXT mode
- Optional: WhisperX + Hugging Face token for diarized transcripts

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

Install Whisper CLI:

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

### WhisperX Diarization (speaker detection)

Install diarization dependencies:

```bash
uv sync --extra diarization
```

Optional performance boost for model downloads:

Some Hugging Face model repos use Xet-backed storage. Installing `hf_xet` can improve download throughput (especially for larger model downloads). This does not change transcript quality; it only affects download behavior.

```bash
uv pip install "huggingface_hub[hf_xet]"
```

Alternative:

```bash
uv pip install hf_xet
```

Set a Hugging Face token for pyannote diarization:

How to create the token and permissions:

1. Create or sign in to your Hugging Face account at `https://huggingface.co`.
2. Open `Settings` -> `Access Tokens` -> `New token`.
3. Create a token with at least `Read` access.
4. If you use fine-grained tokens, allow read access to the pyannote diarization model repos.
5. Visit the pyannote model pages and accept any gated model terms (required before API download works):
	- `https://huggingface.co/pyannote/speaker-diarization`
	- `https://huggingface.co/pyannote/segmentation`
6. Copy the token value (starts with `hf_...`) and set it as `HF_TOKEN`.

Quick verification (optional):

```bash
uv run python -c "import os; t=os.getenv('HF_TOKEN'); print('HF token present' if t and t.startswith('hf_') else 'HF token missing/invalid format')"
```

Windows (PowerShell):

```powershell
$env:HF_TOKEN = "hf_xxx"
```

macOS/Linux:

```bash
export HF_TOKEN="hf_xxx"
```

Performance tuning (system-agnostic):

1. Use GPU acceleration when available.
	- Install a GPU-enabled PyTorch build for your platform and driver using the official selector: `https://pytorch.org/get-started/locally/`
	- Verify GPU availability:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

	- If it prints `True`, run with `--device cuda`.

2. Choose the right model size for your speed/accuracy target.
	- Smaller Whisper/WhisperX models are faster and use less memory.
	- Larger models can improve accuracy but increase runtime.

3. Use compute types that match your hardware.
	- Default mode now checks the compute types supported by `ctranslate2` and chooses the best available option automatically.
	- GPU: `float16` is preferred when supported; older GPUs may fall back to `int8_float32` or `float32`.
	- CPU: `int8` can reduce memory usage and often improves throughput.

4. Constrain speaker search when possible.
	- If you know expected speaker counts, set `--min-speakers` and `--max-speakers` to reduce diarization search work.

5. Help transcription avoid extra autodetection work.
	- Provide `--language` when known.
	- Use `--audio-only` for transcript-focused runs to reduce download/processing overhead.

6. Keep model downloads fast.
	- Keep `hf_xet` installed when using Xet-backed Hugging Face repos.
	- Reuse local model caches to avoid repeated downloads.
	- In one `ytd` process, WhisperX ASR, alignment, and diarization models are cached and reused across playlist items.

Run diarized transcription:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" --format txt-diarize
```

Equivalent syntax:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" --format txt --diarize
```

Optional speaker bounds:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" --format txt --diarize --min-speakers 2 --max-speakers 4
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

Transcribe to TXT with diarization:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" --format txt-diarize
```

Generate transcript + local summary:

```bash
ytd "https://www.youtube.com/watch?v=VIDEO_ID" format summary --ollama-model llama3.1:8b
```

Auto-detect playlist and process videos one at a time:

```bash
ytd "https://www.youtube.com/playlist?list=PLAYLIST_ID" --format txt
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
--format original|mkv|mp3|txt|txt-diarize|summary Output format; can be repeated
--model PATH_OR_NAME                  faster-whisper model path/name
--language en                         Optional language hint
--device cpu|cuda|auto                Transcription device
--compute-type default|int8|float16   faster-whisper compute type
--diarize                             Enable WhisperX speaker diarization for transcript output
--hf-token TOKEN                      Hugging Face token for WhisperX diarization
--min-speakers N                      Optional minimum number of speakers
--max-speakers N                      Optional maximum number of speakers
--ollama-model llama3.1:8b            Local Ollama model for summary mode
--keep-intermediate                   Keep downloaded source media
--audio-only                          Download audio-only source
--print-json                          Print machine-readable result
```

Playlist URLs are auto-detected and expanded to video URLs. Videos are processed sequentially (one at a time), so transcription and conversion workloads do not run concurrently.

## Configuration

Environment variables:

- `YTDL_WHISPER_MODEL`: default `faster-whisper` model path/name
- `YTDL_MODEL_DIR`: preferred directory containing `faster-whisper-base.en`
- `YTD_OLLAMA_MODEL`: default Ollama model for summary mode
- `HF_TOKEN` / `HUGGINGFACE_TOKEN`: token for WhisperX diarization

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
