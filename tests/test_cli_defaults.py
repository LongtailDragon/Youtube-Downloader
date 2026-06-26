from __future__ import annotations

from pathlib import Path

import pytest

from youtube_downloader import cli


def test_resolve_whisper_model_prefers_explicit_model():
    assert cli.resolve_whisper_model("C:/models/explicit") == "C:/models/explicit"


def test_resolve_whisper_model_uses_env_default(monkeypatch):
    monkeypatch.setenv("YTDL_WHISPER_MODEL", "C:/models/from-env")
    assert cli.resolve_whisper_model(None) == "C:/models/from-env"


def test_resolve_whisper_model_uses_global_default_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("YTDL_WHISPER_MODEL", raising=False)
    model_dir = tmp_path / "faster-whisper-base.en"
    model_dir.mkdir()
    monkeypatch.setattr(cli, "DEFAULT_MODEL_DIRS", (tmp_path,))
    assert cli.resolve_whisper_model(None) == str(model_dir)


def test_select_backend_prefers_global_whisper_cli_when_no_model(monkeypatch, tmp_path):
    monkeypatch.delenv("YTDL_WHISPER_MODEL", raising=False)
    monkeypatch.setattr(cli, "DEFAULT_MODEL_DIRS", (tmp_path,))
    monkeypatch.setattr(cli.shutil, "which", lambda name: "C:/Users/Admin/.local/bin/whisper" if name == "whisper" else None)
    assert cli.select_transcription_backend(None) == "whisper-cli"


def test_select_backend_uses_faster_whisper_when_model_is_explicit(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "C:/Users/Admin/.local/bin/whisper")
    assert cli.select_transcription_backend("C:/models/explicit") == "faster-whisper"


def test_transcribe_error_mentions_global_whisper_when_no_backend(monkeypatch, tmp_path):
    monkeypatch.delenv("YTDL_WHISPER_MODEL", raising=False)
    monkeypatch.setattr(cli, "DEFAULT_MODEL_DIRS", (tmp_path,))
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    with pytest.raises(cli.ToolError) as excinfo:
        cli.transcribe_to_txt(
            source=Path("missing.mp4"),
            output_dir=tmp_path,
            base_stem="sample",
            model=None,
            language=None,
            device="cpu",
            compute_type="default",
        )
    message = str(excinfo.value)
    assert "global whisper command" in message
    assert "uv tool install openai-whisper" in message


def test_help_alias_prints_command_list(capsys):
    assert cli.main(["help"]) == 0
    output = capsys.readouterr().out
    assert "Commands:" in output
    assert "ytd URL --format mp3" in output
    assert "ytd URL --format txt" in output
    assert "ytd URL format summary" in output
    assert "ytd help" in output


def test_format_summary_alias_is_normalized():
    assert cli.normalize_argv(["https://example.com/video", "format", "summary"]) == [
        "https://example.com/video",
        "--format",
        "summary",
    ]


def test_summary_format_creates_txt_and_streams_summary(monkeypatch, tmp_path, capsys):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake media")
    transcript = tmp_path / "downloads" / "Example [abc123].txt"

    monkeypatch.setattr(
        cli,
        "download_youtube",
        lambda url, output_dir, audio_only: cli.DownloadedMedia(source, "Example", "abc123", url),
    )

    def fake_transcribe(source, output_dir, base_stem, model, language, device, compute_type):
        output_dir.mkdir(parents=True, exist_ok=True)
        transcript.write_text("This is a local transcript about useful SEO tactics.", encoding="utf-8")
        return transcript

    monkeypatch.setattr(cli, "transcribe_to_txt", fake_transcribe)
    monkeypatch.setattr(cli, "stream_ollama_summary", lambda transcript_path, model: print("SUMMARY: useful SEO tactics"))

    args = cli.parse_args(["https://example.com/video", "format", "summary", "--output-dir", str(tmp_path / "downloads")])
    result = cli.build_outputs(args)

    assert result["outputs"]["txt"] == str(transcript)
    assert result["outputs"]["summary"] == "streamed-to-terminal"
    assert "SUMMARY: useful SEO tactics" in capsys.readouterr().out
