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
    monkeypatch.delenv("YTDL_MODEL_DIR", raising=False)
    model_dir = tmp_path / "faster-whisper-base.en"
    model_dir.mkdir()
    monkeypatch.setattr(cli, "DEFAULT_MODEL_DIRS", (tmp_path,))
    assert cli.resolve_whisper_model(None) == str(model_dir)


def test_resolve_whisper_model_uses_ytdl_model_dir_env(monkeypatch, tmp_path):
    monkeypatch.delenv("YTDL_WHISPER_MODEL", raising=False)
    env_root = tmp_path / "models"
    env_model_dir = env_root / "faster-whisper-base.en"
    env_model_dir.mkdir(parents=True)
    monkeypatch.setenv("YTDL_MODEL_DIR", str(env_root))
    monkeypatch.setattr(cli, "DEFAULT_MODEL_DIRS", ())
    assert cli.resolve_whisper_model(None) == str(env_model_dir)


def test_select_backend_prefers_global_whisper_cli_when_no_model(monkeypatch, tmp_path):
    monkeypatch.delenv("YTDL_WHISPER_MODEL", raising=False)
    monkeypatch.setattr(cli, "DEFAULT_MODEL_DIRS", (tmp_path,))
    monkeypatch.setattr(cli.shutil, "which", lambda name: "C:/tools/whisper/whisper.exe" if name == "whisper" else None)
    assert cli.select_transcription_backend(None) == "whisper-cli"


def test_select_backend_uses_faster_whisper_when_model_is_explicit(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "C:/tools/whisper/whisper.exe")
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


def test_is_playlist_url_detects_youtube_playlist():
    assert cli.is_playlist_url("https://www.youtube.com/playlist?list=PLCxKMhCMh2HlWglySHj7Rrc9qZy9aSG1j")
    assert cli.is_playlist_url("https://www.youtube.com/watch?v=abc123&list=PLCxKMhCMh2HlWglySHj7Rrc9qZy9aSG1j")
    assert not cli.is_playlist_url("https://www.youtube.com/watch?v=abc123")


def test_extract_playlist_info_builds_video_urls(monkeypatch):
    class FakeYDL:
        def __init__(self, _opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, _url, download):
            assert download is False
            return {
                "_type": "playlist",
                "title": "Demo Playlist",
                "entries": [
                    {"id": "one"},
                    {"webpage_url": "https://www.youtube.com/watch?v=two"},
                    {"url": "https://www.youtube.com/watch?v=three"},
                ],
            }

    monkeypatch.setattr(cli, "YoutubeDL", FakeYDL)
    result = cli.extract_playlist_info("https://www.youtube.com/playlist?list=abc")

    assert result is not None
    assert result.title == "Demo Playlist"
    assert result.video_urls == [
        "https://www.youtube.com/watch?v=one",
        "https://www.youtube.com/watch?v=two",
        "https://www.youtube.com/watch?v=three",
    ]


def test_build_outputs_processes_playlist_sequentially(monkeypatch, tmp_path):
    playlist = cli.PlaylistInfo(
        source_url="https://www.youtube.com/playlist?list=abc",
        title="Demo Playlist",
        video_urls=[
            "https://www.youtube.com/watch?v=one",
            "https://www.youtube.com/watch?v=two",
        ],
    )
    monkeypatch.setattr(cli, "extract_playlist_info", lambda url: playlist)

    seen: list[str] = []

    def fake_build_single_output(args, url):
        seen.append(url)
        return {
            "title": f"title-{len(seen)}",
            "video_id": f"id-{len(seen)}",
            "url": url,
            "outputs": {"txt": str(tmp_path / f"out-{len(seen)}.txt")},
        }

    monkeypatch.setattr(cli, "build_single_output", fake_build_single_output)

    args = cli.parse_args([
        "https://www.youtube.com/playlist?list=abc",
        "--format",
        "txt",
        "--output-dir",
        str(tmp_path),
    ])
    result = cli.build_outputs(args)

    assert seen == [
        "https://www.youtube.com/watch?v=one",
        "https://www.youtube.com/watch?v=two",
    ]
    assert result["playlist"]["count"] == 2
    assert len(result["items"]) == 2


def test_main_prints_playlist_summary(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "build_outputs",
        lambda _args: {
            "playlist": {
                "title": "Demo Playlist",
                "url": "https://www.youtube.com/playlist?list=abc",
                "count": 1,
            },
            "items": [
                {
                    "title": "Video 1",
                    "video_id": "abc123",
                    "url": "https://www.youtube.com/watch?v=abc123",
                    "outputs": {"txt": "downloads/Video 1 [abc123].txt"},
                }
            ],
        },
    )

    assert cli.main(["https://www.youtube.com/playlist?list=abc", "--format", "txt"]) == 0
    output = capsys.readouterr().out
    assert "Playlist: Demo Playlist" in output
    assert "Videos processed: 1" in output
