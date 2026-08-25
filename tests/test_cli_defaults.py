from __future__ import annotations

import re
from pathlib import Path
import sys
import types

import pytest

from youtube_downloader import cli


@pytest.fixture(autouse=True)
def clear_whisperx_model_caches():
    cli._WHISPERX_ASR_MODELS.clear()
    cli._WHISPERX_ALIGN_MODELS.clear()
    cli._WHISPERX_DIARIZATION_MODELS.clear()
    yield
    cli._WHISPERX_ASR_MODELS.clear()
    cli._WHISPERX_ALIGN_MODELS.clear()
    cli._WHISPERX_DIARIZATION_MODELS.clear()


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
            diarize=False,
            hf_token=None,
            min_speakers=None,
            max_speakers=None,
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
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake media")
    transcript = tmp_path / "downloads" / "Example [abc123].txt"

    monkeypatch.setattr(
        cli,
        "download_youtube",
        lambda url, output_dir, audio_only: cli.DownloadedMedia(source, "Example", "abc123", url, "2026-08-19T12:34:56Z"),
    )

    def fake_transcribe(
        source,
        output_dir,
        base_stem,
        model,
        language,
        device,
        compute_type,
        diarize,
        hf_token,
        min_speakers,
        max_speakers,
        collapse,
        video_url,
        published_datetime,
        metadata_title,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        assert diarize is False
        assert hf_token is None
        assert min_speakers is None
        assert max_speakers is None
        assert video_url == "https://example.com/video"
        assert published_datetime == "2026-08-19T12:34:56Z"
        assert metadata_title == "Example"
        transcript.write_text("This is a local transcript about useful SEO tactics.", encoding="utf-8")
        return transcript

    monkeypatch.setattr(cli, "transcribe_to_txt", fake_transcribe)
    monkeypatch.setattr(cli, "stream_ollama_summary", lambda transcript_path, model: print("SUMMARY: useful SEO tactics"))

    args = cli.parse_args(["https://example.com/video", "format", "summary", "--output-dir", str(tmp_path / "downloads")])
    result = cli.build_outputs(args)

    assert result["outputs"]["txt"] == str(transcript)
    assert result["outputs"]["summary"] == "streamed-to-terminal"
    assert "SUMMARY: useful SEO tactics" in capsys.readouterr().out


def test_parse_args_accepts_txt_diarize_and_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    args = cli.parse_args([
        "https://example.com/video",
        "--format",
        "txt-diarize",
        "--hf-token",
        "token-123",
        "--min-speakers",
        "2",
        "--max-speakers",
        "5",
    ])

    assert args.formats == ["txt-diarize"]
    assert args.diarize is False
    assert args.hf_token == "token-123"
    assert args.min_speakers == 2
    assert args.max_speakers == 5


def test_parse_args_accepts_optional_collapse_flag():
    args = cli.parse_args([
        "https://example.com/video",
        "--collapse",
    ])

    assert args.collapse is True


def test_help_uses_invoked_command_name(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "argv", ["ytd"])

    with pytest.raises(SystemExit):
        cli.parse_args(["--help"])

    output = capsys.readouterr().out
    assert "usage: ytd" in output


def test_normalize_formats_and_diarize_supports_alias_and_flag():
    normalized, diarize_requested = cli.normalize_formats_and_diarize(["txt-diarize", "mp3"], False)
    assert normalized == ["txt", "mp3"]
    assert diarize_requested is True

    normalized2, diarize_requested2 = cli.normalize_formats_and_diarize(["txt"], True)
    assert normalized2 == ["txt"]
    assert diarize_requested2 is True


def test_validate_args_rejects_diarize_without_transcript_output(tmp_path):
    args = cli.parse_args([
        "https://example.com/video",
        "--format",
        "mp3",
        "--diarize",
        "--output-dir",
        str(tmp_path),
    ])

    with pytest.raises(cli.ToolError) as excinfo:
        cli.validate_args(args)
    assert "Diarization requires transcript output" in str(excinfo.value)


def test_project_dependencies_do_not_include_conflicting_openai_whisper():
    project_toml = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = project_toml.read_text(encoding="utf-8")
    match = re.search(r"^dependencies = \[(.*?)\]", text, re.S | re.M)
    assert match is not None
    dependencies = match.group(1)
    assert '"whisperx>=3.8.7rc1"' in dependencies
    assert 'openai-whisper' not in dependencies.lower()


def test_build_single_output_removes_source_when_output_step_fails(monkeypatch, tmp_path):
    source = tmp_path / "source.webm"
    source.write_bytes(b"fake")

    monkeypatch.setattr(
        cli,
        "download_youtube",
        lambda url, output_dir, audio_only: cli.DownloadedMedia(source, "Demo", "abc123", url, "2026-08-19T12:34:56Z"),
    )
    monkeypatch.setattr(cli, "convert_to_mkv", lambda *args, **kwargs: tmp_path / "out.mkv")
    monkeypatch.setattr(cli, "convert_to_mp3", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(cli, "transcribe_to_txt", lambda *args, **kwargs: tmp_path / "out.txt")
    monkeypatch.setattr(cli, "stream_ollama_summary", lambda *args, **kwargs: None)

    args = types.SimpleNamespace(
        formats=["mp3"],
        diarize=False,
        output_dir=tmp_path,
        keep_intermediate=False,
        model=None,
        language=None,
        device="cpu",
        compute_type="default",
        hf_token=None,
        min_speakers=None,
        max_speakers=None,
        collapse=False,
        audio_only=False,
        ollama_model="llama3.1:8b",
    )

    with pytest.raises(RuntimeError, match="boom"):
        cli.build_single_output(args, "https://example.com/video")

    assert not source.exists()


def test_download_youtube_audio_only_retries_403_with_android_client(monkeypatch, tmp_path):
    calls: list[dict] = []

    def fake_extract_downloaded_youtube(url, ydl_opts):
        calls.append(ydl_opts)
        if len(calls) == 1:
            raise cli.ToolError("ERROR: unable to download video data: HTTP Error 403: Forbidden")
        downloaded = tmp_path / "Recovered [abc123].mp4"
        downloaded.write_bytes(b"fake media")
        return {
            "title": "Recovered",
            "id": "abc123",
            "webpage_url": url,
            "requested_downloads": [{"filepath": str(downloaded)}],
        }

    monkeypatch.setattr(cli, "extract_downloaded_youtube", fake_extract_downloaded_youtube)

    media = cli.download_youtube("https://www.youtube.com/watch?v=abc123", tmp_path, audio_only=True)

    assert media.title == "Recovered"
    assert media.source_path.name == "Recovered [abc123].mp4"
    assert calls[0]["format"] == "bestaudio/best"
    assert calls[1]["format"] == "18/best"
    assert calls[1]["extractor_args"] == {"youtube": {"player_client": ["android"]}}


def test_download_facebook_audio_only_does_not_use_youtube_403_fallback(monkeypatch, tmp_path):
    calls: list[dict] = []

    def fake_extract_downloaded_youtube(url, ydl_opts):
        calls.append(ydl_opts)
        raise cli.ToolError("ERROR: unable to download video data: HTTP Error 403: Forbidden")

    monkeypatch.setattr(cli, "extract_downloaded_youtube", fake_extract_downloaded_youtube)

    with pytest.raises(cli.ToolError, match="HTTP Error 403"):
        cli.download_youtube("https://www.facebook.com/share/v/1c1CCQq3N6/", tmp_path, audio_only=True)

    assert len(calls) == 1
    assert calls[0]["format"] == "bestaudio/best"


def test_is_youtube_url_distinguishes_facebook_urls():
    assert cli.is_youtube_url("https://www.youtube.com/watch?v=abc123")
    assert cli.is_youtube_url("https://youtu.be/abc123")
    assert not cli.is_youtube_url("https://www.facebook.com/share/v/1c1CCQq3N6/")


def test_build_outputs_retries_playlist_items_after_403(monkeypatch):
    playlist = cli.PlaylistInfo(
        source_url="https://example.com/playlist",
        title="Demo Playlist",
        video_urls=["https://example.com/video-1"],
    )
    monkeypatch.setattr(cli, "extract_playlist_info", lambda url: playlist)

    sleep_calls: list[int] = []
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: sleep_calls.append(int(seconds)))

    attempts = {"count": 0}

    def fake_build_single_output(args, url):
        attempts["count"] += 1
        if attempts["count"] in {1, 2}:
            raise cli.ToolError("ERROR: unable to download video data: HTTP Error 403: Forbidden")
        return {"title": "Recovered", "video_id": "abc123", "url": url, "outputs": {}}

    monkeypatch.setattr(cli, "build_single_output", fake_build_single_output)

    args = cli.parse_args(["https://example.com/playlist"])
    result = cli.build_outputs(args)

    assert attempts["count"] == 3
    assert sleep_calls == [60, 120]
    assert result["items"][0]["title"] == "Recovered"


def test_select_whisperx_compute_type_uses_best_supported_cuda_type(monkeypatch):
    fake_ctranslate2 = types.ModuleType("ctranslate2")
    fake_ctranslate2.get_supported_compute_types = lambda device: {"float32", "int8", "int8_float32"}
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ctranslate2)

    assert cli.select_whisperx_compute_type("cuda", "default") == "int8_float32"


def test_select_whisperx_compute_type_keeps_explicit_choice(monkeypatch):
    fake_ctranslate2 = types.ModuleType("ctranslate2")
    fake_ctranslate2.get_supported_compute_types = lambda device: {"int8"}
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ctranslate2)

    assert cli.select_whisperx_compute_type("cuda", "float32") == "float32"


def test_format_published_datetime_prefers_timestamp_and_falls_back_to_upload_date():
    assert cli.format_published_datetime({"timestamp": 1787135696}) == "2026-08-19T10:34:56Z"
    assert cli.format_published_datetime({"upload_date": "20260819"}) == "2026-08-19"
    assert cli.format_published_datetime({}) == "unknown"


def test_format_timestamp_seconds_truncates_subsecond_precision():
    assert cli.format_timestamp_seconds(137.633) == "00:02:17"
    assert cli.format_timestamp_seconds(0.999) == "00:00:00"


def test_collapse_diarized_segments_merges_adjacent_speaker_lines():
    collapsed = cli.collapse_diarized_segments([
        {"start": 137.633, "speaker": "SPEAKER_00", "text": "Hello"},
        {"start": 140.215, "speaker": "SPEAKER_00", "text": "world."},
        {"start": 143.000, "speaker": "SPEAKER_01", "text": "New speaker."},
        {"start": 144.000, "speaker": "SPEAKER_01", "text": "More text."},
        {"start": 145.000, "speaker": "SPEAKER_00", "text": "Back again."},
    ], collapse=True)

    assert collapsed == [
        {"start": 137.633, "speaker": "SPEAKER_00", "text": "Hello world."},
        {"start": 143.0, "speaker": "SPEAKER_01", "text": "New speaker. More text."},
        {"start": 145.0, "speaker": "SPEAKER_00", "text": "Back again."},
    ]


def test_transcribe_with_whisperx_uses_current_diarization_api_and_caches_models(monkeypatch, tmp_path):
    calls = {"asr": 0, "align": 0, "diarize": 0}

    class FakeAsrModel:
        def transcribe(self, audio, language=None):
            assert audio == "AUDIO"
            assert language == "en"
            return {
                "language": "en",
                "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            }

    fake_whisperx = types.ModuleType("whisperx")
    fake_whisperx.load_audio = lambda path: "AUDIO"

    def fake_load_model(model, device, compute_type=None):
        calls["asr"] += 1
        assert model == cli.DEFAULT_WHISPERX_MODEL
        assert device == "cpu"
        assert compute_type == "int8"
        return FakeAsrModel()

    fake_whisperx.load_model = fake_load_model

    def fake_load_align_model(language_code, device):
        calls["align"] += 1
        assert language_code == "en"
        assert device == "cpu"
        return "ALIGN", {"language": language_code}

    fake_whisperx.load_align_model = fake_load_align_model
    fake_whisperx.align = lambda segments, align_model, metadata, audio, device, return_char_alignments=False: {
        "language": metadata["language"],
        "segments": [
            {"start": 137.633, "end": 140.215, "text": "hello"},
            {"start": 141.156, "end": 145.919, "text": "again"},
        ],
    }
    fake_whisperx.assign_word_speakers = lambda diarization, result: {
        "language": result["language"],
        "segments": [
            {"start": 137.633, "end": 140.215, "speaker": "SPEAKER_00", "text": "hello"},
            {"start": 141.156, "end": 145.919, "speaker": "SPEAKER_00", "text": "again"},
        ],
    }

    fake_diarize = types.ModuleType("whisperx.diarize")

    class FakeDiarizationPipeline:
        def __init__(self, token=None, device="cpu"):
            calls["diarize"] += 1
            assert token == "hf_fake"
            assert device == "cpu"

        def __call__(self, audio, **kwargs):
            assert audio == "AUDIO"
            assert kwargs == {"min_speakers": 1, "max_speakers": 2}
            return "DIARIZATION"

    fake_diarize.DiarizationPipeline = FakeDiarizationPipeline
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", fake_diarize)

    source = tmp_path / "source.webm"
    source.write_bytes(b"fake")
    for idx in range(2):
        transcript = cli.transcribe_with_whisperx_diarization(
            source=source,
            output_dir=tmp_path,
            base_stem=f"demo-{idx}",
            model=None,
            language="en",
            device="cpu",
            compute_type="default",
            hf_token="hf_fake",
            min_speakers=1,
            max_speakers=2,
            collapse=True,
            video_url="https://www.youtube.com/watch?v=abc123",
            published_datetime="2026-08-19T12:34:56Z",
            metadata_title="What happened when FL gave $8,000 to families who left government schools",
        )

        text = transcript.read_text(encoding="utf-8")
        assert "Title: What happened when FL gave $8,000 to families who left government schools\n" in text
        assert "_8_000" not in text
        assert "Video URL: https://www.youtube.com/watch?v=abc123" in text
        assert "Published datetime: 2026-08-19T12:34:56Z" in text
        assert "Diarization: WhisperX" in text
        assert "[00:02:17] SPEAKER_00: hello again" in text
        assert "-->" not in text

    assert calls == {"asr": 1, "align": 1, "diarize": 1}


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
