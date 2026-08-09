@echo off
setlocal
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
    echo uv is required but was not found on PATH.
    exit /b 1
)
uv run ytdl-local %*
