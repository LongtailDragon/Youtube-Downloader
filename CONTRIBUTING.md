# Contributing

Thanks for considering a contribution.

## Development Setup

1. Install prerequisites: Python 3.10+, uv, and ffmpeg on PATH.
2. Install dependencies:

```bash
uv sync
```

3. Run tests:

```bash
uv run pytest -q
```

4. Install pre-commit hooks:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

Local wrapper scripts:

- Windows: `run-ytdl-local.bat`
- macOS/Linux: `run-ytdl-local.sh`

## Code Style and Scope

- Keep changes focused and minimal.
- Prefer readable, well-named functions over clever shortcuts.
- Avoid machine-specific paths in code, docs, and tests.
- Maintain cross-platform behavior unless a Windows-only behavior is intentional and documented.

## Testing Expectations

- Add or update tests for any behavior change.
- Ensure existing tests pass before opening a PR.
- For CLI changes, include at least one test for argument parsing or output behavior where practical.
- Ensure GitHub Actions CI passes on Windows, macOS, and Linux.
- Ensure secret scan checks pass.

## Pull Request Guidelines

- Use a clear title and describe what changed and why.
- Include steps to validate the change.
- Link related issues when applicable.
- Keep PRs small enough to review quickly.

## Commit Message Guidance

Use short, imperative subjects.

Examples:

- `docs: improve installation and troubleshooting guide`
- `cli: fix summary format argument normalization`
- `tests: cover whisper backend selection fallback`
