# Copilot Instructions

## Project Overview

`ffmpeg_toolkit` is a Python library providing an object-oriented interface for FFmpeg video processing. Operations are modeled as Pydantic-validated task objects with a `.render()` execution pattern.

## Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run all tests
pytest test/

# Run a single test
pytest test/test_ffmpeg_toolkit.py::TestTimestampConversion::test_timestamp_to_seconds_simple -v

# Run tests matching a pattern
pytest test/ -k "silence"

# Run with coverage
pytest --cov
```

No linter is configured for this project.

## Documentation

Start at [`CONTEXT.md`](../CONTEXT.md). Architecture, task lifecycle, kwargs layering, segment
format, and segment set operations are documented in
[`docs/reference/architecture.md`](../docs/reference/architecture.md); vocabulary in
[`docs/reference/glossary.md`](../docs/reference/glossary.md). Open work goes in
[`docs/plan/BACKLOG.md`](../docs/plan/BACKLOG.md).

## Testing

Tests mock `_ffmpeg()` and `_ffprobe()` with `unittest.mock.patch` — no real FFmpeg needed. See
`test/test_ffmpeg_toolkit.py`.
