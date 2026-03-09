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

## Architecture

Three-layer structure in `src/ffmpeg_toolkit/`:

| File | Role |
|------|------|
| `ffmpeg_toolkit_core.py` | All task classes + FFmpeg/FFprobe subprocess wrappers + segment utilities |
| `ffmpeg_toolkit.py` | Public API: `FF_TASKS` (ClassEnum), `PARTIAL_TASKS` (FunctionEnum), `BatchTask` |
| `ffmpeg_types.py` | Type definitions: `FunctionEnum`, `ClassEnum`, `SegmentOperation`, `VideoSuffix` |
| `__init__.py` | Package exports + FFmpeg PATH setup |

**Task lifecycle:** Instantiate (Pydantic validates) → optional `.override_option()` → `.render()` → returns output path or `FFRenderException`.

**Two API styles exist:**
```python
# Class-based (FF_TASKS)
FF_TASKS.Speedup("in.mp4", "out.mp4", multiple=2).render()

# Factory function (PARTIAL_TASKS) — reusable partial application
speedup = PARTIAL_TASKS.speedup(multiple=2)
speedup("in1.mp4", "out1.mp4")
speedup("in2.mp4", "out2.mp4")
```

## Key Conventions

### Task classes
All tasks inherit from `FFCreateTask` (which extends Pydantic `BaseModel`). Setup logic goes in `model_post_init()`, not `__init__`. Tasks build their FFmpeg kwargs dict there and store it on `self`.

### Kwargs layering
FFmpeg command args are assembled via dict merging with explicit precedence:
```python
ff_kwargs = (
    input_kwargs_default    # hide banner, hwaccel
    | input_kwargs          # user input overrides
    | {"i": input_file}     # input file (always wins)
    | output_kwargs_default # loglevel
    | output_kwargs         # user output overrides
    | {"y": output_file}    # force overwrite (always wins)
)
```

### Segment format
Segments are flat lists `[start1, end1, start2, end2, ...]`. Even indices are segment starts, odd indices are ends. This format is used throughout silence/motion detection and all set operations.

### Segment set operations
`SegmentOperation` enum provides `UNION`, `INTERSECTION`, `SOUND_ONLY`, `MOTION_ONLY`, `XOR`, `COMPLEMENT`. Used by `CutByDetection` to combine silence and motion segment lists.

### Speedup implementation
- `multiple <= 5`: PTS adjustment (better quality)
- `multiple > 5`: Frame selection (faster)

### FFmpeg kwarg key mappings
`cv` → `-c:v`, `ca` → `-c:a` (short aliases expand in `_dic_to_ffmpeg_kwargs()`).

### Temporary files
Tasks write to a `_processing`-suffixed path during execution and rename on success.

### Testing patterns
Tests use `unittest.mock.patch` to mock `_ffmpeg()` and `_ffprobe()` (no real FFmpeg needed). Test classes follow `Test*` / `test_*` naming. See `test/test_ffmpeg_toolkit.py` for examples of how to assert on generated kwargs without invoking FFmpeg.
