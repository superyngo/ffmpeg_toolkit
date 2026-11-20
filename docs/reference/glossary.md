# Glossary

Canonical vocabulary for ffmpeg_toolkit. Code identifiers in backticks.

**Task**:
A Pydantic-validated object (subclass of `FFCreateTask`) describing one FFmpeg/FFprobe operation; executed with `.render()`. Exposed as classes through `FF_TASKS`.
_Avoid_: Job, command.

**Partial task**:
A factory from `PARTIAL_TASKS` that pre-binds a **Task**'s options and returns a callable taking input/output paths; reusable across files.
_Avoid_: Preset, curried task.

**Render**:
Executing a **Task** via `.render()`: writes to a `_processing`-suffixed path, renames on success, returns the output path or raises `FFRenderException`.
_Avoid_: Run, execute.

**Segment list**:
A flat list `[t0, t1, t2, t3, ...]` of timestamps. Even-indexed intervals (0→1, 2→3, …) are kept; odd-indexed intervals (1→2, 3→4, …) are removed.
_Avoid_: Ranges, pairs list.

**Silence**:
An audio interval below the detection threshold; an odd interval of the silence **Segment list** (removed by default).
_Avoid_: Quiet part.

**Motionless**:
A video interval with no detected scene change; an odd interval of the motion **Segment list** (removed by default).
_Avoid_: Static, still segment.

**Segment operation**:
A set operation (`SegmentOperation`: `UNION`, `INTERSECTION`, `SOUND_ONLY`, `MOTION_ONLY`, `XOR`, `COMPLEMENT`) combining the silence and motion **Segment lists** in `CutByDetection`.
_Avoid_: Merge mode.

**Kwargs layering**:
The fixed dict-merge precedence that assembles FFmpeg arguments: input defaults → user input kwargs → input file → output defaults → user output kwargs → output file.
_Avoid_: Arg override chain.
