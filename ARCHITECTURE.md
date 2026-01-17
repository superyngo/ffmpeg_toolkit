# FFmpeg Toolkit Architecture

This document provides a comprehensive technical overview of the FFmpeg Toolkit architecture, designed to help developers understand the system's design patterns, class hierarchies, and implementation details.

## Table of Contents

1. [Overview](#overview)
2. [Class Hierarchy](#class-hierarchy)
3. [Core Concepts](#core-concepts)
4. [Task Types](#task-types)
5. [FFprobe Tasks](#ffprobe-tasks)
6. [Command Generation](#command-generation)
7. [Segment Processing Pipeline](#segment-processing-pipeline)
8. [Batch Processing](#batch-processing)
9. [Error Handling](#error-handling)
10. [Default Values](#default-values)

---

## Overview

FFmpeg Toolkit is a Python wrapper around FFmpeg that provides an object-oriented interface for video processing operations. The toolkit uses Pydantic for parameter validation and follows a consistent pattern for all tasks:

1. Create a task object with configuration
2. Call `render()` to execute the task
3. Receive the output file path or error code

```python
from ffmpeg_toolkit import FF_TASKS

# Basic pattern
result = FF_TASKS.Speedup(
    input_file="input.mp4",
    output_file="output.mp4",
    multiple=2
).render()
```

---

## Class Hierarchy

```
BaseModel (Pydantic)
    |
    +-- FPCreateCommand (FFprobe base)
    |       |
    |       +-- FPCreateRender
    |               |
    |               +-- FPRenderTasks
    |                       Methods: encode(), is_valid_video(), duration(),
    |                                keyframes(), frame_per_s()
    |
    +-- FFCreateCommand (FFmpeg base)
            |
            +-- FFCreateTask (contains render() method)
                    |
                    +-- Custom (passthrough for custom FFmpeg args)
                    +-- Cut (extract video segment)
                    +-- Speedup (change playback speed)
                    +-- Jumpcut (alternating speed effect)
                    +-- Merge (combine multiple videos)
                    +-- SplitSegments (split at timestamps)
                    +-- CutSilenceRerender (remove silence with re-encoding)
                    +-- CutMotionlessRerender (remove motionless with re-encoding)
                    +-- _GetSilenceSegments (internal: detect silence)
                    +-- _GetMotionSegments (internal: detect motion)
                    |
                    +-- [Override render() method]
                    +-- KeepOrRemove (selective segment processing)
                    +-- PartitionVideo (split with per-segment processing)
                    +-- CutSilence (remove silence with stream copy)
                    +-- CutMotionless (remove motionless with stream copy)
```

### Enums and Type Classes

```
Enum (Python)
    +-- FunctionEnum (callable functions as enum values)
    |       +-- PARTIAL_TASKS
    |
    +-- ClassEnum (callable classes as enum values)
    |       +-- FF_TASKS
    |
    +-- DEFAULTS (configuration defaults)
    +-- ERROR_CODE (error code definitions)
    +-- _TASKS (task type identifiers)
    +-- _PROBE_TASKS (probe task identifiers)
    +-- CSFiltersInfo (filter script configurations)

StrEnum
    +-- VideoSuffix (supported video extensions)
```

---

## Core Concepts

### 1. Task Lifecycle

Every task follows this lifecycle:

```
1. Instantiation
   +-- Pydantic validates parameters
   +-- model_post_init() configures task-specific settings
       - Sets task_description
       - Builds output_kwargs
       - Configures exception handlers
       - Sets post_hook if needed

2. Optional Configuration
   +-- override_option() for runtime modifications

3. Execution
   +-- render() method called
       - Handles file paths
       - Checks for exceptions
       - Builds FFmpeg kwargs
       - Executes subprocess
       - Runs post_hook if defined
       - Returns output path or error code
```

### 2. Kwargs Flow

The toolkit uses a layered approach for FFmpeg arguments:

```
Input Kwargs Default     Output Kwargs Default
    |                           |
    v                           v
+-- hide_banner: ""         +-- loglevel: "warning"
+-- hwaccel: "auto"
    |                           |
    v                           v
Custom Input Kwargs         Custom Output Kwargs
    |                           |
    v                           v
Input File                  Output File
    |                           |
    +---------> Merged <--------+
                  |
                  v
         Final FFmpeg Command
```

### 3. File Path Handling

The `_handle_output_file_path()` function handles various scenarios:

| Input | Output |
|-------|--------|
| `None` | Auto-generate based on input filename + task description |
| `"-"` | Return as-is (stdout) |
| Directory | Create file in directory with auto-generated name |
| File path | Use as-is |

### 4. Temporary File Management

- Processing files use `_processing` suffix during execution
- Renamed to final name on success
- Temporary directories use `ffmpeg_toolkit_tmp_` prefix
- Automatic cleanup in post_hook functions

---

## Task Types

### Cut

Extracts a segment from a video.

**Parameters:**
- `ss`: Start time (HH:MM:SS format)
- `to`: End time (HH:MM:SS format)
- `rerender`: Whether to re-encode (False = stream copy)

**FFmpeg Command:**
```bash
# Stream copy (rerender=False)
ffmpeg -hide_banner -hwaccel auto -i input.mp4 \
  -loglevel warning \
  -ss 00:01:00 -to 00:02:00 \
  -c:v copy -c:a copy \
  -y output.mp4

# Re-encode (rerender=True)
ffmpeg -hide_banner -hwaccel auto -i input.mp4 \
  -loglevel warning \
  -ss 00:01:00 -to 00:02:00 \
  -y output.mp4
```

### Speedup

Changes video playback speed.

**Parameters:**
- `multiple`: Speed factor (e.g., 2 = double speed)

**Strategy Selection:**
- `multiple <= 5`: Uses PTS adjustment (better quality)
- `multiple > 5`: Uses frame selection (faster processing)

**FFmpeg Command (PTS method):**
```bash
ffmpeg -hide_banner -hwaccel auto -i input.mp4 \
  -loglevel warning \
  -vf "setpts=0.5*PTS" \
  -af "atempo=2" \
  -map 0 -shortest -fps_mode vfr -async 1 \
  -reset_timestamps 1 \
  -force_key_frames "expr:gte(t,n_forced*2)" \
  -y output.mp4
```

**FFmpeg Command (Frame selection method):**
```bash
ffmpeg -hide_banner -hwaccel auto -i input.mp4 \
  -loglevel warning \
  -vf "select='if(eq(n,0),1,gt(floor(n/10), floor((n-1)/10)))',setpts=N/FRAME_RATE/TB" \
  -af "aselect='if(eq(n,0),1,gt(floor(n/10), floor((n-1)/10)))',asetpts=N/SR/TB" \
  -map 0 -shortest -fps_mode vfr -async 1 \
  -reset_timestamps 1 \
  -force_key_frames "expr:gte(t,n_forced*2)" \
  -y output.mp4
```

### Jumpcut

Creates alternating speed effects.

**Parameters:**
- `b1_duration`: Duration of first segment (seconds)
- `b2_duration`: Duration of second segment (seconds)
- `b1_multiple`: Speed for first segment (0 = remove)
- `b2_multiple`: Speed for second segment (0 = remove)

**FFmpeg Command:**
```bash
ffmpeg -hide_banner -hwaccel auto -i input.mp4 \
  -loglevel warning \
  -vf "select='if(lte(mod(t, 10),5), <expr1>, <expr2>)',setpts=N/FRAME_RATE/TB" \
  -af "aselect='if(lte(mod(t, 10),5), <expr1>, <expr2>)',asetpts=N/SR/TB" \
  -map 0 -shortest -fps_mode vfr -async 1 \
  -reset_timestamps 1 \
  -force_key_frames "expr:gte(t,n_forced*2)" \
  -y output.mp4
```

### Merge

Combines multiple video files.

**Parameters:**
- `input_dir_or_files`: Directory or list of files

**Process:**
1. Creates temporary `input.txt` with file list
2. Uses FFmpeg concat demuxer
3. Cleans up temporary file

**FFmpeg Command:**
```bash
# input.txt format:
# file '/path/to/video1.mp4'
# file '/path/to/video2.mp4'

ffmpeg -hide_banner -hwaccel auto \
  -f concat -safe 0 -i input.txt \
  -loglevel warning \
  -c:v copy -c:a copy \
  -y output.mp4
```

### SplitSegments

Splits video at specified timestamps.

**Parameters:**
- `video_segments`: List of timestamps to split at
- `output_dir`: Output directory

**FFmpeg Command:**
```bash
ffmpeg -hide_banner -hwaccel auto -i input.mp4 \
  -loglevel warning \
  -c:v copy -c:a copy \
  -f segment \
  -segment_times "00:01:00,00:02:00,00:03:00" \
  -segment_format mp4 \
  -reset_timestamps 1 \
  -y output/%d_video.mp4
```

### CutSilence / CutSilenceRerender

Removes silent segments from video.

**Parameters:**
- `dB`: Audio threshold in dB (default: -21)
- `sampling_duration`: Minimum silence duration (default: 0.2s)
- `seg_min_duration`: Minimum segment duration to keep
- `even_further`: Processing for silent segments
- `odd_further`: Processing for non-silent segments

**Process (CutSilence):**
```
1. Detect silence using silencedetect filter
2. Extract keyframes
3. Adjust segments (minimum length, keyframe alignment, merge overlaps)
4. Split video at segment boundaries
5. Process each segment according to even_further/odd_further
6. Merge remaining segments
```

### CutMotionless / CutMotionlessRerender

Removes motionless segments from video.

**Parameters:**
- `threshold`: Motion detection threshold (default: 0.0095)
- `sampling_duration`: Sample interval (default: 0.2s)
- Other parameters same as CutSilence

### KeepOrRemove

Selectively processes video segments.

**Parameters:**
- `video_segments`: List of segment boundaries
- `even_further`: Processing for even-indexed segments
- `odd_further`: Processing for odd-indexed segments
- `remove_temp_handle`: Whether to cleanup temp files

**FurtherMethod Options:**
- `None`: Keep segment as-is
- `"remove"`: Delete segment
- `PARTIAL_TASK`: Apply processing function

### PartitionVideo

Splits video into portions with custom processing.

**Parameters:**
- `count`: Number of equal segments
- `portion_method`: List of (ratio, processing) tuples
- `output_dir`: Output directory

**portion_method Format:**
```python
[
    (1, None),                          # Keep as-is
    (1, PARTIAL_TASKS.speedup(2)),     # Speed up 2x
    (1, "remove"),                      # Remove
]
```

---

## FFprobe Tasks

All probe tasks follow the pattern:
```python
result = FPRenderTasks().method(input_file).render()
```

### duration()

Gets video duration in seconds.

**FFprobe Command:**
```bash
ffprobe -hide_banner -v error \
  -show_entries format=duration \
  -of default=noprint_wrappers=1:nokey=1 \
  -i input.mp4
```

### keyframes()

Gets list of keyframe timestamps.

**FFprobe Command:**
```bash
ffprobe -hide_banner -v error \
  -select_streams v:0 \
  -show_entries packet=pts_time,flags \
  -of json \
  -i input.mp4
```

### encode()

Gets encoding information (codec, bitrate, format).

**FFprobe Command:**
```bash
ffprobe -hide_banner -v error \
  -print_format json \
  -show_format \
  -show_streams \
  -i input.mp4
```

### frame_per_s()

Gets frames per second.

**FFprobe Command:**
```bash
ffprobe -hide_banner -v error \
  -select_streams v \
  -show_entries stream=r_frame_rate \
  -of csv=p=0 \
  -i input.mp4
```

### is_valid_video()

Validates if file is a valid video.

---

## Command Generation

### Dictionary to FFmpeg Args

The `_dic_to_ffmpeg_kwargs()` function converts Python dicts to FFmpeg CLI args:

```python
# Special mappings
arg_map = {
    "cv": "-c:v",       # Video codec
    "ca": "-c:a",       # Audio codec
    "bv": "-b:v",       # Video bitrate
    "ba": "-b:a",       # Audio bitrate
    "filterv": "-filter:v",
    "filtera": "-filter:a",
}

# Example
{"vf": "setpts=0.5*PTS", "map": 0}
# Becomes
["-vf", "setpts=0.5*PTS", "-map", "0"]
```

### Kwargs Merging

The `_create_ff_kwargs()` function merges arguments in order:

```python
ff_kwargs = (
    input_kwargs_default      # hide_banner, hwaccel
    | input_kwargs            # User input kwargs
    | {"i": input_file}       # Input file
    | output_kwargs_default   # loglevel
    | output_kwargs           # User output kwargs
    | {"y": output_file}      # Output file (overwrite)
)
```

---

## Segment Processing Pipeline

The segment adjustment pipeline ensures clean cuts:

```
Input Segments
      |
      v
+-------------------------------------+
| _ensure_minimum_segment_length()   |
| - Ensures each segment >= minimum   |
| - Extends short segments            |
| - Returns [] if too short overall   |
+-------------------------------------+
      |
      v
+-------------------------------------+
| _adjust_segments_to_keyframes()    |
| - Aligns start times to keyframes   |
| - Aligns end times to keyframes     |
| - Prevents black frames             |
+-------------------------------------+
      |
      v
+-------------------------------------+
| _merge_overlapping_segments()      |
| - Sorts segments by start time      |
| - Merges overlapping segments       |
| - Returns clean segment list        |
+-------------------------------------+
      |
      v
Adjusted Segments
```

---

## Batch Processing

### BatchTask Class

```python
class BatchTask(BaseModel):
    input_folder_path: Path           # Source directory
    output_folder_path: Path | None   # Destination directory
    walkthrough: bool = False         # Recurse subdirectories
    valid_extensions: ValidExtensions # File types to process
    input_kwargs: FFKwargs            # Input args for all files
    output_kwargs: FFKwargs           # Output args for all files
    delete_after: bool = False        # Delete originals
    post_hook: Callable | None        # Per-file callback
```

### PARTIAL_TASKS Factory

Creates reusable task functions:

```python
# Create function
speedup_2x = PARTIAL_TASKS.speedup(multiple=2)

# Use directly
speedup_2x("input.mp4", "output.mp4")

# Use with batch
batch.render(speedup_2x)
```

---

## Error Handling

### FFRenderException Structure

```python
FFRenderException = TypedDict('FFRenderException', {
    'code': int | ERROR_CODE,
    'message': str,
    'hook': Callable | None,  # Cleanup function
})
```

### ERROR_CODE Enum

```python
class ERROR_CODE(Enum):
    DURATION_LESS_THAN_ZERO = auto()
    NO_VALID_SEGMENTS = auto()
    FAILED_TO_CUT = auto()
    NO_VIDEO_SEGMENTS = auto()
```

### Exception Flow

```python
def render(self):
    if self.exception is not None:
        logger.error(self.exception["message"])
        self.exception.get("hook", lambda: None)()  # Run cleanup
        return self.exception["code"]
    # ... continue with normal execution
```

---

## Default Values

### DEFAULTS Enum

| Name | Value | Description |
|------|-------|-------------|
| `num_cores` | `os.cpu_count() or 4` | Parallel processing cores |
| `hwaccel` | `"auto"` | Hardware acceleration |
| `loglevel` | `"warning"` | FFmpeg log level |
| `keyframe_interval` | `2` | Forced keyframe interval (seconds) |
| `speedup_multiple` | `2` | Default speedup factor |
| `speedup_task_threshold` | `5` | PTS vs frame selection threshold |
| `db_threshold` | `-21` | Silence detection threshold (dB) |
| `motionless_threshold` | `0.0095` | Motion detection threshold |
| `sampling_duration` | `0.2` | Detection sample interval (seconds) |
| `seg_min_duration` | `0` | Minimum segment duration |
| `temp_prefix` | `"ffmpeg_toolkit_tmp_"` | Temp file prefix |

### Supported Video Formats

```python
class VideoSuffix(StrEnum):
    MP4 = auto()
    MKV = auto()
    AVI = auto()
```

---

## File Structure

```
ffmpeg_toolkit/
+-- src/
|   +-- ffmpeg_toolkit/
|       +-- __init__.py           # Package exports
|       +-- ffmpeg_toolkit.py     # High-level API (FF_TASKS, PARTIAL_TASKS, BatchTask)
|       +-- ffmpeg_toolkit_core.py # Core implementation
|       +-- ffmpeg_types.py       # Type definitions
+-- test/
|   +-- test.py                   # Test examples
+-- README.MD                     # User documentation
+-- ARCHITECTURE.md               # This file
+-- CHANGELOG.md                  # Version history
```
