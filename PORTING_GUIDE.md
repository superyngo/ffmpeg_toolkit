# FFmpeg Toolkit Porting Guide

This guide provides detailed information for developers who want to port the FFmpeg Toolkit to other programming languages or platforms. It covers the core abstractions, algorithms, and patterns that need to be implemented.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Core Abstractions](#core-abstractions)
4. [Key Algorithms](#key-algorithms)
5. [FFmpeg Command Patterns](#ffmpeg-command-patterns)
6. [Output Parsing](#output-parsing)
7. [Segment Processing](#segment-processing)
8. [Implementation Checklist](#implementation-checklist)
9. [Language-Specific Considerations](#language-specific-considerations)

---

## Overview

The FFmpeg Toolkit is essentially a domain-specific language (DSL) for video editing operations that compiles to FFmpeg commands. When porting, you need to implement:

1. **Command Builder**: Convert structured parameters to FFmpeg CLI arguments
2. **Task Classes**: Encapsulate common video operations
3. **Output Parser**: Parse FFmpeg/FFprobe output for analysis tasks
4. **Segment Processor**: Handle video splitting, merging, and segment manipulation
5. **Batch Processor**: Apply operations to multiple files

### Architecture Pattern

```
User API (High-level tasks)
        |
        v
Task Configuration (Pydantic models in Python)
        |
        v
Command Builder (Dictionary to CLI args)
        |
        v
Subprocess Executor (Run FFmpeg/FFprobe)
        |
        v
Output Parser (Extract data from output)
        |
        v
Result Handler (Return paths or error codes)
```

---

## Prerequisites

### Required External Dependencies

- **FFmpeg**: Must be installed and accessible in PATH
- **FFprobe**: Usually bundled with FFmpeg

### Version Compatibility

The toolkit uses FFmpeg features available in version 4.0+. Key features used:

- `hwaccel auto` (hardware acceleration)
- `fps_mode vfr` (variable frame rate)
- `force_key_frames` expression syntax
- `silencedetect` audio filter
- `select` and `aselect` filters
- `metadata=print` for scene detection
- `concat` demuxer with `-safe 0`

---

## Core Abstractions

### 1. Parameter Dictionary (FFKwargs)

The fundamental data structure is a key-value dictionary that maps to FFmpeg arguments:

```python
# Python type
FFKwargs = dict[str, str | Path | float | int]
```

**Equivalent in other languages:**

```typescript
// TypeScript
type FFKwargs = Record<string, string | number>;

// Go
type FFKwargs map[string]interface{}

// Rust
type FFKwargs = HashMap<String, Value>;

// Java
Map<String, Object> ffKwargs;
```

### 2. Task Base Class

Every task must implement:

```pseudocode
class FFCreateTask:
    input_file: Path
    output_file: Path | None
    input_kwargs: FFKwargs
    output_kwargs: FFKwargs
    task_description: String
    exception: Optional[Exception]
    post_hook: Optional[Callback]

    method model_post_init():
        # Called after construction
        # Set task_description
        # Build output_kwargs
        # Configure error handlers

    method override_option(options):
        # Allow runtime configuration changes
        return self

    method render():
        # Validate and transform file paths
        # Check for exceptions
        # Build FFmpeg command
        # Execute subprocess
        # Run post_hook if defined
        # Return output path or error code
```

### 3. Command Builder Function

Converts dictionary to CLI arguments:

```pseudocode
function dic_to_ffmpeg_kwargs(kwargs: Dict) -> List[String]:
    args = []

    # Special argument mappings
    arg_map = {
        "cv": "-c:v",
        "ca": "-c:a",
        "bv": "-b:v",
        "ba": "-b:a",
        "filterv": "-filter:v",
        "filtera": "-filter:a"
    }

    for key, value in kwargs:
        # Get mapped key or add dash prefix
        cli_key = arg_map.get(key, "-" + key)
        args.append(cli_key)

        # Empty string means flag-only argument
        if value != "":
            args.append(str(value))

    return args
```

### 4. Kwargs Merger Function

Combines default and custom arguments:

```pseudocode
function create_ff_kwargs(input_file, output_file, input_kwargs, output_kwargs):
    input_defaults = {
        "hide_banner": "",
        "hwaccel": "auto"
    }

    output_defaults = {
        "loglevel": "warning"
    }

    # Merge in order (later overrides earlier)
    return merge(
        input_defaults,
        input_kwargs,
        {"i": input_file},
        output_defaults,
        output_kwargs,
        {"y": output_file}
    )
```

---

## Key Algorithms

### 1. Timestamp Conversion

```pseudocode
function timestamp_to_seconds(timestamp: String) -> Float:
    # Input: "HH:MM:SS" or "HH:MM:SS.mmm"
    parts = timestamp.split(":")
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

function seconds_to_timestamp(seconds: Float) -> String:
    hours = floor(seconds / 3600)
    remaining = seconds % 3600
    minutes = floor(remaining / 60)
    secs = remaining % 60
    return format("{:02d}:{:02d}:{:06.3f}", hours, minutes, secs)
```

### 2. Speedup Filter Generation

```pseudocode
function create_speedup_kwargs(multiple: Float) -> Dict:
    THRESHOLD = 5  # Switch strategies at this point

    if multiple > THRESHOLD:
        # Frame selection method for large speedups
        vf = "select='if(eq(n,0),1,gt(floor(n/{mult}), floor((n-1)/{mult})))',setpts=N/FRAME_RATE/TB"
        af = "aselect='if(eq(n,0),1,gt(floor(n/{mult}), floor((n-1)/{mult})))',asetpts=N/SR/TB"
    else:
        # PTS manipulation for smaller speedups
        vf = "setpts={factor}*PTS"  # factor = 1/multiple
        af = "atempo={multiple}"

    return {
        "vf": vf.format(mult=multiple, factor=1/multiple),
        "af": af.format(mult=multiple, factor=1/multiple),
        "map": 0,
        "shortest": "",
        "fps_mode": "vfr",
        "async": 1,
        "reset_timestamps": "1",
        "force_key_frames": "expr:gte(t,n_forced*2)"
    }
```

### 3. Jumpcut Filter Generation

```pseudocode
function create_jumpcut_kwargs(b1_duration, b2_duration, b1_multiple, b2_multiple):
    # Generate expression for each block
    function multiple_expr(mult):
        if mult == 0:
            return "0"  # Remove this block
        return "if(eq(n,0),1,gt(floor(n/{mult}), floor((n-1)/{mult})))".format(mult=mult)

    expr1 = multiple_expr(b1_multiple)
    expr2 = multiple_expr(b2_multiple)

    # Alternating selection based on time
    total_period = b1_duration + b2_duration
    frame_select = "if(lte(mod(t, {period}),{b1}), {expr1}, {expr2})"
    frame_select = frame_select.format(
        period=total_period,
        b1=b1_duration,
        expr1=expr1,
        expr2=expr2
    )

    return {
        "vf": "select='" + frame_select + "',setpts=N/FRAME_RATE/TB",
        "af": "aselect='" + frame_select + "',asetpts=N/SR/TB",
        "map": 0,
        "shortest": "",
        "fps_mode": "vfr",
        "async": 1,
        "reset_timestamps": "1",
        "force_key_frames": "expr:gte(t,n_forced*2)"
    }
```

### 4. Segment Partitioning

```pseudocode
function get_segments_from_parts_count(duration, parts_count, portions):
    # portions is optional list of relative sizes
    # e.g., [1, 2, 1] means 25%, 50%, 25%

    if portions is None:
        portions = [1] * parts_count

    if sum(portions) != parts_count:
        raise Error("Portions must sum to parts_count")

    segment_length = duration / parts_count
    split_points = []

    cumulative = 0
    for portion in portions[:-1]:  # Exclude last
        cumulative += portion
        time = segment_length * cumulative
        split_points.append(seconds_to_timestamp(time))

    return split_points
```

---

## FFmpeg Command Patterns

### Detection Commands

#### Silence Detection

```bash
ffmpeg -hide_banner -hwaccel auto -i input.mp4 \
  -af "silencedetect=n={dB}dB:d={min_duration}" \
  -vn \
  -loglevel info \
  -f null -
```

**Output Pattern:**
```
[silencedetect @ 0x...] silence_start: 5.23
[silencedetect @ 0x...] silence_end: 8.71 | silence_duration: 3.48
```

#### Motion Detection

```bash
ffmpeg -hide_banner -hwaccel auto -i input.mp4 \
  -vf "select='not(mod(n,floor({fps}*{sample_duration})))*gte(scene,0)',metadata=print" \
  -an \
  -loglevel info \
  -f null -
```

**Output Pattern:**
```
frame:123 pts:12300 pts_time:5.125
lavfi.scene_score=0.0123
```

### Probe Commands

#### Duration

```bash
ffprobe -hide_banner -v error \
  -show_entries format=duration \
  -of default=noprint_wrappers=1:nokey=1 \
  -i input.mp4
```

**Output:** `123.456` (seconds as float)

#### Keyframes

```bash
ffprobe -hide_banner -v error \
  -select_streams v:0 \
  -show_entries packet=pts_time,flags \
  -of json \
  -i input.mp4
```

**Output:**
```json
{
  "packets": [
    {"pts_time": "0.000000", "flags": "K__"},
    {"pts_time": "2.000000", "flags": "K__"},
    ...
  ]
}
```

#### Encoding Info

```bash
ffprobe -hide_banner -v error \
  -print_format json \
  -show_format \
  -show_streams \
  -i input.mp4
```

### Processing Commands

#### Stream Copy Cut

```bash
ffmpeg -hide_banner -hwaccel auto -i input.mp4 \
  -loglevel warning \
  -ss {start} -to {end} \
  -c:v copy -c:a copy \
  -y output.mp4
```

#### Segment Split

```bash
ffmpeg -hide_banner -hwaccel auto -i input.mp4 \
  -loglevel warning \
  -c:v copy -c:a copy \
  -f segment \
  -segment_times "time1,time2,time3" \
  -segment_format mp4 \
  -reset_timestamps 1 \
  -y output/%d_name.mp4
```

#### Concat Merge

```bash
# Create input.txt:
# file '/path/to/video1.mp4'
# file '/path/to/video2.mp4'

ffmpeg -hide_banner -hwaccel auto \
  -f concat -safe 0 -i input.txt \
  -loglevel warning \
  -c:v copy -c:a copy \
  -y output.mp4
```

#### Filter Script Re-render

```bash
ffmpeg -hide_banner -hwaccel auto -i input.mp4 \
  -loglevel warning \
  -filter_script:v /tmp/video_filter.txt \
  -filter_script:a /tmp/audio_filter.txt \
  -y output.mp4
```

**video_filter.txt:**
```
select='between(t,0,5)+between(t,10,15)+between(t,20,25)', setpts=N/FRAME_RATE/TB
```

**audio_filter.txt:**
```
aselect='between(t,0,5)+between(t,10,15)+between(t,20,25)', asetpts=N/SR/TB
```

---

## Output Parsing

### Silence Detection Parser

```pseudocode
function extract_non_silence_info(output: String):
    # Extract total duration
    duration_match = regex_find(output, r"Duration: (.+?),")
    total_duration = timestamp_to_seconds(duration_match)

    # Extract silence timestamps
    pattern = r"silence_(?:start|end): ([0-9.]+)"
    matches = regex_find_all(output, pattern)
    timestamps = [float(m) for m in matches]

    # Add boundaries
    segments = [0.0] + timestamps + [total_duration]

    # Extract silence durations
    duration_pattern = r"silence_duration: ([0-9.]+)"
    durations = regex_find_all(output, duration_pattern)
    total_silence = sum(float(d) for d in durations)

    return (segments, total_duration, total_silence)
```

### Motion Detection Parser

```pseudocode
function extract_motion_info(output: String):
    # Extract total duration
    duration_match = regex_find(output, r"Duration: (.+?),")
    total_duration = timestamp_to_seconds(duration_match)

    # Extract timestamps
    pts_pattern = r"pts_time:([0-9.]+)"
    pts_matches = regex_find_all(output, pts_pattern)

    # Extract scene scores
    score_pattern = r"lavfi.scene_score=([0-9.]+)"
    score_matches = regex_find_all(output, score_pattern)

    # Create timestamp -> score mapping
    motion_info = {}
    for i in range(min(len(pts_matches), len(score_matches))):
        motion_info[float(pts_matches[i])] = float(score_matches[i])

    return (motion_info, total_duration)

function extract_motion_segments(motion_info: Dict, threshold: Float):
    break_points = []
    prev_above = False

    for time, score in sorted(motion_info.items()):
        # Detect state transitions
        if (score > threshold and not prev_above) or \
           (score <= threshold and prev_above):
            break_points.append(time)
            prev_above = score > threshold

    return break_points
```

---

## Segment Processing

### Minimum Length Enforcement

```pseudocode
function ensure_minimum_segment_length(segments, min_duration, total_duration):
    if min_duration == 0 or len(segments) == 0:
        return segments

    if len(segments) % 2 != 0:
        raise Error("Segments must be pairs of start/end times")

    result = []
    for i in range(0, len(segments), 2):
        start = segments[i]
        end = segments[i + 1]
        duration = end - start

        if duration >= min_duration or len(segments) == 2:
            result.extend([start, end])
            continue

        if i == len(segments) - 2:
            # Last segment - extend backwards
            start = max(0, end - min_duration)
        else:
            # Extend both directions
            diff = min_duration - duration
            start = max(0, start - diff / 2)
            end = min(start + min_duration, total_duration)

        result.extend([start, end])

    # Check if result is long enough overall
    if result[-1] - result[0] < min_duration:
        return []

    return result
```

### Keyframe Alignment

```pseudocode
function adjust_segments_to_keyframes(segments, keyframes):
    result = []
    kf_index = 0

    for i, time in enumerate(segments):
        if i % 2 == 0:  # Start time - find largest keyframe <= time
            while kf_index < len(keyframes) and keyframes[kf_index] <= time:
                kf_index += 1
            adjusted = keyframes[kf_index - 1] if kf_index > 0 else time
        else:  # End time - find smallest keyframe >= time
            while kf_index < len(keyframes) and keyframes[kf_index] < time:
                kf_index += 1
            adjusted = keyframes[kf_index] if kf_index < len(keyframes) else time

        result.append(adjusted)

    return result
```

### Overlap Merging

```pseudocode
function merge_overlapping_segments(segments):
    # Convert to list of (start, end) pairs
    pairs = [(segments[i], segments[i+1]) for i in range(0, len(segments), 2)]
    pairs.sort(key=lambda p: p[0])

    if len(pairs) == 0:
        return []

    result = []
    current_start, current_end = pairs[0]

    for start, end in pairs[1:]:
        if start <= current_end:
            # Overlapping - extend current segment
            current_end = max(current_end, end)
        else:
            # No overlap - save current and start new
            result.extend([current_start, current_end])
            current_start, current_end = start, end

    result.extend([current_start, current_end])
    return result
```

### Complete Pipeline

```pseudocode
function adjust_segments_pipe(segments, min_duration, total_duration, keyframes):
    # Step 1: Ensure minimum length
    if min_duration > 0:
        segments = ensure_minimum_segment_length(segments, min_duration, total_duration)

    # Step 2: Align to keyframes
    segments = adjust_segments_to_keyframes(segments, keyframes)

    # Step 3: Ensure even number of elements
    if len(segments) % 2 == 1:
        segments.append(total_duration)

    # Step 4: Merge overlapping segments
    segments = merge_overlapping_segments(segments)

    return segments
```

### Segment Set Operations

These functions combine segments from different detection sources (e.g., silence and motion detection).

#### Union Segments (A OR B)

```pseudocode
function union_segments(segs1, segs2):
    if not segs1:
        return merge_overlapping_segments(segs2) if segs2 else []
    if not segs2:
        return merge_overlapping_segments(segs1)

    combined = segs1 + segs2
    return merge_overlapping_segments(combined)
```

#### Intersect Segments (A AND B)

```pseudocode
function intersect_segments(segs1, segs2):
    if not segs1 or not segs2:
        return []

    # Convert to (start, end) pairs
    intervals1 = [(segs1[i], segs1[i+1]) for i in range(0, len(segs1), 2)]
    intervals2 = [(segs2[i], segs2[i+1]) for i in range(0, len(segs2), 2)]

    result = []
    i, j = 0, 0

    while i < len(intervals1) and j < len(intervals2):
        start1, end1 = intervals1[i]
        start2, end2 = intervals2[j]

        # Find overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        if overlap_start < overlap_end:
            result.extend([overlap_start, overlap_end])

        # Move pointer for interval that ends first
        if end1 <= end2:
            i += 1
        else:
            j += 1

    return result
```

#### Difference Segments (A - B)

```pseudocode
function difference_segments(segs1, segs2):
    if not segs1:
        return []
    if not segs2:
        return merge_overlapping_segments(segs1)

    intervals1 = [(segs1[i], segs1[i+1]) for i in range(0, len(segs1), 2)]
    intervals2 = sorted([(segs2[i], segs2[i+1]) for i in range(0, len(segs2), 2)])

    result = []

    for start1, end1 in intervals1:
        current_start = start1

        for start2, end2 in intervals2:
            if start2 >= end1:
                break
            if end2 <= current_start:
                continue

            # Add part before overlap
            if start2 > current_start:
                result.extend([current_start, start2])

            # Move past overlap
            current_start = max(current_start, end2)

            if current_start >= end1:
                break

        # Add remaining part
        if current_start < end1:
            result.extend([current_start, end1])

    return result
```

#### XOR Segments

```pseudocode
function xor_segments(segs1, segs2):
    if not segs1:
        return merge_overlapping_segments(segs2) if segs2 else []
    if not segs2:
        return merge_overlapping_segments(segs1)

    # XOR = (A - B) union (B - A)
    diff1 = difference_segments(segs1, segs2)
    diff2 = difference_segments(segs2, segs1)
    return union_segments(diff1, diff2)
```

#### Complement Segments

```pseudocode
function complement_segments(segs, total_duration):
    if not segs:
        return [0.0, total_duration] if total_duration > 0 else []

    merged = merge_overlapping_segments(segs)
    result = []
    current_pos = 0.0

    for i in range(0, len(merged), 2):
        seg_start = merged[i]
        seg_end = merged[i + 1]

        if current_pos < seg_start:
            result.extend([current_pos, seg_start])

        current_pos = seg_end

    if current_pos < total_duration:
        result.extend([current_pos, total_duration])

    return result
```

#### SegmentOperation Enum

```pseudocode
enum SegmentOperation:
    UNION          # A OR B - sound or motion
    INTERSECTION   # A AND B - sound and motion
    SOUND_ONLY     # A - B - sound but not motion
    MOTION_ONLY    # B - A - motion but not sound
    XOR            # (A OR B) - (A AND B) - one but not both
    COMPLEMENT     # NOT (A OR B) - neither sound nor motion
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure

- [ ] Implement subprocess execution wrapper for FFmpeg/FFprobe
- [ ] Implement dictionary to CLI argument converter
- [ ] Implement kwargs merger function
- [ ] Implement timestamp conversion utilities
- [ ] Implement file path handler for various scenarios
- [ ] Set up configuration defaults

### Phase 2: Probe Tasks

- [ ] Implement duration probe
- [ ] Implement keyframes probe
- [ ] Implement encoding info probe
- [ ] Implement frames per second probe
- [ ] Implement video validation probe

### Phase 3: Basic Tasks

- [ ] Implement Cut task (stream copy)
- [ ] Implement Cut task (re-encode)
- [ ] Implement Speedup task (PTS method)
- [ ] Implement Speedup task (frame selection method)
- [ ] Implement Merge task
- [ ] Implement SplitSegments task

### Phase 4: Detection Tasks

- [ ] Implement silence detection
- [ ] Implement silence output parser
- [ ] Implement motion detection
- [ ] Implement motion output parser

### Phase 5: Advanced Tasks

- [ ] Implement Jumpcut task
- [ ] Implement segment adjustment pipeline
- [ ] Implement KeepOrRemove task
- [ ] Implement CutSilence task
- [ ] Implement CutMotionless task
- [ ] Implement CutSilenceRerender task
- [ ] Implement CutMotionlessRerender task
- [ ] Implement PartitionVideo task

### Phase 5.5: Segment Set Operations

- [ ] Implement union segments function
- [ ] Implement intersect segments function
- [ ] Implement difference segments function
- [ ] Implement XOR segments function
- [ ] Implement complement segments function
- [ ] Implement SegmentOperation enum
- [ ] Implement CutByDetection task

### Phase 6: Batch Processing

- [ ] Implement file discovery function
- [ ] Implement batch task executor
- [ ] Implement partial task factory pattern

### Phase 7: Polish

- [ ] Implement comprehensive error handling
- [ ] Implement logging
- [ ] Implement temporary file cleanup
- [ ] Add progress callbacks
- [ ] Write tests

---

## Language-Specific Considerations

### TypeScript / Node.js

```typescript
import { spawn, exec } from 'child_process';
import * as path from 'path';

interface FFKwargs {
  [key: string]: string | number | boolean;
}

function dicToFFmpegKwargs(kwargs: FFKwargs): string[] {
  const argMap: Record<string, string> = {
    cv: '-c:v',
    ca: '-c:a',
    bv: '-b:v',
    ba: '-b:a',
  };

  const args: string[] = [];
  for (const [key, value] of Object.entries(kwargs)) {
    args.push(argMap[key] || `-${key}`);
    if (value !== '' && value !== true) {
      args.push(String(value));
    }
  }
  return args;
}

async function ffmpeg(kwargs: FFKwargs): Promise<string> {
  const args = dicToFFmpegKwargs(kwargs);
  return new Promise((resolve, reject) => {
    const proc = spawn('ffmpeg', args);
    let output = '';
    proc.stderr.on('data', (data) => output += data);
    proc.on('close', (code) => {
      if (code === 0) resolve(output);
      else reject(new Error(output));
    });
  });
}
```

### Go

```go
package ffmpeg

import (
    "fmt"
    "os/exec"
    "strconv"
    "strings"
)

type FFKwargs map[string]interface{}

var argMap = map[string]string{
    "cv": "-c:v",
    "ca": "-c:a",
    "bv": "-b:v",
    "ba": "-b:a",
}

func dicToFFmpegKwargs(kwargs FFKwargs) []string {
    var args []string
    for key, value := range kwargs {
        cliKey := argMap[key]
        if cliKey == "" {
            cliKey = "-" + key
        }
        args = append(args, cliKey)

        switch v := value.(type) {
        case string:
            if v != "" {
                args = append(args, v)
            }
        case int:
            args = append(args, strconv.Itoa(v))
        case float64:
            args = append(args, fmt.Sprintf("%f", v))
        }
    }
    return args
}

func FFmpeg(kwargs FFKwargs) (string, error) {
    args := dicToFFmpegKwargs(kwargs)
    cmd := exec.Command("ffmpeg", args...)
    output, err := cmd.CombinedOutput()
    return string(output), err
}
```

### Rust

```rust
use std::collections::HashMap;
use std::process::Command;

type FFKwargs = HashMap<String, String>;

fn dic_to_ffmpeg_kwargs(kwargs: &FFKwargs) -> Vec<String> {
    let arg_map: HashMap<&str, &str> = [
        ("cv", "-c:v"),
        ("ca", "-c:a"),
        ("bv", "-b:v"),
        ("ba", "-b:a"),
    ].iter().cloned().collect();

    let mut args = Vec::new();
    for (key, value) in kwargs {
        let cli_key = arg_map.get(key.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("-{}", key));
        args.push(cli_key);

        if !value.is_empty() {
            args.push(value.clone());
        }
    }
    args
}

fn ffmpeg(kwargs: FFKwargs) -> Result<String, std::io::Error> {
    let args = dic_to_ffmpeg_kwargs(&kwargs);
    let output = Command::new("ffmpeg")
        .args(&args)
        .output()?;

    Ok(String::from_utf8_lossy(&output.stderr).to_string())
}
```

### Java / Kotlin

```kotlin
typealias FFKwargs = Map<String, Any>

val argMap = mapOf(
    "cv" to "-c:v",
    "ca" to "-c:a",
    "bv" to "-b:v",
    "ba" to "-b:a"
)

fun dicToFFmpegKwargs(kwargs: FFKwargs): List<String> {
    return kwargs.flatMap { (key, value) ->
        val cliKey = argMap[key] ?: "-$key"
        when {
            value == "" -> listOf(cliKey)
            else -> listOf(cliKey, value.toString())
        }
    }
}

fun ffmpeg(kwargs: FFKwargs): String {
    val args = dicToFFmpegKwargs(kwargs)
    val process = ProcessBuilder(listOf("ffmpeg") + args)
        .redirectErrorStream(true)
        .start()
    return process.inputStream.bufferedReader().readText()
}
```

---

## Testing Recommendations

### Unit Tests

1. **Timestamp conversion** - Test edge cases (0, large values, fractional seconds)
2. **Kwargs conversion** - Test all special mappings and empty values
3. **Segment algorithms** - Test minimum length, keyframe alignment, overlap merging
4. **Output parsers** - Test with sample FFmpeg output
5. **Segment set operations** - Test union, intersection, difference, XOR, complement
   - Empty segments handling
   - Non-overlapping segments
   - Fully overlapping segments
   - Partial overlaps
   - Multiple segments

### Integration Tests

1. **Basic operations** - Cut, speedup, merge with actual video files
2. **Detection** - Verify silence/motion detection accuracy
3. **Complex workflows** - PartitionVideo with nested operations
4. **Combined detection** - CutByDetection with various set operations
5. **Error handling** - Invalid files, missing FFmpeg, etc.

### Test Video Files

Create test videos with known properties:
- 10 seconds of audio + 5 seconds silence + 10 seconds audio
- Static image for 5 seconds + motion for 5 seconds
- Multiple resolutions and codecs
