# FFmpeg Converter Toolkit

This project offers tools to enhance your experience with FFmpeg. It includes scripts for tasks such as speeding up videos, removing silence/motionless, performing jump cuts, and probing encoding information. Simplify your video processing workflows with these solutions.

## Installation

```
pip install ffmpeg-toolkit
```

## Usage

### Basic Usage

The toolkit can be used both as a command-line tool and as a Python library.

### Python Library Usage
(Written by Chatpgt so use follow with cautions)

```python
from ffmpeg_toolkit import FF_TASKS, PARTIAL_TASKS

# Speed up a video
FF_TASKS.Speedup(input_file="input.mp4", output_file="output.mp4", multiple=2).render()

# Create a jump cut effect (alternate between normal speed and removing segments)
FF_TASKS.Jumpcut(
    input_file="input.mp4", 
    output_file="output.mp4",
    b1_duration=5,      # Keep 5 seconds
    b2_duration=5,      # Then skip 5 seconds
    b1_multiple=1,      # Play first segment at normal speed
    b2_multiple=0       # Remove second segment
).render()

# Cut a segment from a video
FF_TASKS.Cut(
    input_file="input.mp4", 
    output_file="output.mp4",
    ss="00:01:30",     # Start time
    to="00:02:45",     # End time
    rerender=False     # Use stream copy instead of re-encoding
).render()

# Remove silent parts from a video
FF_TASKS.CutSilence(
    input_file="input.mp4", 
    output_file="output.mp4",
    dB=-21,                 # Audio threshold level in dB
    sampling_duration=0.2,  # Minimum silence duration to detect
    seg_min_duration=0      # Minimum duration of segments to keep
).render()

# Remove motionless parts from a video
FF_TASKS.CutMotionless(
    input_file="input.mp4", 
    output_file="output.mp4",
    threshold=0.0095,       # Scene change threshold
    sampling_duration=0.2,  # Duration between samples
    seg_min_duration=0      # Minimum duration of segments to keep
).render()

# Split a video into multiple segments
FF_TASKS.PartitionVideo(
    input_file="input.mp4",
    count=3,               # Number of equal parts
    output_dir="segments"  # Directory to save split segments
).render()

# Using partial tasks for simpler syntax
speedup_task = PARTIAL_TASKS.speedup(multiple=2)
speedup_task("input.mp4", "output.mp4")

# Probe video information
from ffmpeg_toolkit import FPRenderTasks

# Get video duration
duration = FPRenderTasks().duration("input.mp4").render()

# Get video encoding details
encoding_info = FPRenderTasks().encode("input.mp4").render()

# Check if a file is a valid video
is_valid = FPRenderTasks().is_valid_video("input.mp4").render()
```

### Advanced Usage

```python
from ffmpeg_toolkit import FF_TASKS, PARTIAL_TASKS, FFKwargs

# Custom FFmpeg commands
custom_output_kwargs: FFKwargs = {
    "c:v": "libx264",
    "crf": "23",
    "preset": "medium",
    "c:a": "aac",
    "b:a": "192k"
}

FF_TASKS.Custom(
    input_file="input.mp4",
    output_file="output.mp4",
    output_kwargs=custom_output_kwargs
).render()

# Chaining operations
# First cut a segment, then speed it up
cut_segment = FF_TASKS.Cut(
    input_file="input.mp4",
    output_file="temp.mp4",
    ss="00:01:30",
    to="00:02:45"
).render()

FF_TASKS.Speedup(
    input_file=cut_segment,
    output_file="final.mp4",
    multiple=2
).render()

# Process multiple files with concurrent execution
import glob
from concurrent.futures import ThreadPoolExecutor

speedup_task = PARTIAL_TASKS.speedup(multiple=2)

input_files = glob.glob("input/*.mp4")
with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(speedup_task, input_file, f"output/{i}_output.mp4")
        for i, input_file in enumerate(input_files)
    ]
    for future in futures:
        future.result()
```

PyPi Page[https://pypi.org/project/ffmpeg-toolkit/#description]
Github [https://github.com/superyngo/ffmpeg_toolkit]