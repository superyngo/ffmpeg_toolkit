# FFmpeg Converter Toolkit

A comprehensive Python toolkit for video processing using FFmpeg. This project offers tools to enhance your experience with FFmpeg through a clean, object-oriented interface. It includes functionality for tasks such as speeding up videos, removing silence/motionless sections, performing jump cuts, batch processing, and probing encoding information.

## Features

- **Video Speed Manipulation**: Speed up or slow down videos with customizable factors
- **Jump Cut Effects**: Create dynamic jumpcut effects with configurable patterns
- **Silence Removal**: Automatically detect and remove silent segments from videos
- **Motionless Section Removal**: Detect and remove parts of videos with little or no motion
- **Video Partitioning**: Split videos into multiple parts with custom processing for each segment
- **Batch Processing**: Process multiple videos with the same operations
- **Video Merging**: Combine multiple video files into a single output
- **Video Information**: Probe video files for encoding details, duration, and more
- **Custom FFmpeg Commands**: Create and execute custom FFmpeg operations

## Installation

```
pip install ffmpeg-toolkit
```

The package includes bundled FFmpeg and FFprobe executables for Windows, so you don't need to install them separately.

## Basic Usage

The toolkit can be used as a Python library:

```python
from ffmpeg_toolkit import FF_TASKS

# Speed up a video
FF_TASKS.Speedup(
    input_file="input.mp4",
    output_file="output.mp4",
    multiple=2  # Double the speed
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
    sampling_duration=0.2   # Minimum silence duration to detect
).render()
```

## Task Types

### Speedup

```python
# Speed up a video by 2x
FF_TASKS.Speedup(
    input_file="input.mp4",
    output_file="output.mp4",
    multiple=2
).render()

# Speed up a video by 4x
FF_TASKS.Speedup(
    input_file="input.mp4",
    output_file="output_4x.mp4",
    multiple=4
).render()
```

### Jump Cut

Create a jump cut effect that alternates between segments at different speeds:

```python
# Create a jump cut effect
FF_TASKS.Jumpcut(
    input_file="input.mp4", 
    output_file="output.mp4",
    b1_duration=5,      # Keep 5 seconds
    b2_duration=5,      # Then skip 5 seconds
    b1_multiple=1,      # Play first segment at normal speed
    b2_multiple=0       # Remove second segment
).render()

# Create a creative effect with varying speeds
FF_TASKS.Jumpcut(
    input_file="input.mp4", 
    output_file="creative.mp4",
    b1_duration=3,      # First 3 seconds
    b2_duration=2,      # Next 2 seconds
    b1_multiple=1,      # Normal speed
    b2_multiple=2       # Double speed
).render()
```

### Remove Silent Parts

```python
# Remove silent parts with default settings
FF_TASKS.CutSilence(
    input_file="input.mp4", 
    output_file="output.mp4"
).render()

# More sensitive silence detection
FF_TASKS.CutSilence(
    input_file="input.mp4", 
    output_file="output.mp4",
    dB=-30,                 # Lower threshold (more sensitive)
    sampling_duration=0.1,  # Shorter silence detection window
    seg_min_duration=1      # Keep segments at least 1 second long
).render()

# With re-encoding for better quality
FF_TASKS.CutSilenceRerender(
    input_file="input.mp4", 
    output_file="output.mp4",
    dB=-21
).render()
```

### Remove Motionless Parts

```python
# Remove motionless parts with default settings
FF_TASKS.CutMotionless(
    input_file="input.mp4", 
    output_file="output.mp4"
).render()

# Custom motion detection sensitivity
FF_TASKS.CutMotionless(
    input_file="input.mp4", 
    output_file="output.mp4",
    threshold=0.015,        # Higher threshold (less sensitive)
    sampling_duration=0.3,  # Check motion every 0.3 seconds
    seg_min_duration=0.5    # Keep segments at least 0.5 seconds long
).render()

# With re-encoding for better quality
FF_TASKS.CutMotionlessRerender(
    input_file="input.mp4", 
    output_file="output.mp4",
    threshold=0.0095
).render()
```

### Merge Videos

```python
# Merge multiple video files
FF_TASKS.Merge(
    input_dir_or_files=["part1.mp4", "part2.mp4", "part3.mp4"],
    output_file="merged.mp4"
).render()

# Merge all videos in a directory
FF_TASKS.Merge(
    input_dir_or_files="video_parts/",
    output_file="merged.mp4"
).render()
```

### Partition Video

Split a video into multiple parts:

```python
# Split into 3 equal parts
FF_TASKS.PartitionVideo(
    input_file="input.mp4",
    count=3,               # Number of equal parts
    output_dir="segments"  # Directory to save split segments
).render()

# Custom partitioning with different processing methods
from ffmpeg_toolkit import PARTIAL_TASKS

FF_TASKS.PartitionVideo(
    input_file="input.mp4",
    output_dir="processed_segments",
    # Define 3 segments with custom processing
    portion_method=[
        (1, None),                               # Keep segment 1 as is
        (1, PARTIAL_TASKS.speedup(multiple=2)),  # Speed up segment 2
        (1, "remove")                            # Remove segment 3
    ]
).render()
```

## Simplified Workflow with PARTIAL_TASKS

For simpler syntax and reusable operations, use the `PARTIAL_TASKS` class:

```python
from ffmpeg_toolkit import PARTIAL_TASKS

# Create reusable task functions
speedup_2x = PARTIAL_TASKS.speedup(multiple=2)
remove_silence = PARTIAL_TASKS.cut_silence(dB=-25)

# Apply to files
speedup_2x("input.mp4", "output_2x.mp4")
remove_silence("input.mp4", "output_no_silence.mp4")

# Apply to multiple files
input_files = ["video1.mp4", "video2.mp4", "video3.mp4"]
for i, file in enumerate(input_files):
    speedup_2x(file, f"processed_{i}.mp4")
```

## Batch Processing

Process multiple video files with the same operations using the `BatchTask` class:

```python
from ffmpeg_toolkit import BatchTask, PARTIAL_TASKS, FF_TASKS

# Create a batch processing configuration
batch = BatchTask(
    input_folder_path="videos/",
    output_folder_path="processed_videos/",
    walkthrough=True,              # Process files in subdirectories
    valid_extensions={"mp4", "mkv"} # Only process these extensions
)

# Apply a task to all matching files
batch.render(PARTIAL_TASKS.speedup(multiple=2))
```

### BatchTask Options

```python
# Basic configuration
batch = BatchTask(
    input_folder_path="input_videos/",   # Source directory with videos
    output_folder_path="output_videos/", # Destination directory for processed videos
    walkthrough=True,                    # Also process videos in subdirectories
    valid_extensions={"mp4", "mkv", "avi"}, # Only process these file types
    delete_after=False,                  # Keep original files after processing
)

# Apply custom input/output parameters to all processed files
batch = BatchTask(
    input_folder_path="videos/",
    output_folder_path="compressed/",
    # FFmpeg input parameters for all files
    input_kwargs={"hwaccel": "cuda"},
    # FFmpeg output parameters for all files
    output_kwargs={"c:v": "libx264", "crf": "23", "preset": "medium"}
)

# With post-processing hook for each file
def after_file_processed(output_file):
    print(f"Finished processing {output_file}")
    # Additional operations can be performed here

batch = BatchTask(
    input_folder_path="videos/",
    output_folder_path="processed/",
    post_hook=after_file_processed
)
```

### Batch Processing Examples

```python
# Example 1: Speed up all videos in a directory
batch = BatchTask(
    input_folder_path="lectures/",
    output_folder_path="lectures_fast/",
)
batch.render(PARTIAL_TASKS.speedup(multiple=1.5))

# Example 2: Remove silence from all video files
batch = BatchTask(
    input_folder_path="interviews/",
    output_folder_path="interviews_edited/",
)
batch.render(PARTIAL_TASKS.cut_silence(dB=-30, sampling_duration=0.3))

# Example 3: Create jumpcuts for all videos
batch = BatchTask(
    input_folder_path="recordings/",
    output_folder_path="jumpcut_recordings/",
)
batch.render(PARTIAL_TASKS.jumpcut(
    b1_duration=2,      # Keep 2 seconds
    b2_duration=1,      # Then skip 1 second
    b1_multiple=1,      # Normal speed
    b2_multiple=0       # Remove segments
))

# Example 4: Converting file formats with custom encoding
batch = BatchTask(
    input_folder_path="raw_videos/",
    output_folder_path="converted/",
    output_kwargs={"c:v": "libx264", "preset": "slow", "crf": "18"}
)

# Create a custom task function for conversion
convert_to_mp4 = PARTIAL_TASKS.custom(
    output_kwargs={"c:v": "libx264", "c:a": "aac", "b:a": "128k"}
)
batch.render(convert_to_mp4)

# Example 5: Process all files with different extensions
batch = BatchTask(
    input_folder_path="mixed_media/",
    output_folder_path="processed/",
    valid_extensions={"mp4", "mkv", "avi", "mov", "webm"}
)
batch.render(PARTIAL_TASKS.speedup(multiple=2))
```

### Combining Batch Processing with Other Features

```python
# First define a batch task
batch = BatchTask(
    input_folder_path="raw_videos/",
    output_folder_path="temp_videos/",
    walkthrough=True
)

# First pass: Speed up all videos
batch.render(PARTIAL_TASKS.speedup(multiple=2))

# Create a new batch task for the second pass
second_batch = BatchTask(
    input_folder_path="temp_videos/",
    output_folder_path="final_videos/",
    # Delete intermediate files after processing
    delete_after=True
)

# Second pass: Remove silent parts from the sped-up videos
second_batch.render(PARTIAL_TASKS.cut_silence(dB=-25))
```

## Video Probing

Get information about video files:

```python
from ffmpeg_toolkit.ffmpeg_toolkit_core import FPRenderTasks

# Get video duration
duration = FPRenderTasks().duration("input.mp4").render()
print(f"Video duration: {duration} seconds")

# Get video encoding information
encoding_info = FPRenderTasks().encode("input.mp4").render()
print(f"Video codec: {encoding_info.get('vcodec')}")
print(f"Audio codec: {encoding_info.get('acodec')}")

# Get keyframe positions
keyframes = FPRenderTasks().keyframes("input.mp4").render()
print(f"Keyframes at: {keyframes} seconds")

# Check if a file is a valid video
is_valid = FPRenderTasks().is_valid_video("input.mp4").render()
print(f"Is valid video: {is_valid}")
```

## Advanced Usage

### Custom FFmpeg Commands

```python
from ffmpeg_toolkit import FF_TASKS, FFKwargs

# Custom FFmpeg command with specific encoding parameters
custom_output_kwargs = {
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
```

### Chaining Operations

```python
# First cut a segment, then speed it up
cut_segment = FF_TASKS.Cut(
    input_file="input.mp4",
    ss="00:01:30",
    to="00:02:45"
).render()

final_output = FF_TASKS.Speedup(
    input_file=cut_segment,
    output_file="final.mp4",
    multiple=2
).render()
```

### Parallel Processing

```python
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

## Links

- [PyPI Package](https://pypi.org/project/ffmpeg-toolkit/)
- [GitHub Repository](https://github.com/superyngo/ffmpeg_toolkit)
