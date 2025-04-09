import os
from pathlib import Path
from .ffmpeg_toolkit import FF_TASKS, PARTIAL_TASKS, BatchTask
from . import ffmpeg_types as types

PACKAGE_NAME = "ffmpeg_converter"

# Set custom paths for ffmpeg and ffprobe executables
# Get the absolute path of the current file
current_file_path = Path(__file__).resolve()

# Set the ./bin path to the PATH environment variable
bin_path = current_file_path.parent / "bin"
os.environ["PATH"] = str(bin_path) + os.pathsep + os.environ["PATH"]


__all__: list[str] = ["types", "PARTIAL_TASKS", "FF_TASKS", "BatchTask"]
