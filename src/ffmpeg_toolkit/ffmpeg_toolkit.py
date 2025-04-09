from pathlib import Path
from typing import Callable

from pydantic import BaseModel, computed_field, Field

from .ffmpeg_toolkit_core import (
    DEFAULTS,
    Custom,
    Cut,
    CutMotionless,
    CutMotionlessRerender,
    CutSilence,
    CutSilenceRerender,
    Jumpcut,
    Merge,
    PartitionVideo,
    PortionMethod,
    Speedup,
)
from .ffmpeg_types import (
    ClassEnum,
    FFKwargs,
    FunctionEnum,
    FurtherMethod,
    OptionFFRender,
    PARTIAL_TASK,
    ValidExtensions,
    VideoSuffix,
)

try:
    from .ffmpeg_toolkit_core import logger  # type: ignore
except ImportError:
    # Fallback to a default value
    class logger:
        """Logger class that provides basic logging functionality when the app.common.logger is not available."""

        @classmethod
        def info(cls, message: str) -> None:
            """Log an informational message.

            Args:
                message: The message to log
            """
            print(message)

        @classmethod
        def error(cls, message: str) -> None:
            """Log an error message.

            Args:
                message: The error message to log
            """
            print(message)


class FF_TASKS(ClassEnum):
    """Enumeration of available FFmpeg task classes.

    This enum maps task identifiers to their implementing classes,
    providing a consistent interface for accessing different video processing operations.

    Attributes:
        Merge: For merging multiple video files
        Custom: For custom FFmpeg operations
        Cut: For cutting segments from videos
        Speedup: For changing video playback speed
        Jumpcut: For creating jumpcut effects
        CutSilence: For removing silent segments
        CutSilenceRerender: For removing silent segments with re-encoding
        CutMotionless: For removing motionless segments
        CutMotionlessRerender: For removing motionless segments with re-encoding
        PartitionVideo: For splitting videos into multiple parts

    Examples:
        Basic usage with default parameters:
        ```python
        from ffmpeg_toolkit import FF_TASKS
        from pathlib import Path

        # Speed up a video by the default factor (2x)
        FF_TASKS.Speedup(
            input_file=Path("input.mp4"),
            output_file=Path("output_2x.mp4")
        ).render()

        # Cut a segment from a video
        FF_TASKS.Cut(
            input_file=Path("input.mp4"),
            output_file=Path("cut_segment.mp4"),
            ss="00:01:30",  # Start time
            to="00:02:45"   # End time
        ).render()
        ```

        Customizing task parameters:
        ```python
        # Remove silent parts with custom threshold
        FF_TASKS.CutSilence(
            input_file=Path("input.mp4"),
            output_file=Path("no_silence.mp4"),
            dB=-30,  # More sensitive silence detection
            sampling_duration=0.5  # Check for silence every half second
        ).render()

        # Create a jumpcut effect that keeps normal speed sections
        # and removes sections in between
        FF_TASKS.Jumpcut(
            input_file=Path("input.mp4"),
            output_file=Path("jumpcut.mp4"),
            b1_duration=3,  # 3 seconds of normal speed
            b2_duration=2,  # 2 seconds to skip
            b1_multiple=1,  # Keep normal speed
            b2_multiple=0   # Remove these sections
        ).render()
        ```

        Chaining operations with callbacks:
        ```python
        # First cut out a segment, then speed it up
        segment = FF_TASKS.Cut(
            input_file=Path("input.mp4"),
            ss="00:01:00",
            to="00:02:00"
        ).render()

        FF_TASKS.Speedup(
            input_file=segment,
            output_file=Path("cut_and_speedup.mp4"),
            multiple=4  # 4x speed
        ).render()
        ```

        Partition a video into multiple segments with different processing:
        ```python
        FF_TASKS.PartitionVideo(
            input_file=Path("input.mp4"),
            # Define 3 segments: keep first as-is, speed up second by 2x, remove third
            portion_method=[
                (1, None),  # Keep segment 1 as is
                (1, PARTIAL_TASKS.speedup(multiple=2)),  # Speed up segment 2
                (1, "remove")  # Remove segment 3
            ]
        ).render()
        ```
    """

    Merge = Merge
    Custom = Custom
    Cut = Cut
    Speedup = Speedup
    Jumpcut = Jumpcut
    CutSilenceRerender = CutSilenceRerender
    CutMotionlessRerender = CutMotionlessRerender
    CutSilence = CutSilence
    CutMotionless = CutMotionless
    PartitionVideo = PartitionVideo


class PARTIAL_TASKS(FunctionEnum):
    """Collection of factory functions for creating partially configured FFmpeg task functions.

    This class provides static methods that create and return partially configured functions
    for common FFmpeg video processing tasks. Each method pre-configures task parameters,
    creating a simpler function that only requires input and output file paths.

    The returned functions are suitable for:
    - Direct execution with input/output files
    - Being passed as callbacks to other operations
    - Creating processing pipelines with consistent configurations
    - Use with batch processing operations

    Each factory method returns a callable that accepts:
    - input_file: Path to the source video file
    - output_file: Path where the processed video will be saved
    - options: Optional runtime configuration overrides

    Examples:
        Create a function that will speed up videos by 4x:
        ```python
        fast_forward = PARTIAL_TASKS.speedup(multiple=4)

        # Apply it to a video file
        fast_forward('input.mp4', 'output.mp4')
        ```

        Create a function that removes silent parts with specific threshold:
        ```python
        remove_silence = PARTIAL_TASKS.cut_silence(dB=-30)

        # Apply it to multiple files
        for file in video_files:
            remove_silence(file, f"processed_{file.name}")
        ```

        Use with batch processing:
        ```python
        batch = BatchTasks(
            input_folder_path=Path("videos"),
            output_folder_path=Path("processed_videos")
        )
        batch.render(PARTIAL_TASKS.speedup(multiple=2))
        ```
    """

    @staticmethod
    def custom(
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> PARTIAL_TASK:
        """Create a partially configured custom FFmpeg function.

        This allows for complete customization of FFmpeg command parameters through
        input and output keyword arguments.

        Args:
            input_kwargs: Input-related arguments passed to FFmpeg
            output_kwargs: Output-related arguments passed to FFmpeg

        Returns:
            Function that applies custom FFmpeg processing when called with input and output files
        """

        def _partial(
            input_file: str | Path,
            output_file: str | Path,
            options: OptionFFRender | None = None,
        ):
            kwargs = {
                "input_file": input_file,
                "output_file": output_file,
                "input_kwargs": input_kwargs or {},
                "output_kwargs": output_kwargs or {},
            }
            return Custom(**kwargs).override_option(options=options).render()

        return _partial

    @staticmethod
    def cut(
        ss: str = "00:00:00",
        to: str = "00:00:01",
        rerender: bool = False,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> PARTIAL_TASK:
        """Create a partially configured cut function.

        Extracts a segment from a video based on start and end timestamps.

        Args:
            ss: Start time in format 'HH:MM:SS' or seconds
            to: End time in format 'HH:MM:SS' or seconds
            rerender: Whether to re-encode the video (True) or use stream copy (False)
            input_kwargs: Additional input-related arguments passed to FFmpeg
            output_kwargs: Additional output-related arguments passed to FFmpeg

        Returns:
            Function that cuts the video when called with input and output files
        """

        def _partial(
            input_file: str | Path,
            output_file: str | Path,
            options: OptionFFRender | None = None,
        ):
            kwargs = {
                "input_file": input_file,
                "output_file": output_file,
                "ss": ss,
                "to": to,
                "rerender": rerender,
                "input_kwargs": input_kwargs or {},
                "output_kwargs": output_kwargs or {},
            }
            return Cut(**kwargs).override_option(options=options).render()

        return _partial

    @staticmethod
    def speedup(
        multiple: float | int = DEFAULTS.speedup_multiple.value,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> PARTIAL_TASK:
        """Create a partially configured speedup function.

        Changes the playback speed of a video by the specified multiplier.

        Args:
            multiple: Speed-up factor (e.g., 2 for double speed, 0.5 for half speed)
            input_kwargs: Additional input-related arguments passed to FFmpeg
            output_kwargs: Additional output-related arguments passed to FFmpeg

        Returns:
            Function that applies the speedup effect when called with input and output files
        """

        def _partial(
            input_file: str | Path,
            output_file: str | Path,
            options: OptionFFRender | None = None,
        ):
            kwargs = {
                "input_file": input_file,
                "output_file": output_file,
                "multiple": multiple,
                "input_kwargs": input_kwargs or {},
                "output_kwargs": output_kwargs or {},
            }
            return Speedup(**kwargs).override_option(options=options).render()

        return _partial

    @staticmethod
    def jumpcut(
        b1_duration: float = 5,
        b2_duration: float = 5,
        b1_multiple: float = 1,  # 0 means remove this part
        b2_multiple: float = 0,  # 0 means remove this part
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> PARTIAL_TASK:
        """Create a partially configured jumpcut function.

        Creates a jumpcut effect by alternating between two segments with different speeds.
        Can be used to keep some segments and remove others, or change speeds of alternating segments.

        Args:
            b1_duration: Duration of first part in seconds
            b2_duration: Duration of second part in seconds
            b1_multiple: Speed multiple for first part (0 = remove, 1 = normal speed)
            b2_multiple: Speed multiple for second part (0 = remove, 1 = normal speed)
            input_kwargs: Additional input-related arguments passed to FFmpeg
            output_kwargs: Additional output-related arguments passed to FFmpeg

        Returns:
            Function that applies the jumpcut effect when called with input and output files
        """

        def _partial(
            input_file: str | Path,
            output_file: str | Path,
            options: OptionFFRender | None = None,
        ):
            kwargs = {
                "input_file": input_file,
                "output_file": output_file,
                "b1_duration": b1_duration,
                "b2_duration": b2_duration,
                "b1_multiple": b1_multiple,
                "b2_multiple": b2_multiple,
                "input_kwargs": input_kwargs or {},
                "output_kwargs": output_kwargs or {},
            }
            return Jumpcut(**kwargs).override_option(options=options).render()

        return _partial

    @staticmethod
    def cut_silence_rerender(
        dB: int = DEFAULTS.db_threshold.value,
        sampling_duration: float = DEFAULTS.sampling_duration.value,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> PARTIAL_TASK:
        """Create a partially configured silence cutting function with re-encoding.

        Identifies and removes silent segments from a video by re-encoding the entire file.
        This method provides better quality but is slower than the non-rerender version.

        Args:
            dB: Audio threshold level in dB for identifying silence (lower values = more sensitive)
            sampling_duration: Minimum duration of silence to detect in seconds
            input_kwargs: Additional input-related arguments passed to FFmpeg
            output_kwargs: Additional output-related arguments passed to FFmpeg

        Returns:
            Function that removes silent segments with re-encoding when called with input and output files
        """

        def _partial(
            input_file: str | Path,
            output_file: str | Path,
            options: OptionFFRender | None = None,
        ):
            kwargs = {
                "input_file": input_file,
                "output_file": output_file,
                "dB": dB,
                "sampling_duration": sampling_duration,
                "input_kwargs": input_kwargs or {},
                "output_kwargs": output_kwargs or {},
            }
            return (
                CutSilenceRerender(**kwargs).override_option(options=options).render()
            )

        return _partial

    @staticmethod
    def cut_motionless_rerender(
        threshold: float = DEFAULTS.motionless_threshold.value,
        sampling_duration: float = DEFAULTS.sampling_duration.value,
    ) -> PARTIAL_TASK:
        """Create a partially configured motionless cutting function with re-encoding.

        Identifies and removes segments with no motion by re-encoding the entire file.
        This method provides better quality but is slower than the non-rerender version.

        Args:
            threshold: Scene change threshold for identifying motion (lower values = more sensitive)
            sampling_duration: Duration between motion samples in seconds

        Returns:
            Function that removes motionless segments with re-encoding when called with input and output files
        """

        def _partial(
            input_file: str | Path,
            output_file: str | Path,
            options: OptionFFRender | None = None,
        ):
            kwargs = {
                "input_file": input_file,
                "output_file": output_file,
                "threshold": threshold,
                "sampling_duration": sampling_duration,
            }
            return (
                CutMotionlessRerender(**kwargs)
                .override_option(options=options)
                .render()
            )

        return _partial

    @staticmethod
    def cut_silence(
        dB: int = DEFAULTS.db_threshold.value,
        sampling_duration: float = DEFAULTS.sampling_duration.value,
        seg_min_duration: float = DEFAULTS.seg_min_duration.value,
        even_further: FurtherMethod = "remove",  # For other segments, remove means remove, None means copy
        odd_further: FurtherMethod = None,
    ) -> PARTIAL_TASK:
        """Create a partially configured silence cutting function.

        Identifies silent and non-silent segments and processes them according to the specified methods.
        This version uses segment extraction and concatenation without full re-encoding.

        Args:
            dB: Audio threshold level in dB for identifying silence (lower values = more sensitive)
            sampling_duration: Minimum duration of silence to detect in seconds
            seg_min_duration: Minimum duration of segments to keep in seconds
            even_further: Processing method for even segments (typically silence)
                          Use "remove" to remove these segments, None to keep as-is
            odd_further: Processing method for odd segments (typically non-silence)
                         Use "remove" to remove these segments, None to keep as-is

        Returns:
            Function that removes silent segments when called with input and output files
        """

        def _partial(
            input_file: str | Path,
            output_file: str | Path,
            options: OptionFFRender | None = None,
        ):
            kwargs = {
                "input_file": input_file,
                "output_file": output_file,
                "dB": dB,
                "sampling_duration": sampling_duration,
                "seg_min_duration": seg_min_duration,
                "even_further": even_further,
                "odd_further": odd_further,
            }
            return CutSilence(**kwargs).override_option(options=options).render()

        return _partial

    @staticmethod
    def cut_motionless(
        threshold: float = DEFAULTS.motionless_threshold.value,
        sampling_duration: float = DEFAULTS.sampling_duration.value,
        seg_min_duration: float = DEFAULTS.seg_min_duration.value,
        even_further: FurtherMethod = "remove",  # For other segments, remove means remove, None means copy
        odd_further: FurtherMethod = None,  # For segments, remove means remove, None means copy
    ) -> PARTIAL_TASK:
        """Create a partially configured motionless cutting function.

        Identifies segments with and without motion and processes them according to the specified methods.
        This version uses segment extraction and concatenation without full re-encoding.

        Args:
            threshold: Scene change threshold for identifying motion (lower values = more sensitive)
            sampling_duration: Duration between motion samples in seconds
            seg_min_duration: Minimum duration of segments to keep in seconds
            even_further: Processing method for even segments (typically motionless)
                          Use "remove" to remove these segments, None to keep as-is
            odd_further: Processing method for odd segments (typically with motion)
                         Use "remove" to remove these segments, None to keep as-is

        Returns:
            Function that removes motionless segments when called with input and output files
        """

        def _partial(
            input_file: str | Path,
            output_file: str | Path,
            options: OptionFFRender | None = None,
        ):
            kwargs = {
                "input_file": input_file,
                "output_file": output_file,
                "threshold": threshold,
                "sampling_duration": sampling_duration,
                "seg_min_duration": seg_min_duration,
                "even_further": even_further,
                "odd_further": odd_further,
            }
            return CutMotionless(**kwargs).override_option(options=options).render()

        return _partial

    @staticmethod
    def partion_video(
        count: int = 0,  # Easy way to create portion_method
        portion_method: PortionMethod | None = None,  # Main logic for partitioning
        output_dir: Path | str | None = None,
    ) -> PARTIAL_TASK:
        """Create a partially configured video partitioning function.

        Splits a video into multiple segments based on specified criteria and
        optionally applies different processing to each segment.

        Args:
            count: Number of equal-sized segments to create (simple division)
            portion_method: Custom method specifying segment sizes and processing
                           Format: list of tuples (duration_ratio, processing_method)
            output_dir: Directory to save partitioned segments

        Returns:
            Function that partitions the video when called with input and output files
        """

        def _partial(
            input_file: str | Path,
            output_file: str | Path,
            options: OptionFFRender | None = None,
        ):
            kwargs = {
                "input_file": input_file,
                "output_file": output_file,
                "count": count,
                "portion_method": portion_method,
                "output_dir": output_dir,
            }
            return PartitionVideo(**kwargs).override_option(options=options).render()

        return _partial


def _list_video_files(
    root_path: str | Path,
    valid_extensions: ValidExtensions,
    walkthrough: bool = True,
) -> list[Path]:
    """Find all video files in a directory with the specified extensions.

    Args:
        root_path: Directory to search for video files
        valid_extensions: Set of file extensions to include (without the dot)
        walkthrough: Whether to search recursively through subdirectories

    Returns:
        List of Path objects for all matching video files
    """
    if not valid_extensions:
        valid_extensions = set(VideoSuffix)

    root_path = Path(root_path)
    video_files: list[Path] = []

    # Use rglob to recursively find files with the specified extensions
    video_files = (
        [
            file
            for file in root_path.rglob("*")
            if file.is_file() and file.suffix.lstrip(".").lower() in valid_extensions
        ]
        if walkthrough
        else [
            file
            for file in root_path.iterdir()
            if file.is_file() and file.suffix.lstrip(".").lower() in valid_extensions
        ]
    )

    return video_files


class BatchTask(BaseModel):
    """Batch processing configuration for FFmpeg video tasks.

    Allows applying the same FFmpeg operation to multiple video files in a directory.
    Handles input/output path management, file selection, and optional post-processing.

    Attributes:
        input_folder_path: Directory containing video files to process
        output_folder_path: Directory where processed files will be saved
        walkthrough: Whether to search recursively through subdirectories
        valid_extensions: Set of file extensions to process
        input_kwargs: Input parameters to apply to all files
        output_kwargs: Output parameters to apply to all files
        delete_after: Whether to delete original files after processing
        post_hook: Optional function to call after each file is processed
    """

    input_folder_path: Path
    output_folder_path: Path | None = None
    walkthrough: bool = False
    valid_extensions: ValidExtensions = Field(default_factory=set)
    input_kwargs: FFKwargs = Field(default_factory=dict)
    output_kwargs: FFKwargs = Field(default_factory=dict)
    delete_after: bool = False
    post_hook: Callable | None = None

    def model_post_init(self, *args, **kwargs):
        if self.output_folder_path is None:
            self.output_folder_path = self.input_folder_path

        if self.output_folder_path.suffix == "" and self.output_folder_path.is_file():
            raise ValueError("Output folder path is a file.")

        self.output_folder_path.mkdir(parents=True, exist_ok=True)

        if not self.valid_extensions:
            self.valid_extensions = set(VideoSuffix)

    @computed_field
    @property
    def video_files(self) -> list[Path]:
        """List all video files in the specified folder with valid extensions.

        Returns:
            List of Path objects pointing to valid video files
        """
        files = _list_video_files(
            self.input_folder_path,
            valid_extensions=self.valid_extensions,
            walkthrough=self.walkthrough,
        )
        logger.info(f"Found {len(files)} video files in {self.input_folder_path}")
        return files

    def render(self, task: PARTIAL_TASKS) -> None:
        """Apply the specified task to all video files in the batch.

        Processes each video file using the provided partial task function
        and optionally applies a post-processing hook to each result.

        Args:
            task: A partially configured task function from PARTIAL_TASKS
        """
        for video in self.video_files:
            output_gile = task(
                input_file=video,
                output_file=self.output_folder_path,
                options={"delete_after": self.delete_after},
            )
            if self.post_hook:
                logger.info("Post hooking...")
                self.post_hook(video, output_gile)
