# Standard library imports
import concurrent.futures
import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Generator
from enum import Enum, StrEnum, auto
from functools import wraps
from itertools import accumulate
from pathlib import Path
from types import MethodType, FunctionType
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    NotRequired,
    Optional,
    TypedDict,
    Self,
)

try:
    from app.common import logger  # type: ignore
except ImportError:
    # Fallback to a default value
    class logger:
        @classmethod
        def info(cls, message: str) -> None:
            print(message)

        @classmethod
        def error(cls, message: str) -> None:
            print(message)


# Third-party imports
from pydantic import BaseModel, Field, field_validator

# Local imports
from .ffmpeg_types import EncodeKwargs, VideoSuffix, FFKwargs, FunctionEnum
# import ffmpeg


class DEFAULTS(Enum):
    num_cores = os.cpu_count() or 4
    hwaccel = "auto"
    loglevel = "warning"
    keyframe_interval = 2
    speedup_multiple = 2
    speedup_task_threshold = 5
    db_threshold = -21
    motionless_threshold = 0.0095
    sampling_duration = 0.2
    seg_min_duration = 0
    temp_dir_prefix = "ffmpeg_toolkit_"


class ERROR_CODE(Enum):
    DURATION_LESS_THAN_ZERO = auto()
    NO_VALID_SEGMENTS = auto()
    FAILED_TO_CUT = auto()


def timing(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} took {end - start:.6f} seconds")
        return result

    return wrapper


def _convert_timestamp_to_seconds(timestamp: str) -> float:
    h, m, s = map(float, timestamp.split(":"))
    return h * 3600 + m * 60 + s


def _convert_seconds_to_timestamp(seconds: float) -> str:
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{s:06.3f}"


class _TASKS(StrEnum):
    SPEEDUP = auto()
    JUMPCUT = auto()
    CONVERT = auto()
    CUT = auto()
    KEEP_OR_REMOVE = auto()
    MERGE = auto()
    GET_NON_SILENCE_SEGS = auto()
    GET_MOTION_SEGS = auto()
    CUT_SILENCE = auto()
    CUT_SILENCE_RERENDER = auto()
    CUT_MOTIONLESS = auto()
    CUT_MOTIONLESS_RERENDER = auto()
    SPLIT = auto()
    PARTITION = auto()


class _PROBE_TASKS(StrEnum):
    DURATION = auto()
    ENCODING = auto()
    IS_VALID_VIDEO = auto()
    KEYFRAMES = auto()
    NON_SILENCE_SEGS = auto()
    FRAMES_PER_SECOND = auto()


# basic
type FurtherRenderTask = Callable[..., Any] | PARTIAL_TASKS
type FurtherMethod = FurtherRenderTask | Literal["remove"] | None


def _create_ff_kwargs(
    input_file: Path,
    output_file: Path,
    input_kwargs: FFKwargs,
    output_kwargs: FFKwargs,
) -> FFKwargs:
    input_kwargs_default: FFKwargs = {
        "hide_banner": "",
        "hwaccel": DEFAULTS.hwaccel.value,
    }
    output_kwargs_default: FFKwargs = {"loglevel": DEFAULTS.loglevel.value}

    input_file_kwargs: Mapping[Literal["i"], Path] = {"i": input_file}
    # Handle file path
    if output_file == Path():
        output_file_kwargs: Mapping[Literal["y"], Path] = {}
    else:
        output_file_kwargs = {"y": output_file}
    ff_kwargs: FFKwargs = (
        input_kwargs_default
        | input_kwargs
        | input_file_kwargs
        | output_kwargs_default
        | output_kwargs
        | output_file_kwargs
    )

    return ff_kwargs


def _create_fp_kwargs(
    input_file: Path,
    input_kwargs: FFKwargs,
    output_kwargs: FFKwargs,
) -> FFKwargs:
    input_kwargs_default: FFKwargs = {
        "hide_banner": "",
    }
    output_kwargs_default: FFKwargs = {"loglevel": DEFAULTS.loglevel.value}

    input_file_kwargs: Mapping[Literal["i"], Path] = {"i": input_file}
    ff_kwargs: FFKwargs = (
        input_kwargs_default
        | input_kwargs
        | input_file_kwargs
        | output_kwargs_default
        | output_kwargs
    )

    return ff_kwargs


def _dic_to_ffmpeg_kwargs(kwargs: dict | None = None) -> list[str]:
    """Create ffmpeg args to be executed in subprocess

    Args:
        kwargs (dict | None, optional): _description_. Defaults to None.

    Returns:
        list[str | float]: _description_
    """
    args = []
    if kwargs is None:
        return args
    arg_map = {
        "cv": "-c:v",
        "ca": "-c:a",
        "bv": "-b:v",
        "ba": "-b:a",
        "filterv": "-filter:v",
        "filtera": "-filter:a",
    }

    for k, v in kwargs.items():
        args.append(arg_map.get(k, f"-{k}"))
        if v != "":
            args.append(f"{v}")
    return args


@timing
def _ffmpeg(**ffkwargs) -> subprocess.CompletedProcess[str]:
    command = ["ffmpeg"] + _dic_to_ffmpeg_kwargs(ffkwargs)
    logger.info(f"Executing FFmpeg command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding="utf-8"
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to execute FFmpeg. Error: {e}")
        raise e


@timing
def _ffprobe(**ffkwargs):
    command = ["ffprobe"] + _dic_to_ffmpeg_kwargs(ffkwargs)
    logger.info(f"Executing ffprobe command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding="utf-8"
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to execute ffprobe. Error: {e}")
        raise e


def _handle_output_file_path(
    input_file: Path, output_file: Path | str | None, task_description: str
) -> Path:
    if output_file is None:
        output_file = (
            input_file.parent
            / f"{input_file.stem}_{task_description}{input_file.suffix if input_file.suffix in VideoSuffix else '.' + VideoSuffix.MKV}"
        )
    else:
        output_file = Path(output_file)
        if output_file == Path():
            # Using empty path - let FFmpeg handle output (might be for probe operations)
            pass
        elif output_file.suffix == "" and output_file.is_file():
            # Path exists but isn't a directory - this is an error
            raise ValueError(f"{output_file} exists and is not a directory")
        elif output_file.suffix == "":
            # It's a directory (or should be treated as one)
            # Create directory if needed and build output filename inside it
            output_file.mkdir(exist_ok=True)
            output_file = (
                output_file
                / f"{input_file.stem}_{task_description}{input_file.suffix if input_file.suffix in VideoSuffix else '.' + VideoSuffix.MKV}"
            )
    return output_file


class FFRenderException(TypedDict):
    code: int
    message: str
    hook: NotRequired[Callable[[], Any]]


class FPCreateCommand(BaseModel):
    input_file: Path | str = Path()
    input_kwargs: FFKwargs = Field(default_factory=dict)
    output_kwargs: FFKwargs = Field(default_factory=dict)


class FPCreateRender(FPCreateCommand):
    task_descripton: str = "probe"
    exception: Optional[FFRenderException] = None
    post_task: Optional[Callable[..., Any]] = None

    def render(self) -> Any:
        _shadow = copy.copy(self)
        _shadow.input_file = Path(_shadow.input_file)

        # Exception hadling
        if _shadow.exception is not None:
            match _shadow.exception["code"]:
                case 9:
                    _shadow.exception.get("hook", lambda: None)()
                case _:
                    pass
            logger.error(_shadow.exception["message"])
            return _shadow.exception["code"]

        # Generate ff kwargs in
        unwated_key: list[str] = ["task_descripton", "exception", "post_task"]
        fp_command = {
            k: v for k, v in _shadow.model_dump().items() if k not in unwated_key
        }
        ff_kwargs: FFKwargs = _create_fp_kwargs(**fp_command)

        logger.info(
            f"{_shadow.task_descripton.capitalize()} {_shadow.input_file.name}  with {ff_kwargs}"
        )

        try:
            result: subprocess.CompletedProcess[str] = _ffprobe(**ff_kwargs)
            stripped_result: str = result.stdout.strip() + result.stderr.strip()

            # Return post task if exists
            if _shadow.post_task is not None:
                return _shadow.post_task(stripped_result)

            return stripped_result
        except Exception as e:
            logger.error(
                f"Failed to do {_shadow.task_descripton} videos for {_shadow.input_file}. Error: {e}"
            )
            raise e


class FPRenderTasks(FPCreateRender):
    def encode(
        self,
        input_file: str | Path,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> Self:
        defalut_output_kwargs: FFKwargs = {
            "v": "error",
            "print_format": "json",
            "show_format": "",
            "show_streams": "",
            "i": input_file,
        }
        self.task_descripton = _PROBE_TASKS.ENCODING
        self.input_file = input_file
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        self.output_kwargs = defalut_output_kwargs | (
            {} if output_kwargs is None else output_kwargs
        )

        def post_task(result) -> EncodeKwargs:
            # Initialize the dictionary with default values
            encoding_info: EncodeKwargs = {}
            probe = json.loads(result)

            # Extract video stream information
            video_stream = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "video"
                ),
                None,
            )
            if video_stream:
                encoding_info["video_track_timescale"] = int(
                    video_stream.get("time_base").split("/")[1]
                )
                encoding_info["vcodec"] = video_stream.get("codec_name")
                encoding_info["video_bitrate"] = int(video_stream.get("bit_rate", 0))

            # Extract audio stream information
            audio_stream = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "audio"
                ),
                None,
            )
            if audio_stream:
                encoding_info["acodec"] = audio_stream.get("codec_name")
                encoding_info["ar"] = int(audio_stream.get("sample_rate", 0))

            # Extract format information
            format_info = probe.get("format", {})
            encoding_info["f"] = format_info.get("format_name").split(",")[0]
            cleaned_None = {
                k: v for k, v in encoding_info.items() if v is not None and v != 0
            }
            logger.info(f"{Path(input_file).name} probed: {cleaned_None}")
            return cleaned_None  # type: ignore

        self.post_task = post_task
        return self

    def is_valid_video(
        self,
        input_file: str | Path,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> Self:
        defalut_output_kwargs: FFKwargs = {
            "v": "error",
            "show_entries": "format=duration",
            "of": "default=noprint_wrappers=1:nokey=1",
            "i": input_file,
        }
        self.task_descripton = _PROBE_TASKS.IS_VALID_VIDEO
        self.input_file = input_file
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        self.output_kwargs = defalut_output_kwargs | (
            {} if output_kwargs is None else output_kwargs
        )

        def post_task(result) -> bool:
            if result:
                message = f"Validated file: {input_file}, Status: Valid"
                logger.info(message)
                return True
            else:
                message = f"Validated file: {input_file}, Status: Invalid"
                logger.info(message)
                return False

        self.post_task = post_task
        return self

    def duration(
        self,
        input_file: str | Path,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> Self:
        defalut_output_kwargs: FFKwargs = {
            "v": "error",
            "show_entries": "format=duration",
            "of": "default=noprint_wrappers=1:nokey=1",
            "i": input_file,
        }
        self.task_descripton = _PROBE_TASKS.DURATION
        self.input_file = input_file
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        self.output_kwargs = defalut_output_kwargs | (
            {} if output_kwargs is None else output_kwargs
        )

        def post_task(result):
            logger.info(f"{Path(input_file).name} duration probed: {result}")
            return float(result or 0)

        self.post_task = post_task
        return self

    def keyframes(
        self,
        input_file: str | Path,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> Self:
        defalut_output_kwargs: FFKwargs = {
            "v": "error",
            "select_streams": "v:0",
            "show_entries": "packet=pts_time,flags",
            "of": "json",
            "i": input_file,
        }
        self.task_descripton = _PROBE_TASKS.KEYFRAMES
        self.input_file = input_file
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        self.output_kwargs = defalut_output_kwargs | (
            {} if output_kwargs is None else output_kwargs
        )

        def post_task(result) -> list[float]:
            probe = json.loads(result)
            keyframe_pts: list[float] = [
                float(packet["pts_time"])
                for packet in probe["packets"]
                if "K" in packet["flags"]
            ]
            logger.info(f"{Path(input_file).name} keyframes probed: {keyframe_pts}")
            return keyframe_pts

        self.post_task = post_task
        return self

    def frame_per_s(
        self,
        input_file: str | Path,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> Self:
        defalut_output_kwargs: FFKwargs = {
            "v": "error",
            "select_streams": "v",
            "show_entries": "stream=r_frame_rate",
            "of": "csv=p=0",
            "i": input_file,
        }
        self.task_descripton = _PROBE_TASKS.FRAMES_PER_SECOND
        self.input_file = input_file
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        self.output_kwargs = defalut_output_kwargs | (
            {} if output_kwargs is None else output_kwargs
        )

        def post_task(result):
            logger.info(f"{Path(input_file).name} frames per second probed: {result}")
            return result

        self.post_task = post_task
        return self


class FFCreateCommand(BaseModel):
    input_file: Path | str = Path()
    output_file: Optional[Path | str] = None
    input_kwargs: FFKwargs = Field(default_factory=dict)
    output_kwargs: FFKwargs = Field(default_factory=dict)


class OptionFFRender(TypedDict):
    task_descripton: NotRequired[str]
    delete_after: NotRequired[bool]
    exception: NotRequired[FFRenderException]
    psot_task: NotRequired[Callable[..., Any]]
    return_result: NotRequired[bool]


class FFCreateRender(FFCreateCommand):
    task_descripton: str = "render"
    delete_after: bool = False
    exception: Optional[FFRenderException] = None
    post_task: Optional[Callable[..., Any]] = None
    return_result: bool = False

    def override_option(self, options: OptionFFRender | None = None) -> Self:
        if options is not None:
            for k, v in options.items():
                setattr(self, k, v)

        return self

    def render(self) -> Any:
        _shadow = copy.copy(self)
        _shadow.input_file = Path(_shadow.input_file)

        # Handle output file path
        _shadow.output_file = _handle_output_file_path(
            _shadow.input_file, _shadow.output_file, _shadow.task_descripton
        )

        # Handle temp output file path
        if _shadow.output_file == Path() or r"%d" in str(_shadow.output_file):
            temp_output_file: Path = _shadow.output_file
        else:
            temp_output_file: Path = _shadow.output_file.parent / (
                _shadow.output_file.stem + "_processing" + _shadow.output_file.suffix
            )

        # Exception hadling
        if _shadow.exception is not None:
            match _shadow.exception["code"]:
                case _:
                    pass
            logger.error(_shadow.exception["message"])
            exception_hook = _shadow.exception.get("hook", lambda: None)
            if exception_hook is not None:
                return exception_hook()
            return _shadow.exception["code"]

        # Generate ff kwargs in
        unwated_key: list[str] = [
            "task_descripton",
            "exception",
            "post_task",
            "output_file",
            "delete_after",
            "return_result",
        ]
        fp_command = {
            k: v for k, v in _shadow.model_dump().items() if k not in unwated_key
        } | {"output_file": temp_output_file}
        ff_kwargs: FFKwargs = _create_ff_kwargs(**fp_command)

        logger.info(
            f"{_shadow.task_descripton.capitalize()} {_shadow.input_file.name} to {_shadow.output_file.name} with {ff_kwargs}"
        )

        try:
            result: subprocess.CompletedProcess[str] = _ffmpeg(**ff_kwargs)
            if temp_output_file != _shadow.output_file and r"%" not in str(
                temp_output_file.stem
            ):
                temp_output_file.replace(_shadow.output_file)

            if _shadow.delete_after:
                os.remove(_shadow.input_file)

            # Do post task if exists
            if _shadow.post_task is not None:
                return _shadow.post_task(result)

            if _shadow.return_result:
                return result.stdout.strip() + result.stderr.strip()

            return _shadow.output_file
        except Exception as e:
            logger.error(
                f"Failed to do {_shadow.task_descripton} videos for {_shadow.input_file}. Error: {e}"
            )
            raise e


class FFRenderTasks(FFCreateRender):
    def custom(
        self,
        input_file: Path | str,
        output_file: Optional[Path | str] = None,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> Self:
        self.input_file = input_file
        self.output_file = output_file
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        if output_kwargs is not None:
            self.output_kwargs = output_kwargs
        return self

    def cut(
        self,
        input_file: Path | str,
        output_file: Optional[Path | str] = None,
        ss: str = "00:00:00",
        to: str = "00:00:01",
        rerender: bool = False,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> Self:
        self.task_descripton = f"{_TASKS.CUT}_{_convert_timestamp_to_seconds(ss)}-{_convert_timestamp_to_seconds(to)}"
        self.input_file = input_file
        self.output_file = output_file
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        self.output_kwargs = (
            {
                "ss": ss,
                "to": to,
            }
            | ({} if rerender else {"c:v": "copy", "c:a": "copy"})
            | ({} if output_kwargs is None else output_kwargs)
        )
        return self

    def speedup(
        self,
        input_file: Path | str,
        output_file: Optional[Path | str] = None,
        multiple: float | int = DEFAULTS.speedup_multiple.value,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> Self:
        self.task_descripton = f"{_TASKS.SPEEDUP}_by_{multiple}"
        self.input_file = input_file
        self.output_file = output_file
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        self.output_kwargs = _create_speedup_kwargs(multiple) | (
            {} if output_kwargs is None else output_kwargs
        )

        # error handling
        if multiple == 1:
            if input_file != output_file:
                self.exception = {
                    "code": 0,
                    "message": "Speedup multiple 1, only replace target file",
                }

        if multiple <= 0:
            self.exception = {
                "code": 1,
                "message": "Speedup factor must be greater than 0.",
            }

        return self

    def jumpcut(
        self,
        input_file: Path | str,
        output_file: Path | None = None,
        b1_duration: float = 5,
        b2_duration: float = 5,
        b1_multiple: float = 1,  # 0 means remove this part
        b2_multiple: float = 0,  # 0 means remove this part
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> Self:
        self.task_descripton = f"{_TASKS.JUMPCUT}_b1({b1_duration}x{b1_multiple})_b2({b2_duration}x{b2_multiple})"
        self.input_file = input_file
        self.output_file = output_file
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        self.output_kwargs = _create_jumpcut_kwargs(
            b1_duration, b2_duration, b1_multiple, b2_multiple
        ) | ({} if output_kwargs is None else output_kwargs)

        # error handling
        if any((b1_duration <= 0, b2_duration <= 0)):
            self.exception = {
                "code": 1,
                "message": "Both parts' durations must be greater than 0.",
            }

        if any((b1_multiple < 0, b2_multiple < 0)):
            self.exception = {
                "code": 2,
                "message": "Both multiples must be greater or equal to 0 (0 means remove).",
            }

        return self

    def merge(
        self,
        input_dir_or_files: Path | str | list[Path] | list[str],
        output_file: Path | None = None,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> Self:
        # Create input.txt
        if isinstance(input_dir_or_files, Iterable):
            input_dir_or_files = [Path(video) for video in input_dir_or_files]
        else:
            input_dir_or_files = Path(input_dir_or_files)
        input_txt: Path = create_merge_txt(input_dir_or_files)
        self.task_descripton = _TASKS.MERGE
        self.input_file = input_txt
        self.output_file = output_file
        self.input_kwargs = {
            "f": "concat",
            "safe": 0,
        } | ({} if input_kwargs is None else input_kwargs)
        self.output_kwargs = {
            "c:a": "copy",
            "c:v": "copy",
        } | ({} if output_kwargs is None else output_kwargs)

        def post_task(_result):
            os.remove(input_txt)
            if not any(input_txt.parent.iterdir()):
                os.rmdir(input_txt.parent)
                logger.info(f"removed {input_txt.parent}")

        self.post_task = post_task

        return self

    def get_silence_segs(
        self,
        input_file: Path | str,
        dB: int = DEFAULTS.db_threshold.value,
        sampling_duration: float = DEFAULTS.sampling_duration.value,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> Self:
        self.task_descripton = f"{_TASKS.GET_NON_SILENCE_SEGS}_by_{dB}"
        self.input_file = input_file
        self.output_file = Path()
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        self.output_kwargs = (
            {
                "af": f"silencedetect=n={dB}dB:d={sampling_duration}",
                "vn": "",
                "loglevel": "info",
                "f": "null",
            }
            | ({} if output_kwargs is None else output_kwargs)
            | {"": ""}
        )
        self.return_result = True

        return self

    def get_motion_segs(
        self,
        input_file: Path | str,
        sampling_duration: float = DEFAULTS.sampling_duration.value,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> Self:
        self.task_descripton = f"{_TASKS.GET_MOTION_SEGS}"
        self.input_file = input_file = Path(input_file)
        self.output_file = Path()
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        frame_per_second: str = FPRenderTasks().frame_per_s(input_file).render()
        self.output_kwargs = (
            {
                "vf": f"select='not(mod(n,floor({frame_per_second})*{sampling_duration}))*gte(scene,0)',metadata=print",
                "an": "",
                "loglevel": "info",
                "f": "null",
            }
            | ({} if output_kwargs is None else output_kwargs)
            | {"": ""}
        )
        self.return_result = True

        return self

    def cut_silence_rerender(
        self,
        input_file: Path | str,
        output_file: Path | str | None = None,
        dB: int = DEFAULTS.db_threshold.value,
        sampling_duration: float = DEFAULTS.sampling_duration.value,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> Self:
        self.task_descripton = f"{_TASKS.CUT_SILENCE_RERENDER}_by_{dB}"
        self.input_file = input_file
        self.output_file = output_file
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        self.output_kwargs = output_kwargs = _create_cut_sl_kwargs(
            input_file, dB, sampling_duration
        ) | ({} if output_kwargs is None else output_kwargs)

        def post_task(_result):
            os.remove(str(output_kwargs["filter_script:v"]))
            os.remove(str(output_kwargs["filter_script:a"]))

        self.post_task = post_task

        return self

    def cut_motionless_rerender(
        self,
        input_file: Path | str,
        output_file: Path | str | None = None,
        threshold: float = DEFAULTS.motionless_threshold.value,
        sampling_duration: float = DEFAULTS.sampling_duration.value,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> Self:
        self.task_descripton = f"{_TASKS.CUT_MOTIONLESS_RERENDER}_by_{threshold}"
        self.input_file = input_file
        self.output_file = output_file
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        self.output_kwargs = output_kwargs = _create_cut_motionless_kwargs(
            input_file, threshold, sampling_duration
        ) | ({} if output_kwargs is None else output_kwargs)

        def post_task(_result):
            os.remove(str(output_kwargs["filter_script:v"]))
            os.remove(str(output_kwargs["filter_script:a"]))

        self.post_task = post_task

        return self

    def split_segments(
        self,
        input_file: Path | str,
        video_segments: list[str],
        output_dir: Optional[Path | str] = None,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ) -> Self:
        self.task_descripton = f"{_TASKS.SPLIT}"
        input_file = Path(input_file)
        if output_dir is None:
            output_dir = input_file.parent / f"{input_file.stem}_{self.task_descripton}"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        self.input_file = input_file
        self.output_file = f"{output_dir}/%d_{input_file.stem}{input_file.suffix}"
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        self.output_kwargs = {
            "c:v": "copy",
            "c:a": "copy",
            "f": "segment",
            "segment_times": ",".join(video_segments),
            "segment_format": input_file.suffix.lstrip("."),
            "reset_timestamps": "1",
        } | ({} if output_kwargs is None else output_kwargs)

        if len(video_segments) == 0:
            self.exception = {
                "code": 9,
                "message": f"No video segments provided, just copy {input_file} to {output_dir}",
                "hook": lambda: shutil.copy2(
                    input_file, output_dir / f"0_{input_file.name}"
                ),
            }

        return self


# for fppedup
def _create_force_keyframes_kwargs(
    keyframe_interval: int = DEFAULTS.keyframe_interval.value,
) -> dict[str, str]:
    """_summary_

    Args:
        keyframe_interval (int, optional): _description_. Defaults to DEFAULTS.keyframe_interval.value.

    Returns:
        dict[str, str]: _description_
    """
    return {"force_key_frames": f"expr:gte(t,n_forced*{keyframe_interval})"}


def _create_speedup_kwargs(multiple: float) -> dict[str, str]:
    SPEEDUP_task_THRESHOLD: int = DEFAULTS.speedup_task_threshold.value
    vf: str
    af: str
    if multiple > SPEEDUP_task_THRESHOLD:
        vf = f"select='if(eq(n,0),1,gt(floor(n/{multiple}), floor((n-1)/{multiple})))',setpts=N/FRAME_RATE/TB"
        af = f"aselect='if(eq(n,0),1,gt(floor(n/{multiple}), floor((n-1)/{multiple})))',asetpts=N/SR/TB"
    else:
        vf = f"setpts={1 / multiple}*PTS"
        af = f"atempo={multiple}"
    return (
        {"vf": vf, "af": af}
        | {
            "map": 0,
            "shortest": "",
            "fps_mode": "vfr",
            "async": 1,
            "reset_timestamps": "1",
        }
        | _create_force_keyframes_kwargs()
    )


# for jumpcut
def _create_jumpcut_kwargs(
    b1_duration: float,
    b2_duration: float,
    b1_multiple: float,  # 0 means unwanted cut out
    b2_multiple: float,  # 0 means unwanted cut out
) -> dict[str, str]:
    interval_multiple_expr: str = (
        str(b1_multiple)
        if b1_multiple == 0
        else f"if(eq(n,0),1,gt(floor(n/{b1_multiple}), floor((n-1)/{b1_multiple})))"
    )
    lasting_multiple_expr: str = (
        str(b2_multiple)
        if b2_multiple == 0
        else f"if(eq(n,0),1,gt(floor(n/{b2_multiple}), floor((n-1)/{b2_multiple})))"
    )
    frame_select_expr: str = f"if(lte(mod(t, {b1_duration + b2_duration}),{b1_duration}), {interval_multiple_expr}, {lasting_multiple_expr})"
    args: dict[str, str] = (
        {
            "vf": f"select='{frame_select_expr}',setpts=N/FRAME_RATE/TB",
            "af": f"aselect='{frame_select_expr}',asetpts=N/SR/TB",
        }
        | {
            "map": 0,
            "shortest": "",
            "fps_mode": "vfr",
            "async": 1,
            "reset_timestamps": "1",
        }
        | _create_force_keyframes_kwargs()
    )

    return args


# merge
def create_merge_txt(
    video_files_source: Path | list[Path], output_txt: Path | None = None
) -> Path:
    # Step 0: Set the output txt path
    if output_txt is None:
        temp_output_dir = Path(tempfile.mkdtemp(prefix=DEFAULTS.temp_dir_prefix.value))
        output_txt = temp_output_dir / "input.txt"

    if isinstance(video_files_source, Path):
        if not video_files_source.is_dir():
            raise ValueError(f"{video_files_source} is not a directory")
        video_files: list[Path] = sorted(
            list(
                video
                for video in video_files_source.glob("*")
                if video.suffix.lstrip(".") in VideoSuffix
            ),
            key=lambda video: video.stem,
        )
    else:
        for video in video_files_source:
            if video.suffix.lstrip(".") not in VideoSuffix:
                raise ValueError(f"{video} is not a video file")
        video_files = video_files_source

    # Step 5: Create input.txt for FFmpeg concatenation
    with open(output_txt, "w", encoding="utf-8") as f:
        for video_path in video_files:
            f.write(f"file '{video_path}'\n")

    return output_txt


# keep or remove copy/rendering by split segs
def advanced_keep_or_remove_by_split_segs(
    input_file: Path | str,
    output_file: Path | str | None,
    video_segments: list[str] | list[float],
    even_further: FurtherMethod = "remove",  # For other segments, remove means remove, None means copy
    odd_further: FurtherMethod = None,  # For segments, remove means remove, None means copy
    remove_temp_handle: bool = True,
    delete_after: bool = False,
) -> int | Path:
    task_descripton = _TASKS.KEEP_OR_REMOVE + "_split"
    input_file = Path(input_file)

    # Handle output file path
    output_file = _handle_output_file_path(input_file, output_file, task_descripton)

    logger.info(
        f"{task_descripton.capitalize()} {input_file.name} to {output_file.name} with {even_further = } ,{odd_further = }."
    )

    # convert to timestamp if needed
    video_segments = list(
        _convert_seconds_to_timestamp(s) if isinstance(s, (float, int)) else s
        for s in video_segments
    )

    try:
        # Split videos
        temp_dir: Path = Path(tempfile.mkdtemp(prefix=DEFAULTS.temp_dir_prefix.value))
        FFRenderTasks(delete_after=delete_after).split_segments(
            input_file=input_file,
            video_segments=video_segments,
            output_dir=temp_dir,
        ).render()

        video_files: list[Path] = sorted(
            temp_dir.glob(f"*{input_file.suffix}"),
            key=lambda video_file: int(video_file.stem.split("_")[0]),
        )

        further_methods: dict[int, FurtherMethod] = {
            0: even_further,
            1: odd_further,
        }

        # Use ThreadPoolExecutor to manage rendreing tasks
        # Create the executor once for all tasks
        futures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=DEFAULTS.num_cores.value
        ) as executor:
            for video in video_files:
                index = int(video.stem.split("_")[0])
                i_remainder = index % 2
                further_method: FurtherMethod = further_methods[i_remainder]

                # Remove unwanted segments
                if further_method == "remove":
                    os.remove(video)
                    continue

                # Skip further rendering if the segment is to be copied
                if further_method is None:
                    continue

                # Submit the further render task to the executor properly
                future = executor.submit(
                    further_method, input_file=video, output_file=video
                )
                futures.append(future)

            # Wait for all submitted tasks to complete and then move files
            for future in futures:
                future.result()  # This will raise exceptions if any occurred during task execution

        # Merge the kept segments
        rendered_video_files: list[Path] = sorted(
            temp_dir.glob(f"*{input_file.suffix}"),
            key=lambda video_file: int(video_file.stem.split("_")[0]),
        )
        FFRenderTasks().merge(rendered_video_files, output_file).render()

        # Clean up temporary files and dir
        if remove_temp_handle:
            for video in temp_dir.iterdir():
                os.remove(video)
            os.rmdir(temp_dir)

        return output_file

    except Exception as e:
        logger.error(f"Failed to {task_descripton} for {input_file}. Error: {e}")
        return 1


# Split segments by part
def _get_segments_from_parts_count(
    duration: float | str, parts_count: int, portion: Optional[list[int]] = None
) -> list[str]:
    if parts_count <= 0:
        raise ValueError("parts_count must be greater than 0")

    if isinstance(duration, str):
        duration = _convert_timestamp_to_seconds(duration)

    if portion is None:
        portion = [1] * parts_count
    # Error hadling
    if sum(portion) != parts_count:
        raise ValueError(
            f"Sum of portions ({sum(portion)}) must equal to parts_count ({parts_count})"
        )

    segment_length = duration / parts_count
    split_points = [
        _convert_seconds_to_timestamp(segment_length * ap)
        for ap in accumulate(p for p in portion[:-1])
    ]

    return split_points


# Partitioning
type PortionMethodSpecific = list[tuple[int, FurtherMethod]] | list[tuple[int, None]]
type PortionMethod = PortionMethodSpecific | list[tuple[int, FurtherMethod] | int]


class PartitionConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    count: int = Field(default=0, gt=0)
    portion_method: Optional[PortionMethod] = None

    def model_post_init(self, *args, **kwargs):
        if self.count == 0 and self.portion_method is None:
            self.count = 3
        if self.count == 0 and self.portion_method is not None:
            self.count = sum(p[0] for p in self.portion_method)  # type: ignore
        if self.portion_method is None:
            self.portion_method = [(1, None)] * self.count

    @field_validator("portion_method")
    @classmethod
    def validate_portion_sum(
        cls, portion_method: PortionMethod, info
    ) -> PortionMethodSpecific:
        if portion_method is not None:
            _portion_method: PortionMethodSpecific = [
                (p, None) if isinstance(p, int) else p for p in portion_method
            ]
            if (_sum := sum(p[0] for p in _portion_method)) != (
                _count := info.data["count"]
            ) and info.data["count"] != 0:
                raise ValueError(
                    f"Sum of portions ({_sum}) must equal to count ({_count})"
                )
            return _portion_method


@timing
def partion_video(
    input_file: Path | str,
    partition_config: Optional[PartitionConfig] = None,
    output_dir: Optional[Path | str] = None,
    output_file: Path | str | None = None,
    delete_after: bool = False,
) -> Path | ERROR_CODE | int:
    if partition_config is None:
        partition_config = PartitionConfig()

    task_descripton = _TASKS.PARTITION
    input_file = Path(input_file)
    if output_dir is None:
        output_dir = input_file.parent / f"{input_file.stem}_{task_descripton}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    duration: float = FPRenderTasks().duration(input_file).render()
    video_segments: list[str] = _get_segments_from_parts_count(
        duration,
        partition_config.count,
        [p[0] for p in partition_config.portion_method],  # type: ignore
    )
    logger.info(f"{video_segments = }")
    logger.info(
        f"{task_descripton.capitalize()} {input_file.name} to {output_dir} with {partition_config}."
    )

    try:
        # Split videos
        temp_dir: Path = Path(tempfile.mkdtemp(prefix=DEFAULTS.temp_dir_prefix.value))
        FFRenderTasks(delete_after=delete_after).split_segments(
            input_file=input_file,
            video_segments=video_segments,
            output_dir=temp_dir,
        ).render()

        video_files: list[Path] = sorted(
            temp_dir.glob(f"*{input_file.suffix}"),
            key=lambda video_file: int(video_file.stem.split("_")[0]),
        )

        # Further render videos
        new_video_files: list[Path] = []
        futures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=DEFAULTS.num_cores.value
        ) as executor:
            for video in video_files:
                seg_output_file: Path = output_dir / video.name
                index = int(video.stem.split("_")[0])
                further_method: FurtherMethod = partition_config.portion_method[index][  # type:ignore
                    1
                ]

                if further_method == "remove":
                    os.remove(video)
                    continue

                if further_method is None:
                    shutil.move(str(video), str(seg_output_file))
                    new_video_files.append(seg_output_file)
                    continue

                # Submit the task and store the future
                future = executor.submit(
                    further_method, input_file=video, output_file=video
                )
                futures.append((future, video, seg_output_file))

        # Wait for all submitted tasks to complete and then move files
        for future, video, seg_output_file in futures:
            future.result()  # This will raise exceptions if any occurred during task execution
            shutil.move(str(video), str(seg_output_file))
            new_video_files.append(seg_output_file)

        # Optionally, sort the new video files
        new_video_files = sorted(
            new_video_files, key=lambda video: int(video.stem.split("_")[0])
        )

        os.rmdir(temp_dir)

        if output_file is not None:
            output_file = _handle_output_file_path(
                input_file, output_file, task_descripton
            )
            FFRenderTasks().merge(
                input_dir_or_files=new_video_files,  # type:ignore
                output_file=Path(output_file),
            ).render()  # type:ignore
            return output_file

        return 0

    except Exception as e:
        logger.error(f"Failed to {task_descripton} for {input_file}. Error: {e}")
        raise e


# cut silence copy / render
# Extract non silence segments info
def _extract_non_silence_info(
    non_silence_segs_str: str,
) -> tuple[list[float], float, float]:
    # Total duration
    total_duration_pattern = r"Duration: (.+?),"
    total_duration_match: str | None = re.findall(
        total_duration_pattern, non_silence_segs_str
    )[0]
    total_duration: float = (
        _convert_timestamp_to_seconds(total_duration_match)
        if total_duration_match
        else 0.0
    )

    # Regular expression to find all floats after "silence_start or end: "
    silence_seg_pattern = r"silence_(?:start|end): ([0-9.]+)"
    # Find all matches in the log data
    silence_seg_matches: list[str] = re.findall(
        silence_seg_pattern, non_silence_segs_str
    )
    # Convert matches to a list of floats
    non_silence_segs: list[float] = list(float(match) for match in silence_seg_matches)
    non_silence_segs = [0.0] + non_silence_segs + [total_duration]

    # Regular expression to find all floats after silence_duration: "
    silence_duration_pattern = r"silence_duration: ([0-9.]+)"
    silence_duration_matches: list[str] | Generator[float] = re.findall(
        silence_duration_pattern, non_silence_segs_str
    )
    silence_duration_matches = (float(s) for s in silence_duration_matches)
    total_silence_duration: float = sum(silence_duration_matches)

    return (non_silence_segs, total_duration, total_silence_duration)


# Motion detect
def _extract_motion_info(
    motion_segs_str: str,
) -> tuple[dict[float, float], float]:
    # Total duration
    total_duration_pattern = r"Duration: (.+?),"
    total_duration_match: str | None = re.findall(
        total_duration_pattern, motion_segs_str
    )[0]
    total_duration: float = (
        _convert_timestamp_to_seconds(total_duration_match)
        if total_duration_match
        else 0.0
    )

    # Regular expression to find all floats after "pts_time:"
    motion_seg_pattern = r"pts_time:([0-9.]+)"
    # Find all matches in the log data
    motion_seg_matches: list[str] = re.findall(motion_seg_pattern, motion_segs_str)
    # Convert matches to a list of floats
    motion_segs: list[float] = list(float(match) for match in motion_seg_matches)

    # Regular expression to find all floats after "lavfi.scene_score="
    scene_score_pattern = r"lavfi.scene_score=([0-9.]+)"
    scene_score_matches: list[str] | Generator[float] = re.findall(
        scene_score_pattern, motion_segs_str
    )
    scene_score_segs = list(float(s) for s in scene_score_matches)

    motion_segs_dict: dict[float, float] = dict(zip(motion_segs, scene_score_segs))

    return (motion_segs_dict, total_duration)


def _extract_motion_segments(
    motion_info: dict[float, float],
    threshold: float = DEFAULTS.motionless_threshold.value,
) -> list[float]:
    break_points = []
    prev_above = False  # Track if last added was above the threshold

    for _time, score in motion_info.items():
        if (score > threshold and not prev_above) or (
            score <= threshold and prev_above
        ):
            break_points.append(_time)
            prev_above = score > threshold  # Update the flag
    return break_points


# Adjust segments
class AdjustSegmentsConfig(BaseModel):
    segments: list[float]
    seg_min_duration: float = DEFAULTS.seg_min_duration.value
    total_duration: float
    keyframes: list[float]


def _ensure_minimum_segment_length(
    video_segments: list[float],
    seg_min_duration: float = DEFAULTS.seg_min_duration.value,
    total_duration: float | None = None,
) -> list[float]:
    """
    Ensures that every segment in the video_segments list is at least seg_min_duration seconds long.

    Args:
        video_segments (list[float]): List of start and end times in seconds.
        seg_min_duration (float, optional): Minimum duration for each segment in seconds. Defaults to 2.

    Raises:
        ValueError: If video_segments does not contain pairs of start and end times.

    Returns:
        list[float]: Updated list of start and end times with adjusted segment durations.
    """
    if seg_min_duration == 0 or video_segments == []:
        return video_segments

    if seg_min_duration < 0:
        raise ValueError(
            f"seg_min_duration must greater than 0 but got {seg_min_duration}."
        )

    if len(video_segments) % 2 != 0:
        raise ValueError("video_segments must contain pairs of start and end times.")

    if total_duration is None:
        total_duration = video_segments[-1]

    updated_segments = []
    for i in range(0, len(video_segments), 2):
        start_time = video_segments[i]
        end_time = video_segments[i + 1]
        duration = end_time - start_time

        if duration >= seg_min_duration or len(video_segments) == 2:
            updated_segments.extend([start_time, end_time])
            continue

        if i == len(video_segments) - 2:
            # This is the last segment
            start_time = max(0, end_time - seg_min_duration)
        else:
            # Calculate the difference between the minimum duration and the current duration
            diff = seg_min_duration - duration
            # Adjust the start and end times to increase the duration to the minimum
            start_time = max(0, start_time - diff / 2)
            end_time = min(start_time + seg_min_duration, total_duration)

        updated_segments.extend([start_time, end_time])

    # Ensure the hole video is long enough
    if updated_segments[-1] - updated_segments[0] < seg_min_duration:
        return []

    return updated_segments


def _adjust_segments_to_keyframes(
    video_segments: list[float], keyframes_segments: list[float]
) -> list[float]:
    adjusted_segments = []
    keyframe_index = 0

    for i, _time in enumerate(video_segments):
        if i % 2 == 0:  # start time
            # 找到不大於當前時間的最大關鍵幀時間
            while (
                keyframe_index < len(keyframes_segments)
                and keyframes_segments[keyframe_index] <= _time
            ):
                keyframe_index += 1
            adjusted_time = (
                keyframes_segments[keyframe_index - 1] if keyframe_index > 0 else time
            )
            adjusted_segments.append(adjusted_time)
        else:  # end time
            # 找到不小於當前時間的最小關鍵幀時間
            while (
                keyframe_index < len(keyframes_segments)
                and keyframes_segments[keyframe_index] < _time
            ):
                keyframe_index += 1
            adjusted_time = (
                keyframes_segments[keyframe_index]
                if keyframe_index < len(keyframes_segments)
                else _time
            )
            adjusted_segments.append(adjusted_time)

    return adjusted_segments


def _merge_overlapping_segments(segments: list[float]) -> list[float]:
    """_summary_

    Args:
        segments (list[float]): _description_

    Returns:
        list[float]: _description_
    """
    # Sort segments by start time
    sorted_segments = sorted(
        (segments[i], segments[i + 1]) for i in range(0, len(segments), 2)
    )
    if len(sorted_segments) == 0:
        return []

    merged_segments = []
    current_start, current_end = sorted_segments[0]

    for start, end in sorted_segments[1:]:
        if start <= current_end:
            # Overlapping segments, merge them
            current_end = max(current_end, end)
        else:
            # No overlap, add the current segment and move to the next
            merged_segments.extend([current_start, current_end])
            current_start, current_end = start, end

    # Add the last segment
    merged_segments.extend([current_start, current_end])

    return merged_segments


def _adjust_segments_pipe(
    adjusted_segments_config: AdjustSegmentsConfig,
) -> list[float]:
    ensured_minimum: list[float] = (
        _ensure_minimum_segment_length(
            adjusted_segments_config.segments,
            adjusted_segments_config.seg_min_duration,
            adjusted_segments_config.total_duration,
        )
        if adjusted_segments_config.seg_min_duration > 0
        else adjusted_segments_config.segments
    )

    adjusted_segments: list[float] = _adjust_segments_to_keyframes(
        ensured_minimum,
        adjusted_segments_config.keyframes,
    )

    if len(adjusted_segments) % 2 == 1:
        adjusted_segments.append(adjusted_segments_config.total_duration)

    merged_overlapping_segments: list[float] = _merge_overlapping_segments(
        adjusted_segments
    )

    return merged_overlapping_segments


@timing
def cut_silence(
    input_file: Path | str,
    output_file: Path | str | None = None,
    dB: int = DEFAULTS.db_threshold.value,
    sampling_duration: float = DEFAULTS.sampling_duration.value,
    seg_min_duration: float = DEFAULTS.seg_min_duration.value,
    even_further: FurtherMethod = "remove",  # For other segments, remove means remove, None means copy
    odd_further: FurtherMethod = None,  # For segments, remove means remove, None means copy
    delete_after: bool = False,
) -> Path | ERROR_CODE:
    class ERROR_CODE(Enum):
        DURATION_LESS_THAN_ZERO = auto()
        NO_VALID_SEGMENTS = auto()
        FAILED_TO_CUT = auto()

    if sampling_duration <= 0:
        logger.error("Duration must be greater than 0.")
        return ERROR_CODE.DURATION_LESS_THAN_ZERO

    # init task
    task_descripton = f"{_TASKS.CUT_SILENCE}_by_{dB}"
    input_file = Path(input_file)

    # Handle output file path
    output_file = _handle_output_file_path(input_file, output_file, task_descripton)

    # Handle temp output file
    temp_output_file: Path = output_file.parent / (
        output_file.stem + "_processing" + output_file.suffix
    )

    logger.info(
        f"{task_descripton.capitalize()} {input_file} to {output_file} with {dB = } ,{sampling_duration = }, {seg_min_duration = }."
    )

    non_silence_str: str = str(
        FFRenderTasks()
        .get_silence_segs(
            input_file=input_file,
            dB=dB,
            sampling_duration=sampling_duration,
        )
        .render()
    )
    non_silence_segments, total_duration, _ = _extract_non_silence_info(non_silence_str)
    logger.info(f"{non_silence_segments = }")

    keyframes = FPRenderTasks().keyframes(input_file).render()

    adjusted_segments_config = AdjustSegmentsConfig(
        segments=non_silence_segments,
        seg_min_duration=seg_min_duration,
        total_duration=total_duration,
        keyframes=keyframes,
    )

    adjusted_segments: list[float] = _adjust_segments_pipe(adjusted_segments_config)

    if adjusted_segments == []:
        logger.error(f"No valid segments found for {input_file}.")
        return ERROR_CODE.NO_VALID_SEGMENTS

    try:
        advanced_keep_or_remove_by_split_segs(
            input_file=input_file,
            output_file=temp_output_file,
            video_segments=adjusted_segments,
            even_further=even_further,
            odd_further=odd_further,
            delete_after=delete_after,
        )
        temp_output_file.replace(output_file)

        return output_file

    except Exception as e:
        logger.error(f"Failed to {task_descripton} for {input_file}. Error: {e}")
        raise e
        # return ERROR_CODE.FAILED_TO_CUT


@timing
def cut_motionless(
    input_file: Path | str,
    output_file: Path | str | None = None,
    threshold: float = DEFAULTS.motionless_threshold.value,
    sampling_duration: float = DEFAULTS.sampling_duration.value,
    seg_min_duration: float = DEFAULTS.seg_min_duration.value,
    even_further: FurtherMethod = "remove",  # For other segments, remove means remove, None means copy
    odd_further: FurtherMethod = None,  # For segments, remove means remove, None means copy
    delete_after: bool = False,
) -> Path | ERROR_CODE:
    if sampling_duration <= 0:
        logger.error("Duration must be greater than 0.")
        return ERROR_CODE.DURATION_LESS_THAN_ZERO

    # init task
    task_descripton = f"{_TASKS.CUT_MOTIONLESS}_by_{threshold}"
    input_file = Path(input_file)

    # Handle output file path
    output_file = _handle_output_file_path(input_file, output_file, task_descripton)

    # Handle temp output file
    temp_output_file: Path = output_file.parent / (
        output_file.stem + "_processing" + output_file.suffix
    )

    logger.info(
        f"{task_descripton.capitalize()} {input_file} to {output_file} with {threshold = } ,{sampling_duration = }, {seg_min_duration = }."
    )

    motion_str: str = str(
        FFRenderTasks()
        .get_motion_segs(input_file=input_file, sampling_duration=sampling_duration)
        .render()
    )
    motion_info, total_duration = _extract_motion_info(motion_str)
    motion_segs = _extract_motion_segments(motion_info, threshold)

    if motion_segs == [0.0]:
        logger.error(f"No valid segments found for {input_file}.")
        return ERROR_CODE.NO_VALID_SEGMENTS

    keyframes = FPRenderTasks().keyframes(input_file).render()

    adjusted_segments_config = AdjustSegmentsConfig(
        segments=motion_segs,
        seg_min_duration=seg_min_duration,
        total_duration=total_duration,
        keyframes=keyframes,
    )

    adjusted_segments: list[float] = _adjust_segments_pipe(adjusted_segments_config)

    if adjusted_segments == []:
        logger.error(f"No valid segments found for {input_file}.")
        return ERROR_CODE.NO_VALID_SEGMENTS

    try:
        advanced_keep_or_remove_by_split_segs(
            input_file=input_file,
            output_file=temp_output_file,
            video_segments=adjusted_segments,
            even_further=even_further,
            odd_further=odd_further,
            delete_after=delete_after,
        )
        temp_output_file.replace(output_file)
        return output_file

    except Exception as e:
        logger.error(f"Failed to {task_descripton} for {input_file}. Error: {e}")
        raise e


# cut rerender
class CSFiltersInfo(Enum):
    VIDEO = {
        "filename": f"temp_{time.strftime('%Y%m%d-%H%M%S')}_video_filter_",
        "texts": [
            "select='",
            "', setpts=N/FRAME_RATE/TB",
        ],
    }
    AUDIO = {
        "filename": f"temp_{time.strftime('%Y%m%d-%H%M%S')}_audio_filter_",
        "texts": [
            "aselect='",
            "', asetpts=N/SR/TB",
        ],
    }


def _gen_cut_segs_filter(
    filter_texts: list[str],
    videoSectionTimings: list[float],
) -> Generator[str, None, None]:
    yield filter_texts[0]
    yield from (
        f"between(t,{videoSectionTimings[i]},{videoSectionTimings[i + 1]})"
        + ("+" if i != len(videoSectionTimings) - 2 else "")
        for i in range(0, len(videoSectionTimings), 2)
    )
    yield filter_texts[1]


def _create_cut_segs_filter_tempfile(
    filter_info: CSFiltersInfo,
    videoSectionTimings: list[float],
) -> Path:
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", encoding="UTF-8", prefix=filter_info.value["filename"]
    ) as temp_file:
        for line in _gen_cut_segs_filter(
            filter_info.value["texts"], videoSectionTimings
        ):
            temp_file.write(f"{line}\n")
        path: Path = Path(temp_file.name)
    return path


def _create_cut_sl_kwargs(
    input_file: Path | str, dB: int, sampling_duration: float
) -> dict:
    non_silence_str: str = str(
        FFRenderTasks()
        .get_silence_segs(
            input_file=input_file, dB=dB, sampling_duration=sampling_duration
        )
        .render()
    )
    non_silence_segments, total_duration, _ = _extract_non_silence_info(non_silence_str)
    video_filter_script: Path = _create_cut_segs_filter_tempfile(
        CSFiltersInfo.VIDEO, non_silence_segments
    )
    audio_filter_script: Path = _create_cut_segs_filter_tempfile(
        CSFiltersInfo.AUDIO, non_silence_segments
    )

    return {
        "filter_script:v": video_filter_script,
        "filter_script:a": audio_filter_script,
    }


def _create_cut_motionless_kwargs(
    input_file: Path | str, threshold: float, sampling_duration: float
) -> dict:
    motion_str: str = str(
        FFRenderTasks()
        .get_motion_segs(input_file=input_file, sampling_duration=sampling_duration)
        .render()
    )
    motion_info, total_duration = _extract_motion_info(motion_str)
    motion_segs = _extract_motion_segments(motion_info, threshold)

    if len(motion_segs) == 0:
        logger.error(f"No valid segments found for {input_file}.")
        raise ValueError

    if len(motion_segs) % 2 == 1:
        motion_segs.append(total_duration)

    video_filter_script: Path = _create_cut_segs_filter_tempfile(
        CSFiltersInfo.VIDEO, motion_segs
    )
    audio_filter_script: Path = _create_cut_segs_filter_tempfile(
        CSFiltersInfo.AUDIO, motion_segs
    )

    return {
        "filter_script:v": video_filter_script,
        "filter_script:a": audio_filter_script,
    }


def _partial_render_task(
    task: MethodType | FunctionType | Callable,
    **config,
) -> FurtherRenderTask:
    if "FFRenderTasks" in task.__qualname__:

        def _partial1(
            input_file: str | Path,
            output_file: str | Path,
            options: OptionFFRender | None = None,
        ) -> Any:
            return (
                task(input_file=input_file, output_file=output_file, **config)
                .override_option(options=options)
                .render()
            )

        return _partial1
    else:

        def _partial2(
            input_file: str | Path,
            output_file: str | Path,
            options: OptionFFRender | None = None,
        ) -> Any:
            if options is None:
                options = {}
            return task(
                input_file=input_file, output_file=output_file, **(config | options)
            )

        return _partial2


class PARTIAL_TASKS(FunctionEnum):
    @staticmethod
    def speedup(
        multiple: float | int = DEFAULTS.speedup_multiple.value,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ):
        return _partial_render_task(
            task=FFRenderTasks().speedup,
            multiple=multiple,
            input_kwargs=input_kwargs,
            output_kwargs=output_kwargs,
        )

    @staticmethod
    def jumpcut(
        b1_duration: float = 5,
        b2_duration: float = 5,
        b1_multiple: float = 1,  # 0 means remove this part
        b2_multiple: float = 0,  # 0 means remove this part
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ):
        return _partial_render_task(
            task=FFRenderTasks().jumpcut,
            b1_duration=b1_duration,
            b2_duration=b2_duration,
            b1_multiple=b1_multiple,
            b2_multiple=b2_multiple,
            input_kwargs=input_kwargs,
            output_kwargs=output_kwargs,
        )

    @staticmethod
    def custom(
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ):
        return _partial_render_task(
            task=FFRenderTasks().custom,
            input_kwargs=input_kwargs,
            output_kwargs=output_kwargs,
        )

    @staticmethod
    def cut(
        ss: str = "00:00:00",
        to: str = "00:00:01",
        rerender: bool = False,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ):
        return _partial_render_task(
            task=FFRenderTasks().cut,
            ss=ss,
            to=to,
            rerender=rerender,
            input_kwargs=input_kwargs,
            output_kwargs=output_kwargs,
        )

    @staticmethod
    def cut_silence_rerender(
        dB: int = DEFAULTS.db_threshold.value,
        sampling_duration: float = DEFAULTS.sampling_duration.value,
        input_kwargs: FFKwargs | None = None,
        output_kwargs: FFKwargs | None = None,
    ):
        return _partial_render_task(
            task=FFRenderTasks().cut_silence_rerender,
            dB=dB,
            sampling_duration=sampling_duration,
            input_kwargs=input_kwargs,
            output_kwargs=output_kwargs,
        )

    @staticmethod
    def cut_silence(
        dB: int = DEFAULTS.db_threshold.value,
        sampling_duration: float = DEFAULTS.sampling_duration.value,
        seg_min_duration: float = DEFAULTS.seg_min_duration.value,
        even_further: FurtherMethod = "remove",  # For other segments, remove means remove, None means copy
        odd_further: FurtherMethod = None,
    ):  # For segments, remove means remove, None means copy
        return _partial_render_task(
            task=cut_silence,
            dB=dB,
            sampling_duration=sampling_duration,
            seg_min_duration=seg_min_duration,
            even_further=even_further,
            odd_further=odd_further,
        )

    @staticmethod
    def cut_motionless(
        threshold: float = DEFAULTS.motionless_threshold.value,
        sampling_duration: float = DEFAULTS.sampling_duration.value,
        seg_min_duration: float = DEFAULTS.seg_min_duration.value,
        even_further: FurtherMethod = "remove",  # For other segments, remove means remove, None means copy
        odd_further: FurtherMethod = None,  # For segments, remove means remove, None means copy
    ):
        return _partial_render_task(
            task=cut_motionless,
            threshold=threshold,
            sampling_duration=sampling_duration,
            seg_min_duration=seg_min_duration,
            even_further=even_further,
            odd_further=odd_further,
        )

    @staticmethod
    def partion_video(
        partition_config: Optional[PartitionConfig] = None,
        output_dir: Optional[Path | str] = None,
    ):
        return _partial_render_task(
            task=partion_video,
            partition_config=partition_config,
            output_dir=output_dir,
        )
