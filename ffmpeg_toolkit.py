# import ffmpeg
from typing import (
    Any,
    Iterable,
    NotRequired,
    Sequence,
    Optional,
    Callable,
    Literal,
    Mapping,
    TypedDict,
)
from types import MethodType, FunctionType
from pathlib import Path
from enum import StrEnum, auto
from collections import deque
from collections.abc import Generator
import re
import subprocess
from enum import Enum
import tempfile
import time
import os
import concurrent.futures
import json
from ffmpeg_types import EncodeKwargs, VideoSuffix
import functools
from pydantic import BaseModel, Field, field_validator
import shutil
from itertools import batched, accumulate


class FrozenBaseModel(BaseModel):
    class Config:
        frozen = True


# from app.common import logger
class logger:
    @classmethod
    def info(cls, message: str) -> None:
        print(message)

    @classmethod
    def error(cls, message: str) -> None:
        print(message)


def timing(func: Callable):
    @functools.wraps(func)
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


class _tasks(StrEnum):
    SPEEDUP = auto()
    JUMPCUT = auto()
    CONVERT = auto()
    CUT = auto()
    KEEP_OR_REMOVE = auto()
    MERGE = auto()
    PROBE_ENCODING = auto()
    PROBE_DURATION = auto()
    PROBE_IS_VALID_VIDEO = auto()
    GET_NON_SILENCE_SEGS = auto()
    CUT_SILENCE = auto()
    CUT_SILENCE_RERENDER = auto()
    SPLIT = auto()
    PARTITION = auto()


# basic
type FF_Kwargs_Value = str | Path | float | int
type FF_Kwargs = dict[str, FF_Kwargs_Value]

# dict[str, Path]
# | dict[str, str]
# | dict[str, int]
# | dict[str, float]
# | dict[str, Path | float | str | int]
# | dict[str, str | Path]
# | dict[str, str | int]
# | dict[str, str | float]
# | dict[str, str | Path | float | int]


def _create_ff_kwargs(
    input_file: Path,
    output_file: Path,
    input_kwargs: FF_Kwargs,
    output_kwargs: FF_Kwargs,
) -> FF_Kwargs:
    input_kwargs_default: FF_Kwargs = {"hwaccel": "auto"}
    output_kwargs_default: FF_Kwargs = {"loglevel": "warning"}

    input_file_kwargs: Mapping[Literal["i"], Path] = {"i": input_file}
    # Handle file path
    if output_file == Path():
        output_file_kwargs: Mapping[Literal["y"], Path] = {}
    else:
        output_file_kwargs = {"y": output_file}
    ff_kwargs: FF_Kwargs = (
        input_kwargs_default
        | input_kwargs
        | input_file_kwargs
        | output_kwargs_default
        | output_kwargs
        | output_file_kwargs
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
def _ffmpeg(**ffkwargs) -> str:
    logger.info(f"Executing FFmpeg with {ffkwargs = }")
    command = ["ffmpeg"] + _dic_to_ffmpeg_kwargs(ffkwargs)
    logger.info(f"command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding="utf-8"
        )
        return result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to execute FFmpeg. Error: {e}")
        raise e


def _ffprobe(**ffkwargs):
    logger.info(f"Executing ffprobe with {ffkwargs = }")
    command = ["ffprobe"] + _dic_to_ffmpeg_kwargs(ffkwargs)
    logger.info(f"command: {' '.join(command)}")
    subprocess.run(command, check=True, encoding="utf-8")


# probe
def probe_is_valid_video(input_file: Path, **othertags) -> bool:  #
    """Function to check if a video file is valid using ffprobe."""
    output_kwargs: dict = {
        "v": "error",
        "show_entries": "format=duration",
        "of": "default=noprint_wrappers=1:nokey=1",
        "i": input_file,
    } | othertags
    logger.info(f"Validating {input_file.name} with {output_kwargs = }")
    try:
        command = ["ffprobe"] + _dic_to_ffmpeg_kwargs(output_kwargs)
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding="utf-8"
        ).stdout.strip()
        if result:
            message = f"Validated file: {input_file}, Status: Valid"
            logger.info(message)
            return True
        else:
            message = f"Validated file: {input_file}, Status: Invalid"
            logger.info(message)
            return False
    except Exception as e:
        message = f"Validating file: {input_file}, Error: {str(e)}"
        logger.info(message)
        return False


def probe_duration(input_file: Path, **othertags) -> float:  #
    output_kwargs: dict = {
        "v": "error",
        "show_entries": "format=duration",
        "of": "default=noprint_wrappers=1:nokey=1",
        "i": input_file,
    } | othertags
    logger.info(f"Probing {input_file.name} duration with {output_kwargs = }")
    command = ["ffprobe"] + _dic_to_ffmpeg_kwargs(output_kwargs)
    result = subprocess.run(
        command, capture_output=True, text=True, check=True, encoding="utf-8"
    )
    probe_duration = result.stdout.strip()
    logger.info(f"{input_file.name} duration probed: {probe_duration}")

    return float(probe_duration or 0)


def probe_encoding(input_file: Path, **othertags) -> EncodeKwargs:  #
    output_kwargs: dict = {
        "v": "error",
        "print_format": "json",
        "show_format": "",
        "show_streams": "",
        "i": input_file,
    } | othertags
    logger.info(f"Probing {input_file.name} encoding with {output_kwargs = }")
    # Probe the video file to get metadata
    command = ["ffprobe"] + _dic_to_ffmpeg_kwargs(output_kwargs)
    result = subprocess.run(
        command, capture_output=True, text=True, check=True, encoding="utf-8"
    )
    probe = json.loads(result.stdout)

    # Initialize the dictionary with default values
    encoding_info: EncodeKwargs = {}

    # Extract video stream information
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    if video_stream:
        encoding_info["video_track_timescale"] = int(
            video_stream.get("time_base").split("/")[1]
        )
        encoding_info["vcodec"] = video_stream.get("codec_name")
        encoding_info["video_bitrate"] = int(video_stream.get("bit_rate", 0))

    # Extract audio stream information
    audio_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "audio"), None
    )
    if audio_stream:
        encoding_info["acodec"] = audio_stream.get("codec_name")
        encoding_info["ar"] = int(audio_stream.get("sample_rate", 0))

    # Extract format information
    format_info = probe.get("format", {})
    encoding_info["f"] = format_info.get("format_name").split(",")[0]
    cleaned_None = {k: v for k, v in encoding_info.items() if v is not None and v != 0}
    logger.info(f"{input_file.name} probed: {cleaned_None}")

    return cleaned_None  # type: ignore


def _probe_keyframe(input_file: Path, **othertags) -> list[float]:  #
    output_kwargs: dict = {
        "v": "error",
        "select_streams": "v:0",
        "show_entries": "packet=pts_time,flags",
        "of": "json",
        "i": input_file,
    } | othertags
    logger.info(f"Getting keyframe for {input_file.name} with {output_kwargs = }")
    command = ["ffprobe"] + _dic_to_ffmpeg_kwargs(output_kwargs)
    logger.info(f"command: {' '.join(command)}")
    result = subprocess.run(
        command, capture_output=True, text=True, check=True, encoding="utf-8"
    )
    probe = json.loads(result.stdout)
    keyframe_pts: list[float] = [
        float(packet["pts_time"])
        for packet in probe["packets"]
        if "K" in packet["flags"]
    ]
    return keyframe_pts


# converting core
class FF_Render_Exception(TypedDict):
    code: int
    message: str
    hook: NotRequired[Callable[[], Any]]


class FF_Create_Command(BaseModel):
    input_file: Path | str = Path()
    output_file: Optional[Path | str] = None
    input_kwargs: FF_Kwargs = Field(default_factory=dict)
    output_kwargs: FF_Kwargs = Field(default_factory=dict)


class FF_Create_Render(FF_Create_Command):
    task_descripton: Optional[str] = "render"
    exception: Optional[FF_Render_Exception] = None
    post_task: Optional[Callable[[], None]] = None


class FF_Create_Render_Task(FF_Create_Render):
    def custom(
        self,
        input_file: Path | str,
        output_file: Optional[Path | str] = None,
        input_kwargs: Optional[FF_Kwargs] = None,
        output_kwargs: Optional[FF_Kwargs] = None,
    ) -> FF_Create_Render:
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
        input_kwargs: Optional[FF_Kwargs] = None,
        output_kwargs: Optional[FF_Kwargs] = None,
    ) -> FF_Create_Render:
        self.task_descripton = f"{_tasks.CUT}_{_convert_timestamp_to_seconds(ss)}-{_convert_timestamp_to_seconds(to)}"
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
        multiple: float | int = 2,
        input_kwargs: Optional[FF_Kwargs] = None,
        output_kwargs: Optional[FF_Kwargs] = None,
    ) -> FF_Create_Render:
        self.task_descripton = f"{_tasks.SPEEDUP}_by_{multiple}"
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
        input_kwargs: Optional[FF_Kwargs] = None,
        output_kwargs: Optional[FF_Kwargs] = None,
    ) -> FF_Create_Render:
        self.task_descripton = f"{_tasks.JUMPCUT}_b1({b1_duration}x{b1_multiple})_b2({b2_duration}x{b2_multiple})"
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
        input_kwargs: Optional[FF_Kwargs] = None,
        output_kwargs: Optional[FF_Kwargs] = None,
    ) -> FF_Create_Render:
        # Create input.txt
        if isinstance(input_dir_or_files, Iterable):
            input_dir_or_files = [Path(video) for video in input_dir_or_files]
        else:
            input_dir_or_files = Path(input_dir_or_files)
        input_txt: Path = create_merge_txt(input_dir_or_files)

        self.task_descripton = _tasks.MERGE
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

        def post_task():
            os.remove(input_txt)
            if not any(input_txt.parent.iterdir()):
                os.rmdir(input_txt.parent)
                logger.info(f"{input_txt.parent} removed")

        self.post_task = post_task

        return self

    def get_silence_segs(
        self,
        input_file: Path | str,
        dB: int = -21,
        sl_duration: float = 0.2,
        input_kwargs: Optional[FF_Kwargs] = None,
        output_kwargs: Optional[FF_Kwargs] = None,
    ) -> FF_Create_Render:
        self.task_descripton = f"{_tasks.GET_NON_SILENCE_SEGS}_by_{dB}"
        self.input_file = input_file
        self.output_file = Path()
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        self.output_kwargs = (
            {
                "af": f"silencedetect=n={dB}dB:d={sl_duration}",
                "c:v": "copy",
                "f": "null",
                "loglevel": "info",
            }
            | ({} if output_kwargs is None else output_kwargs)
            | {"": ""}
        )

        return self

    def cut_silence_rerender(
        self,
        input_file: Path | str,
        output_file: Path | str | None = None,
        dB: int = -21,
        sl_duration: float = 0.2,
        input_kwargs: Optional[FF_Kwargs] = None,
        output_kwargs: Optional[FF_Kwargs] = None,
    ) -> FF_Create_Render:
        self.task_descripton = f"{_tasks.CUT_SILENCE_RERENDER}_by_{dB}"
        self.input_file = input_file
        self.output_file = output_file
        if input_kwargs is not None:
            self.input_kwargs = input_kwargs
        self.output_kwargs = output_kwargs = _create_cut_sl_kwargs(
            input_file, dB, sl_duration
        ) | ({} if output_kwargs is None else output_kwargs)

        def post_task():
            os.remove(str(output_kwargs["filter_script:v"]))
            os.remove(str(output_kwargs["filter_script:a"]))

        self.post_task = post_task

        return self

    def split_segments(
        self,
        input_file: Path | str,
        video_segments: Sequence[str],
        output_dir: Optional[Path | str] = None,
        input_kwargs: Optional[FF_Kwargs] = None,
        output_kwargs: Optional[FF_Kwargs] = None,
    ) -> FF_Create_Render:
        self.task_descripton = f"{_tasks.SPLIT}"
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

    def render(self):
        render_task(self)


def ff_render(
    input_file: Path | str,
    output_file: Optional[Path | str],
    task_descripton: _tasks | str,
    input_kwargs: FF_Kwargs,
    output_kwargs: FF_Kwargs,
    exception: Optional[FF_Render_Exception],
    post_task: Optional[Callable[[], None]],
) -> str | int:
    """Executeing ffmpeg with given args for different tasks

    Args:
        input_file (Path): _description_
        output_file (Path | None, optional): _description_. Defaults to None.
        task_name (str, optional): _description_. Defaults to "rendered".
        othertags (dict | None, optional): _description_. Defaults to None.

    Raises:
        e: _description_

    Returns:
        int: _description_
    """

    input_file = Path(input_file)

    # Handle output file path
    if output_file is None:
        output_file = (
            input_file.parent
            / f"{input_file.stem}_{task_descripton}{
                input_file.suffix
                if input_file.suffix in VideoSuffix
                else '.' + VideoSuffix.MKV
            }"
        )
    else:
        output_file = Path(output_file)

    # Handle temp output file path
    if output_file == Path() or r"%d" in str(output_file):
        temp_output_file: Path = output_file
    else:
        temp_output_file: Path = output_file.parent / (
            output_file.stem + "_processing" + output_file.suffix
        )

    # Exception hadling
    if exception is not None:
        match exception["code"]:
            case 0:
                shutil.copyfile(input_file, output_file)
            case 9:
                exception.get("hook", lambda: None)()
            case _:
                pass
        logger.error(exception["message"])
        return exception["code"]

    # Generate ff kwargs in
    ff_kwargs_in: FF_Create_Command = FF_Create_Command(
        input_file=input_file,
        output_file=temp_output_file,
        input_kwargs=input_kwargs,
        output_kwargs=output_kwargs,
    )

    ff_kwargs_out: FF_Kwargs = _create_ff_kwargs(**ff_kwargs_in.model_dump())

    logger.info(
        f"{task_descripton.capitalize()} {input_file.name} to {output_file.name} with {ff_kwargs_out}"
    )

    try:
        result: str = _ffmpeg(**ff_kwargs_out)
        if temp_output_file != output_file and r"%" not in str(temp_output_file.stem):
            temp_output_file.replace(output_file)

        # Handle post task
        if post_task is not None:
            post_task()

        return result
    except Exception as e:
        logger.error(
            f"Failed to do {task_descripton} videos for {input_file}. Error: {e}"
        )
        raise e


# ff_render
def render_task(
    Render_Tasks: FF_Create_Render,
) -> str | int:
    try:
        result = ff_render(**Render_Tasks.model_dump())
        return result
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e


# ff_render: sppedup
def _create_force_keyframes_kwargs(keyframe_times: int = 2) -> dict[str, str]:
    """_summary_

    Args:
        keyframe_times (int, optional): _description_. Defaults to 2.

    Returns:
        dict[str, str]: _description_
    """
    return {"force_key_frames": f"expr:gte(t,n_forced*{keyframe_times})"}


def _create_speedup_kwargs(multiple: float) -> dict[str, str]:
    SPEEDUP_task_THRESHOLD: int = 5
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


# ff_render: jumpcut
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
        temp_output_dir = Path(tempfile.mkdtemp())
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


type FURTHER_KWARGS = None | FF_Kwargs
type Further_Render = Callable[[str | Path], Any]


def my_partial_task(
    task: MethodType | FunctionType | Callable, **config
) -> Further_Render:
    if (
        isinstance(task, MethodType)
        and task.__self__.__class__ == FF_Create_Render_Task
    ):

        def _partial(input_file: str | Path) -> Any:
            return task(
                input_file=input_file, output_file=input_file, **config
            ).render()
    else:

        def _partial(input_file: str | Path) -> Any:
            return task(input_file=input_file, output_file=input_file, **config)

    return _partial


# keep or remove copy/rendering by split segs
def advanced_keep_or_remove_by_split_segs(
    input_file: Path | str,
    output_file: Path | str | None,
    video_segments: Sequence[str] | Sequence[float],
    even_kwargs: Optional[
        FURTHER_KWARGS | Literal["remove"]
    ] = "remove",  # For other segments, remove means remove, None means copy
    odd_kwargs: Optional[
        FURTHER_KWARGS | Literal["remove"]
    ] = None,  # For segments, remove means remove, None means copy
    remove_temp_handle: bool = True,
) -> int:
    task_descripton = _tasks.KEEP_OR_REMOVE + "_split"
    input_file = Path(input_file)

    # Set the output file path
    if output_file is None:
        output_file = input_file.parent / (
            input_file.stem + "_" + task_descripton + input_file.suffix
        )
    else:
        output_file = Path(output_file)

    logger.info(
        f"{task_descripton.capitalize()} {input_file.name} to {output_file.name} with {odd_kwargs = } ,{even_kwargs = }."
    )

    # Double every time point and convert to timestamp if needed
    video_segments = list(
        _convert_seconds_to_timestamp(s) if isinstance(s, (float, int)) else s
        for s in video_segments
    )

    # Split videos
    temp_dir: Path = Path(tempfile.mkdtemp())

    ff_split_task: FF_Create_Render = FF_Create_Render_Task().split_segments(
        input_file=input_file,
        video_segments=video_segments,
        output_dir=temp_dir,
    )
    render_task(ff_split_task)

    splitted_videos: list[Path] = sorted(
        temp_dir.glob(f"*{input_file.suffix}"),
        key=lambda video_file: int(video_file.stem.split("_")[0]),
    )

    # Use ThreadPoolExecutor to manage rendreing tasks
    num_cores = os.cpu_count()
    rerender_handle = any(
        kwargs not in ["remove", None] for kwargs in [even_kwargs, odd_kwargs]
    )
    further_kwargs: dict[int, FURTHER_KWARGS | Literal["remove"] | None] = {
        0: {} if even_kwargs is None and rerender_handle else even_kwargs,
        1: {} if odd_kwargs is None and rerender_handle else odd_kwargs,
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = []

        for video in splitted_videos:
            index: int = int(video.stem.split("_")[0])
            i_remainder = index % 2

            # remove unwanted segments
            if further_kwargs[i_remainder] == "remove":
                os.remove(video)
                continue

            # Skip further rendering if the segment is to be copied
            if further_kwargs[i_remainder] is None:
                continue

            # Further render the segment with giver kwargs
            ff_custom_kwargs: FF_Create_Render = FF_Create_Render_Task().custom(
                input_file=video,
                output_file=video,
                output_kwargs=further_kwargs[i_remainder],  # type: ignore
            )

            # Submit further render task to the executor
            future = executor.submit(render_task, ff_custom_kwargs)
            futures.append(future)  # Store the future for tracking
            future.result()
        # Optionally, wait for all futures to complete
        # concurrent.futures.wait(futures)

    try:
        # Merge the kept segments
        video_files: list[Path] = sorted(
            list(
                video
                for video in temp_dir.glob("*")
                if video.suffix.lstrip(".") in VideoSuffix
            ),
            key=lambda video: int(str(video.stem).split("_")[0]),
        )
        merge_task: FF_Create_Render = FF_Create_Render_Task().merge(
            video_files, output_file
        )
        render_task(merge_task)

        # Clean up temporary files and dir
        if remove_temp_handle:
            for video in temp_dir.iterdir():
                os.remove(video)
            os.rmdir(temp_dir)
        return 0

    except Exception as e:
        logger.error(f"Failed to {task_descripton} for {input_file}. Error: {e}")
        return 1


# keep or remove copy/rendering by cuts
def advanced_keep_or_remove_by_cuts(
    input_file: Path | str,
    output_file: Path | str | None,
    video_segments: Sequence[str] | Sequence[float],
    even_kwargs: Optional[
        FURTHER_KWARGS | Literal["remove"]
    ] = "remove",  # For other segments, remove means remove, None means copy
    odd_kwargs: Optional[
        FURTHER_KWARGS | Literal["remove"]
    ] = None,  # For segments, remove means remove, None means copy
) -> int:
    task_descripton = _tasks.KEEP_OR_REMOVE
    input_file = Path(input_file)

    # Set the output file path
    if output_file is None:
        output_file = input_file.parent / (
            input_file.stem + "_" + task_descripton + input_file.suffix
        )
    else:
        output_file = Path(output_file)

    logger.info(
        f"{task_descripton.capitalize()} {input_file.name} to {output_file.name} with {odd_kwargs = } ,{even_kwargs = }."
    )

    # Double every time point and convert to timestamp if needed
    video_segments = deque(
        _convert_seconds_to_timestamp(s) if isinstance(s, (float, int)) else s
        for o in video_segments
        for s in (o, o)  # double every time point
    )

    # Create a full segment list
    video_segments.appendleft("00:00:00.000")
    video_segments.append(_convert_seconds_to_timestamp(probe_duration(input_file)))
    batched_segments = batched(video_segments, 2)
    # Use ThreadPoolExecutor to manage rendreing tasks
    temp_dir: Path = Path(tempfile.mkdtemp())
    num_cores = os.cpu_count()
    cut_videos = []
    further_kwargs: dict[int, FURTHER_KWARGS | Literal["remove"] | None] = {
        0: even_kwargs,
        1: odd_kwargs,
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = []

        for i, segment in enumerate(batched_segments):
            i_remainder = i % 2

            # remove unwanted segments
            if further_kwargs[i_remainder] == "remove":
                continue

            # cut segments by submitting cut task to the executor
            start_time: str = segment[0]
            end_time: str = segment[1]
            if start_time[:8] == end_time[:8]:
                logger.info(
                    f"Sagment is too short to cut, skipping {start_time} ot {end_time}"
                )
                continue
            seg_output_file = temp_dir / f"{i}{input_file.suffix}"
            cut_videos.append(seg_output_file)
            ff_cut_task: FF_Create_Render = FF_Create_Render_Task().cut(
                input_file=input_file,
                output_file=seg_output_file,
                ss=start_time,
                to=end_time,
            )
            future = executor.submit(render_task, ff_cut_task)
            futures.append(future)  # Store the future for tracking
            future.result()  # Ensures `cut` completes before proceeding

            # Skip further rendering if the segment is to be copied
            if further_kwargs[i_remainder] is None:
                continue

            # Further render the segment with giver kwargs
            ff_custom_task: FF_Create_Render = FF_Create_Render_Task().custom(
                input_file=seg_output_file,
                output_file=seg_output_file,
                output_kwargs=further_kwargs[i_remainder],  # type: ignore
            )

            # Submit further render task to the executor
            future = executor.submit(render_task, ff_custom_task)
            futures.append(future)  # Store the future for tracking
            future.result()
        # Optionally, wait for all futures to complete
        # concurrent.futures.wait(futures)

    try:
        # Merge the kept segments
        # Sort the cut video paths by filename by index order
        cut_videos.sort(key=lambda video_file: int(video_file.stem))
        merge_task: FF_Create_Render = FF_Create_Render_Task().merge(
            cut_videos, output_file
        )
        render_task(merge_task)

        # Clean up temporary files and dir
        for video_path in cut_videos:
            os.remove(video_path)
        os.rmdir(temp_dir)
        return 0

    except Exception as e:
        logger.error(f"Failed to {task_descripton} for {input_file}. Error: {e}")
        return 1


# Split segments by part
def _get_segments_from_parts_count(
    duration: float | str, parts_count: int, portion: Optional[list[int]] = None
) -> Sequence[str]:
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


type Method = Further_Render | Literal["remove"] | None
type Portion_Method_Specific = list[tuple[int, Method]] | list[tuple[int, None]]
type Portion_Method = Portion_Method_Specific | list[tuple[int, Method] | int]


class Partition_Config(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    count: int = Field(default=0, gt=0)
    portion_method: Optional[Portion_Method] = None

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
        cls, portion_method: Portion_Method, info
    ) -> Portion_Method_Specific:
        if portion_method is not None:
            _portion_method: Portion_Method_Specific = [
                (p, None) if isinstance(p, int) else p for p in portion_method
            ]
            if (_sum := sum(p[0] for p in _portion_method)) != (
                _count := info.data["count"]
            ) and info.data["count"] != 0:
                raise ValueError(
                    f"Sum of portions ({_sum}) must equal to count ({_count})"
                )
            return _portion_method


def partion_video(
    input_file: Path | str,
    partition_config: Optional[Partition_Config] = None,
    output_dir: Optional[Path | str] = None,
) -> int:
    if partition_config is None:
        partition_config = Partition_Config()

    task_descripton = _tasks.PARTITION
    input_file = Path(input_file)
    if output_dir is None:
        output_dir = input_file.parent / f"{input_file.stem}_{task_descripton}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    duration: float = probe_duration(input_file)
    video_segments: Sequence[str] = _get_segments_from_parts_count(
        duration,
        partition_config.count,
        [p[0] for p in partition_config.portion_method],  # type: ignore
    )
    logger.info(f"{video_segments = }")
    logger.info(
        f"{task_descripton.capitalize()} {input_file.name} to {output_dir} with {partition_config}."
    )

    # Split videos
    temp_dir: Path = Path(tempfile.mkdtemp())
    ff_split_task: FF_Create_Render = FF_Create_Render_Task().split_segments(
        input_file=input_file,
        video_segments=video_segments,
        output_dir=temp_dir,
    )
    render_task(ff_split_task)

    video_files: list[Path] = sorted(
        list(
            video
            for video in temp_dir.glob("*")
            if video.suffix.lstrip(".") in VideoSuffix
        ),
        key=lambda video: int(str(video.stem).split("_")[0]),
    )

    # Further render videos
    try:
        for video in video_files:
            i_remainder = int(video.stem.split("_")[0])
            _method: Method = (
                partition_config.portion_method[i_remainder][1]  # type: ignore
            )
            if _method == "remove":
                os.remove(video)
                continue

            if _method is not None:
                _method(input_file=video)

            output_path = output_dir / video.name
            shutil.move(str(video), str(output_path))

        os.rmdir(temp_dir)
        return 0

    except Exception as e:
        logger.error(f"Failed to {task_descripton} for {input_file}. Error: {e}")
        return 1


# ff_render: cut silence copy / render
# Extract non silence segments info
def _extract_non_silence_segs_info(
    non_silence_segs_str: str,
) -> tuple[Sequence[float], float, float]:
    # Total duration
    total_duration_pattern = r"Duration: (.+?),"
    total_duration_match: str | None = re.findall(
        total_duration_pattern, non_silence_segs_str
    )[0]
    total_duration: float = _convert_timestamp_to_seconds(
        total_duration_match if total_duration_match else "0.0"
    )

    # Regular expression to find all floats after "silence_start or end: "
    silence_seg_pattern = r"silence_(?:start|end): ([0-9.]+)"
    # Find all matches in the log data
    silence_seg_matches: list[str] = re.findall(
        silence_seg_pattern, non_silence_segs_str
    )
    # Convert matches to a list of floats
    non_silence_segs: deque[float] = deque(
        float(match) for match in silence_seg_matches
    )
    # Handle silence start and end to represent non silence
    non_silence_segs.appendleft(0.0)
    non_silence_segs.append(total_duration)

    # Regular expression to find all floats after silence_duration: "
    silence_duration_pattern = r"silence_duration: ([0-9.]+)"
    silence_duration_matches: list[str] | Generator[float] = re.findall(
        silence_duration_pattern, non_silence_segs_str
    )
    silence_duration_matches = (float(s) for s in silence_duration_matches)
    total_silence_duration: float = sum(silence_duration_matches)

    return (non_silence_segs, total_duration, total_silence_duration)


def _adjust_segments_to_keyframes(
    video_segments: Sequence[float], keyframe_times: Sequence[float]
) -> Sequence[float]:
    adjusted_segments = []
    keyframe_index = 0

    for i, _time in enumerate(video_segments):
        if i % 2 == 0:  # start time
            # 找到不大於當前時間的最大關鍵幀時間
            while (
                keyframe_index < len(keyframe_times)
                and keyframe_times[keyframe_index] <= _time
            ):
                keyframe_index += 1
            adjusted_time = (
                keyframe_times[keyframe_index - 1] if keyframe_index > 0 else time
            )
            adjusted_segments.append(adjusted_time)
        else:  # end time
            # 找到不小於當前時間的最小關鍵幀時間
            while (
                keyframe_index < len(keyframe_times)
                and keyframe_times[keyframe_index] < _time
            ):
                keyframe_index += 1
            adjusted_time = (
                keyframe_times[keyframe_index]
                if keyframe_index < len(keyframe_times)
                else _time
            )
            adjusted_segments.append(adjusted_time)

    return adjusted_segments


def _ensure_minimum_segment_length(
    video_segments: Sequence[float],
    seg_min_duration: float = 1,
    total_duration: float | None = None,
) -> Sequence[float]:
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


def _merge_overlapping_segments(segments: Sequence[float]) -> Sequence[float]:
    """_summary_

    Args:
        segments (Sequence[float]): _description_

    Returns:
        Sequence[float]: _description_
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


@timing
def cut_silence(
    input_file: Path | str,
    output_file: Path | str | None = None,
    dB: int = -21,
    sl_duration: float = 0.2,
    seg_min_duration: float = 0,
    even_kwargs: Optional[
        FURTHER_KWARGS | Literal["remove"]
    ] = "remove",  # For other segments, None means copy, remove means remove
    odd_kwargs: Optional[
        FURTHER_KWARGS | Literal["remove"]
    ] = None,  # For other segments, None means copy, remove means remove
) -> int | Enum:
    class error_code(Enum):
        DURATION_LESS_THAN_ZERO = auto()
        NO_VALID_SEGMENTS = auto()
        FAILED_TO_CUT = auto()

    if sl_duration <= 0:
        logger.error("Duration must be greater than 0.")
        return error_code.DURATION_LESS_THAN_ZERO

    # init task
    task_descripton = f"{_tasks.CUT_SILENCE}_by_{dB}"
    input_file = Path(input_file)

    if output_file is None:
        output_file = input_file.parent / (
            input_file.stem + "_" + task_descripton + input_file.suffix
        )
    else:
        output_file = Path(output_file)
    temp_output_file: Path = output_file.parent / (
        output_file.stem + "_processing" + output_file.suffix
    )
    logger.info(
        f"{task_descripton.capitalize()} {input_file} to {output_file} with {dB = } ,{sl_duration = }, {seg_min_duration = }."
    )

    get_silence_segs_task: FF_Create_Render = FF_Create_Render_Task().get_silence_segs(
        input_file=input_file, dB=dB, sl_duration=sl_duration
    )
    non_silence_str: str = str(render_task(get_silence_segs_task))
    non_silence_segments, total_duration, _ = _extract_non_silence_segs_info(
        non_silence_str
    )

    adjusted_segments: Sequence[float] = _adjust_segments_to_keyframes(
        _ensure_minimum_segment_length(
            non_silence_segments, seg_min_duration, total_duration
        ),
        _probe_keyframe(input_file),
    )

    merged_overlapping_segments: Sequence[float] = _merge_overlapping_segments(
        adjusted_segments
    )
    if merged_overlapping_segments == []:
        logger.error(f"No valid segments found for {input_file}.")
        return error_code.NO_VALID_SEGMENTS

    try:
        advanced_keep_or_remove_by_split_segs(
            input_file=input_file,
            output_file=temp_output_file,
            video_segments=merged_overlapping_segments,
            odd_kwargs=odd_kwargs,
            even_kwargs=even_kwargs,
        )
        temp_output_file.replace(output_file)

    except Exception as e:
        logger.error(f"Failed to {task_descripton} for {input_file}. Error: {e}")
        raise e
        # return error_code.FAILED_TO_CUT
    return 0


# cut silence rerender
def _gen_cut_sl_filter(
    filter_text: Sequence[str],
    videoSectionTimings: Sequence[float],
) -> Generator[str, None, None]:
    yield filter_text[0]
    yield from (
        f"between(t,{videoSectionTimings[i]},{videoSectionTimings[i + 1]})"
        + ("+" if i != len(videoSectionTimings) - 2 else "")
        for i in range(0, len(videoSectionTimings), 2)
    )
    yield filter_text[1]


def _create_cut_sl_filter_tempfile(
    filter_info: Sequence[str],
    videoSectionTimings: Sequence[float],
) -> Path:
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", encoding="UTF-8", prefix=filter_info[2]
    ) as temp_file:
        for line in _gen_cut_sl_filter(filter_info, videoSectionTimings):
            temp_file.write(f"{line}\n")
        path: Path = Path(temp_file.name)
    return path


def _create_cut_sl_kwargs(input_file: Path | str, dB: int, sl_duration: float) -> dict:
    get_silence_segs_task: FF_Create_Render = FF_Create_Render_Task().get_silence_segs(
        input_file=input_file, dB=dB, sl_duration=sl_duration
    )
    non_silence_str: str = str(render_task(get_silence_segs_task))
    non_silence_segments: Sequence[float] = _extract_non_silence_segs_info(
        non_silence_str
    )[0]

    class CSFiltersInfo(Enum):
        VIDEO = [
            "select='",
            "', setpts=N/FRAME_RATE/TB",
            f"temp_{time.strftime('%Y%m%d-%H%M%S')}_video_filter_",
        ]
        AUDIO = [
            "aselect='",
            "', asetpts=N/SR/TB",
            f"temp_{time.strftime('%Y%m%d-%H%M%S')}_audio_filter_",
        ]

    video_filter_script: Path = _create_cut_sl_filter_tempfile(
        CSFiltersInfo.VIDEO.value, non_silence_segments
    )
    audio_filter_script: Path = _create_cut_sl_filter_tempfile(
        CSFiltersInfo.AUDIO.value, non_silence_segments
    )

    return {
        "filter_script:v": video_filter_script,
        "filter_script:a": audio_filter_script,
    }
