# import ffmpeg
from typing import (
    Iterable,
    Sequence,
    TypedDict,
    Optional,
    Callable,
    NotRequired,
    Required,
)
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

from attr import asdict
from ffmpeg_types import EncodeKwargs, VideoSuffix
import functools
from dataclasses import dataclass, field


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


# basic
def _dic_to_ffmpeg_args(kwargs: dict | None = None) -> list:
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
def _ffmpeg(**kwargs) -> str:
    front_default_kwargs = {"hwaccel": "auto"}
    behind_default_kwargs = {"loglevel": "warning"}
    output_kwargs: dict = front_default_kwargs | kwargs | behind_default_kwargs
    logger.info(f"Executing FFmpeg with {output_kwargs = }")
    command = ["ffmpeg"] + _dic_to_ffmpeg_args(output_kwargs)
    result = subprocess.run(
        command, capture_output=True, text=True, check=True, encoding="utf-8"
    )
    return result.stdout + result.stderr


def _ffprobe(**kwargs):
    default_kwargs = {}
    output_kwargs: dict = kwargs | default_kwargs
    logger.info(f"Executing ffprobe with {output_kwargs = }")
    command = ["ffprobe"] + _dic_to_ffmpeg_args(output_kwargs)
    subprocess.run(command, check=True, encoding="utf-8")


# probe
def probe_is_valid_video(input_file: Path, **othertags) -> bool:  # command
    """Function to check if a video file is valid using ffprobe."""
    output_kwargs: dict = {
        "v": "error",
        "show_entries": "format=duration",
        "of": "default=noprint_wrappers=1:nokey=1",
        "i": input_file,
    } | othertags
    logger.info(f"Validating {input_file.name} with {output_kwargs = }")
    try:
        command = ["ffprobe"] + _dic_to_ffmpeg_args(output_kwargs)
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


def probe_duration(input_file: Path, **othertags) -> float:  # command
    output_kwargs: dict = {
        "v": "error",
        "show_entries": "format=duration",
        "of": "default=noprint_wrappers=1:nokey=1",
        "i": input_file,
    } | othertags
    logger.info(f"Probing {input_file.name} duration with {output_kwargs = }")
    command = ["ffprobe"] + _dic_to_ffmpeg_args(output_kwargs)
    result = subprocess.run(
        command, capture_output=True, text=True, check=True, encoding="utf-8"
    )
    probe_duration = result.stdout.strip()
    logger.info(f"{input_file.name} duration probed: {probe_duration}")

    return float(probe_duration or 0)


def probe_encoding(input_file: Path, **othertags) -> EncodeKwargs:  # command
    output_kwargs: dict = {
        "v": "error",
        "print_format": "json",
        "show_format": "",
        "show_streams": "",
        "i": input_file,
    } | othertags
    logger.info(f"Probing {input_file.name} encoding with {output_kwargs = }")
    # Probe the video file to get metadata
    command = ["ffprobe"] + _dic_to_ffmpeg_args(output_kwargs)
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


def _probe_keyframe(input_file: Path, **othertags) -> list[float]:  # command
    output_kwargs: dict = {
        "v": "error",
        "select_streams": "v:0",
        "show_entries": "packet=pts_time,flags",
        "of": "json",
        "i": input_file,
    } | othertags
    logger.info(f"Getting keyframe for {input_file.name} with {output_kwargs = }")
    command = ["ffprobe"] + _dic_to_ffmpeg_args(output_kwargs)
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
type T_Othertags = dict


@dataclass(frozen=True)
class T_Cut_Output_args:
    start_time: str = "00:00:00"
    end_time: str = "00:00:01"
    ca: str = "copy"
    cv: str = "copy"


@dataclass(frozen=True)
class T_FF_Args_In:
    input_file: Path
    output_file: Optional[Path] = None
    input_tags: T_Othertags = field(default_factory=dict)
    output_tags: T_Othertags | T_Cut_Output_args = field(default_factory=dict)


type T_FF_Args_Out = dict[str, Path | str | float]


@dataclass(frozen=True)
class T_FF_Render_Args(T_FF_Args_In):
    task_descripton: Optional[_tasks | str] = "rendered"


def _create_ff_args(
    input_file: Path,
    output_file: Path,
    input_tags: T_Othertags,
    output_tags: T_Othertags,
) -> T_FF_Args_Out:
    """Create args for converting by ffmpeg

    Args:
        input_file (Path): _description_
        output_file (Path): _description_

    Returns:
        Mapping[str, str | Path]: _description_
    """
    if output_file == Path():
        output_file_tags: dict = {}
    else:
        output_file_tags = {"y": output_file}
    input_tags = asdict(input_tags)
    output_tags = asdict(output_tags)
    return input_tags | {"i": input_file} | output_tags | output_file_tags


def ff_render(
    input_file: Path,
    output_file: Optional[Path],
    task_descripton: _tasks | str,
    input_tags: T_Othertags,
    output_tags: T_Othertags,
) -> str:  # command
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

    # Handle temp output file path
    if output_file == Path():
        temp_output_file: Path = output_file
    else:
        temp_output_file: Path = output_file.parent / (
            output_file.stem + "_processing" + output_file.suffix
        )
    ff_args_in: T_FF_Args_In = T_FF_Args_In(
        input_file=input_file,
        output_file=temp_output_file,
        input_tags=input_tags,
        output_tags=output_tags,
    )

    ff_args_out: T_FF_Args_Out = _create_ff_args(**asdict(ff_args_in))

    logger.info(
        f"{task_descripton.capitalize()} {input_file.name} to {output_file.name} with {ff_args_out}"
    )

    try:
        result: str = _ffmpeg(**ff_args_out)
        if temp_output_file != output_file:
            temp_output_file.replace(output_file)
        return result
    except Exception as e:
        logger.error(
            f"Failed to do {task_descripton} videos for {input_file}. Error: {e}"
        )
        raise e


# ff_render: cut
def _create_cut_args(start_time: str, end_time: str) -> dict[str, str]:
    return {
        "ss": start_time,
        "to": end_time,
    }


@dataclass(frozen=True)
class T_Cut_Args(T_FF_Render_Args):
    task_descripton = _tasks.CUT
    output_tags: T_Cut_Output_args = field(default_factory=T_Cut_Output_args)


def cut(  # command
    input_file: Path | str,
    output_file: Path | None = None,
    start_time: str = "00:00:00",
    end_time: str = "00:00:01",
    rerender: bool = False,
    **othertags: EncodeKwargs,
) -> int:
    """Cut a video file using ffmpeg-python.

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """

    # init task
    task_descripton = f"{_tasks.CUT}_from_{_convert_timestamp_to_seconds(start_time)}_to_{_convert_timestamp_to_seconds(end_time)}"
    input_file = Path(input_file)

    ff_render_args: T_Cut_Args = T_Cut_Args(
        input_file=input_file,
        output_file=output_file,
        task_descripton=task_descripton,
        output_tags=T_Cut_Output_args(start_time, end_time),
    )

    try:
        ff_render(**asdict(ff_render_args))
    except Exception as e:
        logger.error(f"Failed to {task_descripton} videos for {input_file}. Error: {e}")
        raise e
    return 0


# ff_render: sppedup
def _create_force_keyframes_args(keyframe_times: int = 2) -> dict[str, str]:
    """_summary_

    Args:
        keyframe_times (int, optional): _description_. Defaults to 2.

    Returns:
        dict[str, str]: _description_
    """
    return {"force_key_frames": f"expr:gte(t,n_forced*{keyframe_times})"}


def _create_speedup_args(multiple: float) -> dict[str, str]:
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
        }
        | _create_force_keyframes_args()
    )


def speedup(  # command
    input_file: Path | str,
    output_file: Optional[Path] = None,
    multiple: float | int = 2,
    **othertags: T_Othertags,
) -> int:
    """_summary_

    Args:
        output_file (Path | None, optional): _description_. Defaults to None.
        multiple (float | int, optional): _description_. Defaults to 2.

    Raises:
        e: _description_

    Returns:
        int: _description_
    """
    # init task
    task_descripton = f"{_tasks.SPEEDUP}_by_{multiple}"
    input_file = Path(input_file)

    # error handling
    if multiple <= 0:
        logger.error("Speedup factor must be greater than 0.")
        return 1

    if multiple == 1:
        if input_file != output_file and output_file is not None:
            input_file.replace(output_file)
        logger.error("Speedup multiple 1, only replace target file")
        return 0

    output_kwargs: T_FF_Render_Args = {
        "input_file": input_file,
        "output_file": output_file,
        "task_name": task_descripton,
        "othertags": _create_speedup_args(multiple) | othertags,
    }

    try:
        ff_render(**output_kwargs)
    except Exception as e:
        logger.error(
            f"Failed to {task_descripton} videos for {input_file}. Error: {str(e)}"
        )
        raise e
    return 0


# ff_render: jumpcut
def _create_jumpcut_args(
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
        }
        | _create_force_keyframes_args()
    )

    return args


def jumpcut(  # command
    input_file: Path | str,
    output_file: Path | None = None,
    b1_duration: float = 5,
    b2_duration: float = 5,
    b1_multiple: float = 1,  # 0 means unwanted cut out
    b2_multiple: float = 0,  # 0 means unwanted cut out
    **othertags: T_Othertags,
) -> int:
    """_summary_

    Args:
        output_file (Path | None): _description_
        b1_duration (float): _description_
        b2_duration (float): _description_
        b1_multiple (float, optional): _description_. Defaults to 0.

    Returns:
        int: _description_
    """
    # error handling
    if any((b1_duration <= 0, b2_duration <= 0)):
        logger.error("Both 'interval' and 'lasting' must be greater than 0.")
        return 1

    if any((b1_multiple < 0, b2_multiple < 0)):
        logger.error(
            "Both 'interval_multiple' and 'lasting_multiple' must be greater or equal to 0."
        )
        return 2

    # init task
    task_descripton = f"{_tasks.JUMPCUT}_b1({b1_duration}x{b1_multiple})_b2({b2_duration}x{b2_multiple})"
    input_file = Path(input_file)

    output_kwargs: T_FF_Render_Args = {
        "input_file": input_file,
        "output_file": output_file,
        "task_name": task_descripton,
        "othertags": _create_jumpcut_args(
            b1_duration, b2_duration, b1_multiple, b2_multiple
        )
        | othertags,
    }

    try:
        ff_render(**output_kwargs)
    except Exception as e:
        logger.error(
            f"Failed to {task_descripton} videos for {input_file}. Error: {str(e)}"
        )
        return 2
    return 0


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
            video
            for video in video_files_source.glob("*")
            if video.suffix.lstrip(".") in VideoSuffix
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


def merge(
    input_dir_or_files: Path | str | list[Path] | list[str],
    output_file: Path | None = None,
    **othertags,
) -> int:  # command
    # init task
    task_descripton = f"{_tasks.MERGE}"
    if isinstance(input_dir_or_files, Iterable):
        input_dir_or_files = [Path(video) for video in input_dir_or_files]
    else:
        input_dir_or_files = Path(input_dir_or_files)
    input_txt: Path = create_merge_txt(input_dir_or_files)

    output_kwargs: T_FF_Render_Args = {
        "input_file": input_txt,
        "output_file": output_file,
        "task_name": task_descripton,
        "input_tags": {
            "f": "concat",
            "safe": 0,
        },
        "othertags": {
            "c:a": "copy",
            "c:v": "copy",
        }
        | othertags,
    }

    try:
        ff_render(**output_kwargs)
        os.remove(input_txt)
        if not any(input_txt.parent.iterdir()):
            os.rmdir(input_txt.parent)
            print(f"{input_txt.parent} removed")
        return 0
    except Exception as e:
        logger.error(f"Failed merging {input_txt}. Error: {str(e)}")
        raise e


# keep or remove copy/rendering
def advanced_keep_or_remove_by_cuts(
    input_file: Path,
    output_file: Path | None,
    video_segments: Sequence[str] | Sequence[float],
    odd_args: None | dict[str, str | float],  # For segments, None means remove
    even_args: None | dict[str, str | float] = None,  # For other segments
) -> int:
    task_descripton = _tasks.KEEP_OR_REMOVE

    if output_file is None:
        output_file = input_file.parent / (
            input_file.stem + "_" + task_descripton + input_file.suffix
        )
    temp_output_file: Path = output_file.parent / (
        output_file.stem + "_processing" + output_file.suffix
    )
    logger.info(
        f"{task_descripton.capitalize()} {input_file.name} to {output_file.name} with {odd_args = } ,{even_args = }."
    )

    # Step 1:ff_render video segments if needed and double them
    video_segments = deque(
        _convert_seconds_to_timestamp(s) if isinstance(s, (float, int)) else s
        for o in video_segments
        for s in (o, o)
    )  # type: ignore

    # Step 2: create a full segment list
    video_segments.appendleft("00:00:00.000")  # type: ignore
    video_segments.append(_convert_seconds_to_timestamp(probe_duration(input_file)))  # type: ignore

    # Use ThreadPoolExecutor to manage the threads
    temp_dir: Path = Path(tempfile.mkdtemp())
    num_cores = os.cpu_count()
    cut_videos = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for i in range(0, len(video_segments), 2):
            if even_args is None and i % 4 == 0:
                continue
            if odd_args is None and i % 4 == 2:
                continue
            start_time: str = video_segments[i]  # type:ignore
            end_time: str = video_segments[i + 1]  # type:ignore
            if start_time[:8] == end_time[:8]:
                continue
            seg_output_file = temp_dir / f"{i}{input_file.suffix}"
            cut_videos.append(seg_output_file)

            # Submit the cut task to the executor
            future = executor.submit(
                cut, input_file, seg_output_file, start_time, end_time
            )
            futures.append(future)  # Store the future for tracking
            future.result()  # Ensures `cut` completes before proceeding

            further_args = even_args if i % 4 == 0 else odd_args

            if not further_args:
                continue

            output_kwargs: T_FF_Render_Args = {
                "input_file": seg_output_file,
                "output_file": seg_output_file,
                "othertags": further_args,
            }

            # Submit further render task to the executor
            future = executor.submit(ff_render, **output_kwargs)
            futures.append(future)  # Store the future for tracking
            future.result()

        # Optionally, wait for all futures to complete
        # concurrent.futures.wait(futures)

    # Step 4: Sort the cut video paths by filename (index order)
    cut_videos.sort(key=lambda video_file: int(video_file.stem))

    # Step 4: Merge the kept segments
    try:
        merge(cut_videos, temp_output_file)
        temp_output_file.replace(output_file)
        # Step 5: Clean up temporary files
        for video_path in cut_videos:
            os.remove(video_path)
        os.rmdir(temp_dir)
    except Exception as e:
        logger.error(f"Failed to {task_descripton} for {input_file}. Error: {e}")
        return 1
    return 0


#  Get non silence segments
def get_non_silence_segs(  # command
    input_file: Path, dB: int = -21, sl_duration: float = 0.2, **othertags
) -> tuple[Sequence[float], float, float]:
    """_summary_

    Args:
        input_file (Path): _description_
        dB (int, optional): _description_. Defaults to -35.
        sl_duration (float, optional): _description_. Defaults to 1.

    Returns:
        tuple[Sequence[float], float, float]: (non_silence_segs, total_duration, total_silence_duration)
    """
    # init task
    task_descripton = f"{_tasks.GET_NON_SILENCE_SEGS}_by_{dB}"
    input_file = Path(input_file)

    output_kwargs: T_FF_Render_Args = {
        "input_file": input_file,
        "output_file": Path(""),
        "task_name": task_descripton,
        "othertags": {
            "af": f"silencedetect=n={dB}dB:d={sl_duration}",
            "c:v": "copy",
            "f": "null",
        }
        | othertags
        | {"": ""},
    }

    try:
        non_silence_segs_str: str = ff_render(**output_kwargs)
    except Exception as e:
        raise Exception(
            f"Failed to {task_descripton} videos for {input_file}. Error: {str(e)}"
        )

    # output_kwargs1: dict = (
    #     {
    #         "hwaccel": "auto",
    #         "i": input_file,
    #         "af": f"silencedetect=n={dB}dB:d={sl_duration}",
    #         "c:v": "copy",
    #         "f": "null",
    #     }
    #     | othertags
    #     | {"": ""}
    # )

    # logger.info(
    #     f"Detecting silences of {input_file.name} by {dB = } and {output_kwargs = }"
    # )

    # command = ["ffmpeg"] + _dic_to_ffmpeg_args(output_kwargs)
    # result = subprocess.run(
    #     command, capture_output=True, text=True, check=True, encoding="utf-8"
    # )
    # output = result.stdout + result.stderr

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


# ff_render: cut silenct copy
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
                else time
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


def cut_silence(
    input_file: Path | str,
    output_file: Path | None = None,
    dB: int = -21,
    sl_duration: float = 0.2,
    seg_min_duration: float = 0,
    odd_args: dict[str, str | float] = {},
    even_args: dict[str, str | float] | None = None,
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
    temp_output_file: Path = output_file.parent / (
        output_file.stem + "_processing" + output_file.suffix
    )
    logger.info(
        f"{task_descripton.capitalize()} {input_file} to {output_file} with {dB = } ,{sl_duration = }, {seg_min_duration = }."
    )

    non_silence_segments: Sequence[float]
    total_duration: float
    non_silence_segments, total_duration, _ = get_non_silence_segs(
        input_file, dB, sl_duration
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
        advanced_keep_or_remove_by_cuts(
            input_file=input_file,
            output_file=temp_output_file,
            video_segments=merged_overlapping_segments,
            odd_args=odd_args,
            even_args=even_args,
        )
        temp_output_file.replace(output_file)

    except Exception as e:
        logger.error(f"Failed to {task_descripton} for {input_file}. Error: {e}")
        return error_code.FAILED_TO_CUT
    return 0


# keep or remove copy
def _split_segments_cut(
    input_file: Path,
    video_segments: Sequence[float] | Sequence[str],
    output_dir: Path | None = None,
    **othertags,
) -> int:
    """_summary_

    Args:
        input_file (Path): _description_
        video_segments (Sequence[float] | Sequence[str]): _description_
        output_dir (Path | None, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        tuple[Sequence[Path], Path]: (cut_videos, input_txt_path)
    """

    # Step 1: Validate input
    if len(video_segments) % 2 != 0:
        raise ValueError(
            "video_segments must contain an even number of elements (start and end times)."
        )

    # Step 2: Create a temporary folder for storing cut videos
    if output_dir is None:
        output_dir = input_file.parent / f"{input_file.stem}_segments"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: Use threading to process video segments
    # Get the number of CPU cores
    num_cores = os.cpu_count()

    # Use ThreadPoolExecutor to manage the threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for i in range(0, len(video_segments), 2):
            if isinstance(video_segments[i], float):
                start_time = _convert_seconds_to_timestamp(
                    video_segments[i]  # type:ignore
                )
                end_time = _convert_seconds_to_timestamp(
                    video_segments[i + 1]  # type:ignore
                )
            else:
                start_time: str = video_segments[i]  # type:ignore
                end_time: str = video_segments[i + 1]  # type:ignore
            if start_time == end_time:
                continue
            output_file: Path = output_dir / f"{i // 2}{input_file.suffix}"

            # Submit the cut task to the executor
            future = executor.submit(
                cut, input_file, output_file, start_time, end_time, **othertags
            )
            futures.append(future)

        # Optionally, wait for all futures to complete
        concurrent.futures.wait(futures)

    return 0


def keep_or_remove_by_cuts(
    input_file: Path,
    output_file: Path | None,
    video_segments: Sequence[str] | Sequence[float],
    keep_handle: bool = True,  # True means keep, False means remove
):
    if output_file is None:
        output_file = input_file.parent / (
            input_file.stem + "_" + _tasks.KEEP_OR_REMOVE + input_file.suffix
        )
    temp_output_file: Path = output_file.parent / (
        output_file.stem + "_processing" + output_file.suffix
    )
    logger.info(f"{_tasks.KEEP_OR_REMOVE} {input_file.name} to {output_file.name}")

    # Step 0: Create a temporary folder for storing cut videos
    temp_dir: Path = Path(tempfile.mkdtemp())

    # Step 1:ff_render video segments if needed
    video_segments = deque(
        _convert_timestamp_to_seconds(s) if isinstance(s, str) else s
        for s in video_segments
    )

    # Step 2: rearrange video_segments if keep_handle == False
    if not keep_handle:
        video_segments.appendleft(0.0)
        video_segments.append(probe_duration(input_file))

    # Step 3: Cut the video into segments based on the provided start and end times
    _do: int = _split_segments_cut(
        input_file,
        video_segments,
        temp_dir,
    )

    # Step 4: Merge the kept segments
    try:
        merge(temp_dir, temp_output_file)
        temp_output_file.replace(output_file)
        # Step 5: Clean up temporary files
        for video in temp_dir.iterdir():
            os.remove(video)
        os.rmdir(temp_dir)
    except Exception as e:
        logger.error(f"Failed to {_tasks.KEEP_OR_REMOVE} for {input_file}. Error: {e}")
        return 1
    return 0


# cut silence rerendering
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


def _create_cut_sl_args(input_file: Path, dB: int, sl_duration: float) -> dict:
    non_silence_segments: Sequence[float] = get_non_silence_segs(
        input_file, dB, sl_duration
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


def cut_silence_rerender(  # command
    input_file: Path | str,
    output_file: Path | None = None,
    dB: int = -21,
    sl_duration: float = 0.2,
    **othertags,
) -> int:
    # error handling
    if sl_duration <= 0:
        logger.error("Duration must be greater than 0.")
        return 1

    # init task
    task_descripton = f"{_tasks.CUT_SILENCE_RERENDER}_by_{dB}"
    input_file = Path(input_file)

    othertags = _create_cut_sl_args(input_file, dB, sl_duration) | othertags
    output_kwargs: T_FF_Render_Args = {
        "input_file": input_file,
        "output_file": output_file,
        "task_name": task_descripton,
        "othertags": othertags,
    }

    try:
        ff_render(**output_kwargs)
        os.remove(othertags["filter_script:v"])
        os.remove(othertags["filter_script:a"])
    except Exception as e:
        logger.error(f"Failed to cut silence for {input_file}. Error: {e}")
        return 2
    return 0


# not useful
def _split_segments(  # command
    input_file: Path,
    video_segments: Sequence[float],
    output_dir: Path | None = None,
    **othertags,
) -> Sequence[Path]:
    """
    Cuts the input video into segments based on the provided start and end times.
    """
    if output_dir is None:
        output_dir = input_file.parent / f"{input_file.stem}_segments"

    # Step 2: Use ffmpeg to cut the video into segments
    output_kwargs: dict = (
        {
            "i": input_file,
            "c:v": "copy",
            "c:a": "copy",
            "f": "segment",
            "segment_times": ",".join(map(str, video_segments)),
            "segment_format": input_file.suffix.lstrip("."),
            "reset_timestamps": "1",
        }
        | othertags
        | {"y": f"{output_dir}/%d_{input_file.stem}{input_file.suffix}"}
    )
    logger.info(f"Split {input_file.name} to {output_dir} with {output_kwargs = }")
    try:
        # Execute the FFmpeg command
        _ffmpeg(**output_kwargs)
    except Exception as e:
        logger.error(f"Failed to cut videos for {input_file}. Error: {e}")
        raise e

    # Step 3: Collect the cut video segments
    cut_videos: list[Path] = sorted(
        output_dir.glob(f"*{input_file.suffix}"),
        key=lambda video_file: int(video_file.stem.split("_")[0]),
    )

    return cut_videos


def keep_or_remove_by_split_segs(
    input_file: Path,
    output_file: Path | None,
    video_segments: Sequence[str] | Sequence[float],
    keep_handle: bool = True,  # True means keep, False means remove
) -> int:
    """remove a segment from a video

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    if output_file is None:
        output_file = input_file.parent / (
            input_file.stem + "_" + _tasks.KEEP_OR_REMOVE + input_file.suffix
        )
    temp_output_file: Path = output_file.parent / (
        output_file.stem + "_processing" + output_file.suffix
    )
    logger.info(f"{_tasks.KEEP_OR_REMOVE} {input_file.name} to {output_file.name}")

    # Step 0: Create a temporary folder for storing cut videos
    temp_dir: Path = Path(tempfile.mkdtemp())

    # Step 1: Cut the video into segments based on the provided start and end times
    cut_videos: Sequence[Path] = _split_segments(
        input_file,
        [
            _convert_timestamp_to_seconds(s) if isinstance(s, str) else s
            for s in video_segments
        ],
        temp_dir,
    )

    # Step 2: Sort the cut videos into two lists(0 and 1) based on index % 2
    cut_videos_dict: dict[int, list[Path]] = {}
    for index, path in enumerate(cut_videos):
        cut_videos_dict.setdefault(index % 2, []).append(path)

    # Step 3: Decide which segments to keep and which to remove
    keep_key: int = int(keep_handle)
    remove_key: int = abs(1 - keep_key)

    # Step 4: Remove the unwanted segments
    for video_path in cut_videos_dict[remove_key]:
        os.remove(video_path)

    # Step 5: Create input.txt for FFmpeg concatenation
    temp_dir = cut_videos[0].parent
    input_txt_path: Path = temp_dir / "input.txt"
    with open(input_txt_path, "w") as f:
        for video_path in cut_videos_dict[keep_key]:
            f.write(f"file '{video_path}'\n")

    # Step 6: Merge the kept segments
    try:
        merge(input_txt_path, temp_output_file)
        temp_output_file.replace(output_file)
        # Step 7: Clean up temporary files
        for video_path in cut_videos_dict[keep_key]:
            os.remove(video_path)
        os.remove(input_txt_path)
        os.rmdir(temp_dir)
    except Exception as e:
        logger.error(f"Failed to cut silence for {input_file}. Error: {e}")
        return 1
    return 0
