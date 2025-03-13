from typing import TypedDict, NotRequired
from enum import StrEnum, auto


class EncodeKwargs(TypedDict):
    video_track_timescale: NotRequired[int]
    vcodec: NotRequired[str]
    video_bitrate: NotRequired[int]
    acodec: NotRequired[str]
    ar: NotRequired[int]
    f: NotRequired[str]


class VideoSuffix(StrEnum):
    MP4 = auto()
    MKV = auto()
    AVI = auto()
