from typing import TypedDict, NotRequired
from types import FunctionType
from functools import partial
from enum import StrEnum, auto, Enum
from pathlib import Path


class FunctionEnum(Enum):
    @classmethod
    def register(cls, name, func, *args, **kwargs):
        if not isinstance(func, FunctionType):
            raise TypeError("Only functions can be registered.")

        partial_func = partial(func, *args, **kwargs)
        new_enum = Enum(
            cls.__name__, {**{e.name: e.value for e in cls}, name: partial_func}
        )
        cls._member_map_.update(new_enum._member_map_)

    def __call__(self, *args, **kwargs):
        # If the value is callable, invoke it to get a fresh instance
        return self.value


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


type FFKwargsValue = str | Path | float | int
type FFKwargs = dict[str, FFKwargsValue] | dict[str, str]
