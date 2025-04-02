from typing import TypedDict, NotRequired
from types import FunctionType
from functools import partial
from enum import StrEnum, auto, Enum
from pathlib import Path


class FunctionEnum(Enum):
    """
    An enum class that allows registering functions as enum values.
    Functions are stored as partial objects, allowing for preset arguments.
    """

    @classmethod
    def register(cls, name, func, *args, **kwargs):
        """
        Register a function as a new enum value.

        Args:
            name: The name of the new enum value
            func: The function to register
            *args: Positional arguments to be preset for the function
            **kwargs: Keyword arguments to be preset for the function

        Raises:
            TypeError: If the provided value is not a function
        """
        if not isinstance(func, FunctionType):
            raise TypeError("Only functions can be registered.")

        partial_func = partial(func, *args, **kwargs)
        new_enum = Enum(
            cls.__name__, {**{e.name: e.value for e in cls}, name: partial_func}
        )
        cls._member_map_.update(new_enum._member_map_)

    def __call__(self, *args, **kwargs):
        """
        Call the function stored in the enum value.

        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result of calling the stored function
        """
        # If the value is callable, invoke it to get a fresh instance
        return self.value


class ClassEnum(Enum):
    """
    An enum class that allows registering classes as enum values.
    Classes are stored as partial objects, allowing for preset constructor arguments.
    """

    @classmethod
    def register(cls, name, class_type, *args, **kwargs):
        """
        Register a class as a new enum value.

        Args:
            name: The name of the new enum value
            class_type: The class to register
            *args: Positional arguments to be preset for the class constructor
            **kwargs: Keyword arguments to be preset for the class constructor

        Raises:
            TypeError: If the provided value is not a class
        """
        if not isinstance(class_type, type):
            raise TypeError("Only classes can be registered.")

        partial_class = partial(class_type, *args, **kwargs)
        new_enum = Enum(
            cls.__name__, {**{e.name: e.value for e in cls}, name: partial_class}
        )
        cls._member_map_.update(new_enum._member_map_)

    def __call__(self, *args, **kwargs):
        """
        Create a new instance of the class stored in the enum value.

        Args:
            *args: Positional arguments to pass to the class constructor
            **kwargs: Keyword arguments to pass to the class constructor

        Returns:
            A new instance of the stored class
        """
        # If the value is callable, invoke it to create a new instance
        return self.value(*args, **kwargs)


class EncodeKwargs(TypedDict):
    """
    TypedDict defining the optional keyword arguments for video encoding operations.
    Contains various encoding parameters like codecs, bitrates, and format.
    """

    video_track_timescale: NotRequired[int]
    vcodec: NotRequired[str]
    video_bitrate: NotRequired[int]
    acodec: NotRequired[str]
    ar: NotRequired[int]
    f: NotRequired[str]


class VideoSuffix(StrEnum):
    """
    String enumeration of common video file suffixes/extensions.
    """

    MP4 = auto()
    MKV = auto()
    AVI = auto()


# Type aliases
type FFKwargsValue = str | Path | float | int
"""Type alias for possible values in FFmpeg keyword arguments."""

type FFKwargs = dict[str, FFKwargsValue] | dict[str, str]
"""Type alias for FFmpeg keyword arguments dictionary."""
