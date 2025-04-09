# Standard library imports
from enum import Enum, StrEnum, auto
from functools import partial
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Literal, NotRequired, TypedDict


class FunctionEnum(Enum):
    """
    An enum class that allows registering functions as enum values.

    Functions are stored as partial objects, allowing for preset arguments.
    This enables defining a set of related operations as enum members that can be
    called directly.
    """

    @classmethod
    def register(cls, name, func, *args, **kwargs):
        """
        Register a function as a new enum value.

        Args:
            name (str): The name of the new enum value to be registered
            func (FunctionType): The function to register as the enum value
            *args: Positional arguments to be preset for the function
            **kwargs: Keyword arguments to be preset for the function

        Raises:
            TypeError: If the provided value is not a function

        Returns:
            None: Updates the enum class in-place
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

        This method makes the enum instance callable, forwarding any
        arguments to the underlying stored function.

        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Any: The result of calling the stored function
        """
        return self.value(*args, **kwargs)


class ClassEnum(Enum):
    """
    An enum class that allows registering classes as enum values.

    Classes are stored as partial objects, allowing for preset constructor arguments.
    This enables defining a set of related class types as enum members that can be
    instantiated directly.
    """

    @classmethod
    def register(cls, name, class_type, *args, **kwargs):
        """
        Register a class as a new enum value.

        Args:
            name (str): The name of the new enum value to be registered
            class_type (type): The class to register as the enum value
            *args: Positional arguments to be preset for the class constructor
            **kwargs: Keyword arguments to be preset for the class constructor

        Raises:
            TypeError: If the provided value is not a class

        Returns:
            None: Updates the enum class in-place
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

        This method makes the enum instance callable, forwarding any
        arguments to the underlying stored class constructor.

        Args:
            *args: Positional arguments to pass to the class constructor
            **kwargs: Keyword arguments to pass to the class constructor

        Returns:
            Any: A new instance of the stored class
        """
        return self.value(*args, **kwargs)


class EncodeKwargs(TypedDict):
    """
    TypedDict defining the optional keyword arguments for video encoding operations.

    Contains various encoding parameters like codecs, bitrates, and format settings
    that can be passed to FFmpeg encoding operations.

    Attributes:
        video_track_timescale (int, optional): Timescale for video tracks in Hz
        vcodec (str, optional): Video codec to use for encoding (e.g., 'h264', 'vp9')
        video_bitrate (int, optional): Target bitrate for video in bits per second
        acodec (str, optional): Audio codec to use for encoding (e.g., 'aac', 'mp3')
        ar (int, optional): Audio sample rate in Hz
        f (str, optional): Output format specification (e.g., 'mp4', 'mkv')
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

    These values can be used for type checking and validation of video file extensions.
    The enum uses auto() to automatically assign values that match the member names.
    """

    MP4 = auto()
    MKV = auto()
    AVI = auto()


type ValidExtensions = set[VideoSuffix] | set[str]
"""
Type representing valid file extensions for video operations.

Can be either a set of VideoSuffix enum values or a set of string extensions.
Used for validating input and output file types.
"""


class FFRenderException(TypedDict):
    """
    Type definition for render exceptions in FFmpeg operations.

    Provides a structured way to handle and report errors during rendering,
    with an optional hook for custom error handling.

    Attributes:
        code (int): Error code for the exception
        message (str): Human-readable error message describing the issue
        hook (Callable[[], Any], optional): Optional callback function to execute when this exception occurs
    """

    code: int
    message: str
    hook: NotRequired[Callable[[], Any]]


class OptionFFRender(TypedDict):
    """
    Type definition for optional render parameters in FFmpeg operations.

    Provides configuration options for rendering tasks, including exception handling,
    post-processing hooks, and file management options.

    Attributes:
        task_descripton (str, optional): Human-readable description of the rendering task
        delete_after (bool, optional): Whether to delete the input file after processing
        exception (FFRenderException, optional): Exception information configuration, if any
        post_hook (Callable[..., Any], optional): Function to process results after command execution
    """

    task_descripton: NotRequired[str]
    delete_after: NotRequired[bool]
    exception: NotRequired[FFRenderException]
    post_hook: NotRequired[Callable[..., Any]]


# Type aliases
type FFKwargsValue = str | Path | float | int
"""
Type alias for possible values in FFmpeg keyword arguments.

Represents the valid data types that can be used as values in FFmpeg
command parameters, including strings, paths, and numeric values.
"""

type FFKwargs = dict[str, FFKwargsValue] | dict[str, str]
"""
Type alias for FFmpeg keyword arguments dictionary.

Represents a dictionary of parameter names to their values, which can be
either a general FFKwargsValue type or specifically a string value.
"""

type PARTIAL_TASK = Callable[..., Any]
"""
Type alias for a callable that can be partially applied.

Represents a function or callable object that can be invoked with
arbitrary arguments and returns any type of value.
"""

type FurtherMethod = PARTIAL_TASK | Literal["remove"] | None
"""
Type alias for methods that can be applied to elements after processing.

Can be a callable function, the string literal "remove" to indicate
removal of the element, or None to indicate no further processing.
"""

type PortionMethod = (
    list[int]
    | list[tuple[int, None]]
    | list[tuple[int, PARTIAL_TASK]]
    | list[tuple[int, Literal["remove"]]]
    | list[
        int
        | tuple[int, None]
        | tuple[int, PARTIAL_TASK]
        | tuple[int, Literal["remove"]]
    ]
)
"""
Type alias for specifying how to handle portions of data.

Defines various ways to identify and process specific portions:
- Simple list of indices
- List of tuples pairing indices with None (no action)
- List of tuples pairing indices with callables (process with function)
- List of tuples pairing indices with "remove" (remove these portions)
- Mixed list containing any of the above forms
"""
