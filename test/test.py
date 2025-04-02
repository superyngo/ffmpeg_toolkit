import os
import ffmpeg_toolkit
from ffmpeg_toolkit import FF_TASKS, PARTIAL_TASKS
from pathlib import Path
from functools import partial
from typing import Iterable, Sequence, Optional, Callable, Literal, Mapping, TypedDict
from pydantic import BaseModel, Field, field_validator
import code

os.environ["PYTHONUTF8"] = "1"
# Get the absolute path of the current file
current_file_path = Path(__file__).resolve()
# Set the ./bin path to the PATH environment variable
bin_path = current_file_path.parent / "bin"
os.environ["PATH"] = a = str(bin_path) + os.pathsep + os.environ["PATH"]


input_file = Path(r"C:\Users\user\Downloads\2031-12-12_1954875155_merged.mkv")
# input_file = Path(r"C:\Users\user\Downloads\IMG_2078.mp4")
dir = Path(r"C:\Users\user\AppData\Local\Temp\tmpsn52hpxj")

jumpcut_args = {
    # "input_file": input_file,
    "b1_duration": 0.5,
    "b2_duration": 1.5,
    "b1_multiple": 3,
    "b2_multiple": 10.5,
}
portion_method3 = [
    (1, "remove"),
    (1, PARTIAL_TASKS.custom()),
    (
        2,
        PARTIAL_TASKS.cut_silence(
            even_further=PARTIAL_TASKS.cut_motionless(),
            odd_further=PARTIAL_TASKS.jumpcut(),
        ),
    ),
]

portion_method2 = [
    (
        1,
        PARTIAL_TASKS.cut_silence(
            even_further=PARTIAL_TASKS.cut_silence_rerender(),
            odd_further=PARTIAL_TASKS.jumpcut(),
        ),
    ),
    (4, PARTIAL_TASKS.speedup(multiple=5)),
    (
        4,
        PARTIAL_TASKS.partion_video(
            # portion_method=portion_method3,
            output_dir=input_file.parent / "can",
        ),
    ),
]

portion_method1 = [
    (1, PARTIAL_TASKS.cut_silence_rerender()),
    (4, PARTIAL_TASKS.speedup(multiple=5)),
    # (
    #     4,
    #     PARTIAL_TASKS.partion_video(
    #         portion_method=portion_method2,
    #         output_dir=input_file.parent / "can",
    #     ),
    # ),
]

# ffmpeg_toolkit.PartitionVideo(
#     input_file=input_file,
#     output_file=input_file.parent,
#     portion_method=portion_method1,
#     output_dir=input_file.parent / "can",
# ).render()

ffmpeg_toolkit.FF_TASKS.cut_motionless(
    input_file=input_file,
    output_file=input_file,
    even_further=PARTIAL_TASKS.cut_motionless(),
    odd_further=PARTIAL_TASKS.cut_silence(),
).render()
# code.interact(local=globals())
