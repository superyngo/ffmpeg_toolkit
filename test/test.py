import os
import ffmpeg_toolkit
from ffmpeg_toolkit import FPRenderTasks
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


input_file = Path(r"C:\Users\user\Downloads\IMG_2180.mp4")
# input_file = Path(r"C:\Users\user\Downloads\IMG_2078.mp4")
dir = Path(r"C:\Users\user\AppData\Local\Temp\tmpsn52hpxj")

jumpcut_args = {
    # "input_file": input_file,
    "b1_duration": 0.5,
    "b2_duration": 1.5,
    "b1_multiple": 3,
    "b2_multiple": 10.5,
}
# ffmpeg_toolkit.FFRenderTasks().jumpcut(**jumpcut_args).render()
# ffmpeg_toolkit.FFRenderTasks().speedup(input_file=input_file, multiple=5).render()


# ffmpeg_toolkit.FFRenderTasks().cut_silence_rerender(input_file).render()
# ffmpeg_toolkit.cut_silence(input_file)
# ffmpeg_toolkit.probe_is_valid_video(input_file)
# ffmpeg_toolkit.FPRenderTasks().is_valid_video(input_file).render()
# cut_motionless_config = {
#     "input_file": input_file,
#     "threshold": 0.005,
#     "odd_further": ffmpeg_toolkit.PARTIAL_TASKS.custom(),
# # }
# # # # ffmpeg_toolkit.cut_motionless(**cut_motionless_config)

partition_config = ffmpeg_toolkit.PartitionConfig(
    portion_method=[
        (1, ffmpeg_toolkit.PARTIAL_TASKS.custom()),
        (1, ffmpeg_toolkit.PARTIAL_TASKS.jumpcut(**jumpcut_args)),
        (1, ffmpeg_toolkit.PARTIAL_TASKS.cut_silence()),
        (1, ffmpeg_toolkit.PARTIAL_TASKS.cut_motionless()),
        (1, ffmpeg_toolkit.PARTIAL_TASKS.speedup(multiple=2)),
    ]
)
# ffmpeg_toolkit.cut_silence(
#     input_file,
#     odd_further=ffmpeg_toolkit.PARTIAL_TASKS.speedup(),
#     even_further=ffmpeg_toolkit.PARTIAL_TASKS.jumpcut(**jumpcut_args),
# )


ffmpeg_toolkit.partion_video(
    input_file,
    partition_config,
    output_file=input_file.parent,
)


# code.interact(local=globals())
