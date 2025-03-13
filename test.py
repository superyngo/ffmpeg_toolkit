import os

import ffmpeg_converter

os.environ["PYTHONUTF8"] = "1"

# import code
from pathlib import Path

file = r"F:\Users\user\Downloads\IMG_2078.mp4"
dir = Path(r"F:\Users\user\Downloads\新增資料夾")
# multiple = 4.1
# ffmpeg_converter.speedup(file, multiple=multiple)
jumpcut_args = {
    # "input_file": file,
    "b1_duration": 0.5,
    "b2_duration": 1.5,
    "b1_multiple": 3,
    "b2_multiple": 10.5,
}
# ffmpeg_converter.jumpcut(**jumpcut_args)

# ffmpeg_converter.cut(file)

# ffmpeg_converter.cut_silence_rerender(file)

# code.interact(local=globals())
# ffmpeg_converter.merge(dir)
cut_silence_args = {
    "input_file": file,
    "odd_args": ffmpeg_converter._create_speedup_args(1),
    "even_args": ffmpeg_converter._create_speedup_args(60),
}

ffmpeg_converter.cut_silence(**cut_silence_args)
