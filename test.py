import os
import ffmpeg_converter
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
# Get the absolute path of the current file
current_file_path = Path(__file__).resolve()
# Set the ./bin path to the PATH environment variable
bin_path = current_file_path.parent / "bin"
os.environ["PATH"] = a = str(bin_path) + os.pathsep + os.environ["PATH"]


# import code

file = r"C:\Users\user\Downloads\IMG_2078.mp4"
dir = Path(r"F:\Users\user\Downloads\新增資料夾")
# multiple = 4.1
# ffmpeg_converter.speedup(file, multiple=multiple)
# jumpcut_args = {
#     # "input_file": file,
#     "b1_duration": 0.5,
#     "b2_duration": 1.5,
#     "b1_multiple": 3,
#     "b2_multiple": 10.5,
# }
# ffmpeg_converter.jumpcut(**jumpcut_args)

# ff_cut_kwargs_In: ffmpeg_converter.FF_Render_In = (
#     ffmpeg_converter.FF_Render_Tasks().cut(input_file=file, to="00:00:10")
# )
# ffmpeg_converter.cut(ff_cut_kwargs_In)

ff_kwargs: ffmpeg_converter.FF_Render_In = ffmpeg_converter.FF_Render_Tasks().jumpcut(
    input_file=file, b1_duration=0.5
)
ffmpeg_converter.render(ff_kwargs)

# ffmpeg_converter.cut_silence_rerender(file)

# code.interact(local=globals())
# ffmpeg_converter.merge(dir)
# cut_silence_args = {
#     "input_file": file,
#     "odd_args": ffmpeg_converter._create_speedup_args(1),
#     "even_args": ffmpeg_converter._create_speedup_args(60),
# }

# ffmpeg_converter.cut_silence(**cut_silence_args)
