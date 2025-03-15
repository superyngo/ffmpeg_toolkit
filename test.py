import os
import ffmpeg_toolkit
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
# Get the absolute path of the current file
current_file_path = Path(__file__).resolve()
# Set the ./bin path to the PATH environment variable
bin_path = current_file_path.parent / "bin"
os.environ["PATH"] = a = str(bin_path) + os.pathsep + os.environ["PATH"]


# import code

input_file = Path(r"F:\Users\user\Downloads\IMG_2078.mp4")
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

# ff_render_task: ffmpeg_converter.FF_Create_Render = (
#     ffmpeg_converter.FF_Create_Render_Task().get_silence_segs(input_file=file)
# )
# non_silence: str = str(ffmpeg_converter.render_task(ff_render_task))
# silence_info = ffmpeg_converter.extract_non_silence_segs_info(non_silence)


# ffmpeg_converter.advanced_keep_or_remove_by_cuts(file, None, silence_info[0])

# ffmpeg_converter.cut_silence_rerender(file)

# code.interact(local=globals())
# ffmpeg_converter.merge(dir)
# cut_silence_args = {
#     "input_file": file,
#     "odd_args": ffmpeg_converter._create_speedup_args(1),
#     "even_args": ffmpeg_converter._create_speedup_args(60),
# }

# ffmpeg_converter.cut_silence(**cut_silence_args)
# ff_render_task: ffmpeg_converter.FF_Create_Render = (
#     ffmpeg_converter.FF_Create_Render_Task().merge(dir, dir / "output.mp4")
# )
# ffmpeg_converter.render_task(ff_render_task)

# ff_render_task: FF_Create_Render = FF_Create_Render_Task().cut_silence_rerender(
#     input_file
# )
# render_task(ff_render_task)

ffmpeg_toolkit.cut_silence(
    input_file,
    even_kwargs=ffmpeg_toolkit._create_speedup_kwargs(60),
    odd_kwargs=ffmpeg_toolkit._create_jumpcut_kwargs(1.5, 3, 2, 1),
)
