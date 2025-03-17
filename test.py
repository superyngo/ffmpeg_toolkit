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


input_file = Path(r"C:\Users\user\Downloads\2025-02-17_1739743880_merged.mp4")
# input_file = Path(r"C:\Users\user\Downloads\IMG_2078.mp4")
dir = Path(r"F:\Users\user\Downloads\新增資料夾")

# jumpcut_args = {
#     "input_file": input_file,
#     "b1_duration": 0.5,
#     "b2_duration": 1.5,
#     "b1_multiple": 3,
#     "b2_multiple": 10.5,
# }
# ff_render_task = ffmpeg_toolkit.FF_Create_Render_Task().jumpcut(**jumpcut_args)
# ffmpeg_toolkit.render_task(ff_render_task)


# ff_render_task = ffmpeg_toolkit.FF_Create_Render_Task().speedup(input_file, multiple=4)
# ffmpeg_toolkit.render_task(ff_render_task)

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


# code.interact(local=globals())
# ffmpeg_converter.merge(dir)
cut_silence_args = {
    "input_file": input_file,
    "odd_kwargs": ffmpeg_toolkit._create_speedup_kwargs(2),
    "even_kwargs": ffmpeg_toolkit._create_jumpcut_kwargs(2, 5, 4, 6),
}

ffmpeg_toolkit.cut_silence(**cut_silence_args)
# ff_render_task: ffmpeg_converter.FF_Create_Render = (
#     ffmpeg_converter.FF_Create_Render_Task().merge(dir, dir / "output.mp4")
# )
# ffmpeg_converter.render_task(ff_render_task)

# ff_render_task: ffmpeg_toolkit.FF_Create_Render = (
#     ffmpeg_toolkit.FF_Create_Render_Task().cut_silence_rerender(input_file)
# )
# ffmpeg_toolkit.render_task(ff_render_task)

# ffmpeg_toolkit.cut_silence(
#     input_file,
# )
# (input_file.parent / "segs").mkdir(exist_ok=True)
# ff_split_task: ffmpeg_toolkit.FF_Create_Render = (
#     ffmpeg_toolkit.FF_Create_Render_Task().split_segments(
#         input_file=input_file,
#         video_segments=[2, 4, 10],  # type: ignore
#         output_dir=input_file.parent / "segs",
#     )
# )
# ffmpeg_toolkit.render_task(ff_split_task)
# ffmpeg_toolkit.cut_silence(
#     input_file,
# )
