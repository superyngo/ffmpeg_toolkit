# keep or remove copy/rendering by cuts
# def advanced_keep_or_remove_by_cuts(
#     input_file: Path | str,
#     output_file: Path | str | None,
#     video_segments: list[str] | list[float],
#     even_further: FurtherMethod = "remove",  # For other segments, remove means remove, None means copy
#     odd_further: FurtherMethod = None,  # For segments, remove means remove, None means copy
# ) -> int:
#     task_descripton = _TASKS.KEEP_OR_REMOVE
#     input_file = Path(input_file)

#     # Set the output file path
#     if output_file is None:
#         output_file = input_file.parent / (
#             input_file.stem + "_" + task_descripton + input_file.suffix
#         )
#     else:
#         output_file = Path(output_file)

#     logger.info(
#         f"{task_descripton.capitalize()} {input_file.name} to {output_file.name} with {even_further = } ,{odd_further = }."
#     )

#     # Double every time point and convert to timestamp if needed
#     video_segments = list(
#         _convert_seconds_to_timestamp(s) if isinstance(s, (float, int)) else s
#         for o in video_segments
#         for s in (o, o)  # double every time point
#     )

#     # Create a full segment list
#     video_segments = ["00:00:00.000"] + video_segments
#     duration = FPRenderTasks().duration(input_file).render()
#     video_segments.append(_convert_seconds_to_timestamp(duration))
#     batched_segments = batched(video_segments, 2)
#     # Use ThreadPoolExecutor to manage rendreing tasks
#     temp_dir: Path = Path(tempfile.mkdtemp(prefix=DEFAULTS.temp_dir_prefix.value))
#     cut_videos = []
#     further_render_tasks: dict[int, FurtherMethod] = {
#         0: even_further,
#         1: odd_further,
#     }
#     with concurrent.futures.ThreadPoolExecutor(
#         max_workers=DEFAULTS.num_cores.value
#     ) as executor:
#         futures = []

#         for i, segment in enumerate(batched_segments):
#             i_remainder = i % 2

#             # remove unwanted segments
#             if further_render_tasks[i_remainder] == "remove":
#                 continue

#             # cut segments by submitting cut task to the executor
#             start_time: str = segment[0]
#             end_time: str = segment[1]
#             if start_time[:8] == end_time[:8]:
#                 logger.info(
#                     f"Sagment is too short to cut, skipping {start_time} ot {end_time}"
#                 )
#                 continue
#             seg_output_file = temp_dir / f"{i}{input_file.suffix}"
#             cut_videos.append(seg_output_file)
#             ff_cut_task: FFCreateRender = FFRenderTasks().cut(
#                 input_file=input_file,
#                 output_file=seg_output_file,
#                 ss=start_time,
#                 to=end_time,
#             )
#             future = executor.submit(ff_cut_task.render)
#             futures.append(future)  # Store the future for tracking
#             future.result()  # Ensures `cut` completes before proceeding

#             # Skip further rendering if the segment is to be copied
#             if further_render_tasks[i_remainder] is None:
#                 continue

#             # Submit further render task to the executor
#             future = executor.submit(
#                 further_render_tasks[i_remainder],  # type: ignore
#                 input_file=seg_output_file,  # type: ignore
#             )
#             futures.append(future)  # Store the future for tracking
#             future.result()
#         # Optionally, wait for all futures to complete
#         # concurrent.futures.wait(futures)

#     try:
#         # Merge the kept segments
#         # Sort the cut video paths by filename by index order
#         cut_videos.sort(key=lambda video_file: int(video_file.stem))
#         FFRenderTasks().merge(cut_videos, output_file).render()

#         # Clean up temporary files and dir
#         for video_path in cut_videos:
#             os.remove(video_path)
#         os.rmdir(temp_dir)
#         return 0

#     except Exception as e:
#         logger.error(f"Failed to {task_descripton} for {input_file}. Error: {e}")
#         return 1
