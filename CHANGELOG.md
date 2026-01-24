# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-01-24

### Fixed

- **BREAKING CHANGE**: Fixed critical motion detection semantics bug in segment set operations
  - `_extract_motion_segments()` now returns segment boundaries with consistent semantics:
    - Even-indexed intervals (0-1, 2-3, ...) represent MOTION segments (to keep)
    - Odd-indexed intervals (1-2, 3-4, ...) represent MOTIONLESS segments (to remove)
  - This aligns with `_extract_non_silence_info()` format where even intervals = sound (keep)
  - Updated `CutMotionless` class: swapped `even_further` and `odd_further` default values
  - Updated `CutByDetection`: removed manual segment boundary padding (now handled in `_extract_motion_segments`)
  - Updated `_create_cut_motionless_kwargs()`: removed manual boundary padding
  - Added `total_duration` parameter to `_extract_motion_segments()` for proper boundary handling
  - Updated test expectations in `test_ffmpeg_toolkit.py::test_extract_motion_segments`
  - **Impact**: This fixes incorrect behavior in set operations (UNION, INTERSECTION, etc.) that previously combined sound and motion segments with opposite semantic meanings
  - **Migration**: All code using `CutByDetection`, `CutMotionless`, or `CutMotionlessRerender` will produce different results after this fix. The new results are correct; previous results were inverted.

## [0.3.0] - 2026-01-17

### Added

- Added segment set operations feature for combining silence and motion detection
  - New utility functions for segment set operations:
    - `_union_segments()` - Union of two segment sets (A OR B)
    - `_intersect_segments()` - Intersection of two segment sets (A AND B)
    - `_difference_segments()` - Difference of two segment sets (A - B)
    - `_xor_segments()` - Symmetric difference (XOR) of two segment sets
    - `_complement_segments()` - Complement of a segment set
  - New `SegmentOperation` enum with operations: UNION, INTERSECTION, SOUND_ONLY, MOTION_ONLY, XOR, COMPLEMENT
  - New `CutByDetection` task class for combined silence and motion detection with set operations
  - New `PARTIAL_TASKS.cut_by_detection()` factory method
  - Export `SegmentOperation` from package `__init__.py`
- Updated documentation:
  - `ARCHITECTURE.md` - Added CutByDetection section and segment set operations algorithms
  - `PORTING_GUIDE.md` - Added segment set operations implementation guide

## [0.2.2] - 2026-01-17

### Added

- **2026-01-17**: Added comprehensive documentation files:
  - `ARCHITECTURE.md` - Technical architecture documentation with class hierarchy, core concepts, and implementation details
  - `PORTING_GUIDE.md` - Guide for porting the toolkit to other programming languages/platforms
- **2026-01-17**: Added comprehensive test suite with pytest (`test/test_ffmpeg_toolkit.py`)
  - Unit tests for timestamp conversion, kwargs generation, segment processing
  - Mocked tests for FFmpeg/FFprobe execution
  - Tests for Pydantic model initialization and validation
- **2026-01-17**: Added missing video format extensions to `VideoSuffix` enum:
  - MOV, WEBM, M4V, TS

### Fixed

- **2026-01-17**: Fixed typo `task_descripton` -> `task_description` across all files
  - `ffmpeg_types.py`: `OptionFFRender` TypedDict
  - `ffmpeg_toolkit_core.py`: `FPCreateRender` and `FFCreateTask` classes and all usages
- **2026-01-17**: Fixed typo `_defalut_output_kwargs` -> `_default_output_kwargs` in ffmpeg_toolkit_core.py
- **2026-01-17**: Fixed typo `defalut_output_kwargs` -> `default_output_kwargs` in probe methods
- **2026-01-17**: Fixed comment typos:
  - `# Exception hadling` -> `# Exception handling`
  - `# Error hadling` -> `# Error handling`
  - `# Handle inout and output file path` -> `# Handle input and output file path`

### Improved

- **2026-01-17**: Added `FileNotFoundError` handling in `_ffmpeg()` and `_ffprobe()` functions
  - Now provides clear error message when FFmpeg/FFprobe executables are not found in PATH

## [0.2.1] - 2025-06-04

### Fixed

- 修復 BatchTask.render 方法的型別標註和實現問題
  - 修正參數型別標註錯誤：將 `BatchTask.render()` 方法的參數從 `task: PARTIAL_TASKS` 修正為 `task: PARTIAL_TASK`
    - `PARTIAL_TASKS` 是一個類別，而 `PARTIAL_TASK` 才是正確的型別別名
  - 修復變數名稱拼寫錯誤：修正 `BatchTask.render()` 方法中的 `output_gile` 變數名稱為 `output_file`
    - 確保 `post_hook` 函數能夠接收正確的參數
  - 改進 `PARTIAL_TASK` 型別定義精確度：
    - 將回傳型別從 `Callable[..., Any]` 精確化為 `Callable[..., Path | "ERROR_CODE" | int | Any]`
    - 加入 `TYPE_CHECKING` 匯入以避免循環匯入問題
    - 更準確地反映實際的回傳型別，包括成功的檔案路徑、錯誤代碼和其他可能的結果

### Changed

- 影響範圍：
  - `ffmpeg_toolkit.py` - BatchTask 類別
  - `ffmpeg_types.py` - 型別定義
- 相容性：保持向後相容，不影響現有 API

## [0.2.0] - Previous Version

- Initial release with core functionality
