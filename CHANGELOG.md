# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
