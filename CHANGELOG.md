# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
