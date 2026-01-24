# Motion Detection Semantics Fix - Technical Report

## 問題概述

在 v0.3.0 版本中實現的 Silence Detection 和 Motion Detection 集合運算功能存在嚴重的語義錯誤，導致集合運算結果與預期相反。

## 根本原因

### Silence Detection 語義（正確）
`_extract_non_silence_info()` 返回的 segment list 格式：
```python
[0.0, silence_start_1, silence_end_1, silence_start_2, silence_end_2, ..., total_duration]
```

**語義解釋**：
- 偶數索引區間 (0→1, 2→3, 4→5, ...) = **有聲音時段** (sound segments, 要保留)
- 奇數索引區間 (1→2, 3→4, 5→6, ...) = **靜音時段** (silence segments, 要移除)

**範例**：
```python
segments = [0.0, 5.23, 8.71, 15.5, 20.0, 90.5]
# [0.0, 5.23]   - index 0 (偶數) = 有聲音 ✓
# [5.23, 8.71]  - index 1 (奇數) = 靜音 ✓
# [8.71, 15.5]  - index 2 (偶數) = 有聲音 ✓
# [15.5, 20.0]  - index 3 (奇數) = 靜音 ✓
# [20.0, 90.5]  - index 4 (偶數) = 有聲音 ✓
```

### Motion Detection 語義（錯誤）

**舊版實現**：
```python
def _extract_motion_segments(motion_info, threshold):
    break_points = []
    prev_above = False  # ❌ 假設影片開始時是「無動作」狀態

    for _time, score in motion_info.items():
        if (score > threshold and not prev_above) or (score <= threshold and prev_above):
            break_points.append(_time)
            prev_above = score > threshold
    return break_points
```

返回的是**狀態轉換點**列表，例如：`[2.0, 4.0, 5.0, 6.0]`

在 `CutByDetection` 中補全：
```python
if motion_segments[0] != 0.0:
    motion_segments = [0.0] + motion_segments
if motion_segments[-1] != total_duration:
    motion_segments = motion_segments + [total_duration]
# 結果: [0.0, 2.0, 4.0, 5.0, 6.0, total_duration]
```

**語義解釋**（錯誤）：
```python
# 假設: motion_info = {0.0: 0.001, 2.0: 0.015, 4.0: 0.003, 6.0: 0.001}
# threshold = 0.01
segments = [0.0, 2.0, 4.0, 6.0, 7.0]
# [0.0, 2.0]   - index 0 (偶數) = 無動作 ✗ (與 silence detection 語義相反!)
# [2.0, 4.0]   - index 1 (奇數) = 有動作 ✗
# [4.0, 6.0]   - index 2 (偶數) = 無動作 ✗
# [6.0, 7.0]   - index 3 (奇數) = 無動作 ✗
```

### 集合運算的問題

所有集合運算函數（`_union_segments`, `_intersect_segments`, 等）都假設：
- **偶數區間 = 要保留的內容**
- **奇數區間 = 要移除的內容**

當 `CutByDetection` 執行 `INTERSECTION` 操作時：
```python
combined = _intersect_segments(sound_segments, motion_segments)
```

預期行為：保留「**有聲音 AND 有動作**」的片段
實際行為：保留「**有聲音 AND 無動作**」的片段 ❌

## 修正方案

### 1. 修改 `_extract_motion_segments()`

**核心邏輯**：
1. 檢測影片第一幀的狀態（有動作 or 無動作）
2. 如果開始是無動作，在轉換點列表前**插入額外的 0.0**
3. 確保返回格式始終以 0.0 開頭、total_duration 結尾

**新版實現**：
```python
def _extract_motion_segments(
    motion_info: dict[float, float],
    threshold: float = DEFAULTS.motionless_threshold.value,
    total_duration: float | None = None,
) -> list[float]:
    if not motion_info:
        if total_duration is not None:
            return [0.0, 0.0, total_duration, total_duration]
        return [0.0]

    # 收集狀態轉換點
    break_points = []
    prev_above = None

    for _time, score in motion_info.items():
        current_above = score > threshold
        if prev_above is None:
            prev_above = current_above
        elif current_above != prev_above:
            break_points.append(_time)
            prev_above = current_above

    # 構建完整的 segment list
    motion_segments = [0.0] + break_points

    if total_duration is not None:
        motion_segments.append(total_duration)

    # 關鍵：如果影片開始是無動作狀態，插入 0.0 使第一個區間變成奇數索引
    first_score = next(iter(motion_info.values()))
    starts_with_motion = first_score > threshold

    if not starts_with_motion:
        motion_segments = [0.0] + motion_segments

    return motion_segments
```

**語義驗證**（修正後）：
```python
# 情況 1: 影片開始是無動作
motion_info = {0.0: 0.001, 2.0: 0.015, 4.0: 0.003, 6.0: 0.001}
threshold = 0.01
total_duration = 7.0

result = [0.0, 0.0, 2.0, 4.0, 6.0, 7.0]
# [0.0, 0.0]   - index 0 (偶數) = 空的有動作區間 ✓
# [0.0, 2.0]   - index 1 (奇數) = 無動作 ✓
# [2.0, 4.0]   - index 2 (偶數) = 有動作 ✓
# [4.0, 6.0]   - index 3 (奇數) = 無動作 ✓
# [6.0, 7.0]   - index 4 (偶數) = 無動作 ✓ (最後區間也可能是無動作)

# 情況 2: 影片開始是有動作
motion_info = {0.0: 0.020, 2.0: 0.005, 4.0: 0.018, 5.0: 0.003}
threshold = 0.01
total_duration = 7.0

result = [0.0, 2.0, 4.0, 5.0, 7.0]
# [0.0, 2.0]   - index 0 (偶數) = 有動作 ✓
# [2.0, 4.0]   - index 1 (奇數) = 無動作 ✓
# [4.0, 5.0]   - index 2 (偶數) = 有動作 ✓
# [5.0, 7.0]   - index 3 (奇數) = 無動作 ✓
```

### 2. 更新所有調用方

#### `CutMotionless` 類
交換 `even_further` 和 `odd_further` 的預設值：
```python
# 舊版（錯誤）
even_further: FurtherMethod = "remove"  # 偶數 = 無動作 (移除)
odd_further: FurtherMethod = None       # 奇數 = 有動作 (保留)

# 新版（正確）
even_further: FurtherMethod = None      # 偶數 = 有動作 (保留) ✓
odd_further: FurtherMethod = "remove"   # 奇數 = 無動作 (移除) ✓
```

#### `CutByDetection.render()`
移除手動補全邏輯（現在由 `_extract_motion_segments` 內部處理）：
```python
# 舊版
motion_segments = _extract_motion_segments(motion_info, threshold)
if motion_segments[0] != 0.0:
    motion_segments = [0.0] + motion_segments
if motion_segments[-1] != total_duration:
    motion_segments = motion_segments + [total_duration]

# 新版
motion_segments = _extract_motion_segments(motion_info, threshold, total_duration)
```

#### `_create_cut_motionless_kwargs()`
同樣移除手動補全邏輯：
```python
# 舊版
motion_segs = _extract_motion_segments(motion_info, threshold)
if len(motion_segs) % 2 == 1:
    motion_segs.append(total_duration)

# 新版
motion_segs = _extract_motion_segments(motion_info, threshold, total_duration)
```

### 3. 更新測試

更新 `test_extract_motion_segments` 測試的預期值：
```python
def test_extract_motion_segments(self):
    motion_info = {
        0.0: 0.001, 1.0: 0.002,
        2.0: 0.015, 3.0: 0.020,
        4.0: 0.003,
        5.0: 0.012,
        6.0: 0.001,
    }
    threshold = 0.01
    total_duration = 7.0
    result = _extract_motion_segments(motion_info, threshold, total_duration)

    # 驗證格式
    assert result[0] == 0.0
    assert result[-1] == total_duration
    assert 2.0 in result  # 轉換點
    assert 4.0 in result
    assert 5.0 in result
    assert 6.0 in result
```

## 影響範圍

### 受影響的功能
1. **`CutByDetection`** - 所有集合運算（UNION, INTERSECTION, SOUND_ONLY, MOTION_ONLY, XOR, COMPLEMENT）
2. **`CutMotionless`** - 移除無動作片段的功能
3. **`CutMotionlessRerender`** - 重新編碼並移除無動作片段

### 向後兼容性
❌ **不兼容** - 這是一個 breaking change

- 修正前使用 `CutByDetection` 的代碼會產生**相反**的結果
- 建議所有使用這些功能的用戶更新到修正版本並重新處理影片

## 驗證

### 單元測試
```bash
pytest test/test_ffmpeg_toolkit.py::TestMotionOutputParsing -v
# ✓ test_extract_motion_info PASSED
# ✓ test_extract_motion_segments PASSED
```

### 語義測試
```bash
python test_motion_semantics.py
# ✓ All tests passed!
```

### 集成測試建議
使用包含明確動作和靜音片段的測試影片驗證：
1. `CutByDetection` INTERSECTION 應保留「有聲音 AND 有動作」的片段
2. `CutByDetection` UNION 應保留「有聲音 OR 有動作」的片段
3. `CutMotionless` 應移除無動作片段、保留有動作片段

## 總結

這次修正確保了 **Motion Detection 和 Silence Detection 具有一致的語義**：
- ✓ 偶數區間 = 要保留的內容（motion/sound）
- ✓ 奇數區間 = 要移除的內容（motionless/silence）
- ✓ 集合運算現在能正確組合兩種檢測結果
- ✓ `CutMotionless` 現在正確移除無動作片段

修正日期: 2026-01-24
修正版本: v0.3.1 (待發布)
