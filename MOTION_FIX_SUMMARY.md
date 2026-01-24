# Motion Detection 時間切片語義修正總結

## 問題發現
前次實現的 Silence Detection 和 Motion Detection 集合運算功能中，兩種檢測返回的時間切片語義不一致，導致集合運算結果錯誤。

## 時間切片語義

### Silence Detection（正確）
```python
segments = [0.0, silence_start_1, silence_end_1, ..., total_duration]
```
- **偶數索引區間** (0→1, 2→3, ...) = **有聲音時段** (保留)
- **奇數索引區間** (1→2, 3→4, ...) = **靜音時段** (移除)

### Motion Detection（修正前 - 錯誤）
```python
segments = [0.0, transition_1, transition_2, ..., total_duration]
```
- **偶數索引區間** = **無動作時段** ✗
- **奇數索引區間** = **有動作時段** ✗

### Motion Detection（修正後 - 正確）
```python
segments = [0.0, transition_1, transition_2, ..., total_duration]
# 如果影片開始是無動作，會插入額外的 0.0: [0.0, 0.0, ...]
```
- **偶數索引區間** = **有動作時段** (保留) ✓
- **奇數索引區間** = **無動作時段** (移除) ✓

## 修正內容

### 1. 核心函數修改
- `_extract_motion_segments()`: 新增 `total_duration` 參數，調整返回格式確保語義一致

### 2. 調用方更新
- `CutMotionless`: 交換 `even_further` 和 `odd_further` 預設值
- `CutByDetection`: 移除手動邊界補全邏輯
- `_create_cut_motionless_kwargs()`: 移除手動邊界補全邏輯

### 3. 測試更新
- `test_extract_motion_segments`: 更新預期值以符合新語義

## 影響範圍

### 受影響的功能
1. `CutByDetection` - 所有集合運算
2. `CutMotionless` - 移除無動作片段
3. `CutMotionlessRerender` - 重新編碼並移除無動作片段

### 向後兼容性
⚠️ **Breaking Change** - 修正前的結果與修正後完全相反

## 驗證

- ✅ 單元測試通過
- ✅ 語義測試驗證通過
- ✅ Silence 和 Motion Detection 語義現已一致

## 文件
- 詳細技術報告: `MOTION_DETECTION_FIX.md`
- 驗證腳本: `test_motion_semantics.py`
- 更新日誌: `CHANGELOG.md`

修正日期: 2026-01-24
