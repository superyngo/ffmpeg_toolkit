"""
Comprehensive test suite for FFmpeg Toolkit.

This module provides unit tests for the core functionality of the FFmpeg Toolkit,
including utility functions, command generation, output parsing, and segment processing.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

# Import the modules to test
from ffmpeg_toolkit.ffmpeg_toolkit_core import (
    _convert_timestamp_to_seconds,
    _convert_seconds_to_timestamp,
    _dic_to_ffmpeg_kwargs,
    _create_ff_kwargs,
    _create_fp_kwargs,
    _create_speedup_kwargs,
    _create_jumpcut_kwargs,
    _get_segments_from_parts_count,
    _ensure_minimum_segment_length,
    _adjust_segments_to_keyframes,
    _merge_overlapping_segments,
    _extract_non_silence_info,
    _extract_motion_info,
    _extract_motion_segments,
    DEFAULTS,
    ERROR_CODE,
    Cut,
    Speedup,
    Jumpcut,
    SplitSegments,
    FPRenderTasks,
)
from ffmpeg_toolkit.ffmpeg_types import VideoSuffix, FFKwargs


class TestTimestampConversion:
    """Tests for timestamp conversion functions."""

    def test_timestamp_to_seconds_simple(self):
        """Test basic timestamp to seconds conversion."""
        assert _convert_timestamp_to_seconds("00:00:00") == 0.0
        assert _convert_timestamp_to_seconds("00:00:01") == 1.0
        assert _convert_timestamp_to_seconds("00:01:00") == 60.0
        assert _convert_timestamp_to_seconds("01:00:00") == 3600.0

    def test_timestamp_to_seconds_complex(self):
        """Test complex timestamp values."""
        assert _convert_timestamp_to_seconds("01:30:45") == 5445.0
        assert _convert_timestamp_to_seconds("00:00:30.5") == 30.5
        assert _convert_timestamp_to_seconds("02:15:30.123") == 8130.123

    def test_seconds_to_timestamp_simple(self):
        """Test basic seconds to timestamp conversion."""
        assert _convert_seconds_to_timestamp(0) == "00:00:00.000"
        assert _convert_seconds_to_timestamp(1) == "00:00:01.000"
        assert _convert_seconds_to_timestamp(60) == "00:01:00.000"
        assert _convert_seconds_to_timestamp(3600) == "01:00:00.000"

    def test_seconds_to_timestamp_complex(self):
        """Test complex second values."""
        assert _convert_seconds_to_timestamp(5445.5) == "01:30:45.500"
        assert _convert_seconds_to_timestamp(90.123) == "00:01:30.123"

    def test_roundtrip_conversion(self):
        """Test that converting back and forth preserves value."""
        test_values = [0, 1, 60, 3600, 5445.5, 90.123, 7265.999]
        for value in test_values:
            timestamp = _convert_seconds_to_timestamp(value)
            result = _convert_timestamp_to_seconds(timestamp)
            assert abs(result - value) < 0.001  # Allow for floating point precision


class TestDicToFFmpegKwargs:
    """Tests for dictionary to FFmpeg arguments conversion."""

    def test_empty_dict(self):
        """Test with empty dictionary."""
        assert _dic_to_ffmpeg_kwargs(None) == []
        assert _dic_to_ffmpeg_kwargs({}) == []

    def test_simple_args(self):
        """Test simple argument conversion."""
        kwargs = {"vf": "setpts=0.5*PTS", "map": 0}
        result = _dic_to_ffmpeg_kwargs(kwargs)
        assert "-vf" in result
        assert "setpts=0.5*PTS" in result
        assert "-map" in result
        assert "0" in result

    def test_special_mappings(self):
        """Test special argument mappings (cv -> c:v, etc.)."""
        kwargs = {"cv": "libx264", "ca": "aac", "bv": "5M", "ba": "128k"}
        result = _dic_to_ffmpeg_kwargs(kwargs)
        assert "-c:v" in result
        assert "libx264" in result
        assert "-c:a" in result
        assert "aac" in result
        assert "-b:v" in result
        assert "5M" in result
        assert "-b:a" in result
        assert "128k" in result

    def test_flag_only_args(self):
        """Test flag-only arguments (empty string value)."""
        kwargs = {"hide_banner": "", "y": "output.mp4"}
        result = _dic_to_ffmpeg_kwargs(kwargs)
        assert "-hide_banner" in result
        assert "-y" in result
        assert "output.mp4" in result

    def test_path_values(self):
        """Test Path object values."""
        kwargs = {"i": Path("/path/to/input.mp4")}
        result = _dic_to_ffmpeg_kwargs(kwargs)
        assert "-i" in result


class TestCreateFFKwargs:
    """Tests for FFmpeg kwargs creation and merging."""

    def test_default_kwargs(self):
        """Test that default kwargs are included."""
        result = _create_ff_kwargs(
            input_file=Path("input.mp4"),
            output_file=Path("output.mp4"),
            input_kwargs={},
            output_kwargs={},
        )
        assert result["hide_banner"] == ""
        assert result["hwaccel"] == DEFAULTS.hwaccel.value
        assert result["loglevel"] == DEFAULTS.loglevel.value
        assert result["i"] == Path("input.mp4")
        assert result["y"] == Path("output.mp4")

    def test_custom_kwargs_override(self):
        """Test that custom kwargs override defaults."""
        result = _create_ff_kwargs(
            input_file=Path("input.mp4"),
            output_file=Path("output.mp4"),
            input_kwargs={"hwaccel": "cuda"},
            output_kwargs={"loglevel": "error"},
        )
        assert result["hwaccel"] == "cuda"
        assert result["loglevel"] == "error"


class TestSpeedupKwargs:
    """Tests for speedup kwargs generation."""

    def test_small_multiple_uses_pts(self):
        """Test that small multiples use PTS adjustment."""
        result = _create_speedup_kwargs(2)
        assert "setpts=" in result["vf"]
        assert "atempo=" in result["af"]
        assert "select=" not in result["vf"]

    def test_large_multiple_uses_select(self):
        """Test that large multiples use frame selection."""
        result = _create_speedup_kwargs(10)
        assert "select=" in result["vf"]
        assert "aselect=" in result["af"]

    def test_common_kwargs(self):
        """Test that common speedup kwargs are included."""
        result = _create_speedup_kwargs(2)
        assert result["map"] == 0
        assert result["shortest"] == ""
        assert result["fps_mode"] == "vfr"
        assert result["reset_timestamps"] == "1"
        assert "force_key_frames" in result


class TestJumpcutKwargs:
    """Tests for jumpcut kwargs generation."""

    def test_basic_jumpcut(self):
        """Test basic jumpcut configuration."""
        result = _create_jumpcut_kwargs(5, 5, 1, 0)
        assert "select=" in result["vf"]
        assert "aselect=" in result["af"]
        assert "mod(t," in result["vf"]

    def test_jumpcut_remove_segment(self):
        """Test jumpcut with segment removal (multiple=0)."""
        result = _create_jumpcut_kwargs(5, 5, 1, 0)
        # When b2_multiple is 0, it should just use "0" in the expression
        assert ", 0)" in result["vf"]


class TestSegmentPartitioning:
    """Tests for segment partitioning functions."""

    def test_equal_segments(self):
        """Test splitting into equal segments."""
        result = _get_segments_from_parts_count(100, 4)
        assert len(result) == 3  # 4 parts = 3 split points
        # Each segment should be 25 seconds
        times = [_convert_timestamp_to_seconds(t) for t in result]
        expected = [25.0, 50.0, 75.0]
        for actual, exp in zip(times, expected):
            assert abs(actual - exp) < 0.001

    def test_custom_portions(self):
        """Test splitting with custom portions."""
        result = _get_segments_from_parts_count(100, 4, [1, 2, 1])
        times = [_convert_timestamp_to_seconds(t) for t in result]
        expected = [25.0, 75.0]  # 1:2:1 ratio = 25:50:25
        for actual, exp in zip(times, expected):
            assert abs(actual - exp) < 0.001

    def test_invalid_parts_count(self):
        """Test that invalid parts count raises error."""
        with pytest.raises(ValueError):
            _get_segments_from_parts_count(100, 0)

    def test_portions_sum_mismatch(self):
        """Test that portions not summing to parts_count raises error."""
        with pytest.raises(ValueError):
            _get_segments_from_parts_count(100, 4, [1, 1, 1])


class TestSegmentAdjustment:
    """Tests for segment adjustment functions."""

    def test_ensure_minimum_segment_length_no_change(self):
        """Test that segments meeting minimum length are unchanged."""
        segments = [0.0, 10.0, 20.0, 30.0]
        result = _ensure_minimum_segment_length(segments, 5.0, 30.0)
        assert result == segments

    def test_ensure_minimum_segment_length_extend(self):
        """Test that short segments are extended."""
        segments = [0.0, 2.0, 10.0, 20.0]  # First segment is 2s, too short
        result = _ensure_minimum_segment_length(segments, 5.0, 20.0)
        # First segment should be extended
        assert result[1] - result[0] >= 5.0

    def test_ensure_minimum_segment_length_empty_input(self):
        """Test with empty input."""
        assert _ensure_minimum_segment_length([], 5.0) == []

    def test_ensure_minimum_segment_length_zero_min(self):
        """Test with zero minimum duration."""
        segments = [0.0, 1.0, 10.0, 11.0]
        assert _ensure_minimum_segment_length(segments, 0) == segments

    def test_merge_overlapping_segments(self):
        """Test merging overlapping segments."""
        segments = [0.0, 10.0, 5.0, 15.0, 20.0, 30.0]
        result = _merge_overlapping_segments(segments)
        # First two segments overlap (0-10, 5-15), should merge to 0-15
        assert result == [0.0, 15.0, 20.0, 30.0]

    def test_merge_no_overlap(self):
        """Test that non-overlapping segments are unchanged."""
        segments = [0.0, 10.0, 20.0, 30.0]
        result = _merge_overlapping_segments(segments)
        assert result == segments

    def test_adjust_to_keyframes(self):
        """Test segment alignment to keyframes."""
        segments = [1.5, 8.7, 12.3, 18.9]
        keyframes = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        result = _adjust_segments_to_keyframes(segments, keyframes)
        # Start times should be adjusted to keyframes <= original
        # End times should be adjusted to keyframes >= original
        assert result[0] == 0.0  # 1.5 -> 0.0 (largest keyframe <= 1.5)
        assert result[1] == 10.0  # 8.7 -> 10.0 (smallest keyframe >= 8.7)


class TestSilenceOutputParsing:
    """Tests for silence detection output parsing."""

    def test_extract_non_silence_info(self):
        """Test parsing silence detection output."""
        sample_output = """
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'input.mp4':
  Duration: 00:01:30.50, start: 0.000000, bitrate: 1234 kb/s
[silencedetect @ 0x55555] silence_start: 5.23
[silencedetect @ 0x55555] silence_end: 8.71 | silence_duration: 3.48
[silencedetect @ 0x55555] silence_start: 15.5
[silencedetect @ 0x55555] silence_end: 20.0 | silence_duration: 4.5
        """
        segments, total_duration, total_silence = _extract_non_silence_info(sample_output)

        # Should include boundaries: [0, silence_start1, silence_end1, silence_start2, silence_end2, total_duration]
        assert 0.0 in segments
        assert 5.23 in segments
        assert 8.71 in segments
        assert 15.5 in segments
        assert 20.0 in segments
        assert abs(total_duration - 90.5) < 0.1
        assert abs(total_silence - 7.98) < 0.1


class TestMotionOutputParsing:
    """Tests for motion detection output parsing."""

    def test_extract_motion_info(self):
        """Test parsing motion detection output."""
        sample_output = """
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'input.mp4':
  Duration: 00:00:30.00, start: 0.000000, bitrate: 1234 kb/s
frame:1 pts:0 pts_time:0.000000
lavfi.scene_score=0.001
frame:5 pts:50 pts_time:2.000000
lavfi.scene_score=0.015
frame:10 pts:100 pts_time:4.000000
lavfi.scene_score=0.002
        """
        motion_info, total_duration = _extract_motion_info(sample_output)

        assert abs(total_duration - 30.0) < 0.1
        assert 0.0 in motion_info
        assert 2.0 in motion_info
        assert 4.0 in motion_info
        assert abs(motion_info[0.0] - 0.001) < 0.0001
        assert abs(motion_info[2.0] - 0.015) < 0.0001

    def test_extract_motion_segments(self):
        """Test extracting motion segments from motion info."""
        motion_info = {
            0.0: 0.001,
            1.0: 0.002,
            2.0: 0.015,  # Above threshold
            3.0: 0.020,  # Above threshold
            4.0: 0.003,
            5.0: 0.012,  # Above threshold
            6.0: 0.001,
        }
        threshold = 0.01
        total_duration = 7.0
        result = _extract_motion_segments(motion_info, threshold, total_duration)

        # New behavior: returns complete segment list [0.0, ..., total_duration]
        # Even intervals = motion, odd intervals = motionless
        # Video starts motionless (0.001 < 0.01), so prepend 0.0
        # Expected: [0.0, 0.0, 2.0, 4.0, 5.0, 6.0, 7.0]
        #   [0.0, 0.0] = empty motion (even index 0)
        #   [0.0, 2.0] = motionless (odd index 1)
        #   [2.0, 4.0] = motion (even index 2)
        #   [4.0, 5.0] = motionless (odd index 3)
        #   [5.0, 6.0] = motion (even index 4)
        #   [6.0, 7.0] = motionless (odd index 5)
        assert result[0] == 0.0  # Always starts with 0.0
        assert result[-1] == total_duration  # Always ends with total_duration
        assert 2.0 in result  # Transition from motionless to motion
        assert 4.0 in result  # Transition from motion to motionless
        assert 5.0 in result  # Transition from motionless to motion
        assert 6.0 in result  # Transition from motion to motionless


class TestVideoSuffix:
    """Tests for VideoSuffix enum."""

    def test_all_extensions(self):
        """Test that all expected extensions are present."""
        extensions = {e.value for e in VideoSuffix}
        assert "mp4" in extensions
        assert "mkv" in extensions
        assert "avi" in extensions
        assert "mov" in extensions
        assert "webm" in extensions
        assert "m4v" in extensions
        assert "ts" in extensions


class TestTaskModels:
    """Tests for task Pydantic models."""

    def test_cut_task_initialization(self):
        """Test Cut task initialization."""
        task = Cut(
            input_file=Path("input.mp4"),
            output_file=Path("output.mp4"),
            ss="00:01:00",
            to="00:02:00",
            rerender=False,
        )
        assert "cut" in task.task_description.lower()
        assert task.output_kwargs.get("ss") == "00:01:00"
        assert task.output_kwargs.get("to") == "00:02:00"
        assert task.output_kwargs.get("c:v") == "copy"

    def test_cut_task_rerender(self):
        """Test Cut task with rerender=True."""
        task = Cut(
            input_file=Path("input.mp4"),
            output_file=Path("output.mp4"),
            ss="00:01:00",
            to="00:02:00",
            rerender=True,
        )
        # When rerendering, should not have copy codec
        assert task.output_kwargs.get("c:v") is None

    def test_speedup_task_initialization(self):
        """Test Speedup task initialization."""
        task = Speedup(
            input_file=Path("input.mp4"),
            output_file=Path("output.mp4"),
            multiple=2,
        )
        assert "speedup" in task.task_description.lower()
        assert "vf" in task.output_kwargs
        assert "af" in task.output_kwargs

    def test_speedup_task_error_multiple_zero(self):
        """Test Speedup task error for multiple <= 0."""
        task = Speedup(
            input_file=Path("input.mp4"),
            output_file=Path("output.mp4"),
            multiple=0,
        )
        assert task.exception is not None
        assert task.exception["code"] == 1

    def test_jumpcut_task_initialization(self):
        """Test Jumpcut task initialization."""
        task = Jumpcut(
            input_file=Path("input.mp4"),
            output_file=Path("output.mp4"),
            b1_duration=5,
            b2_duration=5,
            b1_multiple=1,
            b2_multiple=0,
        )
        assert "jumpcut" in task.task_description.lower()

    def test_jumpcut_task_error_invalid_duration(self):
        """Test Jumpcut task error for invalid duration."""
        task = Jumpcut(
            input_file=Path("input.mp4"),
            output_file=Path("output.mp4"),
            b1_duration=0,  # Invalid
            b2_duration=5,
        )
        assert task.exception is not None
        assert task.exception["code"] == 1


class TestFFmpegExecutionMocked:
    """Tests for FFmpeg execution with mocked subprocess."""

    @patch("ffmpeg_toolkit.ffmpeg_toolkit_core.subprocess.run")
    def test_ffmpeg_not_found_error(self, mock_run):
        """Test FileNotFoundError when FFmpeg is not installed."""
        from ffmpeg_toolkit.ffmpeg_toolkit_core import _ffmpeg

        mock_run.side_effect = FileNotFoundError("ffmpeg not found")

        with pytest.raises(FileNotFoundError) as excinfo:
            _ffmpeg(i="input.mp4", y="output.mp4")

        assert "FFmpeg executable not found" in str(excinfo.value)

    @patch("ffmpeg_toolkit.ffmpeg_toolkit_core.subprocess.run")
    def test_ffprobe_not_found_error(self, mock_run):
        """Test FileNotFoundError when FFprobe is not installed."""
        from ffmpeg_toolkit.ffmpeg_toolkit_core import _ffprobe

        mock_run.side_effect = FileNotFoundError("ffprobe not found")

        with pytest.raises(FileNotFoundError) as excinfo:
            _ffprobe(i="input.mp4")

        assert "FFprobe executable not found" in str(excinfo.value)


# Optional: Integration tests (require actual FFmpeg installation)
class TestIntegration:
    """Integration tests that require FFmpeg to be installed."""

    @pytest.fixture
    def check_ffmpeg(self):
        """Skip if FFmpeg is not available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("FFmpeg not available")

    @pytest.mark.skip(reason="Integration test - requires actual video file")
    def test_probe_duration(self, check_ffmpeg):
        """Test probing video duration."""
        # This test would require an actual video file
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
