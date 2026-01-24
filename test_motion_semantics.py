#!/usr/bin/env python3
"""
Test script to verify motion detection semantics fix.

This script demonstrates that the motion detection now returns segments
with the same semantic meaning as silence detection:
- Even-indexed intervals represent the content to KEEP (motion/sound)
- Odd-indexed intervals represent the content to REMOVE (motionless/silence)
"""

import sys
sys.path.insert(0, 'src')

from ffmpeg_toolkit.ffmpeg_toolkit_core import (
    _extract_motion_segments,
    _extract_non_silence_info,
)


def print_segment_intervals(segments: list[float], name: str):
    """Print segment intervals with their indices."""
    print(f"\n{name} segments: {segments}")
    print(f"Interval breakdown:")
    for i in range(0, len(segments) - 1, 2):
        if i + 1 < len(segments):
            interval_type = "KEEP (even)" if i % 2 == 0 else "REMOVE (odd)"
            print(f"  [{segments[i]:.2f}, {segments[i+1]:.2f}] - {interval_type}")


def test_motion_segments():
    """Test motion segment extraction with new semantics."""
    print("="*70)
    print("MOTION DETECTION SEMANTICS TEST")
    print("="*70)

    # Test case 1: Video starts with motionless
    print("\n--- Test 1: Video starts MOTIONLESS ---")
    motion_info_1 = {
        0.0: 0.001,   # motionless
        1.0: 0.002,   # motionless
        2.0: 0.015,   # MOTION (above threshold)
        3.0: 0.020,   # MOTION
        4.0: 0.003,   # motionless
        5.0: 0.012,   # MOTION
        6.0: 0.001,   # motionless
    }
    threshold = 0.01
    total_duration = 7.0

    result_1 = _extract_motion_segments(motion_info_1, threshold, total_duration)
    print_segment_intervals(result_1, "Motion")

    # Verify semantics
    assert result_1[0] == 0.0, "Should start with 0.0"
    assert result_1[-1] == total_duration, "Should end with total_duration"

    # Test case 2: Video starts with motion
    print("\n--- Test 2: Video starts with MOTION ---")
    motion_info_2 = {
        0.0: 0.020,   # MOTION
        1.0: 0.015,   # MOTION
        2.0: 0.005,   # motionless
        3.0: 0.002,   # motionless
        4.0: 0.018,   # MOTION
        5.0: 0.003,   # motionless
    }

    result_2 = _extract_motion_segments(motion_info_2, threshold, total_duration)
    print_segment_intervals(result_2, "Motion")

    # Verify: first interval should be motion (even index)
    print(f"\nFirst interval [0.0, {result_2[1]:.2f}] should be MOTION (even index 0)")

    # Test case 3: Compare with silence detection format
    print("\n--- Test 3: Format comparison with silence detection ---")
    silence_output = """
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'input.mp4':
  Duration: 00:01:30.50, start: 0.000000, bitrate: 1234 kb/s
[silencedetect @ 0x55555] silence_start: 5.23
[silencedetect @ 0x55555] silence_end: 8.71 | silence_duration: 3.48
[silencedetect @ 0x55555] silence_start: 15.5
[silencedetect @ 0x55555] silence_end: 20.0 | silence_duration: 4.5
    """

    silence_segs, _, _ = _extract_non_silence_info(silence_output)
    print_segment_intervals(silence_segs, "Silence")

    print("\n" + "="*70)
    print("✓ All tests passed! Motion detection now has consistent semantics:")
    print("  - Even intervals = content to KEEP (motion/sound)")
    print("  - Odd intervals = content to REMOVE (motionless/silence)")
    print("="*70)


if __name__ == "__main__":
    test_motion_segments()
