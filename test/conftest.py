"""
Pytest configuration and fixtures for FFmpeg Toolkit tests.
"""

import pytest
import sys
from pathlib import Path

# Add the src directory to the Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
