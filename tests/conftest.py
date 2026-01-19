#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pytest configuration for DeepTutor tests
Handles import path setup to avoid logging module conflicts
"""

from pathlib import Path
import sys

# Add project root to path (not src/) to avoid shadowing stdlib modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set up environment variables for testing
import os

os.environ["DEEPTUTOR_LOG_LEVEL"] = "ERROR"
os.environ["DEEPTUTOR_DISABLE_LOGGING"] = "1"
