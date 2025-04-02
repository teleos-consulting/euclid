#!/usr/bin/env python3
"""
Euclid - A CLI tool for interacting with local Ollama models

This script is a shortcut to run the Euclid CLI application.
"""
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path if running as a script
parent_dir = Path(__file__).resolve().parent
if parent_dir not in sys.path:
    sys.path.insert(0, str(parent_dir))

from euclid.cli import app

if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        import traceback
        import pretty_traceback
        pretty_traceback.install()
        traceback.print_exc()
        sys.exit(1)
