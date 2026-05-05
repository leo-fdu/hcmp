#!/usr/bin/env python
"""Compatibility entrypoint for HCMP pretraining."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("06_train_hcmp.py")), run_name="__main__")
