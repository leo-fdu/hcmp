"""Configuration loading helpers for HCMP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hcmp.utils.io import load_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    return load_yaml(path)
