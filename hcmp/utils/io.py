"""Small I/O helpers used by HCMP scripts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs.
    yaml = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        text = handle.read()
    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        data = _parse_simple_yaml_mapping(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}.")
    return data


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path``."""

    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def slugify(value: str, fallback: str = "molecule") -> str:
    """Convert a molecule name or SMILES string into a safe filename stem."""

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return slug or fallback


def _parse_simple_yaml_mapping(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by HCMP configs when PyYAML is absent."""

    lines: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip()))
    if not lines:
        return {}
    parsed, index = _parse_yaml_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ValueError("Could not parse full YAML document.")
    if not isinstance(parsed, dict):
        raise ValueError("Expected YAML mapping at document root.")
    return parsed


def _parse_yaml_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if lines[index][1] == "-" or lines[index][1].startswith("- "):
        values = []
        while index < len(lines) and lines[index][0] == indent and (
            lines[index][1] == "-" or lines[index][1].startswith("- ")
        ):
            item_text = "" if lines[index][1] == "-" else lines[index][1][2:].strip()
            index += 1
            if item_text == "":
                item, index = _parse_yaml_block(lines, index, lines[index][0])
            elif ":" in item_text and not item_text.startswith("["):
                key, raw_value = item_text.split(":", 1)
                item = {key.strip(): _parse_scalar(raw_value.strip())}
                if raw_value.strip() == "" and index < len(lines) and lines[index][0] > indent:
                    child, index = _parse_yaml_block(lines, index, lines[index][0])
                    item[key.strip()] = child
                while index < len(lines) and lines[index][0] > indent:
                    child_indent = lines[index][0]
                    child, index = _parse_yaml_block(lines, index, child_indent)
                    if isinstance(child, dict):
                        item.update(child)
                    else:
                        raise ValueError("List item child must be a mapping.")
            else:
                item = _parse_scalar(item_text)
            values.append(item)
        return values, index

    mapping: dict[str, Any] = {}
    while index < len(lines) and lines[index][0] == indent and not (
        lines[index][1] == "-" or lines[index][1].startswith("- ")
    ):
        text = lines[index][1]
        if ":" not in text:
            raise ValueError(f"Cannot parse YAML line: {text!r}")
        key, raw_value = text.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1
        if raw_value == "":
            if index < len(lines) and lines[index][0] > indent:
                value, index = _parse_yaml_block(lines, index, lines[index][0])
            else:
                value = {}
        else:
            value = _parse_scalar(raw_value)
        mapping[key] = value
    return mapping, index


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~", ""}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip("'\"")
