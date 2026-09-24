"""YAML configs with key=value overrides (CLAUDE.md conventions)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Load a YAML config and apply overrides like `seed=3` or `clinc.k_max=8` (values parsed as YAML)."""
    config = yaml.safe_load(Path(path).read_text())
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep or not key:
            raise ValueError(f"override {override!r}: expected key=value")
        *parents, leaf = key.split(".")
        node = config
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise ValueError(f"override {override!r}: {part!r} is not a section of the config")
            node = node[part]
        if leaf not in node:
            raise ValueError(f"override {override!r}: unknown key {key!r}")
        node[leaf] = yaml.safe_load(raw)
    return config
