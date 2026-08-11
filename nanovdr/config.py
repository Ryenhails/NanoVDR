"""YAML config loading with environment expansion.

Configs name paths like ``${DATA_ROOT}/train``. Expanding them here keeps the
committed files free of anyone's directory layout.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config", "expand"]


def expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return expand(cfg)
