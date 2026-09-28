"""Small dependency-light YAML configuration loader with dotted overrides."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import re
from typing import Any

import yaml


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        # Support the common ${VAR:-fallback} spelling without making a
        # dataset/model path a hidden repository constant.
        def replace(match: re.Match[str]) -> str:
            name, fallback = match.group(1), match.group(2)
            return os.environ.get(name, fallback if fallback is not None else match.group(0))
        value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}", replace, value)
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _parse_scalar(value: str) -> Any:
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        return value
    return value if parsed is None and value.lower() not in {"null", "none"} else parsed


def set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {config_path}")
    defaults = data.pop("defaults", []) or []
    if isinstance(defaults, str):
        defaults = [defaults]
    result: dict[str, Any] = {}
    for default in defaults:
        default_path = Path(default)
        if not default_path.is_absolute():
            default_path = config_path.parent / default_path
        default_config = load_config(default_path)
        default_config.pop("_meta", None)
        deep_update(result, default_config)
    deep_update(result, _expand_env(data))
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override must be key=value, got {override!r}")
        key, raw_value = override.split("=", 1)
        set_dotted(result, key, _parse_scalar(raw_value))
    result.setdefault("_meta", {})["config_path"] = str(config_path)
    return result


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
