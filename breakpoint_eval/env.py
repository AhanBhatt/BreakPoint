from __future__ import annotations

import os
from pathlib import Path


TRUTHY = {"1", "true", "yes", "on"}


def load_env(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load simple KEY=VALUE entries from .env without adding a runtime dependency."""

    env_path = Path(path) if path else find_env()
    if not env_path or not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


def find_env(start: str | Path | None = None) -> Path | None:
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for directory in [current, *current.parents]:
        candidate = directory / ".env"
        if candidate.exists():
            return candidate
    return None


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUTHY


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
