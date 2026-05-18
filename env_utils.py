from __future__ import annotations

import os
from pathlib import Path


def load_project_env(start: str | os.PathLike | None = None, filename: str = ".env") -> Path | None:
    """Load key=value pairs from the nearest project .env without overriding real env vars."""
    current = Path(start or os.getcwd()).resolve()
    if current.is_file():
        current = current.parent

    for directory in (current, *current.parents):
        env_path = directory / filename
        if env_path.is_file():
            _load_env_file(env_path)
            return env_path
    return None


def _load_env_file(env_path: Path) -> None:
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
