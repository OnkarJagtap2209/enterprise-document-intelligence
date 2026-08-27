"""Environment-backed application settings."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without replacing process variables."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration shared by future application components."""

    app_env: str
    document_dir: Path
    extracted_dir: Path
    chroma_db_path: Path
    log_level: str

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> Settings:
        _load_env_file(Path(env_file))
        return cls(
            app_env=os.getenv("APP_ENV", "development"),
            document_dir=Path(os.getenv("DOCUMENT_DIR", "data/documents")),
            extracted_dir=Path(os.getenv("EXTRACTED_DIR", "data/extracted")),
            chroma_db_path=Path(os.getenv("CHROMA_DB_PATH", "chroma_db")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one settings instance for the current process."""
    return Settings.from_env()
