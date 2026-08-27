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
    chunks_dir: Path
    chroma_db_path: Path
    chroma_collection_name: str
    embedding_model_name: str
    embedding_batch_size: int
    semantic_top_k: int
    bm25_top_k: int
    hybrid_top_k: int
    hybrid_candidate_depth: int
    rrf_k: int
    chunk_max_chars: int
    chunk_overlap_chars: int
    log_level: str

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> Settings:
        _load_env_file(Path(env_file))
        return cls(
            app_env=os.getenv("APP_ENV", "development"),
            document_dir=Path(os.getenv("DOCUMENT_DIR", "data/documents")),
            extracted_dir=Path(os.getenv("EXTRACTED_DIR", "data/extracted")),
            chunks_dir=Path(os.getenv("CHUNKS_DIR", "data/chunks")),
            chroma_db_path=Path(os.getenv("CHROMA_DB_PATH", "chroma_db")),
            chroma_collection_name=os.getenv(
                "CHROMA_COLLECTION_NAME", "enterprise_financial_chunks"
            ),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
            semantic_top_k=int(os.getenv("SEMANTIC_TOP_K", "5")),
            bm25_top_k=int(os.getenv("BM25_TOP_K", "10")),
            hybrid_top_k=int(os.getenv("HYBRID_TOP_K", "5")),
            hybrid_candidate_depth=int(
                os.getenv("HYBRID_CANDIDATE_DEPTH", "10")
            ),
            rrf_k=int(os.getenv("RRF_K", "60")),
            chunk_max_chars=int(os.getenv("CHUNK_MAX_CHARS", "1600")),
            chunk_overlap_chars=int(os.getenv("CHUNK_OVERLAP_CHARS", "200")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one settings instance for the current process."""
    return Settings.from_env()
