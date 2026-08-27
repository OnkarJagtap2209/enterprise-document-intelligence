"""Embed and upsert one persisted Phase 3 chunk artifact."""

import argparse
import logging
from pathlib import Path

from enterprise_rag.config import get_settings
from enterprise_rag.embeddings import SentenceTransformerEmbeddingService
from enterprise_rag.stores import ChromaVectorStore, ChunkIndexer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunk_artifact", type=Path)
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    embedding_service = SentenceTransformerEmbeddingService(
        settings.embedding_model_name,
        batch_size=settings.embedding_batch_size,
    )
    vector_store = ChromaVectorStore(
        db_path=settings.chroma_db_path,
        collection_name=settings.chroma_collection_name,
        embedding_model=settings.embedding_model_name,
        embedding_dimension=embedding_service.dimension,
    )
    result = ChunkIndexer(
        embedding_service, vector_store, settings.embedding_batch_size
    ).index(args.chunk_artifact)
    print(
        f"artifact={result.artifact_path}\n"
        f"chunks={result.chunk_count}\n"
        f"embeddings={result.embeddings_created}\n"
        f"dimension={result.embedding_dimension}\n"
        f"collection={result.collection_name}\n"
        f"records={result.collection_record_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
